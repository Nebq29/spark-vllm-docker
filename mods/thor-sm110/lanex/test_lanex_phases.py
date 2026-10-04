"""test_lanex_phases.py — split one lanex all_reduce into its parts.

Launch exactly like test_lanex1.py (2 ranks, gloo control plane). Prints, on rank 0:
  - the GPU's memory-model attributes (cuDeviceGetAttribute)
  - median times for:
      d2h / h2d / add          GPU copies and the add (CUDA events)
      xchg <buffer type>       lanex_xchg on: torch pinned (cudaHostAlloc), pageable
                               malloc, mmap+cudaHostRegister, cuMemAllocManaged
      peer-wait                part of xchg spent waiting for the peer's first byte
      all_reduce               the full lanex.all_reduce (LANEX_ASYNC / LANEX_HOSTBUF as set)

Needs lanex_core.so built from lanex_core.c v2 (exports lanex_last_stats).
Env: TEST_MB (default 17.3), TEST_IT (default 40)
"""
import ctypes
import os
import statistics
import time

import torch
import torch.distributed as dist

import lanex

MB = float(os.environ.get("TEST_MB", "17.3"))
IT = int(os.environ.get("TEST_IT", "40"))
STRESS = int(os.environ.get("TEST_STRESS", "50"))
BF = torch.bfloat16


def gpu_us(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(IT):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def xchg_us(lx, A, B, nb):
    tot, wait = [], []
    w, t = ctypes.c_double(), ctypes.c_double()
    for i in range(IT + 2):
        dist.barrier()
        rc = lanex._lib.lanex_xchg(lx._h,
                                   ctypes.cast(A.data_ptr(), ctypes.c_char_p),
                                   ctypes.cast(B.data_ptr(), ctypes.c_char_p),
                                   nb)
        if rc != 0:
            raise RuntimeError(f"lanex_xchg errno={rc}")
        lanex._lib.lanex_last_stats(lx._h, ctypes.byref(w), ctypes.byref(t))
        if i >= 2:  # skip warm-up (first touch of pages)
            tot.append(t.value)
            wait.append(w.value)
    return statistics.median(tot), statistics.median(wait)


def both_ok(ok):
    flag = torch.tensor([1 if ok else 0])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() == 1


def buffer_case(lx, mode, x, C, nb, stream):
    """Time d2h, h2d and xchg for one host-buffer type. Returns dict or error str."""
    err = None
    try:
        A, ka = lanex.host_buffer(nb, mode)
        B, kb = lanex.host_buffer(nb, mode)
    except Exception as e:
        err = repr(e)
    if not both_ok(err is None):
        return err or "peer rank failed to allocate"
    if mode == "hostmem":
        y = x.clone()
        d2h = gpu_us(lambda: lanex.gpu_copy_out(x, A))
        h2d = gpu_us(lambda: lanex.gpu_add_in(y, B))  # NOTE: includes the add
    elif mode == "managed":
        d2h = gpu_us(lambda: lanex.copy_async(A.data_ptr(), x.data_ptr(), nb, stream))
        h2d = gpu_us(lambda: lanex.copy_async(C.data_ptr(), B.data_ptr(), nb, stream))
    else:
        d2h = gpu_us(lambda: A.view(BF).copy_(x, non_blocking=True))
        h2d = gpu_us(lambda: C.view(BF).copy_(B.view(BF), non_blocking=True))
    torch.cuda.synchronize()
    xt, xw = xchg_us(lx, A, B, nb)
    res = dict(d2h=d2h, h2d=h2d, xchg=xt, wait=xw, pinned=bool(A.is_pinned()))
    lanex.free_host_buffer(A, ka)
    lanex.free_host_buffer(B, kb)
    return res


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(0)
    lx = lanex.LaneX(rank)
    stream = torch.cuda.current_stream()

    nb = int(MB * 1024 * 1024) // 2 * 2
    x = torch.randn(nb // 2, dtype=BF, device="cuda")
    y = x.clone()
    C = torch.empty(nb, dtype=torch.uint8, device="cuda")
    attrs = lanex.cuda_mem_attrs()

    rows = {}
    rows["pinned (alloc)"] = buffer_case(lx, "alloc", x, C, nb, stream)
    # pageable malloc: no async GPU copies possible, xchg only
    Am = torch.ones(nb, dtype=torch.uint8)
    Bm = torch.zeros(nb, dtype=torch.uint8)
    xt, xw = xchg_us(lx, Am, Bm, nb)
    rows["pageable malloc"] = dict(d2h=None, h2d=None, xchg=xt, wait=xw, pinned=False)
    rows["registered"] = buffer_case(lx, "register", x, C, nb, stream)
    rows["managed"] = buffer_case(lx, "managed", x, C, nb, stream)
    rows["hostmem"] = buffer_case(lx, "hostmem", x, C, nb, stream)
    add = gpu_us(lambda: y.add_(C.view(BF)))

    # Stress correctness of the configured mode: fresh data every iteration, same
    # seed on both ranks, so every result must equal exactly 2*a.
    bad = 0
    q8 = lanex.LANEX_COMPRESS == "int8" and lanex.LANEX_HOSTBUF == "hostmem"
    worst = 0.0
    for i in range(STRESS):
        g = torch.Generator(device="cuda").manual_seed(1000 + i)
        a = torch.randn(nb // 2, dtype=BF, device="cuda", generator=g)
        out = lx.all_reduce(a.clone())
        if not q8:
            if not torch.equal(out, (a.float() * 2).to(BF)):
                bad += 1
            continue
        # int8 mode: (1) close to the int8 reference, (2) bit-identical on both ranks
        ref = (lanex.q8_roundtrip_ref(a) * 2).to(BF).float()
        err = ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
        worst = max(worst, err)
        chk = torch.tensor([float(out.view(torch.int16).to(torch.float64).sum().item())],
                           dtype=torch.float64)
        hi, lo = chk.clone(), chk.clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        if err > 2e-2 or hi.item() != lo.item():
            bad += 1
    if q8 and rank == 0:
        print(f"  int8 mode: worst max-abs error vs int8 reference = {worst:.2e} "
              f"(relative to max |x|); ranks bit-identical in every passing iteration")

    z = x.clone()
    for _ in range(3):
        lx.all_reduce(z)
    torch.cuda.synchronize()
    ts = []
    for _ in range(IT):
        dist.barrier()
        t0 = time.perf_counter()
        lx.all_reduce(z)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ar = statistics.median(ts)

    if rank == 0:
        print(f"GPU memory attributes: {attrs}")
        print(f"{MB} MB ({nb} B), median of {IT}:")
        print(f"  {'buffer':<16} {'d2h us':>8} {'h2d us':>8} {'xchg us':>8} "
              f"{'GB/s':>6} {'peer-wait':>9}  is_pinned")
        for name, r in rows.items():
            if isinstance(r, str):
                print(f"  {name:<16} FAILED: {r}")
                continue
            f = lambda v: f"{v:8.0f}" if v is not None else f"{'-':>8}"
            print(f"  {name:<16} {f(r['d2h'])} {f(r['h2d'])} {r['xchg']:8.0f} "
                  f"{nb/r['xchg']/1e3:6.2f} {r['wait']:9.0f}  {r['pinned']}")
        print(f"  add {add:.0f} us   (hostmem h2d column = in-place add from host memory)")
        print(f"  stress correctness [{lanex.LANEX_HOSTBUF}]: "
              f"{STRESS - bad}/{STRESS} exact")
        mode = "async" if lanex.LANEX_ASYNC else "sync"
        print(f"  all_reduce {ar:.0f} us  [{mode}, hostbuf={lanex.LANEX_HOSTBUF}]")
        print("TEST_LANEX_PHASES_DONE", flush=True)
    dist.barrier()
    lx.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

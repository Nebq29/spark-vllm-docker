"""test_bufclass.py (round 6) — which host-buffer setting makes the CPU send fast?

Launch exactly like test_lanex1.py (2 ranks, gloo control plane). Standalone, no vLLM.
Times lanex_xchg (17.3 MB, 4 lanes) for each hostmem allocation/page-size setting,
used the way lanex really uses it (the GPU writes A and reads B through the page
tables), plus pinned for reference. Two interleaved passes.

For every send buffer A it also prints what the kernel actually gave it, from
/proc/self/smaps: mapping size, resident kB, AnonHugePages kB, VmFlags, backing.
  'sh' in flags = shared mapping; 'hg'/'nh' = MADV_HUGEPAGE/NOHUGEPAGE.

Needs lanex.py (round 6) and lanex_core.so (v3; zerocopy stays off).
Env: TEST_MB (default 17.3), TEST_IT (default 30)
"""
import ctypes
import os
import statistics

import torch
import torch.distributed as dist

import lanex

MB = float(os.environ.get("TEST_MB", "17.3"))
IT = int(os.environ.get("TEST_IT", "30"))
BF = torch.bfloat16


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
            return None, None, f"errno {rc} ({os.strerror(rc)})"
        lanex._lib.lanex_last_stats(lx._h, ctypes.byref(w), ctypes.byref(t))
        if i >= 2:
            tot.append(t.value)
            wait.append(w.value)
    return statistics.median(tot), statistics.median(wait), None


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(0)
    lx = lanex.LaneX(rank)
    lanex._lib.lanex_set_zerocopy(lx._h, 0)
    nb = int(MB * 1024 * 1024) // 2 * 2
    x = torch.randn(nb // 2, dtype=BF, device="cuda")
    x_cpu = x.view(torch.uint8).cpu()

    def hostmem(alloc, thp, gpu=True):
        def build():
            A, ka = lanex.hostmem_buffer(nb, alloc, thp)
            B, kb = lanex.hostmem_buffer(nb, alloc, thp)
            if gpu:
                lanex.gpu_copy_out(x, A)      # GPU writes A (as lanex does)
                y = x.clone()
                lanex.gpu_add_in(y, B)        # GPU reads B (as lanex does)
            else:
                A.copy_(x_cpu)
            torch.cuda.synchronize()
            return A, B, (ka, kb)
        return build

    def pinned():
        A, ka = lanex.host_buffer(nb, "alloc")
        B, kb = lanex.host_buffer(nb, "alloc")
        A.copy_(x_cpu)
        return A, B, (ka, kb)

    variants = [
        ("malloc default",        hostmem("malloc", "default")),
        ("malloc nohuge",         hostmem("malloc", "nohuge")),
        ("malloc huge",           hostmem("malloc", "huge")),
        ("mmap default",          hostmem("mmap", "default")),
        ("mmap nohuge",           hostmem("mmap", "nohuge")),
        ("mmap huge",             hostmem("mmap", "huge")),
        ("mmap_shared default",   hostmem("mmap_shared", "default")),
        ("mmap_shared huge (r4)", hostmem("mmap_shared", "huge")),
        ("malloc nohuge, no GPU", hostmem("malloc", "nohuge", gpu=False)),
        ("pinned / pinned",       pinned),
    ]

    only = [s.strip() for s in os.environ.get("TEST_ONLY", "").split(",") if s.strip()]
    if only:
        variants = [v for v in variants if v[0] in only]
    res = {name: [] for name, _ in variants}
    vma = {}
    for p in range(2):
        for name, build in variants:
            err = None
            try:
                A, B, keep = build()
            except Exception as e:
                err = repr(e)
            flag = torch.tensor([0 if err else 1])
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if flag.item() == 0:
                res[name].append((None, None, err or "peer failed to allocate"))
                continue
            if p == 0:
                vma[name] = lanex.vma_info(A.data_ptr())
            t, w, xerr = xchg_us(lx, A, B, nb)
            flag = torch.tensor([0 if xerr else 1])
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            res[name].append((t, w, xerr or (None if flag.item() else "peer xchg failed")))
            if flag.item() == 0:
                if rank == 0:
                    print(f"STOP: {name}: {xerr or 'peer xchg failed'}; sockets may be "
                          f"out of sync", flush=True)
                _print(rank, res, vma, nb)
                return
            del A, B, keep
    _print(rank, res, vma, nb)
    if rank == 0 and hasattr(lanex._lib, "lanex_sockbuf"):
        s, r = ctypes.c_int(), ctypes.c_int()
        lanex._lib.lanex_sockbuf.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                             ctypes.POINTER(ctypes.c_int)]
        lanex._lib.lanex_sockbuf(lx._h, ctypes.byref(s), ctypes.byref(r))
        print(f"LANEX_SOCKBUF={os.environ.get('LANEX_SOCKBUF', '(unset=32M)')}  "
              f"effective SO_SNDBUF={s.value} SO_RCVBUF={r.value}", flush=True)
    dist.barrier()
    lx.close()
    dist.destroy_process_group()


def _print(rank, res, vma, nb):
    if rank != 0:
        return
    print(f"xchg {MB} MB, 4 lanes, median of {IT} per pass (us)")
    print(f"  {'variant':<22} {'pass1':>7} {'pass2':>7} {'GB/s':>6} {'p-wait':>6} | "
          f"A mapping: size_kB rss_kB hugepage_kB  backing  flags")
    for name, runs in res.items():
        if not runs:
            continue
        cells, ok, note = [], [], ""
        for r in runs:
            if r[2]:
                cells.append(f"{'ERR':>7}")
                note = r[2]
            else:
                cells.append(f"{r[0]:7.0f}")
                ok.append(r)
        while len(cells) < 2:
            cells.append(f"{'-':>7}")
        gbs = f"{nb / statistics.mean([r[0] for r in ok]) / 1e3:6.2f}" if ok else f"{'-':>6}"
        pw = f"{statistics.mean([r[1] for r in ok]):6.0f}" if ok else f"{'-':>6}"
        v = vma.get(name)
        vs = (f"{v['size']:>8} {v['rss']:>6} {v['huge']:>11}  {v['path']}  {v['flags']}"
              if v else "n/a")
        print(f"  {name:<22} {cells[0]} {cells[1]} {gbs} {pw} | {vs} {note}")
    print("TEST_BUFCLASS_DONE", flush=True)


if __name__ == "__main__":
    main()

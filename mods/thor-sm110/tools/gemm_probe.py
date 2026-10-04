"""gemm_probe.py — which tensor-core GEMM paths work on this GPU, and how fast?

Model-agnostic. Run on ONE board inside the vLLM container (no vLLM server
running, GPU otherwise idle):   python3 gemm_probe.py

For a few prefill-sized shapes it times:
  bf16           torch.matmul (cuBLAS), the reference
  fp8 torch      torch._scaled_mm, per-tensor FP8 e4m3
  fp8 vllm       vLLM cutlass_scaled_mm (FP8)
  nvfp4 vllm     vLLM scaled_fp4_quant + cutlass_scaled_fp4_mm (NVFP4)
  nvfp4 torch    torch._scaled_mm with float4_e2m1fn_x2 + e4m3 block scales
  mxfp4 torch    torch._scaled_mm with float4_e2m1fn_x2 + e8m0 block scales
Each path is tried independently; failures are reported, not fatal.
Prints TFLOPS and a numerical sanity check (relative error vs a float reference).
Also prints versions and vLLM's own "is this supported on this capability" answers.
"""
import math
import traceback

import torch

SHAPES = [  # (M tokens, N, K)
    (2048, 4096, 4096),
    (8192, 4096, 4096),
    (8192, 2048, 7168),
    (8192, 7168, 2048),
]
dev = "cuda"


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters  # ms


def relerr(out, ref):
    out, ref = out.float(), ref.float()
    if not torch.isfinite(out).all():
        return float("nan")
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def to_blocked(m):
    """torch's swizzled 128x4 block-scale layout (as in torch's own tests)."""
    rows, cols = m.shape
    nr, nc = math.ceil(rows / 128), math.ceil(cols / 4)
    p = torch.zeros(nr * 128, nc * 4, dtype=m.dtype, device=m.device)
    p[:rows, :cols] = m
    blocks = p.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def main():
    torch.manual_seed(0)
    cap = torch.cuda.get_device_capability()
    capi = cap[0] * 10 + cap[1]
    print(f"device: {torch.cuda.get_device_name()}  capability sm_{capi}")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    for mod in ("vllm", "triton", "flashinfer", "b12x"):
        try:
            m = __import__(mod)
            print(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as ex:
            print(f"{mod}: not importable ({type(ex).__name__})")

    ops = None
    try:
        from vllm import _custom_ops as ops
        for q in ("cutlass_scaled_mm_supports_fp8", "cutlass_scaled_mm_supports_block_fp8",
                  "cutlass_scaled_mm_supports_fp4"):
            f = getattr(ops, q, None)
            try:
                print(f"vllm {q}({capi}) = {f(capi) if f else 'n/a'}")
            except Exception as ex:
                print(f"vllm {q}: error {ex!r}")
    except Exception as ex:
        print(f"vllm _custom_ops not importable: {ex!r}")

    one = torch.tensor(1.0, device=dev)
    print("\n  M     N     K   | path          ms      TFLOPS  relerr  note")
    for M, N, K in SHAPES:
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        flop = 2 * M * N * K
        ref = None

        def report(name, fn, check=None):
            try:
                ms = bench(fn)
                err = check() if check else float("nan")
                print(f"{M:5d} {N:5d} {K:5d} | {name:<12} {ms:8.3f} {flop/ms/1e9:9.1f}  {err:6.3f}")
            except Exception as ex:
                msg = str(ex).strip().splitlines()[0][:90] if str(ex).strip() else type(ex).__name__
                print(f"{M:5d} {N:5d} {K:5d} | {name:<12}      FAILED: {msg}")

        report("bf16", lambda: a @ w.t())
        if ref is None:
            ref = (a[:256].float() @ w.float().t())

        a8 = a.to(torch.float8_e4m3fn)
        w8 = w.to(torch.float8_e4m3fn)
        ref8 = a8[:256].float() @ w8.float().t()
        report("fp8 torch",
               lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
               lambda: relerr(torch._scaled_mm(a8[:256], w8.t(), scale_a=one, scale_b=one,
                                               out_dtype=torch.bfloat16), ref8))
        if ops is not None and hasattr(ops, "cutlass_scaled_mm"):
            s1 = torch.ones(1, device=dev, dtype=torch.float32)
            report("fp8 vllm",
                   lambda: ops.cutlass_scaled_mm(a8, w8.t(), s1, s1, torch.bfloat16),
                   lambda: relerr(ops.cutlass_scaled_mm(a8[:256].contiguous(), w8.t(), s1, s1,
                                                        torch.bfloat16), ref8))

        if ops is not None and hasattr(ops, "scaled_fp4_quant") and hasattr(ops, "cutlass_scaled_fp4_mm"):
            def nvfp4_setup(x):
                gs = (448.0 * 6.0) / x.abs().max().float()
                q, sf = ops.scaled_fp4_quant(x, gs.reshape(1))
                return q, sf, gs
            try:
                aq, asf, ags = nvfp4_setup(a)
                wq, wsf, wgs = nvfp4_setup(w)
                alpha = (1.0 / (ags * wgs)).reshape(1).float()
                a2q, a2sf, a2gs = nvfp4_setup(a[:256].contiguous())
                alpha2 = (1.0 / (a2gs * wgs)).reshape(1).float()
                report("nvfp4 vllm",
                       lambda: ops.cutlass_scaled_fp4_mm(aq, wq, asf, wsf, alpha, torch.bfloat16),
                       lambda: relerr(ops.cutlass_scaled_fp4_mm(a2q, wq, a2sf, wsf, alpha2,
                                                                torch.bfloat16), ref))
            except Exception as ex:
                print(f"{M:5d} {N:5d} {K:5d} | nvfp4 vllm        FAILED (quant): "
                      f"{str(ex).strip().splitlines()[0][:80] if str(ex).strip() else type(ex).__name__}")

        f4 = getattr(torch, "float4_e2m1fn_x2", None)
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if f4 is not None:
            # random packed fp4 (e2m1 has no NaN/Inf), unit scales: speed only
            ap = torch.randint(0, 256, (M, K // 2), device=dev, dtype=torch.uint8).view(f4)
            wp = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8).view(f4)
            sa16 = to_blocked(torch.ones(M, K // 16, device=dev).to(torch.float8_e4m3fn))
            sw16 = to_blocked(torch.ones(N, K // 16, device=dev).to(torch.float8_e4m3fn))
            report("nvfp4 torch",
                   lambda: torch._scaled_mm(ap, wp.t(), sa16, sw16, out_dtype=torch.bfloat16))
            if e8m0 is not None:
                sa32 = to_blocked(torch.full((M, K // 32), 127, device=dev, dtype=torch.uint8).view(e8m0))
                sw32 = to_blocked(torch.full((N, K // 32), 127, device=dev, dtype=torch.uint8).view(e8m0))
                report("mxfp4 torch",
                       lambda: torch._scaled_mm(ap, wp.t(), sa32, sw32, out_dtype=torch.bfloat16))
        else:
            print(f"{M:5d} {N:5d} {K:5d} | fp4 torch         n/a (torch has no float4_e2m1fn_x2)")
        print()
    print("relerr: fp8 rows vs exact float product of the fp8 inputs (expect ~0.00x);")
    print("nvfp4 vllm vs bf16 product (expect ~0.1-0.2 from 4-bit rounding; NaN or >0.5 = broken);")
    print("torch fp4 rows are speed-only (random data).")
    print("GEMM_PROBE_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()

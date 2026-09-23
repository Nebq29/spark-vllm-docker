#!/usr/bin/env python3
"""Extend the existing E8M0 -> fp32 upcast to CUDA SM110 (Jetson Thor).

The image ALREADY contains the correct fix and rationale:

    # Triton cannot currently bind E8M0 scale tensors directly. On ROCm,
    # DeepSeek-V4 checkpoints store block scales in exponent-only E8M0 format,
    # so decode them to fp32 before launching the kernel.
    if current_platform.is_rocm() or current_platform.is_xpu():
        if As.dtype == torch.float8_e8m0fnu:
            As = _upcast_e8m0_to_fp32(As).contiguous()
        if Bs.dtype == torch.float8_e8m0fnu:
            Bs = _upcast_e8m0_to_fp32(Bs).contiguous()

Our DeepSeek-V4-Flash-0731 checkpoint also ships UE8M0 block scales
(config.quantization_config.scale_fmt == "ue8m0"), and Triton 3.7 on CUDA has the
same pointer-binding limitation. So the same decode is required on Thor.

This patch adds `or current_platform.get_device_capability().major == 11` to that
gate, reusing the existing helper. Idempotent.
"""
import ast
import sys

FP8_UTILS = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py"

OLD_GATE = "    if current_platform.is_rocm() or current_platform.is_xpu():\n"
NEW_GATE = (
    "    if (\n"
    "        current_platform.is_rocm()\n"
    "        or current_platform.is_xpu()\n"
    "        # [THOR-OVERLAY] Thor (SM110) DSv4 checkpoints also carry UE8M0 block\n"
    "        # scales; Triton 3.7 cannot bind float8_e8m0fnu pointers on CUDA either.\n"
    "        or current_platform.get_device_capability().major == 11\n"
    "    ):\n"
)
MARK = "[THOR-OVERLAY] Thor (SM110) DSv4 checkpoints also carry UE8M0 block"

src = open(FP8_UTILS).read()
if MARK in src:
    print("e8m0 CUDA/SM110 gate already extended")
else:
    if OLD_GATE not in src:
        print("ERROR: existing ROCm/XPU e8m0 gate not found — image differs from "
              "expectation; do not guess a replacement", file=sys.stderr)
        sys.exit(2)
    if src.count(OLD_GATE) != 1:
        print(f"ERROR: expected exactly 1 gate occurrence, found {src.count(OLD_GATE)}",
              file=sys.stderr)
        sys.exit(2)
    src = src.replace(OLD_GATE, NEW_GATE, 1)
    ast.parse(src)
    open(FP8_UTILS, "w").write(src)
    print("extended e8m0->fp32 upcast gate to include CUDA SM110")

# ---- verify the helper exists and the gate now covers SM110 -------------------
try:
    from vllm.model_executor.layers.quantization.utils import fp8_utils as U
    assert hasattr(U, "_upcast_e8m0_to_fp32"), "helper missing"
    print("helper present: _upcast_e8m0_to_fp32")
    import inspect
    body = inspect.getsource(U.w8a8_triton_block_scaled_mm)
    assert "major == 11" in body, "gate not applied to the function"
    print("gate applied inside w8a8_triton_block_scaled_mm")
except Exception as e:
    print("VERIFY FAILED:", type(e).__name__, e, file=sys.stderr)
    sys.exit(3)

# ---- GPU smoke test (skipped when no GPU, e.g. during docker build) ---------
try:
    import torch
    if not torch.cuda.is_available():
        print("no GPU in this environment; skipping e8m0 smoke test")
        raise SystemExit(0)
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )
    torch.manual_seed(0)
    M, N, K = 32, 64, 64
    bn, bk = 32, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    # shapes must satisfy: As.shape[:-1]==A.shape[:-1], cdiv(K,bk)==As.shape[-1],
    #                    cdiv(N,bn)==Bs.shape[0], cdiv(K,bk)==Bs.shape[1]
    As = torch.ones(M, K // bk, device="cuda").to(torch.float8_e8m0fnu)
    Bs = torch.ones(N // bn, K // bk, device="cuda").to(torch.float8_e8m0fnu)
    out = w8a8_triton_block_scaled_mm(A, B, As, Bs, [bn, bk], torch.bfloat16)
    print("smoke OK:", tuple(out.shape), out.dtype,
          "finite:", bool(torch.isfinite(out).all()))
    ref = (A.to(torch.float32) @ B.to(torch.float32).t()).to(torch.bfloat16)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"vs unit-scale reference maxdiff: {diff:.5f}")
    print("E8M0 TRITON PATH VALID" if diff < 1.0 else "E8M0 TRITON PATH SUSPECT")
except SystemExit:
    raise
except Exception as e:
    print("SMOKE FAILED:", type(e).__name__, e, file=sys.stderr)
    sys.exit(3)

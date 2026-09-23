#!/usr/bin/env python3
"""Let the Triton block-scaled-MM kernel accept torch.float8_e8m0fnu scales.

Root cause: DeepSeek-V4 FP4 checkpoints store FP8 linear weight scales as
torch.float8_e8m0fnu (vllm/models/deepseek_v4/quant_config.py:
``is_scale_e8m0 == (expert_dtype == "fp4")``). Triton 3.7.1's
type_canonicalisation_dict has no entry for float8_e8m0fnu, so
triton._utils.canonicalize_dtype raises:

    KeyError: 'float8_e8m0fnu'

Fix: widen e8m0 scales to float32 immediately before the Triton launch.
e8m0 is a pure power-of-two (8-bit exponent, no mantissa) scale, so the
widening is EXACT -- verified round-trip on the target image. The Triton
kernel then multiplies with the float32 scale as it already does for
float32-scaled (Flash-Base / fp8-expert) checkpoints.

Applied at the Triton scaled_mm kernel boundary so it covers every caller
without touching the DeepGEMM or CUTLASS paths.

Idempotent. Refuses to write unparsable Python.
"""
import ast
import sys

TRITON_MM = ("/usr/local/lib/python3.12/dist-packages/vllm/"
             "model_executor/kernels/linear/scaled_mm/triton.py")
MARK = "[THOR-OVERLAY] e8m0 scale widening"

ANCHOR = (
    "    def apply_block_scaled_mm(\n"
    "        self,\n"
    "        A: torch.Tensor,\n"
    "        B: torch.Tensor,\n"
    "        As: torch.Tensor,\n"
    "        Bs: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
)
INJECT = (
    "        # " + MARK + "\n"
    "        # Triton 3.7.1 cannot canonicalise torch.float8_e8m0fnu. e8m0 is a\n"
    "        # power-of-two scale, so widening to float32 is exact.\n"
    "        if getattr(As, \"dtype\", None) == torch.float8_e8m0fnu:\n"
    "            As = As.to(torch.float32)\n"
    "        if getattr(Bs, \"dtype\", None) == torch.float8_e8m0fnu:\n"
    "            Bs = Bs.to(torch.float32)\n"
)

src = open(TRITON_MM).read()

if MARK in src:
    print("already patched:", TRITON_MM)
    sys.exit(0)

if ANCHOR not in src:
    print("ERROR: apply_block_scaled_mm anchor not found", file=sys.stderr)
    sys.exit(2)

patched = src.replace(ANCHOR, ANCHOR + INJECT, 1)
ast.parse(patched)  # refuse unparsable output
open(TRITON_MM, "w").write(patched)
print("patched:", TRITON_MM)

# verify: patch is present and syntactically valid (no import/reload here --
# reloading a vllm module re-registers torch.ops and raises "duplicate registration")
try:
    check = open(TRITON_MM).read()
    assert MARK in check, "marker missing after write"
    assert "torch.float8_e8m0fnu" in check, "e8m0 guard missing"
    ast.parse(check)
    print("verified: marker present, file parses")
except AssertionError as e:
    print("VERIFY FAILED:", e, file=sys.stderr)
    sys.exit(3)

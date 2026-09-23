#!/usr/bin/env python3
"""Port the upstream SM110 Triton fp8_einsum into this image and rewire O-proj.

Problem: on Thor the DSv4 O-projection calls `vllm.utils.deep_gemm.fp8_einsum`
directly, which asserts in DeepGEMM's layout heuristics
(`layout.hpp:39: t.dim() == N`) for this checkpoint's scale layout.

Upstream's Thor SM110 branch (vllm-project/vllm @ 9c9fef7f) solves this with a
dedicated Triton fallback:
  vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py
    deepseek_v4_fp8_einsum_config(major, minor)  -> recipe + tma_aligned
    deepseek_v4_fp8_einsum(...)                 -> Triton on (11,0), DeepGEMM elsewhere
  and o_proj.py routes through it.

This overlay:
  1. installs that fp8_einsum.py module (copied verbatim from the Thor branch)
  2. points compute_fp8_einsum_recipe() at deepseek_v4_fp8_einsum_config
  3. swaps the fp8_einsum() call for deepseek_v4_fp8_einsum()

All of the module's imports already exist in this image. Idempotent.
"""
import ast
import os
import shutil
import sys

VP = "/usr/local/lib/python3.12/dist-packages/vllm"
SRC_MOD = "/opt/thor_overlay/thor_fp8_einsum.py"
DST_MOD = f"{VP}/models/deepseek_v4/nvidia/ops/fp8_einsum.py"
OPROJ = f"{VP}/models/deepseek_v4/nvidia/ops/o_proj.py"

changed = []

# ---- 1. install the Thor fp8_einsum module ---------------------------------
if not os.path.exists(SRC_MOD):
    print(f"ERROR: {SRC_MOD} not found", file=sys.stderr)
    sys.exit(2)
mod_src = open(SRC_MOD).read()
ast.parse(mod_src)
cur = open(DST_MOD).read() if os.path.exists(DST_MOD) else ""
if cur != mod_src:
    shutil.copyfile(SRC_MOD, DST_MOD)
    changed.append(f"installed {DST_MOD} ({len(mod_src)} bytes)")
else:
    print("fp8_einsum module already installed")

# ---- 2/3. rewire o_proj.py -------------------------------------------------
op = open(OPROJ).read()
if "deepseek_v4_fp8_einsum" not in op:
    # 2a. imports
    imp_old = "from vllm.utils.deep_gemm import fp8_einsum\n"
    imp_new = (
        "from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (  # [THOR-OVERLAY]\n"
        "    deepseek_v4_fp8_einsum,\n"
        "    deepseek_v4_fp8_einsum_config,\n"
        ")\n"
    )
    if imp_old not in op:
        print("ERROR: fp8_einsum import anchor not found in o_proj.py", file=sys.stderr)
        sys.exit(2)
    op = op.replace(imp_old, imp_new, 1)

    # 2b. recipe selection -> Thor-aware config
    rec_old = (
        "    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)\n"
        "    tma_aligned_scales = cap.major >= 10\n"
        "    return einsum_recipe, tma_aligned_scales\n"
    )
    rec_new = (
        "    # [THOR-OVERLAY] route SM110 to the Triton recipe (1,128,128)/non-TMA\n"
        "    return deepseek_v4_fp8_einsum_config(cap.major, cap.minor)\n"
    )
    if rec_old not in op:
        print("ERROR: recipe selection block not found in o_proj.py", file=sys.stderr)
        sys.exit(2)
    op = op.replace(rec_old, rec_new, 1)

    # 2c. the einsum call itself
    call_old = (
        "    fp8_einsum(\n"
        '        "bhr,hdr->bhd",\n'
        "        (o_fp8, o_scale),\n"
        "        (wo_a.weight, weight_scale),\n"
        "        z,\n"
        "        recipe=einsum_recipe,\n"
        "    )\n"
    )
    call_new = (
        "    # [THOR-OVERLAY] Triton fallback on SM110, DeepGEMM elsewhere\n"
        "    deepseek_v4_fp8_einsum(\n"
        "        o_fp8,\n"
        "        o_scale,\n"
        "        wo_a.weight,\n"
        "        weight_scale,\n"
        "        z,\n"
        '        "bhr,hdr->bhd",\n'
        "        list(einsum_recipe),\n"
        "    )\n"
    )
    if call_old not in op:
        print("ERROR: fp8_einsum call site not found in o_proj.py", file=sys.stderr)
        sys.exit(2)
    op = op.replace(call_old, call_new, 1)

    ast.parse(op)
    open(OPROJ, "w").write(op)
    changed.append("rewired o_proj.py to deepseek_v4_fp8_einsum")
else:
    print("o_proj.py already rewired")

print("THOR FP8_EINSUM OVERLAY APPLIED:" if changed else "nothing changed")
for c in changed:
    print("  -", c)

# ---- verify ---------------------------------------------------------------
try:
    from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
        deepseek_v4_fp8_einsum,
        deepseek_v4_fp8_einsum_config,
    )
    print("config(11,0):", deepseek_v4_fp8_einsum_config(11, 0))
    print("config(10,0):", deepseek_v4_fp8_einsum_config(10, 0))
    from vllm.models.deepseek_v4.nvidia.ops import o_proj
    import inspect
    assert "deepseek_v4_fp8_einsum" in inspect.getsource(o_proj.deep_gemm_fp8_o_proj)
    print("o_proj routed through Thor einsum: OK")
except Exception as e:
    print("VERIFY FAILED:", type(e).__name__, e, file=sys.stderr)
    sys.exit(3)

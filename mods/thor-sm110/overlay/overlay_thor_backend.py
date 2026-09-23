#!/usr/bin/env python3
"""Overlay the upstream Thor SM110 DSv4 sparse-MLA (Triton) backend onto a Thor vLLM image.

Source of truth: vllm-project/vllm commit 9c9fef7f018578e396372d363503edd83ccca85e
("Merge upstream main into Thor SM110 adaptation"), which ships
vllm/models/deepseek_v4/nvidia/thor.py -- a Triton sparse-MLA fallback declaring
supports_compute_capability == DeviceCapability(11, 0).

All thor.py imports already exist in the target image (verified separately), so this
is a pure Python overlay:
  1. install thor.py
  2. register THOR_MLA_SPARSE_DSV4 in AttentionBackendEnum
  3. short-circuit _select_dsv4_attn_cls to the Thor class on SM110

Idempotent. Refuses to write unparsable Python.
"""
import ast
import os
import sys

VP = "/usr/local/lib/python3.12/dist-packages/vllm"
THOR_SRC = "/opt/thor_overlay/thor.py"
REGISTRY = f"{VP}/v1/attention/backends/registry.py"
MODEL = f"{VP}/models/deepseek_v4/nvidia/model.py"
THOR_DST = f"{VP}/models/deepseek_v4/nvidia/thor.py"

REG_ANCHOR = (
    '    ROCM_FLASHMLA_SPARSE_DSV4 = (\n'
    '        "vllm.models.deepseek_v4.amd.rocm.DeepseekV4ROCMAiterMLASparseBackend"\n'
    '    )\n'
)
REG_ENTRY = (
    '    THOR_MLA_SPARSE_DSV4 = (\n'
    '        "vllm.models.deepseek_v4.nvidia.thor.DeepseekV4ThorSparseBackend"\n'
    '    )\n'
)

IMPORT_ANCHOR = (
    "from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention\n"
)
IMPORT_NEW = (
    "from vllm.models.deepseek_v4.nvidia.thor import DeepseekV4ThorAttention\n"
)

# inject immediately after this line inside _select_dsv4_attn_cls
FACTORY_ANCHOR = "    device_capability = current_platform.get_device_capability()\n"
FACTORY_INJECT = (
    "    # [THOR-OVERLAY] SM110 -> Triton sparse-MLA fallback (upstream thor.py)\n"
    "    if device_capability is not None and device_capability.major == 11:\n"
    "        return DeepseekV4ThorAttention\n"
)

changed = []

# ---- 1. install thor.py ------------------------------------------------------
if not os.path.exists(THOR_SRC):
    print(f"ERROR: {THOR_SRC} not found", file=sys.stderr)
    sys.exit(2)
thor_src = open(THOR_SRC).read()
ast.parse(thor_src)
cur = open(THOR_DST).read() if os.path.exists(THOR_DST) else ""
if cur != thor_src:
    open(THOR_DST, "w").write(thor_src)
    changed.append(f"installed {THOR_DST} ({len(thor_src)} bytes)")
else:
    print("thor.py already installed")

# ---- 2. register the enum ---------------------------------------------------
rs = open(REGISTRY).read()
if "THOR_MLA_SPARSE_DSV4" not in rs:
    if REG_ANCHOR not in rs:
        print("ERROR: registry anchor not found", file=sys.stderr)
        sys.exit(2)
    rs = rs.replace(REG_ANCHOR, REG_ANCHOR + REG_ENTRY, 1)
    ast.parse(rs)
    open(REGISTRY, "w").write(rs)
    changed.append("registered THOR_MLA_SPARSE_DSV4 in AttentionBackendEnum")
else:
    print("backend enum already registered")

# ---- 3. route SM110 to the Thor class --------------------------------------
ms = open(MODEL).read()
if "DeepseekV4ThorAttention" not in ms:
    if IMPORT_ANCHOR not in ms:
        print("ERROR: import anchor not found in model.py", file=sys.stderr)
        sys.exit(2)
    ms = ms.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_NEW, 1)
    if FACTORY_ANCHOR not in ms:
        print("ERROR: factory anchor not found in model.py", file=sys.stderr)
        sys.exit(2)
    ms = ms.replace(FACTORY_ANCHOR, FACTORY_ANCHOR + FACTORY_INJECT, 1)
    ast.parse(ms)
    open(MODEL, "w").write(ms)
    changed.append("routed SM110 -> DeepseekV4ThorAttention in _select_dsv4_attn_cls")
else:
    print("model.py already routed")

print("OVERLAY APPLIED:" if changed else "nothing changed (already overlaid)")
for c in changed:
    print("  -", c)

# ---- 4. verify -------------------------------------------------------------
try:
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    print("enum has THOR_MLA_SPARSE_DSV4:", hasattr(AttentionBackendEnum, "THOR_MLA_SPARSE_DSV4"))
    from vllm.models.deepseek_v4.nvidia.thor import (
        DeepseekV4ThorSparseBackend,
        DeepseekV4ThorAttention,
    )
    print("supports (11,0):",
          DeepseekV4ThorSparseBackend.supports_compute_capability(DeviceCapability(11, 0)))
    print("backend name:", DeepseekV4ThorSparseBackend.get_name())
    print("has _forward_decode:", hasattr(DeepseekV4ThorAttention, "_forward_decode"))
    print("has _forward_prefill:", hasattr(DeepseekV4ThorAttention, "_forward_prefill"))
except Exception as e:
    print("VERIFY FAILED:", type(e).__name__, e, file=sys.stderr)
    sys.exit(3)

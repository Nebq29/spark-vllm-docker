#!/usr/bin/env python3
"""Route the DSv4 sparse indexer to the Triton MQA-logits fallback on SM110.

Our image's `sparse_attn_indexer.py` is NEWER than the upstream Thor branch
(it carries `compress_ratio`, DCP property caching, JIT-warmup registration), so
the Thor branch's file cannot be dropped in wholesale. Instead we apply the
minimal set of changes that branch made for SM110:

  1. import `is_deep_gemm_supported` and the Triton MQA-logits kernels
  2. compute `use_deep_gemm = is_deep_gemm_supported()` in the op body
  3. prefill: `else:` -> `elif use_deep_gemm:` + Triton `else:` branch
  4. decode: `else:` -> `elif use_deep_gemm:` + Triton `else:` branch
  5. cooperative_topk: `num_rows <= 64` -> `<= 32` (Thor cannot launch the
     thread-block-cluster cooperative top-k at larger sizes; upstream
     commit bd6aa9ae "fix(thor): disable cooperative topk on SM110")
  6. drop the hard `if ... not has_deep_gemm(): raise` guard so the Triton
     path is reachable with VLLM_USE_DEEP_GEMM=0

Idempotent; refuses to write unparsable Python.
"""
import ast
import sys

IDX = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py"
MARK = "[THOR-OVERLAY] Triton MQA-logits fallback"

EDITS = [
    # 1a. deep_gemm import: has_deep_gemm -> is_deep_gemm_supported
    (
        "from vllm.utils.deep_gemm import (\n"
        "    fp8_fp4_mqa_logits,\n"
        "    fp8_fp4_paged_mqa_logits,\n"
        "    has_deep_gemm,\n"
        ")\n",
        "from vllm.utils.deep_gemm import (\n"
        "    fp8_fp4_mqa_logits,\n"
        "    fp8_fp4_paged_mqa_logits,\n"
        "    is_deep_gemm_supported,  # [THOR-OVERLAY]\n"
        ")\n",
    ),
    # 1b. import the Triton kernels
    (
        "from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n",
        "from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton\n"
        "from vllm.v1.attention.ops.mqa_logits_triton import (  # [THOR-OVERLAY]\n"
        "    fp8_mqa_logits_triton,\n"
        "    fp8_paged_mqa_logits_triton,\n"
        ")\n",
    ),
    # 2. define use_deep_gemm right after the topk buffer clear
    (
        "    if not skip_topk_buffer_clear:\n"
        "        topk_indices_buffer[: hidden_states.shape[0]] = -1\n",
        "    if not skip_topk_buffer_clear:\n"
        "        topk_indices_buffer[: hidden_states.shape[0]] = -1\n"
        "    # [THOR-OVERLAY] Triton MQA-logits fallback when DeepGEMM is disabled\n"
        "    use_deep_gemm = is_deep_gemm_supported()\n"
        "    if not use_deep_gemm:\n"
        "        assert not use_fp4_cache, (\n"
        "            \"Triton sparse-MLA fallback does not support FP4 KV cache\"\n"
        "        )\n",
    ),
    # 3. prefill: route the non-XPU branch through Triton when no DeepGEMM
    (
        "                else:\n"
        "                    logits = fp8_fp4_mqa_logits(\n"
        "                        (q_slice_cast, q_scale_slice),\n"
        "                        (k_quant_cast, k_scale_cast),\n"
        "                        weights[chunk.token_start : chunk.token_end],\n"
        "                        cu_seqlen_ks,\n"
        "                        cu_seqlen_ke,\n"
        "                        clean_logits=False,\n"
        "                    )\n",
        "                elif use_deep_gemm:\n"
        "                    logits = fp8_fp4_mqa_logits(\n"
        "                        (q_slice_cast, q_scale_slice),\n"
        "                        (k_quant_cast, k_scale_cast),\n"
        "                        weights[chunk.token_start : chunk.token_end],\n"
        "                        cu_seqlen_ks,\n"
        "                        cu_seqlen_ke,\n"
        "                        clean_logits=False,\n"
        "                    )\n"
        "                else:\n"
        "                    logits = fp8_mqa_logits_triton(\n"
        "                        q_slice_cast,\n"
        "                        (k_quant_cast, k_scale_cast),\n"
        "                        weights[chunk.token_start : chunk.token_end],\n"
        "                        cu_seqlen_ks,\n"
        "                        cu_seqlen_ke,\n"
        "                        clean_logits=False,\n"
        "                    )\n",
    ),
    # 4. decode: route the paged branch through Triton when no DeepGEMM
    (
        "        else:\n"
        "            logits = fp8_fp4_paged_mqa_logits(\n"
        "                (padded_q_quant_cast, padded_q_scale),\n"
        "                kv_cache,\n"
        "                weights[:num_padded_tokens],\n"
        "                seq_lens,\n"
        "                decode_metadata.block_table,\n"
        "                decode_metadata.schedule_metadata,\n"
        "                max_model_len=max_model_len,\n"
        "                clean_logits=False,\n"
        "                indices=decode_metadata.indices,\n"
        "            )\n",
        "        elif use_deep_gemm:\n"
        "            logits = fp8_fp4_paged_mqa_logits(\n"
        "                (padded_q_quant_cast, padded_q_scale),\n"
        "                kv_cache,\n"
        "                weights[:num_padded_tokens],\n"
        "                seq_lens,\n"
        "                decode_metadata.block_table,\n"
        "                decode_metadata.schedule_metadata,\n"
        "                max_model_len=max_model_len,\n"
        "                clean_logits=False,\n"
        "                indices=decode_metadata.indices,\n"
        "            )\n"
        "        else:\n"
        "            logits = fp8_paged_mqa_logits_triton(\n"
        "                padded_q_quant_cast,\n"
        "                kv_cache,\n"
        "                weights[:num_padded_tokens],\n"
        "                seq_lens[:, -1] if seq_lens.ndim == 2 else seq_lens,\n"
        "                decode_metadata.block_table,\n"
        "                max_model_len=attn_metadata_narrowed.max_seq_len,\n"
        "                clean_logits=False,\n"
        "            )\n",
    ),
    # 5. cooperative top-k: Thor CANNOT launch the thread-block-cluster kernel.
    #    Upstream commit bd6aa9ae "fix(thor): disable cooperative topk on SM110"
    #    adds `not is_device_capability_family(110)` to the guard.
    (
        "            and current_platform.has_device_capability(90)\n"
        "            and not current_platform.is_device_capability_family(120)\n",
        "            and current_platform.has_device_capability(90)\n"
        "            and not current_platform.is_device_capability_family(110)  # [THOR-OVERLAY] bd6aa9ae\n"
        "            and not current_platform.is_device_capability_family(120)\n",
    ),
]

src = open(IDX).read()
if MARK in src:
    print("indexer already patched")
else:
    for i, (old, new) in enumerate(EDITS, 1):
        n = src.count(old)
        if n != 1:
            print(f"ERROR: edit #{i} anchor found {n} times (expected 1) — "
                  f"image differs from expectation, aborting without writing",
                  file=sys.stderr)
            sys.exit(2)
        src = src.replace(old, new, 1)

    # 6. replace the hard DeepGEMM requirement with the upstream warning, so the
    #    Triton fallback path is reachable with VLLM_USE_DEEP_GEMM=0
    guard_old = (
        "        if current_platform.is_cuda() and not has_deep_gemm():\n"
        "            raise RuntimeError(\n"
        '                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "\n'
        '                "the current vLLM environment."\n'
        "            )\n"
    )
    guard_new = (
        "        if current_platform.is_cuda() and not is_deep_gemm_supported():\n"
        "            # [THOR-OVERLAY] upstream Thor SM110 branch: warn instead of raise\n"
        "            # so the Triton MQA-logits fallback can serve this platform.\n"
        "            logger.warning_once(\n"
        '                "DeepGEMM not supported on this platform; using Triton fallback "\n'
        '                "for sparse attention indexer."\n'
        "            )\n"
    )
    if src.count(guard_old) == 1:
        src = src.replace(guard_old, guard_new, 1)
        print("replaced hard DeepGEMM raise with upstream warning")
    else:
        print(f"ERROR: hard guard anchor found {src.count(guard_old)} times "
              "(expected 1) — aborting", file=sys.stderr)
        sys.exit(2)

    ast.parse(src)
    open(IDX, "w").write(src)
    print("indexer patched: Triton MQA-logits fallback wired for SM110")

# ---- verify ---------------------------------------------------------------
try:
    import inspect
    import vllm.model_executor.layers.sparse_attn_indexer as S
    body = inspect.getsource(S)
    for token in ("fp8_paged_mqa_logits_triton", "fp8_mqa_logits_triton",
                 "use_deep_gemm = is_deep_gemm_supported()",
                 "is_device_capability_family(110)"):
        assert token in body, f"missing: {token}"
    print("all expected tokens present")
    # signature must still accept compress_ratio (our newer attention.py needs it)
    sig = inspect.signature(S.SparseAttnIndexer.__init__)
    assert "compress_ratio" in sig.parameters, "compress_ratio param lost!"
    print("SparseAttnIndexer signature intact (compress_ratio present)")
    from vllm.utils.deep_gemm import is_deep_gemm_supported
    print("is_deep_gemm_supported():", is_deep_gemm_supported())
except Exception as e:
    print("VERIFY FAILED:", type(e).__name__, e, file=sys.stderr)
    sys.exit(3)

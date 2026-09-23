# Thor SM110 DSv4-Flash serving image — consolidated build
#
# Reproduces the full runtime chain that was previously layered across
# thor-dsv4 / -ov / -e8m0 / -spec / -spin images, from a fresh flash:
#
#   docker build -f mods/thor-sm110/Dockerfile -t thor-dsv4:latest .
#
# Base: NVIDIA jetson-thor vLLM (0.28.1rc1, sm_110 kernels precompiled).
# Ray pinned for the multi-node TP=2 path.
# DeepGEMM is NOT used at serve time (VLLM_USE_DEEP_GEMM=0 in the launch
# scripts); the Triton fallbacks installed here are the serving path.
#
# Overlay steps (all idempotent, each refuses to write unparsable Python):
#   1. overlay_thor_backend.py  — installs thor.py (Triton sparse-MLA,
#      THOR_MLA_SPARSE_DSV4) + registers the backend enum + routes
#      _select_dsv4_attn_cls to the Thor class on SM110.
#   2. patch_e8m0_triton.py     — Triton 3.7.1 lacks float8_e8m0fnu in its
#      type canonicalisation; widen e8m0 scales to fp32 before launch (exact,
#      pure power-of-two). NOTE: the committed vllm fork (Nebq29/vllm main)
#      carries this fix as source; this patch applies it to the base image.
#   3. patch_thor_einsum.py     — O-proj fp8_einsum DeepGEMM layout assert
#      workaround via the Thor Triton einsum fallback.
#   4. mqa_logits_triton.py     — Triton MQA-logits kernels for the sparse
#      indexer (replaces the DeepGEMM-only path).
#   5. patch_thor_indexer.py    — routes sparse_attn_indexer prefill/decode
#      to the Triton MQA-logits fallback on SM110.
#   6. fp8_thor_configs/*.json  — 7 on-device-tuned W8A8 block-GEMM tile
#      configs for NVIDIA_Thor (image ships none).
#   7. patch_spin_wait.py       — shm_broadcast busy_loop_s 1 -> 0.002s;
#      removes up-to-1s sleep per engine step on 2-node TP (spec-decode
#      round-trips especially).

FROM ghcr.io/nvidia-ai-iot/vllm:qwen3.8-next-jetson-thor-latest

RUN pip install --no-cache-dir "ray[default]==2.44.1"

# --- overlay scripts + sources ---
COPY mods/thor-sm110/overlay/ /opt/thor_overlay/

RUN python3 /opt/thor_overlay/overlay_thor_backend.py \
 && python3 /opt/thor_overlay/patch_e8m0_triton.py \
 && python3 /opt/thor_overlay/patch_thor_einsum.py \
 && cp /opt/thor_overlay/thor_mqa_logits_triton.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/mqa_logits_triton.py \
 && VLLM_USE_DEEP_GEMM=0 python3 /opt/thor_overlay/patch_thor_indexer.py \
 && python3 /opt/thor_overlay/patch_triton_e8m0.py \
 && python3 /opt/thor_overlay/patch_spin_wait.py

# --- tuned FP8 block-GEMM configs (7 files, verified count) ---
COPY mods/thor-sm110/configs/ /tmp/thor_cfgs/
RUN python3 -c "import glob,os,shutil; d='/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/configs'; fs=glob.glob('/tmp/thor_cfgs/*NVIDIA_Thor*'); assert len(fs)==7, f'expected 7 Thor configs, got {len(fs)}'; [shutil.copy(f,d) for f in fs]; print('Thor FP8 configs installed:', len(glob.glob(os.path.join(d,'*NVIDIA_Thor*'))))"

# --- sanity gate: the Thor backend must import and register ---
RUN python3 -c "from vllm.models.deepseek_v4.nvidia.thor import DeepseekV4ThorAttention; from vllm.v1.attention.backends.registry import AttentionBackendEnum; assert AttentionBackendEnum.THOR_MLA_SPARSE_DSV4 is not None; print('THOR backend OK')"

#!/bin/bash
# DSV4-Flash dual-Thor TP=2 + DSpark k=4 + LL128 + DeepGEMM MoE (D2M shape).
# Round-9 validated: prefill +17-18% vs marlin, decode -19%. Use for
# prefill-heavy workloads; see mods/thor-sm110/dg/README.md.
# Requires: jumbo MTU on mgbe0_0, ~/dg_wheel/ + ~/patches/dg/ present.
# Launch SECOND on the master node (.230), after rank1.
set -e
RANK=${1:-0}
NAME=dsv4f-rank$RANK
docker rm -f $NAME >/dev/null 2>&1 || true
SPEC='{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
COMP='{"cudagraph_capture_sizes":[1,2,4,5,6,8,12,16],"cudagraph_num_of_warmups":1}'
MASTER=10.0.0.1  # rank0 is master on the TP rail
docker run -d --name $NAME \
  -v /home/nebq29/models:/models:ro \
  -v /home/nebq29/thor-cache/triton:/root/.triton \
  -v /home/nebq29/thor-cache/deep_gemm:/root/.dg_cache \
  -v /home/nebq29/dg_wheel:/wheels:ro \
  -v /home/nebq29/patches/dg/cuda_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py:ro \
  -v /home/nebq29/patches/dg/deep_gemm_mod_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py:ro \
  -v /home/nebq29/patches/dg/import_utils_dg_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/import_utils.py:ro \
  -v /home/nebq29/patches/dg/sparse_attn_indexer_dg.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
  -v /home/nebq29/patches/dg/tilelang_dg.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py:ro \
  -v /home/nebq29/patches/dg/deep_gemm_moe_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:ro \
  --network host --ipc host --shm-size 8g --gpus all \
  -e NCCL_SOCKET_IFNAME=mgbe0_0 \
  -e NCCL_PROTO=LL128 \
  -e GLOO_SOCKET_IFNAME=mgbe0_0 \
  -e VLLM_USE_DEEP_GEMM=1 \
  -e VLLM_MOE_USE_DEEP_GEMM=1 \
  -e DG_JIT_CACHE_DIR=/root/.dg_cache \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=512 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  -e OPENBLAS_CORETYPE=NEOVERSEV2 \
  -e TORCH_CUDA_ARCH_LIST=11.0a \
  -e CUDA_VISIBLE_DEVICES=0 \
  --entrypoint bash \
  thor-dsv4:latest -lc "pip install --no-cache-dir /wheels/deep_gemm-*.whl && vllm serve /models/DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 --port 19038 --served-model-name deepseek-v4-flash \
    --trust-remote-code --tokenizer-mode deepseek_v4 \
    --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --load-format safetensors --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 13958643712 \
    --kv-cache-dtype fp8 \
    --attention-backend THOR_MLA_SPARSE_DSV4 \
    --linear-backend triton --moe-backend deep_gemm \
    --compilation-config '$COMP' \
    --speculative-config '$SPEC' \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --all2all-backend allgather_reducescatter \
    --nnodes 2 --node-rank $RANK \
    --master-addr $MASTER --master-port 29679 \
    --distributed-timeout-seconds 1800 \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4"
echo "rank$RANK started (DSpark k=4 + LL128 + deep_gemm MoE)"

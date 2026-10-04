#!/bin/bash
# Round 10 step 3: D3 retest — deep_gemm LINEAR backend (vs D2 which was MoE-only).
# Same G mounts as r10_ab.sh but --linear-backend deep_gemm + DG_JIT_DEBUG=1.
# Usage: r10_d3.sh <rank 0|1> [mtp on|off]
set -e
RANK=${1:?rank}
MTP=${2:-off}
NAME=dsv4f-rank$RANK
docker rm -f $NAME >/dev/null 2>&1 || true

if [ "$MTP" = "on" ]; then
  SPEC='{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
else
  SPEC=''
fi
COMP='{"cudagraph_capture_sizes":[1,2,4,5,6,8,12,16],"cudagraph_num_of_warmups":1}'
MASTER=10.0.0.1
if [ "$RANK" = "0" ]; then
  LANEX_LOCAL_IPS=10.0.0.1,10.0.1.1,10.0.2.1,10.0.3.1
  LANEX_PEER_IPS=10.0.0.2,10.0.1.2,10.0.2.2,10.0.3.2
else
  LANEX_LOCAL_IPS=10.0.0.2,10.0.1.2,10.0.2.2,10.0.3.2
  LANEX_PEER_IPS=10.0.0.1,10.0.1.1,10.0.2.1,10.0.3.1
fi

MOUNTS="-v /home/nebq29/thor-cache/deep_gemm:/root/.dg_cache \
  -v /home/nebq29/dg_wheel:/wheels:ro \
  -v /home/nebq29/patches/dg/cuda_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py:ro \
  -v /home/nebq29/patches/dg/deep_gemm_mod_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py:ro \
  -v /home/nebq29/patches/dg/import_utils_dg_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/import_utils.py:ro \
  -v /home/nebq29/patches/dg/sparse_attn_indexer_dg.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro \
  -v /home/nebq29/patches/dg/tilelang_dg.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py:ro \
  -v /home/nebq29/patches/dg/deep_gemm_moe_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:ro \
  -v /home/nebq29/fp8_einsum_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py:ro"

docker run -d --name $NAME \
  -v /home/nebq29/models:/models:ro \
  -v /home/nebq29/thor-cache/triton:/root/.triton \
  -v /home/nebq29/lanex:/opt/lanex \
  -v /tmp:/tmp \
  $MOUNTS \
  --network host --ipc host --shm-size 8g --runtime nvidia --gpus all \
  -e NCCL_SOCKET_IFNAME=mgbe0_0 \
  -e NCCL_PROTO=LL128 \
  -e GLOO_SOCKET_IFNAME=mgbe0_0 \
  -e VLLM_USE_DEEP_GEMM=1 -e VLLM_MOE_USE_DEEP_GEMM=1 -e DG_JIT_CACHE_DIR=/root/.dg_cache \
  -e DG_JIT_DEBUG=1 \
  -e DG_EINSUM_DEBUG=${DG_EINSUM_DEBUG:-0} \
  -e LANEX_ENABLE=1 -e LANEX_ASYNC=1 -e LANEX_HOSTBUF=hostmem \
  -e LANEX_HOSTMEM_ALLOC=mmap -e LANEX_HOSTMEM_THP=huge -e LANEX_SOCKBUF=0 \
  -e LANEX_LOCAL_IPS=$LANEX_LOCAL_IPS -e LANEX_PEER_IPS=$LANEX_PEER_IPS \
  -e LANEX_LIB=/opt/lanex/lanex_core.so -e PYTHONPATH=/opt/lanex \
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
    --max-model-len 40960 --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --load-format safetensors --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 13958643712 \
    --kv-cache-dtype fp8 \
    --attention-backend THOR_MLA_SPARSE_DSV4 \
    --linear-backend deep_gemm --moe-backend deep_gemm \
    --compilation-config '$COMP' \
    ${SPEC:+--speculative-config '$SPEC'} \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --all2all-backend allgather_reducescatter \
    --nnodes 2 --node-rank $RANK \
    --master-addr $MASTER --master-port 29679 \
    --distributed-timeout-seconds 1800 \
    $( [ "$RANK" = "0" ] && echo "--enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4" || echo "--headless" )"
echo "rank$RANK D3 (linear=deep_gemm) mtp=$MTP started"

#!/bin/bash
# DSV4-Flash dual-Thor TP=2 + DSpark k=4 + LL128 proto (tuned network).
# Requires jumbo MTU on mgbe0_0 (see set_jumbo_tp.sh). Launch FIRST on node2 (.230).
set -e
docker rm -f dsv4f-rank1 >/dev/null 2>&1 || true
SPEC='{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
COMP='{"cudagraph_capture_sizes":[1,2,4,5,6,8,12,16],"cudagraph_num_of_warmups":1}'
docker run -d --name dsv4f-rank1 \
  -v /home/nebq29/models:/models:ro \
  -v /home/nebq29/thor-cache/triton:/root/.triton \
  --network host --ipc host --shm-size 8g --gpus all \
  -e NCCL_SOCKET_IFNAME=mgbe0_0 \
  -e NCCL_PROTO=LL128 \
  -e GLOO_SOCKET_IFNAME=mgbe0_0 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e VLLM_DEEP_GEMM_WARMUP=skip \
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
  thor-dsv4-spin:latest -lc "vllm serve /models/DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 --port 19038 --served-model-name deepseek-v4-flash \
    --trust-remote-code --tokenizer-mode deepseek_v4 \
    --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --load-format safetensors --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 13958643712 \
    --kv-cache-dtype fp8 \
    --attention-backend THOR_MLA_SPARSE_DSV4 \
    --linear-backend triton --moe-backend marlin \
    --compilation-config '$COMP' \
    --speculative-config '$SPEC' \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --all2all-backend allgather_reducescatter \
    --nnodes 2 --node-rank 1 \
    --master-addr 10.0.0.1 --master-port 29679 \
    --distributed-timeout-seconds 1800 --headless"
echo "rank1 started (DSpark k=4 + LL128)"

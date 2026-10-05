#!/bin/bash
# Round 13 step 3: SM100-class gate A/B. lanex shipping env, int8 OFF.
# Usage: r13_ab.sh <M|G0|G1|G2|G3> <rank 0|1> <mtp on|off>
#   M  = marlin, no deep_gemm (reference; patch shouldn't change it)
#   G0 = ee84db0 wheel + round-12 patch set (round-11/12 baseline)
#   G1 = ee84db0 wheel + round-13 gate patches (varlen indexer + moe gates)
#   G2 = new (r13) wheel  + round-13 gate patches
#   G3 = new (r13) wheel  + round-13 gate patches + DG_THOR_SHRINK_ALIGN=1
set -e
CFG=${1:?M|G0|G1|G2|G3}
RANK=${2:?rank}
MTP=${3:-on}
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

V=/usr/local/lib/python3.12/dist-packages/vllm
BASE_MOUNTS="-v /home/nebq29/thor-cache/deep_gemm:/root/.dg_cache \
  -v /home/nebq29/patches/dg/cuda_patched.py:$V/platforms/cuda.py:ro \
  -v /home/nebq29/patches/dg/deep_gemm_mod_patched.py:$V/utils/deep_gemm.py:ro \
  -v /home/nebq29/patches/dg/import_utils_dg_patched.py:$V/utils/import_utils.py:ro \
  -v /home/nebq29/patches/dg/sparse_attn_indexer_dg.py:$V/model_executor/layers/sparse_attn_indexer.py:ro \
  -v /home/nebq29/patches/dg/tilelang_dg.py:$V/model_executor/kernels/mhc/tilelang.py:ro"

case "$CFG" in
  M)
    MOUNTS="-v /home/nebq29/patches/tilelang_patched.py:$V/model_executor/kernels/mhc/tilelang.py:ro \
      -v /home/nebq29/patches/import_utils_patched.py:$V/utils/import_utils.py:ro"
    PIP=""; MOE="--moe-backend marlin"; DGENV=""
    ;;
  G0)
    MOUNTS="$BASE_MOUNTS -v /home/nebq29/patches/dg/deep_gemm_moe_patched.py:$V/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:ro \
      -v /home/nebq29/dg_wheel_ee84db0:/wheels:ro"
    PIP="pip install --no-cache-dir /wheels/deep_gemm-*.whl && "
    MOE="--moe-backend deep_gemm"
    DGENV="-e VLLM_USE_DEEP_GEMM=1 -e VLLM_MOE_USE_DEEP_GEMM=1 -e DG_JIT_CACHE_DIR=/root/.dg_cache"
    ;;
  G1|G2|G3)
    WHEEL=/home/nebq29/dg_wheel_ee84db0
    if [ "$CFG" = "G2" ] || [ "$CFG" = "G3" ]; then WHEEL=/home/nebq29/dg_r13/dist; fi
    ALIGN=""
    [ "$CFG" = "G3" ] && ALIGN="-e DG_THOR_SHRINK_ALIGN=1"
    MOUNTS="$BASE_MOUNTS \
      -v /home/nebq29/patches/dg/r13_indexer.py:$V/v1/attention/backends/mla/indexer.py:ro \
      -v /home/nebq29/patches/dg/r13_deep_gemm_moe.py:$V/model_executor/layers/fused_moe/experts/deep_gemm_moe.py:ro \
      -v /home/nebq29/patches/dg/r13_batched_deep_gemm_moe.py:$V/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py:ro \
      -v $WHEEL:/wheels:ro"
    PIP="pip install --no-cache-dir /wheels/deep_gemm-*.whl && "
    MOE="--moe-backend deep_gemm"
    DGENV="-e VLLM_USE_DEEP_GEMM=1 -e VLLM_MOE_USE_DEEP_GEMM=1 -e DG_JIT_CACHE_DIR=/root/.dg_cache $ALIGN"
    ;;
esac

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
  $DGENV \
  -e LANEX_ENABLE=1 -e LANEX_ASYNC=1 -e LANEX_HOSTBUF=hostmem \
  -e LANEX_HOSTMEM_ALLOC=mmap -e LANEX_HOSTMEM_THP=huge -e LANEX_SOCKBUF=0 \
  -e LANEX_PROF=${LANEX_PROF:-0} -e LANEX_PROF_EVERY=${LANEX_PROF_EVERY:-89} \
  -e LANEX_LOCAL_IPS=$LANEX_LOCAL_IPS -e LANEX_PEER_IPS=$LANEX_PEER_IPS \
  -e LANEX_LIB=/opt/lanex/lanex_core.so -e PYTHONPATH=/opt/lanex \
  -e LANEX_COMPRESS=none \
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
  thor-dsv4:latest -lc "$PIP vllm serve /models/DeepSeek-V4-Flash-0731 \
    --profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/tmp/vprof\"}' \
    --host 0.0.0.0 --port 19038 --served-model-name deepseek-v4-flash \
    --trust-remote-code --tokenizer-mode deepseek_v4 \
    --max-model-len 40960 --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --load-format safetensors --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.90 --kv-cache-memory-bytes 13958643712 \
    --kv-cache-dtype fp8 \
    --attention-backend THOR_MLA_SPARSE_DSV4 \
    --linear-backend deep_gemm $MOE \
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
echo "rank$RANK $CFG-LINEAR mtp=$MTP started"

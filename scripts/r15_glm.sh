#!/bin/bash
# Round 15: GLM-5.3-Flash NVFP4 dual-Thor TP=2 on the NATIVE MX indexer path.
# Same lanex env as DSv4 shipping config; mounts the r15 fork chain instead of
# the round-14 patched wrapper.
# Usage: r15_glm.sh <0|1> [mtp_k]   (mtp_k: 0=off, else num_speculative_tokens)
#   LEGACY_FP8=1 -> VLLM_GLM_INDEXER_LEGACY_FP8=1 (quality reference, old wheel)
set -e
RANK=${1:?rank}
MTPK=${2:-0}
NAME=r15-glm-rank$RANK
docker rm -f $NAME >/dev/null 2>&1 || true

if [ "$MTPK" = "0" ]; then
  SPEC=''
else
  SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTPK}"
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

LEGACY_ENV=""
if [ "${LEGACY_FP8:-0}" = "1" ]; then
  LEGACY_ENV="-e VLLM_GLM_INDEXER_LEGACY_FP8=1"
fi
# cooperative top-k uses CUDA thread-block clusters that fail on Thor sm_110.
TOPK_BACKEND="${TOPK_BACKEND:-persistent}"

if [ "$RANK" = "0" ]; then HEADLESS=""; else HEADLESS="--headless"; fi

S=${STAGE_DIR:-/home/nebq29/r15_stage}

docker run -d --name $NAME \
  -v /home/nebq29/models:/models:ro \
  -v /home/nebq29/thor-cache/triton:/root/.triton \
  -v /home/nebq29/lanex:/opt/lanex \
  -v /tmp:/tmp \
  -v /home/nebq29/patches/tilelang_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py:ro \
  -v /home/nebq29/patches/dg/import_utils_dg_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/import_utils.py:ro \
  -v /home/nebq29/patches/dg/r13_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py:ro \
  -v /home/nebq29/patches/glm/thor_sparse.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/thor_sparse.py:ro \
  -v /home/nebq29/patches/glm/registry.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/registry.py:ro \
  -v $S/common/attention.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/attention.py:ro \
  -v $S/common/sparse_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/sparse_indexer.py:ro \
  -v $S/shim_sparse_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/sparse_indexer.py:ro \
  -v $S/nvidia_sparse_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/sparse_indexer.py:ro \
  -v $S/kpool_compress.py:/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/ops/kpool_compress.py:ro \
  -v $S/indexer_topk.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/indexer_topk.py:ro \
  -v $S/deep_gemm_fork.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py:ro \
  -v $S/sparse_mla_ops_thorpatch.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py:ro \
  -v ${DG_WHEEL_DIR:-/home/nebq29/dg_wheel}:/wheels:ro \
  -v /home/nebq29/thor-cache/deep_gemm:/root/.dg_cache \
  --network host --ipc host --shm-size 8g --runtime nvidia --gpus all \
  -e NCCL_SOCKET_IFNAME=mgbe0_0 \
  -e NCCL_PROTO=LL128 \
  -e GLOO_SOCKET_IFNAME=mgbe0_0 \
  -e VLLM_USE_DEEP_GEMM=1 \
  -e DG_JIT_CACHE_DIR=/root/.dg_cache \
  -e LANEX_ENABLE=1 -e LANEX_ASYNC=1 -e LANEX_HOSTBUF=hostmem \
  -e LANEX_HOSTMEM_ALLOC=mmap -e LANEX_HOSTMEM_THP=huge -e LANEX_SOCKBUF=0 \
  -e LANEX_PROF=${LANEX_PROF:-0} -e LANEX_PROF_EVERY=${LANEX_PROF_EVERY:-89} \
  -e LANEX_LOCAL_IPS=$LANEX_LOCAL_IPS -e LANEX_PEER_IPS=$LANEX_PEER_IPS \
  -e LANEX_LIB=/opt/lanex/lanex_core.so -e PYTHONPATH=/opt/lanex \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  -e OPENBLAS_CORETYPE=NEOVERSEV2 \
  -e TORCH_CUDA_ARCH_LIST=11.0a \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_GLM_INDEXER_TOPK_BACKEND=$TOPK_BACKEND \
  $LEGACY_ENV \
  --entrypoint bash \
  thor-dsv4:latest -lc "mkdir -p /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common && touch /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/__init__.py && pip install --no-cache-dir --root-user-action=ignore /wheels/deep_gemm-*.whl >/dev/null 2>&1 && vllm serve /models/GLM-5.3-Flash-NVFP4 \
    --host 0.0.0.0 --port 19039 --served-model-name glm-5.3-flash \
    --max-model-len ${MAX_LEN:-32768} --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --gpu-memory-utilization ${GPU_UTIL:-0.88} \
    --load-format safetensors --safetensors-load-strategy lazy \
    --kv-cache-dtype ${KV_DTYPE:-bfloat16} \
    --attention-backend THOR_MLA_SPARSE_GLM \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --compilation-config '$COMP' \
    ${SPEC:+--speculative-config '$SPEC'} \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --nnodes 2 --node-rank $RANK \
    --master-addr $MASTER --master-port 29681 \
    --distributed-timeout-seconds 1800 \
    $HEADLESS"
echo "rank$RANK r15 GLM mtp_k=$MTPK legacy=${LEGACY_FP8:-0} started"

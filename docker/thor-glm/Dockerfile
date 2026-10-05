# thor-glm: GLM-5.3-Flash serving image for the Jetson Thor pair (round 16).
#
# Built FROM the immutable thor-dsv4:latest base (never modified). Layers:
#   1. DeepGEMM thor-sm110-20261005 wheel (2.8.1+local)
#   2. Nebq29/vllm thor-sm110-wip Python tree overlaid on the image's
#      compiled vllm package (all .py files; the image's *.abi3.so
#      compiled extensions are KEPT — the repo's Python was verified to
#      import against them, see round-16 Step 1).
#   3. lanex 4-lane allreduce, compiled from source with the round-16
#      LANEX_TIMEOUT_MS change (default 600 s; was hard-coded 30 s, which
#      killed engines when a peer rank JIT-compiled a new shape mid-exchange).
#
# Seam fixes from round 15 are NOT image-specific shims here: they live in
# the fork commits (56e516adbf, 9d8ddca5a2, 98b74a0d43) and ride in with
# the Python overlay. The one image-specific note: we do NOT overlay
# vllm/_version.py (generated at the image's build time) and we keep the
# image's compiled extensions, so `import vllm` reports the base build's
# version string (0.28.1rc1.dev580+g385dce36b) even though the Python
# tree is from thor-sm110-wip@<label>. The labels below are authoritative.
FROM thor-dsv4:latest

ARG VLLM_COMMIT=56e516adbf
ARG DEEPGEMM_TAG=thor-sm110-20261005
ARG BUILD_DATE=2026-10-06

LABEL org.opencontainers.image.title="thor-glm" \
      org.opencontainers.image.description="GLM-5.3-Flash NVFP4 serving on dual Jetson Thor (sm_110): MX kpool indexer + THOR_MLA_SPARSE_GLM + lanex" \
      com.nebq29.vllm-commit="${VLLM_COMMIT}" \
      com.nebq29.deepgemm-tag="${DEEPGEMM_TAG}" \
      com.nebq29.build-date="${BUILD_DATE}"

# --- DeepGEMM (MX attention kernels, sm_110 dispatch) ---
COPY deep_gemm-2.8.1+local-cp312-cp312-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir --root-user-action=ignore /tmp/deep_gemm-*.whl && rm /tmp/deep_gemm-*.whl

# --- vLLM Python overlay (repo thor-sm110-wip; keeps image .so extensions) ---
COPY r16_py_overlay.tgz /tmp/
RUN tar xf /tmp/r16_py_overlay.tgz -C /usr/local/lib/python3.12/dist-packages/ \
    && rm /tmp/r16_py_overlay.tgz \
    && python3 -c "import vllm; from vllm.v1.attention.backends.registry import AttentionBackendEnum; assert AttentionBackendEnum.THOR_MLA_SPARSE_GLM; import vllm.models.glm5next.nvidia.ops.kpool_compress as k; assert hasattr(k, 'fwht128_quant_fp8_mx'); print('overlay self-check OK')"

# --- lanex: compile from source with LANEX_TIMEOUT_MS (default 600000) ---
COPY lanex_core.c lanex_async.c lanex.py sitecustomize.py /opt/lanex/
# NOTE: lanex.py dlopens libcuda.so.1 at import, so a full `import lanex`
# is impossible during docker build (no GPU driver in the build sandbox).
# lanex_core.so itself is pure sockets/pthread — verify it loads directly;
# the runtime import is exercised by the launch scripts + warm-up.
RUN cd /opt/lanex && gcc -O2 -shared -fPIC -pthread -o lanex_core.so lanex_core.c lanex_async.c -ldl \
    && python3 -c "import ctypes; l=ctypes.CDLL('/opt/lanex/lanex_core.so'); print('lanex_core.so loads OK')" \
    && python3 -m py_compile /opt/lanex/lanex.py /opt/lanex/sitecustomize.py \
    && echo "lanex py syntax OK"

# sitecustomize auto-runs at interpreter startup when /opt/lanex is on the
# path; LANEX_ENABLE gates the actual hook install (same contract as r15).
ENV PYTHONPATH=/opt/lanex \
    LANEX_LIB=/opt/lanex/lanex_core.so

# Default service port for the GLM endpoint (launch scripts override as needed).
EXPOSE 1234

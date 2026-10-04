# tools — measurement harnesses for the Thor DSv4 cluster

Standalone probes used across rounds 7-10. Run from a board (thorc1 preferred);
none require vLLM running except where noted.

| file | what it does |
|---|---|
| `lane_xchg.c` | single-lane raw TCP exchange bench (the 3.6 GB/s bare-transport reference) |
| `lane_xchg_mt.c` | multi-threaded 4-lane exchange bench; prints p50/p95 xchg ms + GB/s. Baseline for lanex comparisons |
| `test_bufclass.py` | host-buffer class matrix (cudaHostAlloc / shared mmap / private-huge / managed) — established private-anon-THP as the only fast class for CPU lane threads |
| `test_lanex_phases.py` | per-phase timing of a lanex all-reduce (D2H / xchg / H2D / add) to find where time goes |
| `gemm_probe.py` | on-device GEMM roofline probe: bf16 / fp8 torch / fp8 CUTLASS(vllm) / nvfp4 / mxfp4 on sm_110. Run inside thor-dsv4 image with `--entrypoint python3` |
| `prof_buckets.py` | torch profiler trace analyzer: GPU busy/idle, kernel buckets (lanex-aware: comm bucket = "GPU kernels only"; host-callback exchange shows as "stream blocked ... likely lanex exchange"), idle-by-cause (CPU busy vs blocked launch), DeepGEMM kernel listing, 15 largest gaps. Usage: `python3 prof_buckets.py <trace.json.gz>` |

## prof_buckets.py notes

- Traces come from vLLM `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vprof"}'`
  + `/start_profile` / `/stop_profile` (this build has NO `VLLM_TORCH_PROFILER_DIR`).
- Inside `bash -lc` the JSON needs single-quotes-around + escaped double quotes or the shell eats them.
- With lanex, the true communication share = LANEX_PROF xchg totals from the
  container log, NOT the GPU-kernel comm bucket (the exchange is a host callback;
  the GPU just idles).
- Warm up every prompt length before profiling or `load_binary` (lazy CUDA
  module load) pollutes the idle table with ~0.8 s of first-shape cost.

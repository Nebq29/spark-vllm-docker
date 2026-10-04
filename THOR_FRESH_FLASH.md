# Fresh flash → DSv4-Flash serving on 2× Jetson Thor (SM110)

Everything needed to go from a freshly-flashed Thor pair to a serving
DeepSeek-V4-Flash TP=2 cluster over the 10GbE rail. Validated 2026-09-22:
**~44 tok/s decode (DSpark k=4), ~735 tok/s cold prefill.**

## What this repo provides

| Piece | Path |
|---|---|
| Consolidated serving image build | `mods/thor-sm110/Dockerfile` |
| Runtime overlay sources | `mods/thor-sm110/overlay/` |
| Tuned Thor FP8 GEMM configs (7) | `mods/thor-sm110/configs/` |
| Serving recipe (validated flags) | `recipes/thor-v4.yaml` |
| Jumbo MTU setup | `scripts/set_jumbo_tp.sh` |
| Launch scripts | `scripts/thor_rank{1,0}_spec_k4_ll128.sh` |

The vLLM source itself lives in **Nebq29/vllm** branch `main`
(= `thor-dsv4-serving`): upstream + Thor SM110 sparse-MLA backend
(`vllm/models/deepseek_v4/nvidia/thor.py`) + the e8m0 Triton widening fix.
DeepGEMM Thor arch dispatch is in **Nebq29/DeepGEMM** branch `thor-sm110`
(not used at serve time — `VLLM_USE_DEEP_GEMM=0`).

## Procedure (both nodes)

1. **Flash** JetPack 7.2 (L4T r39.2.1) — stock, no PCIe/ODMDATA changes
   needed for the 10GbE path.
2. **Clone**: `git clone git@github.com:Nebq29/spark-vllm-docker.git && cd spark-vllm-docker`
3. **Build image** (same on both nodes):
   `docker build -f mods/thor-sm110/Dockerfile -t thor-dsv4:latest .`
   The build self-verifies: Thor backend import + 7-config count gate.
4. **Model**: place `DeepSeek-V4-Flash-0731` under `/models/` (or adjust
   the bind mount in the launch scripts).
5. **Network** (both nodes, one time, survives reboot): install
   `mods/thor-sm110/net/` — `net_tune.sh` + `nvethernet_ensure_fixed.sh` +
   `nvethernet_rx_lost_irq.patch` (build the patched nvethernet.ko first;
   build-id `08bf9206…`), then enable
   `forecr-net-tune.service` (unit text in `mods/thor-sm110/net/README.md`).
   This sets MTU 8966, `rx-usecs 8 rx-frames 1`, **RX ring 16384**,
   threaded NAPI, DMA-FQ, CPU `scaling_min` pin, lane IRQ spread (lane N ->
   cpu N+1), and GPU devfreq pins (gpc 1575 / nvd 1692).
   Verify: `ping -M do -s 8972 <peer>` = 0% loss; `ethtool -c mgbe0_0`
   shows rx-frames 1; `nstat TcpRetransSegs` stays flat under load.
6. **lanex** (optional, +15-17% prefill): build
   `mods/thor-sm110/lanex/` per its README, deploy to `~/lanex/` on both
   boards, launch with the env in `scripts/thor_rank{0,1}_dg_spec_k4_ll128.sh`
   (`LANEX_ENABLE=1 LANEX_ASYNC=1 LANEX_HOSTBUF=hostmem
   LANEX_HOSTMEM_ALLOC=mmap LANEX_HOSTMEM_THP=huge LANEX_SOCKBUF=0`).
7. **Launch**: rank1 (worker node) FIRST, wait ~30 s, then rank0 (head).
   `bash scripts/thor_rank1_spec_k4_ll128.sh` then `bash scripts/thor_rank0_spec_k4_ll128.sh`
8. **Verify**: `curl http://<head>:19038/health` → 200; model id
   `deepseek-v4-flash`. Decode should be ~44-46 tok/s (lanex+tuned rail);
   if it's 8-16 tok/s, suspect RX ring overflow or unpinned clocks (see
   `mods/thor-sm110/net/README.md` gotchas 1-2-4).

## Critical flags (do not trim — each fixes a measured failure)

- `--attention-backend THOR_MLA_SPARSE_DSV4` — the only working DSv4 MLA
  path on sm_110 (cute-dsl and trtllm-gen hard-gate to SM100/103/107).
- `--linear-backend triton --moe-backend marlin` — CUTLASS rejects this
  checkpoint's 128×128 block scales on Thor; Marlin handles the INT8 experts.
- `VLLM_USE_DEEP_GEMM=0` + `VLLM_MOE_USE_DEEP_GEMM=0` — DeepGEMM layout
  asserts on this checkpoint's einsum/scale layouts.
- `NCCL_PROTO=LL128` — removes the 32 KB small-message knee on 10GbE
  (−34% per-collective at k=4 payload).
- `--safetensors-load-strategy lazy` + `CUDA_MODULE_LOADING=LAZY` — load
  time on unified memory.
- spin-wait patch (in image) — `busy_loop_s` 1→0.002 s in shm_broadcast;
  the default adds up to 1 s per engine step on 2-node TP.
- e8m0 widening (in image + vllm fork) — Triton 3.7.1 lacks
  `float8_e8m0fnu`; widening to fp32 is exact (power-of-two scales).

## Explicitly NOT included (tested, rejected)

- **bf16-accum block-FP8 GEMM** — safe and 1.1–1.3× on isolated prefill
  shapes, but e2e wash on fresh-boot A/B (43.5 vs 43.3 tok/s). fp16
  accumulation is UNSAFE (raw e4m3 code dots overflow fp16's 65504 max).
- **MX `tl.dot_scaled` overlay** — kernel-level 1.4× negated by wrapper
  prep-pass overhead; caused decode regressions under graph capture.
- **k=3 speculative decode** — reproducibly pathological on this build.
- Debug/probe scripts from the investigation live in the workspace
  (`impl/speed_probe/`), not in the serving repos.

## Known open issues

- Prefix caching: works on this build (hits observed), but was a no-op on
  older Thor builds — re-verify after any vllm rebase.
- 10GbE remains ~17% of the decode step and ~55% of prefill; the PCIe
  upgrade path is documented in the workspace `pcie_research/` folder.

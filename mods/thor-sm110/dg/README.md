# Thor DeepGEMM MoE variant (round 9, validated 2026-10-03)

DeepGEMM MoE serving path for DeepSeek-V4-Flash on 2x Jetson Thor (sm_110),
TP=2. Validated on thorc1 (192.168.1.230) / thorc2 (192.168.1.24).

## Measured (lanex hostmem + SOCKBUF=0 + IRQ spread, 8192/40960)

| config | 2K prefill | 8K prefill | 32K prefill | decode (147/128) | MTP accept |
|---|---|---|---|---|---|
| marlin (shipping) | ~1040 | 910.4 | 866.1 | 44–45.6 tok/s | len 3.36 |
| **D2** (deep_gemm MoE, MTP off) | — | **1156.3** | — | — | — |
| **D2M** (deep_gemm MoE, MTP k=4) | 1082.1 | **1065.8** | **1021.9** | ~36 tok/s ⚠️ | len 3.51–3.54, 63% |

**Verdict: prefill +17–18%, decode −19% vs marlin.** deep_gemm's contiguous
m-grouped path crushes marlin at prefill sizes (222–265 TF standalone vs ~85
effective); its masked small-m decode path is DRAM-roofline-bound and loses.
One `moe_backend` per engine — pick by workload. Keep marlin for
decode-heavy interactive serving.

**Linear-side deep_gemm does NOT work with this checkpoint:** D1/D3 fail at
model load with `layout.hpp:97: sf.size(-2) == ceil_div(mn, gran_mn)` — the
ue8m0 block scales aren't in the TMA-packed layout the linear kernels expect.
Same family as the original einsum assert. Needs a scale-conversion in the
loader; not attempted.

## The six gate files (patches/)

Bind-mount each over the image copy (paths in Dockerfile.dg). Three gates had
to open; two fallbacks had to stay pinned:

1. `cuda_patched.py` — `support_deep_gemm()` += family 110
2. `deep_gemm_mod_patched.py` — `is_deep_gemm_supported()` + UE8M0 on 110
3. `import_utils_dg_patched.py` — `has_deep_gemm()` True on major 11
   (replaces the Sep-25 forced-False override)
4. `deep_gemm_moe_patched.py` — `DeepGemmFP4Experts._supports_current_device()`
   += family 110 (**the one that mattered**; error was misleading)
5. `sparse_attn_indexer_dg.py` — indexer stays on Triton fallback on cap 11
6. `tilelang_dg.py` — mhc stays on torch fallback on Thor

## Launch (bind-mount workflow, no rebuild)

Env (both ranks): `VLLM_USE_DEEP_GEMM=1 VLLM_MOE_USE_DEEP_GEMM=1` and
**remove** `VLLM_DEEP_GEMM_WARMUP=skip` (kernels JIT at startup; first boot
slow, cache `~/.dg_cache` makes later boots fast — mount it persistent).

Flags: `--moe-backend deep_gemm --linear-backend triton` (D2 shape).

Wheel: `deep_gemm-2.5.0+ee84db0-cp312-cp312-linux_aarch64.whl`
(Nebq29/DeepGEMM@thor-sm110), `pip install` at entrypoint or bake via
Dockerfile.dg.

## Fork standalone numbers (B3, test_fp8_fp4.py PASS)

- normal FP8 GEMM: 240–282 TF (CUTLASS parity)
- m-grouped contiguous (MoE prefill): 222–265 TF
- m-grouped masked (MoE decode): ~230 GB/s ≈ 90% DRAM roofline
- k-grouped contiguous: 150–183 TF
- test_bf16.py: timeout >25 min (not needed for DSv4)

## Rollback

Remove the six `-v` mounts + the two env vars → round-8 shipping behavior.
Base image `thor-dsv4:latest` is never modified by this variant.

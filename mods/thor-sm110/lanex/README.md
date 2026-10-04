# lanex — custom 4-lane allreduce transport for dual-Thor TP=2

Host-memory TCP transport that replaces NCCL for the 2-rank all-reduce on the
Forecr Thor pair's 4x 10GbE QSFP lanes. C core (GIL-free) + Python hook via
`sitecustomize.py` patching `GroupCoordinator.all_reduce` when
`LANEX_ENABLE=1`, world_size==2, and msg >= `LANEX_MIN_BYTES`.

Flow per all-reduce: D2H -> pinned host buf | C lane threads xchg over 4 lanes
| H2D -> GPU scratch | GPU add. Bit-exact vs NCCL (fp add commutative).

## Build

```bash
gcc -O2 -shared -fPIC -pthread -o lanex_core.so lanex_core.c lanex_async.c -ldl
```

## Shipping env (round 7, validated)

```
LANEX_ENABLE=1 LANEX_ASYNC=1 LANEX_HOSTBUF=hostmem \
LANEX_HOSTMEM_ALLOC=mmap LANEX_HOSTMEM_THP=huge LANEX_SOCKBUF=0
```

- `LANEX_SOCKBUF=0` = kernel autotune. REQUIRED: `net.core.wmem_max=212992`
  silently clamps any explicit SO_SNDBUF to 416 KB; only autotune reaches the
  real 4 MB/6 MB tcp_wmem/tcp_rmem ceilings. Never trust a requested buffer
  size — read back getsockopt.
- Host buffer MUST be private anonymous mmap with THP (`hostmem`+`mmap`+`huge`).
  Shared/dev_zero/cudaHostAlloc/managed mappings run the C lane threads at
  ~2.1 GB/s vs ~2.9 GB/s for private-huge (measured, rounds 2-6).
- shmem THP is `[never]` on these boards, so shared mappings can never get
  huge pages — this is why the round-4 shared-mmap variant failed.

## Deployment layout (on each board)

`~/lanex/` holds `lanex_core.so`, `lanex.py`, `sitecustomize.py`; rank
scripts bind-mount it at `/opt/lanex` with
`LANEX_LIB=/opt/lanex/lanex_core.so PYTHONPATH=/opt/lanex`.

## Measured results

- Prefill +15-17% at 2K vs no-lanex (round 6, bit-exact).
- Round 7 (with IRQ spread, see ../net): standalone xchg 6.9 -> 5.6 ms
  (2.55 -> 3.18 GB/s, near bare-transport 3.6); in-vLLM transfer
  11.5-13.0 -> 9.3-11.9 ms/xchg; prefill retransmits -58%.
- Communication is ~2% of GPU kernels but the exchange shows up as GPU idle
  ("stream blocked") in profiles — see tools/prof_buckets.py note and
  LANEX_PROF lines for the true share.
- MSG_ZEROCOPY on nvethernet: genuinely zero-copy but 4-8x SLOWER. Dead end.
- bf16 has no numpy dtype: stage as real dtype, view(uint8) for bytes; do the
  final add on GPU (CPU bf16 add is a scalar loop).
- sendall-then-recv deadlocks >= 1 MB/lane (wmem bound); the C core
  interleaves send/recv in one poll loop.

Full round-by-round data lives in the user's `~/claude/LANEX_RESULTS_ROUND{2..7}.md`.

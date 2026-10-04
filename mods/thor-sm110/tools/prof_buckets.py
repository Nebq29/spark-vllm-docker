"""prof_buckets.py — where does GPU time go in a torch-profiler trace?

Usage:  python3 prof_buckets.py <trace.json or trace.json.gz> [more traces...]

Reads the Chrome-trace JSON that torch.profiler / vLLM's profiler writes, takes
every GPU kernel, memcpy and memset event, and prints:
  - total time per category (MoE GEMM, linear GEMM, attention, communication, ...)
  - the 25 most expensive kernels by total time (name, count, total, share)
  - GPU busy time (union of all GPU intervals) vs the span from first to last
    GPU event, i.e. how much of the window the GPU sat idle.
Categories are matched on kernel names, first match wins; check the top-25 list to
see whether anything landed in the wrong bucket.
"""
import gzip
import json
import re
import sys
from collections import defaultdict

BUCKETS = [
    ("marlin GEMM (MoE or linear)", r"marlin"),
    ("MoE other (routing/grouped)", r"moe|expert|topk_softmax|grouped|align_block"),
    ("attention / indexer", r"attn|attention|mla|flash|fmha|sparse|indexer|paged|kv_cache|cache"),
    # NOTE: with lanex, only its GPU copy kernels (copy_out/add_in) appear here; the exchange
    # itself runs as a host callback and shows up as GPU idle ("stream blocked"). Use the
    # LANEX_PROF lines for the true communication time.
    ("communication (GPU kernels only)", r"nccl|allreduce|all_reduce|allgather|reduce_scatter|lanex|copy_out|add_in"),
    ("GEMM (linear/dense)", r"gemm|matmul|cutlass|cublas|xmma|nvjet|_mm_|scaled_mm|sm\d+_.*tensorop|wgmma|tcgen"),
    ("norm / rope / activation", r"norm|rope|rotary|silu|gelu|act_and_mul|swiglu|softmax"),
    ("quantize / dequant", r"quant|fp8|fp4|cvt|convert"),
    ("memcpy / memset", r"^memcpy|^memset|Memcpy|Memset"),
    ("elementwise / reduce / other triton", r"elementwise|reduce|triton|fused|vectorized|unrolled|index|cat|copy"),
]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def bucket(name):
    for b, rx in BUCKETS:
        if re.search(rx, name, re.I):
            return b
    return "other"


def gap_report(gaps, allev, top=15):
    """Explain GPU idle gaps using CPU-side events in the same trace.

    For each gap: the CUDA runtime/driver call that overlaps it most (e.g. a
    synchronize or a blocking memcpy means the CPU was WAITING), and the most
    specific CPU op covering it (if the CPU was busy launching, this is what it
    was doing)."""
    import bisect
    rt = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", "?"))
                for e in allev
                if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"))
    ops = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", "?"))
                 for e in allev
                 if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function"))
    rts = [r[0] for r in rt]
    ots = [o[0] for o in ops]
    maxrt = max((r[1] - r[0] for r in rt), default=0)
    maxop = max((o[1] - o[0] for o in ops), default=0)

    def overl(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    def best(lst, starts, maxd, g0, g1, specific):
        i = bisect.bisect_right(starts, g1)
        cand = None
        j = i - 1
        while j >= 0 and lst[j][0] >= g0 - maxd:
            s, e, n = lst[j]
            ov = overl(s, e, g0, g1)
            if ov > 0:
                if specific:
                    # smallest event that still covers >= 80% of the gap
                    if ov >= 0.8 * (g1 - g0) and (cand is None or (e - s) < (cand[1] - cand[0])):
                        cand = (s, e, n, ov)
                elif cand is None or ov > cand[3]:
                    cand = (s, e, n, ov)
            j -= 1
        return cand

    by_cause = defaultdict(float)
    total = sum(g1 - g0 for g0, g1 in gaps)
    big = sorted(gaps, key=lambda g: g[0] - g[1])[:max(top, 300)]
    rows = []
    for g0, g1 in big:
        r = best(rt, rts, maxrt, g0, g1, False)
        o = best(ops, ots, maxop, g0, g1, True)
        if r and r[3] >= 0.5 * (g1 - g0):
            cause = f"waiting in {r[2]}"
            if "Launch" in r[2] and (g1 - g0) > 5000:
                # A launch call that blocks for >5 ms means the launch queue is full:
                # the stream is parked on something that isn't a kernel, e.g. lanex's
                # host-callback exchange. This is NOT CPU launch overhead.
                cause = f"stream blocked (queue full in {r[2]}): likely lanex exchange / host callback"
        else:
            cause = f"CPU busy: {o[2]}" if o else "unattributed"
            if o and "load_binary" in o[2]:
                cause = "lazy CUDA module load (first use of a shape; warm up to remove)"
        by_cause[cause[:110]] += g1 - g0
        rows.append((g1 - g0, cause, r[2] if r else "-", o[2] if o else "-"))
    if not rt and not ops:
        print("(no CPU-side events in this trace; gap attribution unavailable)\n")
        return
    covered = sum(g1 - g0 for g0, g1 in big)
    print(f"GPU idle gaps: {len(gaps)} total, {total/1e3:.1f} ms; "
          f"the largest {len(big)} cover {covered/1e3:.1f} ms")
    print("idle time by apparent cause (largest gaps):")
    for c, v in sorted(by_cause.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v/1e3:8.1f} ms  {c}")
    print(f"\n{top} largest gaps:")
    print(f"{'ms':>8}  cause  | longest-overlapping runtime call | most specific CPU op")
    for d, c, rn, on in rows[:top]:
        print(f"{d/1e3:8.2f}  {c[:60]} | {rn[:40]} | {on[:60]}")
    print()


def analyse(path):
    allev = load(path)
    evs = [e for e in allev
           if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not evs:
        print(f"{path}: no GPU kernel events found")
        return
    tot = defaultdict(float)
    per_k = defaultdict(lambda: [0, 0.0])
    ivs = []
    for e in evs:
        name, dur, ts = e.get("name", "?"), float(e.get("dur", 0)), float(e.get("ts", 0))
        if e.get("cat") != "kernel":
            name = ("memcpy " if e["cat"] == "gpu_memcpy" else "memset ") + name
        tot[bucket(name)] += dur
        per_k[name][0] += 1
        per_k[name][1] += dur
        ivs.append((ts, ts + dur))
    ivs.sort()
    busy, cs, ce = 0.0, ivs[0][0], ivs[0][1]
    gaps = []
    for s, e in ivs[1:]:
        if s > ce:
            busy += ce - cs
            if s - ce > 20:  # ignore sub-20 us launch spacing
                gaps.append((ce, s))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = max(e for _, e in ivs) - ivs[0][0]
    ksum = sum(tot.values())

    print(f"=== {path}")
    print(f"GPU window {span/1e3:.1f} ms, GPU busy {busy/1e3:.1f} ms "
          f"({100*busy/span:.1f}%), idle {(span-busy)/1e3:.1f} ms; "
          f"sum of kernel times {ksum/1e3:.1f} ms (>{busy/1e3:.1f} means overlapping streams)")
    print(f"\n{'category':<38} {'ms':>9} {'share':>7}")
    for b, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"{b:<38} {v/1e3:9.1f} {100*v/ksum:6.1f}%")
    print(f"\ntop 25 kernels by total time")
    print(f"{'ms':>9} {'count':>6} {'share':>6}  category / name")
    for name, (n, v) in sorted(per_k.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"{v/1e3:9.2f} {n:6d} {100*v/ksum:5.1f}%  [{bucket(name)}] {name[:140]}")
    print()
    dg = [(n, c, v) for n, (c, v) in per_k.items() if re.search(r"deep_gemm|sm100_|sm110_|fp8_fp4|m_grouped", n, re.I)]
    print(f"DeepGEMM-looking kernels: {len(dg)}"
          + ("" if not dg else f", {sum(v for _, _, v in dg)/1e3:.1f} ms total, e.g. {dg[0][0][:80]}"))
    print()
    gap_report(gaps, allev)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        analyse(p)

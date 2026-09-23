#!/usr/bin/env python3
"""Spin-wait hotfix for Jetson Thor TP runs (upstream vllm issue #79 family).

Default `busy_loop_s: float = 1` in vllm/distributed/device_communicators/
shm_broadcast.py makes the shared-memory
broadcast reader sleep (block) rather than spin for up to 1s while waiting for the
peer rank's message. On a 2-node Thor TP=2 pipeline this adds measurable latency to
EVERY engine step, and it is especially costly for speculative decoding, where each
step involves extra draft->verify round-trips.

The dual-Thor DSv4 tuning recipe (SonicBotMan/dsv4-dual-thor-tuning) patches this
to 0.002s and lists it as part of the production "stability trio".

Idempotent: backs up once to <file>.bak-pre-spin and skips if already patched.
"""
import glob
import shutil
import sys

PATTERNS = [
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py",
    "/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/shm_broadcast.py",
]

OLD = "busy_loop_s: float = 1,"
NEW = "busy_loop_s: float = 0.002,"

found = []
for p in PATTERNS:
    found.extend(glob.glob(p))
if not found:
    found = glob.glob("/**/vllm/distributed/device_communicators/shm_broadcast.py", recursive=True)[:1]

if not found:
    print("ERROR: shm_broadcast.py not found", file=sys.stderr)
    sys.exit(1)

path = found[0]
src = open(path).read()

if NEW in src:
    print(f"spin-wait patch ALREADY applied: {path}")
    sys.exit(0)

if OLD not in src:
    print(f"WARNING: pattern {OLD!r} not found in {path} (upstream may have changed)")
    sys.exit(2)

bak = path + ".bak-pre-spin"
try:
    shutil.copy2(path, bak)
except FileNotFoundError:
    pass  # backup already exists

src = src.replace(OLD, NEW, 1)
open(path, "w").write(src)
print(f"spin-wait patch applied: {path}  (busy_loop_s 1 -> 0.002)")

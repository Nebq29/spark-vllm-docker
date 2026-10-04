# sitecustomize.py — auto-installed by Python at interpreter startup.
# When LANEX_ENABLE=1, installs the 4-lane all_reduce hook into vLLM's
# GroupCoordinator before any worker code runs.
import os

if os.environ.get("LANEX_ENABLE", "0") == "1":
    try:
        import lanex
        lanex.install()
    except Exception as e:  # never break interpreter startup
        print(f"[sitecustomize] lanex install failed: {e!r}", flush=True)

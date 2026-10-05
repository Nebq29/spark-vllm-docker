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

# DG_MK_ALIGN: override DeepGEMM's runtime mk_alignment_for_contiguous_layout
# (default 128) in WORKER processes before vLLM first queries it. Applied
# lazily on the first call to vllm.utils.deep_gemm.get_mk_alignment_for_contiguous_layout
# via a one-shot import hook, so the API-server process (which never runs MoE
# GEMMs) is unaffected and the set happens after deep_gemm is importable.
_dg_align = os.environ.get("DG_MK_ALIGN")
if _dg_align:
    try:
        import importlib.abc
        import importlib.util
        import sys

        class _DGAlignFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name != "vllm.utils.deep_gemm":
                    return None
                try:
                    sys.meta_path.remove(self)
                except ValueError:
                    pass
                spec = importlib.util.find_spec(name)
                if spec is None or spec.loader is None:
                    return None
                loader = spec.loader

                class _PatchingLoader:
                    def create_module(self, s):
                        if hasattr(loader, "create_module"):
                            return loader.create_module(s)
                        return None

                    def exec_module(self, m):
                        loader.exec_module(m)
                        _orig_get = m.get_mk_alignment_for_contiguous_layout
                        _applied = []
                        align_val = int(_dg_align or 0)

                        def _get():
                            if not _applied:
                                _applied.append(1)
                                try:
                                    m.set_mk_alignment_for_contiguous_layout(
                                        align_val
                                    )
                                    print(
                                        f"[sitecustomize] DG_MK_ALIGN={align_val} "
                                        f"applied (pid={os.getpid()})",
                                        flush=True,
                                    )
                                except Exception as e:
                                    print(
                                        f"[sitecustomize] DG_MK_ALIGN failed: {e!r}",
                                        flush=True,
                                    )
                            return _orig_get()

                        m.get_mk_alignment_for_contiguous_layout = _get
                        return None

                spec.loader = _PatchingLoader()
                return spec

        sys.meta_path.insert(0, _DGAlignFinder())
    except Exception as e:  # never break interpreter startup
        print(f"[sitecustomize] DG_MK_ALIGN hook install failed: {e!r}", flush=True)

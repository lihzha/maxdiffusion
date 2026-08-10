# Venv-scoped test-environment workaround (exp_06 F3, 2026-08-10) — NOT repo code.
# grain 0.2.18 / array_record 0.8.3 macOS wheels segfault or deadlock at native-extension load
# in every version combination tried (incl. the production-exact tf 2.21.0 + protobuf 6.33.6 set;
# production Linux is unaffected). Nothing in the exp_06 M1 boundary, its suite, or the 80-probe
# battery uses grain (verified by grep 2026-08-10): it is imported only transitively by
# maxdiffusion's generic HF-streaming input pipeline. This stub lets that import complete and
# FAILS LOUDLY if any code ever actually touches a grain attribute.
import os, sys, types

if os.environ.get("POS_SITECUSTOMIZE_SKIP") != "1" and "grain" not in sys.modules:

    class _LoudStub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)  # introspection (hasattr __file__ etc.) sees a normal module
            raise RuntimeError(
                f"grain.{name} accessed, but grain is STUBBED in this macOS test venv "
                "(native wheel segfaults — see exp_06 rollout_adapter_worklog 2026-08-10). "
                "Any test that genuinely needs grain must not run in this venv."
            )

    _MSG = (
        "grain is STUBBED in this macOS test venv (native wheel segfaults - see exp_06 "
        "rollout_adapter_worklog 2026-08-10); constructing/using grain objects is forbidden here."
    )

    class _StubBase:
        """Subclassable at import time (class Foo(grain.X)); any construction or use fails loudly."""

        def __init__(self, *a, **k):
            raise RuntimeError(_MSG)

        def __init_subclass__(cls, **k):
            pass  # subclass definitions are harmless; only instantiation raises

    _CLASS_NAMES = (
        "RandomAccessDataSource",
        "ArrayRecordDataSource",
        "MapTransform",
        "Batch",
        "DataLoader",
        "IndexSampler",
        "ReadOptions",
        "ShardOptions",
    )

    _g = _LoudStub("grain")
    _gp = _LoudStub("grain.python")
    for _n in _CLASS_NAMES:
        setattr(_g, _n, type(_n, (_StubBase,), {}))
        setattr(_gp, _n, getattr(_g, _n))
    _g.python = _gp
    _g.__path__ = []  # mark as package so 'import grain.python' resolves
    sys.modules["grain"] = _g
    sys.modules["grain.python"] = _gp

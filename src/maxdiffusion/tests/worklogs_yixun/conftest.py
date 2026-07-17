"""Shared pytest fixtures/setup for the exp_01 full-FT worklog tests.

Hoisted from the round-1 test file so both ``test_full_ft_overfit_shared_objective``
and ``test_full_ft_overfit_denoising_loss`` rely on a single copy (Codex round-1
review flagged the inline duplicate). Runs at conftest import — i.e. before pytest
imports the test modules in this directory — so the grain stub is installed before
any test module pulls in the trainer (which imports ``input_pipeline_interface``).
"""

from __future__ import annotations

import sys
import types

if sys.platform == "darwin" and "grain" not in sys.modules:
    # The tensorflow and grain macOS-arm64 wheels segfault when imported into one
    # process (native symbol clash; reproduced with TF 2.21 + grain 0.2.16/17/18,
    # both import orders: `python -c "import tensorflow, grain.python"` exits 139).
    # Importing the trainer module pulls both via input_pipeline_interface, though
    # the side-adapter tfrecord path never executes grain code. Stub just enough of
    # ``grain.python`` for those module-level imports (two class bases). On linux
    # (TPU / CI) the real grain imports fine and no stub is installed.
    _grain = types.ModuleType("grain")
    _grain_python = types.ModuleType("grain.python")
    _grain_python.MapTransform = type("MapTransform", (), {})
    _grain_python.RandomAccessDataSource = type("RandomAccessDataSource", (), {})
    _grain.python = _grain_python
    sys.modules["grain"] = _grain
    sys.modules["grain.python"] = _grain_python

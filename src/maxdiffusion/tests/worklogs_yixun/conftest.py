"""Shared fixtures for exp_06's evaluation tests.

The ``gs://`` round trip is needed by three files (the anchor protocol, the gates, and the
end-to-end path), and importing a fixture across test modules makes the fixture name look like a
redefinition to every linter. It lives here instead, which is pytest's own answer.

Nothing here is auto-applied: a fixture is inert until a test asks for it by name, so the other
experiments' files in this directory are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class FakeGfile:
    """An in-memory stand-in for ``tensorflow.io.gfile``, so a ``gs://`` round trip is executable.

    Local paths fall through to the real filesystem: the J0 manifests this evaluator reads are
    checked into the tree, and a fake that swallowed them would be testing itself.
    """

    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.dirs: list[str] = []

    @staticmethod
    def _remote(path):
        from maxdiffusion import pos_rollout_support

        return pos_rollout_support.is_remote(path)

    def exists(self, path):
        return str(path) in self.blobs if self._remote(path) else Path(str(path)).exists()

    def makedirs(self, path):
        if self._remote(path):
            self.dirs.append(str(path))
        else:
            Path(str(path)).mkdir(parents=True, exist_ok=True)

    def GFile(self, path, mode="rb"):  # noqa: N802 - the name is tensorflow's
        if not self._remote(path):
            return open(str(path), mode)  # noqa: SIM115 - the caller owns the handle
        blobs, key = self.blobs, str(path)

        class _Handle:
            def read(self):
                return blobs[key]

            def write(self, payload):
                blobs[key] = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Handle()


@pytest.fixture
def fake_gs(monkeypatch):
    """Route exp_06's storage layer at an in-memory bucket for the duration of one test."""
    from maxdiffusion import pos_rollout_support

    fake = FakeGfile()
    monkeypatch.setattr(pos_rollout_support, "_gfile", lambda path: fake)
    return fake

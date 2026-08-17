"""exp_06 `rollout_adapter` — F5: a cell is published the moment it is measured, and adopted the
moment it is verified.

**The production failure this round answers.** M1's ladder is 16 cells x 2 arms x 2 trials with a
full backbone reload per cell — about 3.5 hours — and it published its authorization table ONLY at
the end. The us-east1-d zone killed seven VMs in one day at lifetimes of 30 minutes to 2 hours;
attempt 2 measured **24 of 32 cells** and then took a chip-level fatal, and `fit_probe/` was left
EMPTY. Five attempts have since re-measured the same cells byte-identically — the pipeline is
deterministic (rollout mb=8 k=2 stepped 25.347 / 25.353 / 25.356 s across attempts, peaks
bit-identical) — so every one of those hours bought a number the campaign already had.

Two changes, and the second is the one with teeth:

1. **Publish per cell.** As soon as a cell's trials finish, that cell's trials, peak, peak source,
   step times and the context they were measured under are written to the attempt root as
   ``cells/<arm>_m<microbatch>_k<k>.json``, digest-bound exactly the way the run-level table is,
   with a ``.digest`` sidecar written LAST so that "adoptable" is observable rather than hoped for.
2. **Adopt what is already published.** Before measuring a cell the probe looks for a published
   artifact for that cell under a configured adoption root and adopts it instead of re-measuring —
   but only after the digest verifies over the content, the recorded context is byte-for-byte the
   context this process derived (which is what carries the code SHA, the recipe fingerprint, the
   model revision, the geometry AND the device count), the run identity matches, and the trial count
   is the one this ladder asks for. **Anything else re-measures.**

**Adoption is bound to the CONTEXT DIGEST, and that is a deliberately coarse policy.** The context
digest contains ``code_sha``, so *any* commit — a one-line comment, a docs-only descendant — makes
every published cell unadoptable. That over-refuses: it costs a re-measure of cells whose footprint
certainly did not change. The alternative is a curated list of "code that matters", which is a list
somebody has to remember to extend, and the failure mode of forgetting is publishing an HBM
authorization for a program nobody measured. This module already chose noisy over-binding once (the
``FINGERPRINT_EXCLUSIONS`` denylist); this is the same choice about the same risk.

**The six known defects of adopt-if-published designs, and where each is refused here.**

* (a) *adopting an artifact produced by different code* — the context digest carries ``code_sha``;
  :func:`test_a_cell_measured_on_another_commit_is_re_measured`.
* (b) *adopting a partial or corrupt write* — the digest covers the whole payload, the content file
  is staged and renamed rather than streamed into place, and the ``.digest`` sidecar is the commit
  marker; :func:`test_a_corrupt_cell_artifact_is_re_measured`,
  :func:`test_a_content_file_without_its_sidecar_is_not_adoptable`.
* (c) *another job's cells leaking in* — every cell artifact names the run that measured it;
  :func:`test_a_foreign_job_identity_is_refused_and_re_measured`.
* (d) *the empty-adoption case must behave exactly like today* —
  :func:`test_with_no_adoption_root_the_probe_measures_exactly_what_it_measured_before`.
* (e) *a cell measured under a different device topology* — ``device_count`` is inside the context
  digest; :func:`test_a_cell_measured_on_a_different_topology_is_re_measured`.
* (f) *a trust-chain gap around adopted content* — adopted trials enter the evidence as
  measurements, so the run-level digest covers them and ``load_authorization`` re-decides them;
  :func:`test_the_run_level_digest_covers_adopted_content`.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.machinery
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

from maxdiffusion import pos_rollout_fit_probe as probe
from maxdiffusion import pos_rollout_support as support

# ---------------------------------------------------------------------------------------------
# The same import shims the fit-probe suite installs: `max_utils` reaches four third-party packages
# this environment does not have, none of them on the measurement path. Copied rather than imported
# so this file stands alone (importing helpers across test modules makes collection order load
# bearing, which is the last thing an adoption test should depend on).
# ---------------------------------------------------------------------------------------------


def _stub_leaf(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__file__ = f"<stub {name}>"

    class _Stub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"{name} is stubbed: it is not part of the fit probe")

    def _getattr(attribute, _stub=_Stub):
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        return _stub

    module.__getattr__ = _getattr
    return module


for _name in ("transformers", "safetensors"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = _stub_leaf(_name)


_CONFIG_PATH = Path(probe.__file__).resolve().parent / "configs" / "base_wan_5b_pos_rollout.yml"
_CAPACITY = 32 * 1024**3
_RUN = "exp06-m1-20260812"


class _Device:
    def __init__(self, kind="v6e"):
        self.device_kind = kind


@functools.lru_cache(maxsize=1)
def _local_model_dir() -> str:
    """Provenance is CONTENT-bound, so every test config names a real directory with real bytes."""
    directory = Path(tempfile.mkdtemp(prefix="exp06_cellpub_model_")) / "snapshot"
    (directory / "transformer").mkdir(parents=True)
    (directory / "transformer" / "weights.safetensors").write_bytes(b"w" * 512)
    (directory / "model_index.json").write_text('{"_class_name": "test"}')
    return str(directory)


def _config(**overrides):
    import yaml

    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values["pretrained_model_name_or_path"] = _local_model_dir()
    values.setdefault("run_name", _RUN)
    values["run_name"] = values["run_name"] or _RUN
    values.update(overrides)

    class _Config:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

        def get_keys(self):
            return dict(self.__dict__)

    return _Config(values)


def _context(config=None, devices=None, **overrides):
    derived = probe.derive_probe_context(
        config or _config(), devices=devices or [_Device() for _ in range(8)], environ={}
    )
    return dataclasses.replace(derived, **overrides) if overrides else derived


def _measurement(context, cell, *, peak=20 * 1024**3, step=3.5):
    return probe.CellMeasurement(
        cell=cell,
        context_digest=context.binding_digest(),
        compile_seconds=480.0,
        step_seconds=step,
        eval_seconds=600.0,
        checkpoint_seconds=90.0,
        peak_bytes=peak,
        capacity_bytes=_CAPACITY,
        reservation_failures=0,
        # F10: the authorized bound is the compiled analysis and the watermark is the cross-check,
        # so the fixture standing in for a fitting cell carries both, in M1-9's measured shape (the
        # runtime mark far below the analysis).
        peak_source=probe.PEAK_SOURCE_ANALYSIS,
        peak_attribution=probe.PEAK_ATTRIBUTION_NONE,
        analysis_bytes=peak,
        watermark_bytes=4 * 1024**3,
        watermark_before_bytes=3 * 1024**3,
    )


class _CountingMeasurer:
    """A deterministic host stand-in that COUNTS. Adoption's whole claim is a call that never happens.

    The step time is a function of the cell alone, so two attempts measure byte-identically — which
    is what production does (the campaign measured 25.347 / 25.353 / 25.356 s for one cell across
    three attempts and bit-identical peaks) and what makes "the table is the same table" checkable.
    """

    def __init__(self, *, peaks=None, die_after=None):
        self.calls: list[probe.FitCell] = []
        self.peaks = peaks or {}
        self.die_after = die_after

    def __call__(self, *, cell, context, config):
        if self.die_after is not None and len(self.calls) >= self.die_after:
            raise _ZoneKilledTheVM(f"the VM died after {len(self.calls)} trials")
        self.calls.append(cell)
        peak = self.peaks.get((cell.arm, cell.microbatch, cell.k_b), 20 * 1024**3)
        return _measurement(context, cell, peak=peak, step=3.5 * cell.k_b + cell.microbatch / 100.0)

    @property
    def cells(self) -> list[probe.FitCell]:
        return sorted(set(self.calls), key=lambda cell: (cell.arm, cell.microbatch, cell.k_b))


class _ZoneKilledTheVM(RuntimeError):
    """Not a bug in the probe: the failure mode this whole round exists to survive."""


#: In LADDER_ORDER, which is the order it EXECUTES in: F9c sorts every requested sequence into the
#: declared order, so `rollout` mb=8 -- the one cell above the headroom floor -- is always last.
#: The `die_after` tests below depend on which cells finish first, so the order has to be visible.
_LADDER = (probe.FitCell("one_step", 8, 2), probe.FitCell("rollout", 16, 2), probe.FitCell("rollout", 8, 2))


def _run(config, measurer, *, cells=_LADDER, trials=2, devices=None):
    return probe.run_fit_probe(
        config,
        measurer=measurer,
        cells=list(cells),
        trials=trials,
        devices=devices or [_Device() for _ in range(8)],
    )


def _attempt(tmp_path: Path, name: str) -> str:
    """One attempt root shaped the way the launcher shapes it: the authorization beside a cells/ dir."""
    root = tmp_path / "m1" / _RUN / "fit_probe" / "attempts" / name
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "fit_authorization.json")


def _adoption_root(tmp_path: Path) -> str:
    return str(tmp_path / "m1")


def _without_provenance(table: dict) -> dict:
    stripped = {key: value for key, value in table.items() if key not in ("cell_provenance", "sha256")}
    return stripped


# =============================================================================================
# 1. The storage primitives adoption needs: a bounded listing, and a publication that is never
#    observed half-written.
# =============================================================================================


def test_the_storage_layer_can_list_a_directorys_children(tmp_path):
    (tmp_path / "att-1").mkdir()
    (tmp_path / "att-2").mkdir()
    (tmp_path / "loose.json").write_text("{}")
    assert support.storage_list_children(str(tmp_path)) == ["att-1", "att-2", "loose.json"]
    assert support.storage_list_children(str(tmp_path / "nothing-here")) == []


def test_a_publication_is_staged_and_renamed_rather_than_streamed_into_place(tmp_path):
    """Defect (b). A reader must never see a prefix of the bytes. The staging file is gone after."""
    target = tmp_path / "deep" / "cells" / "rollout_m8_k2.json"
    support.storage_publish_bytes(str(target), b'{"payload": 1}')
    assert target.read_bytes() == b'{"payload": 1}'
    leftovers = [path for path in (tmp_path / "deep" / "cells").rglob("*") if path != target]
    assert leftovers == [], f"the staging write was left behind: {leftovers}"


def test_a_failed_publication_leaves_no_artifact_at_the_destination(tmp_path, monkeypatch):
    """The half-written artifact is the one that gets adopted. It must not exist to be found."""
    target = tmp_path / "cells" / "rollout_m8_k2.json"

    def explode(*args, **kwargs):
        raise OSError("the VM died between the write and the rename")

    monkeypatch.setattr(support, "_storage_rename", explode)
    with pytest.raises(OSError):
        support.storage_publish_bytes(str(target), b"partial")
    assert not target.exists(), "a destination that exists after a failed publish is an adoptable lie"


def test_the_storage_primitives_work_over_a_gs_uri(tmp_path, fake_gs):
    """Production publishes to ``gs://``; a primitive that only works on a laptop proves nothing."""
    support.storage_publish_bytes("gs://bucket/m1/att-1/cells/rollout_m8_k2.json", b"content")
    assert support.storage_read_bytes("gs://bucket/m1/att-1/cells/rollout_m8_k2.json") == b"content"
    assert "cells" in support.storage_list_children("gs://bucket/m1/att-1")
    assert support.storage_list_children("gs://bucket/m1/att-1/cells") == ["rollout_m8_k2.json"]


# =============================================================================================
# 2. The cell artifact: what a finished cell writes down, and what makes it readable again.
# =============================================================================================


def test_a_finished_cell_is_published_immediately_with_a_verifying_digest(tmp_path):
    """The headline. One cell measured on the host stand-in, and its artifact is on disk BEFORE the
    ladder ends — the 24-of-32 attempt would have banked 24 cells instead of nothing."""
    path = _attempt(tmp_path, "att-1")
    config = _config(pos_fit_authorization=path)
    cell = probe.FitCell("one_step", 8, 2)  # cell 1 of the declared order (F9c)
    measurer = _CountingMeasurer(die_after=2)  # two trials of cell 1, then the VM dies

    with pytest.raises(_ZoneKilledTheVM):
        _run(config, measurer)

    marker = probe.cell_marker_path(path, cell)
    assert Path(marker).exists(), "the cell that finished must be committed although the ladder died"
    content = _content_of(marker)
    assert content.exists(), "the marker commits an object that is there"

    stored = json.loads(content.read_text())
    digest = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode()).hexdigest()
    assert digest == stored["sha256"] == Path(marker).read_text().strip()
    assert digest in content.name, "content objects are named by their WHOLE digest (F5c MINOR)"

    payload = stored["payload"]
    assert payload["protocol"] == probe.CELL_PROTOCOL
    assert payload["cell"] == cell.as_payload()
    assert payload["job_identity"] == _RUN
    assert payload["trial_count"] == 2 and len(payload["trials"]) == 2
    context = _context(config)
    assert payload["context"] == context.as_payload() and payload["context_digest"] == context.digest()
    assert payload["recipe_fingerprint"] == context.recipe_fingerprint
    assert payload["device_count"] == 8 and payload["code_sha"] == context.code_sha
    assert payload["context"]["manifest_digest"] == probe.deployed_manifest_digest()


def test_the_published_content_is_the_measurement_that_was_taken(tmp_path):
    """Not a summary of it: the trials themselves, so the aggregation on adoption is the same one."""
    path = _attempt(tmp_path, "att-1")
    config = _config(pos_fit_authorization=path)
    cell = probe.FitCell("rollout", 16, 2)
    measurer = _CountingMeasurer()
    _run(config, measurer, cells=[cell])

    artifact = probe.load_cell_artifact(probe.cell_marker_path(path, cell))
    context = _context(config)
    expected = [_measurement(context, cell, step=3.5 * 2 + 0.16).as_payload() for _ in range(2)]
    assert [trial.as_payload() for trial in artifact.trials] == expected
    assert artifact.trials[0].peak_bytes == 20 * 1024**3
    assert artifact.trials[0].peak_source == probe.PEAK_SOURCE_ANALYSIS
    assert artifact.trials[0].analysis_bytes == 20 * 1024**3, "F10: the bound is banked with the cell"
    assert artifact.trials[0].watermark_bytes == 4 * 1024**3, "...and so is the reading that cross-checks it"


def test_a_content_object_without_its_marker_is_not_adoptable(tmp_path):
    """Defect (b), the torn pair. The content object lands first and the marker commits it; a crash
    between them leaves an object nothing points at, which is the safe way round."""
    path = _attempt(tmp_path, "att-1")
    config = _config(pos_fit_authorization=path)
    cell = probe.FitCell("rollout", 8, 2)
    _run(config, measurer=_CountingMeasurer(), cells=[cell])
    marker = probe.cell_marker_path(path, cell)
    assert _content_of(marker).exists()
    Path(marker).unlink()

    with pytest.raises(ValueError, match="did not finish"):
        probe.load_cell_artifact(marker)


def test_a_cell_artifact_is_published_once_and_never_rewritten(tmp_path, capsys):
    """Issue #10's rule, applied where it belongs. The run-level authorization RAISES on a differing
    republication because it is the gate; a cell artifact is a cache in front of a 3.5-hour ladder,
    so a collision keeps the published bytes, says so, and lets the ladder finish."""
    path = _attempt(tmp_path, "att-1")
    config = _config(pos_fit_authorization=path)
    cell = probe.FitCell("rollout", 8, 2)
    context = _context(config)
    marker = probe.cell_marker_path(path, cell)

    first = probe.CellArtifact(
        cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=1.0),)
    )
    probe.publish_cell(marker, first)
    second = probe.CellArtifact(
        cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=2.0),)
    )
    probe.publish_cell(marker, second)

    assert probe.load_cell_artifact(marker).trials[0].step_seconds == 1.0, "the first publication stands"
    assert "never rewritten" in capsys.readouterr().out


def test_an_incomplete_publication_is_completed_rather_than_treated_as_published(tmp_path):
    """The other half of "published once": content without its sidecar is NOT a published artifact.

    A crash between the content rename and the sidecar write leaves a path that
    :func:`load_cell_artifact` will never accept. If publication treated that path as taken, the cell
    would be unadoptable for the life of the tree — a cache poisoned by the exact crash this round
    exists to survive. An incomplete publication is therefore re-published, and issue #10's
    never-rewrite rule applies to COMPLETE artifacts, which is what it was always about."""
    path = _attempt(tmp_path, "att-1")
    config = _config(pos_fit_authorization=path)
    cell = probe.FitCell("rollout", 8, 2)
    context = _context(config)
    marker = probe.cell_marker_path(path, cell)
    artifact = probe.CellArtifact(
        cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=1.0),)
    )
    probe.publish_cell_content(marker, artifact)  # the crash before the marker committed

    probe.publish_cell(marker, artifact)
    assert Path(marker).exists(), "the wreck must be completed, not left uncommitted"
    assert probe.load_cell_artifact(marker).trials[0].step_seconds == 1.0


# =============================================================================================
# 3. Finding a published cell: the scan, and its bounds.
# =============================================================================================


def test_the_scan_finds_cells_under_both_root_layouts(tmp_path):
    """The launcher derives ``<OUTPUT_DIR>/<RUN_NAME>/fit_probe/attempts/att-X/`` and the submit
    wrapper adds ANOTHER attempt level above ``OUTPUT_DIR`` — so a cells/ directory sits 2 levels
    under the attempts root and 6 under the wrapper's M1 root. Both are this job's own tree."""
    shallow = tmp_path / "attempts" / "att-1" / "cells"
    deep = tmp_path / "att-A" / _RUN / "fit_probe" / "attempts" / "att-B" / "cells"
    for directory in (shallow, deep):
        directory.mkdir(parents=True)
        (directory / "rollout_m8_k2.json.digest").write_text("0" * 64)

    cell = probe.FitCell("rollout", 8, 2)
    marker = "rollout_m8_k2.json.digest"
    assert probe.adoption_candidates(str(tmp_path / "attempts"), cell) == (str(shallow / marker),)
    assert str(deep / marker) in probe.adoption_candidates(str(tmp_path), cell)


def test_the_scan_is_depth_bounded_and_visit_bounded(tmp_path, capsys):
    """An adoption root pointed at a bucket root must not walk the bucket. Both bounds are declared
    constants, and exceeding the visit bound says so rather than continuing quietly."""
    too_deep = tmp_path.joinpath(*[f"level{i}" for i in range(probe.ADOPTION_SCAN_DEPTH)], "cells")
    too_deep.mkdir(parents=True)
    (too_deep / "rollout_m8_k2.json.digest").write_text("0" * 64)
    assert probe.adoption_candidates(str(tmp_path), probe.FitCell("rollout", 8, 2)) == ()

    wide = tmp_path / "wide"
    for index in range(12):
        (wide / f"d{index:05d}").mkdir(parents=True)
    assert probe.adoption_candidates(str(wide), probe.FitCell("rollout", 8, 2), limit=8) == ()
    assert "adoption scan stopped" in capsys.readouterr().out
    assert probe.ADOPTION_SCAN_LIMIT >= 64, "the deployed bound must be far above any real attempt tree"


def test_an_absent_adoption_root_is_not_an_error(tmp_path):
    assert probe.adoption_candidates(str(tmp_path / "never-created"), probe.FitCell("rollout", 8, 2)) == ()
    assert probe.adoption_candidates("", probe.FitCell("rollout", 8, 2)) == ()


# =============================================================================================
# 4. The round trip: a ladder that dies, restarted, measures only what is left.
# =============================================================================================


def _first_attempt(tmp_path, *, die_after):
    config = _config(pos_fit_authorization=_attempt(tmp_path, "att-1"), pos_fit_adoption_root=_adoption_root(tmp_path))
    measurer = _CountingMeasurer(die_after=die_after)
    with pytest.raises(_ZoneKilledTheVM):
        _run(config, measurer)
    return measurer


def test_a_restart_adopts_the_finished_cells_and_measures_only_the_rest(tmp_path):
    """The round trip, and the claim is a call count: an adopted cell is a cell the measurer is never
    asked about. Attempt 1 dies with 2 of 3 cells finished; attempt 2 measures exactly one."""
    first = _first_attempt(tmp_path, die_after=4)  # 2 cells x 2 trials, then the VM dies
    assert first.cells == [probe.FitCell("one_step", 8, 2), probe.FitCell("rollout", 16, 2)]

    second_config = _config(
        pos_fit_authorization=_attempt(tmp_path, "att-2"), pos_fit_adoption_root=_adoption_root(tmp_path)
    )
    second = _CountingMeasurer()
    table = _run(second_config, second)

    assert second.cells == [probe.FitCell("rollout", 8, 2)], "the two published cells were not re-measured"
    assert len(second.calls) == 2, "one cell, two trials — and nothing else"
    assert len(table["measured_cells"]) == 3, "the table still covers the whole ladder"
    assert len(table["authorized_cells"]) == 3


def test_the_restarted_table_is_the_uninterrupted_table(tmp_path):
    """Adoption must not change WHAT is authorized or the numbers it is authorized from. Everything
    outside the provenance record is compared as canonical bytes, which is the comparison the digest
    makes."""
    _first_attempt(tmp_path, die_after=4)
    resumed = _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "att-2"), pos_fit_adoption_root=_adoption_root(tmp_path)),
        _CountingMeasurer(),
    )
    uninterrupted = _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "clean")),
        _CountingMeasurer(),
    )
    assert json.dumps(_without_provenance(resumed), sort_keys=True) == json.dumps(
        _without_provenance(uninterrupted), sort_keys=True
    )


def test_the_table_records_which_cells_were_adopted_and_from_where(tmp_path):
    _first_attempt(tmp_path, die_after=4)
    path = _attempt(tmp_path, "att-2")
    table = _run(
        _config(pos_fit_authorization=path, pos_fit_adoption_root=_adoption_root(tmp_path)), _CountingMeasurer()
    )

    provenance = {(row["arm"], row["microbatch"], row["k_b"]): row["provenance"] for row in table["cell_provenance"]}
    assert provenance[("rollout", 8, 2)] == probe.PROVENANCE_MEASURED
    for cell in (("one_step", 8, 2), ("rollout", 16, 2)):
        assert provenance[cell].startswith(probe.ADOPTED_PREFIX), provenance[cell]
        assert "att-1" in provenance[cell], "an adoption names the artifact it adopted"
    assert probe.load_authorization(path)["cell_provenance"] == table["cell_provenance"]


def test_an_adopted_cell_is_not_republished_into_the_new_attempt(tmp_path):
    """A cell artifact is evidence about one measurement, not a thing to copy around: the adopting
    attempt records where it came from instead of minting a second copy that looks like a second
    measurement."""
    _first_attempt(tmp_path, die_after=4)
    path = _attempt(tmp_path, "att-2")
    _run(_config(pos_fit_authorization=path, pos_fit_adoption_root=_adoption_root(tmp_path)), _CountingMeasurer())
    written = sorted(Path(probe.cell_publication_dir(path)).glob("*.json"))
    assert len(written) == 1 and written[0].name.startswith("rollout_m8_k2."), written


def test_with_no_adoption_root_the_probe_measures_exactly_what_it_measured_before(tmp_path):
    """Defect (d). A fresh run with no adoption root does the same work in the same order and
    publishes the same table — the only difference is the per-cell artifacts it now leaves behind."""
    _first_attempt(tmp_path, die_after=4)
    measurer = _CountingMeasurer()
    table = _run(_config(pos_fit_authorization=_attempt(tmp_path, "att-3")), measurer)
    assert len(measurer.calls) == 6, "no adoption root means nothing is adopted, however much is published"
    assert [row["provenance"] for row in table["cell_provenance"]] == [probe.PROVENANCE_MEASURED] * 3


# =============================================================================================
# 5. The refusal paths. Every one of them re-measures rather than failing the run.
# =============================================================================================


def _publish_one(tmp_path, attempt="att-1", *, cell=probe.FitCell("rollout", 8, 2), config=None):
    path = _attempt(tmp_path, attempt)
    config = config or _config(pos_fit_authorization=path)
    _run(config, _CountingMeasurer(), cells=[cell])
    return probe.cell_marker_path(path, cell)


def _resume(tmp_path, *, cell=probe.FitCell("rollout", 8, 2), attempt="att-2"):
    measurer = _CountingMeasurer()
    table = _run(
        _config(pos_fit_authorization=_attempt(tmp_path, attempt), pos_fit_adoption_root=_adoption_root(tmp_path)),
        measurer,
        cells=[cell],
    )
    return measurer, table


def _content_of(marker: str) -> Path:
    """The content object the marker commits to."""
    digest = Path(marker).read_text().strip()
    return Path(probe._content_for_marker(marker, digest))


def _damage(marker: str, mutate, *, resync=True):
    """Edit a published cell the way somebody who wanted it adopted would: consistently.

    The artifact's header repeats three fields of its own context, and its digest covers everything,
    so an edit that leaves any of those stale is caught by the artifact disagreeing with ITSELF —
    which proves nothing about whether adoption checks it against THIS process. Every derived field
    is therefore recomputed here before the re-hash, so the only thing left to refuse the artifact is
    the comparison this round is about.

    **F10e: the derived fields are computed from the value production WOULD PARSE, not the raw one,
    and without going through ``ProbeContext``** (which now refuses the malformed payloads these
    forgeries need to write). Hashing the raw ``8.5`` would make the artifact self-inconsistent under
    a REGRESSED production that truncated it to ``8`` — so the case would still be refused, on three
    digest mismatches, while being blind to the truncation it exists to catch.
    """
    content = _content_of(marker)
    stored = json.loads(content.read_text())
    payload = stored["payload"]
    mutate(payload)
    if resync and isinstance(payload.get("context"), dict):
        recorded = payload["context"]
        normalized = dict(recorded)
        try:  # what a bare-`int()` reader would rebuild
            normalized["device_count"] = int(recorded["device_count"])
        except (TypeError, ValueError, KeyError):
            pass
        payload["context_digest"] = probe._digest(normalized)
        payload["code_sha"] = normalized["code_sha"]
        payload["device_count"] = normalized["device_count"]
        payload["recipe_fingerprint"] = normalized["recipe_fingerprint"]
        binding = probe._digest({key: normalized[key] for key in probe.ProbeContext.BINDING_FIELDS})
        for trial in payload["trials"]:
            trial["context_digest"] = binding
    # The forger REPUBLISHES properly -- new content object at its own digest, marker moved to it,
    # old object removed. Anything less would be refused by the content-addressing rather than by the
    # binding under test, and the test would then be measuring the wrong refusal.
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    Path(probe._content_for_marker(marker, digest)).write_text(
        json.dumps({"payload": payload, "sha256": digest}, sort_keys=True)
    )
    content.unlink(missing_ok=True)
    Path(marker).write_text(digest + "\n")


def test_a_corrupt_cell_artifact_is_re_measured(tmp_path, capsys):
    """Defect (b). The digest does not describe the bytes, so the bytes are not evidence."""
    published = _publish_one(tmp_path)
    content = _content_of(published)
    stored = json.loads(content.read_text())
    stored["payload"]["trials"][0]["peak_bytes"] = 1
    content.write_text(json.dumps(stored, sort_keys=True))  # digest NOT recomputed

    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2, "a cell whose digest does not describe it is measured again"
    assert "digest" in capsys.readouterr().out


def test_a_truncated_cell_artifact_is_re_measured(tmp_path):
    published = _publish_one(tmp_path)
    content = _content_of(published)
    content.write_text(content.read_text()[: len(content.read_text()) // 2])
    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2


@pytest.mark.parametrize(
    "knob, value",
    [
        ("pos_rollout_support_salt", 9),
        ("side_adapter_sampling_steps", 41),
        ("pos_logical_batch", 128),
    ],
    ids=["support-salt", "sampling-steps", "logical-batch"],
)
def test_a_recipe_knob_that_moves_the_fingerprint_forces_a_re_measure(tmp_path, knob, value):
    """The fingerprint is what the adoption policy is bound to: a footprint-bearing knob moves it,
    and a cell measured under the old recipe is a measurement of another program."""
    published = _publish_one(tmp_path)
    assert Path(published).exists()
    measurer = _CountingMeasurer()
    _run(
        _config(
            pos_fit_authorization=_attempt(tmp_path, "att-2"),
            pos_fit_adoption_root=_adoption_root(tmp_path),
            **{knob: value},
        ),
        measurer,
        cells=[probe.FitCell("rollout", 8, 2)],
    )
    assert len(measurer.calls) == 2, f"{knob} moved the recipe fingerprint and the cell must be re-measured"


def test_a_cell_measured_on_a_different_topology_is_re_measured(tmp_path, capsys):
    """Defect (e). A v6e-8 peak is not a v6e-64 peak; ``device_count`` is inside the context digest."""
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload["context"].update(device_count=64))
    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2
    assert "device_count" in capsys.readouterr().out


def test_a_foreign_job_identity_is_refused_and_re_measured(tmp_path, capsys):
    """Defect (c). Another run's cells sitting under a shared root do not become this run's evidence."""
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload.update(job_identity="exp06-m1-SOMEBODY-ELSE"))
    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2
    assert "job" in capsys.readouterr().out.lower()


def test_an_unidentified_run_can_neither_adopt_nor_be_adopted(tmp_path):
    """An empty run identity would match every other empty one, so it is refused on both sides rather
    than being treated as a wildcard. The named control is the other half: identical in every respect
    except that the run says who it is, and it adopts."""
    cell = probe.FitCell("rollout", 8, 2)
    for named, expected in ((False, 2), (True, 0)):
        root = tmp_path / ("named" if named else "nameless")
        run_name = _RUN if named else ""
        _run(
            _config(pos_fit_authorization=_attempt(root, "att-1"), run_name=run_name),
            _CountingMeasurer(),
            cells=[cell],
        )
        measurer = _CountingMeasurer()
        _run(
            _config(
                pos_fit_authorization=_attempt(root, "att-2"),
                pos_fit_adoption_root=_adoption_root(root),
                run_name=run_name,
            ),
            measurer,
            cells=[cell],
        )
        assert len(measurer.calls) == expected, f"run_name={run_name!r}"


def test_a_cell_published_with_fewer_trials_than_this_ladder_runs_is_re_measured(tmp_path):
    """Two trials exist because one cannot show a cell that only fits when the neighbours are idle.
    Adopting a one-trial artifact into a two-trial ladder would quietly weaken that."""
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload.update(trials=payload["trials"][:1], trial_count=1))
    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2


def test_an_artifact_describing_another_cell_is_refused(tmp_path):
    """The path says one cell and the content says another: the content decides, and it is refused."""
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload.update(cell={"arm": "one_step", "microbatch": 64, "k_b": 4}))
    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2


def test_a_foreign_protocol_is_refused(tmp_path):
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload.update(protocol="exp06.fit_cell.v0"))
    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2


def test_a_trial_bound_to_another_context_is_refused(tmp_path):
    """The artifact's own header can agree with this process and its trials still be foreign: the
    binding is checked per trial, exactly as ``build_evidence`` checks it."""
    published = _publish_one(tmp_path)

    def swap(payload):
        for trial in payload["trials"]:
            trial["context_digest"] = "f" * 64

    _damage(published, swap, resync=False)
    measurer, _ = _resume(tmp_path)
    assert len(measurer.calls) == 2


def test_a_refused_cell_is_published_and_adopted_like_any_other(tmp_path):
    """A cell that MISSED is a measurement too — re-measuring it costs the same 6 minutes and
    reaches the same verdict. It is published, adopted, and still refused by the headroom rule."""
    path = _attempt(tmp_path, "att-1")
    cell = probe.FitCell("rollout", 8, 2)
    _run(
        _config(pos_fit_authorization=path),
        _CountingMeasurer(peaks={("rollout", 8, 2): 31 * 1024**3}),
        cells=[cell],
    )
    measurer, table = _resume(tmp_path)
    assert len(measurer.calls) == 0, "a refused cell is evidence and does not need re-measuring"
    assert table["authorized_cells"] == []
    assert table["refused_cells"][0]["reasons"] == ["headroom"]


# =============================================================================================
# 6. The trust chain: adopted content is inside the run-level digest, and the gate is unmoved.
# =============================================================================================


def test_the_run_level_digest_covers_adopted_content(tmp_path):
    """Defect (f). An adopted trial enters the evidence as a measurement, so the published table's
    numbers change with it and its digest changes with them — there is no adopted-content bypass."""
    _first_attempt(tmp_path, die_after=4)
    honest = _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "att-2"), pos_fit_adoption_root=_adoption_root(tmp_path)),
        _CountingMeasurer(),
    )

    published = probe.cell_marker_path(_attempt(tmp_path, "att-1"), probe.FitCell("one_step", 8, 2))
    _damage(published, lambda payload: [trial.update(step_seconds=99.0) for trial in payload["trials"]])
    tampered = _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "att-4"), pos_fit_adoption_root=_adoption_root(tmp_path)),
        _CountingMeasurer(),
    )
    assert tampered["sha256"] != honest["sha256"], "adopted numbers are inside the run-level digest"
    assert [entry for entry in tampered["measurements"] if entry["step_seconds"] == 99.0]


def test_an_adopted_table_still_re_decides_from_its_own_numbers(tmp_path):
    """``load_authorization`` rebuilds the whole artifact from the recorded measurements. Adoption
    must not produce a payload that fails to re-decide — the provenance record included — and an
    edit to that record without a re-hash is refused like any other edit.

    Stated plainly, because it is a limit and not a property: provenance is a RECORD, not a
    derivation. The artifact a cell was adopted from is not recomputable from the numbers, so a
    re-hashed edit to the provenance line is indistinguishable from the truth. What the digest
    protects, and what the gate depends on, is the NUMBERS."""
    _first_attempt(tmp_path, die_after=4)
    path = _attempt(tmp_path, "att-2")
    table = _run(
        _config(pos_fit_authorization=path, pos_fit_adoption_root=_adoption_root(tmp_path)), _CountingMeasurer()
    )
    assert probe.load_authorization(path)["sha256"] == table["sha256"]

    stored = json.loads(Path(path).read_text())
    stored["payload"]["cell_provenance"][0]["provenance"] = probe.PROVENANCE_MEASURED
    Path(path).write_text(json.dumps(stored, sort_keys=True))  # digest NOT recomputed
    with pytest.raises(ValueError, match="edited"):
        probe.load_authorization(path)


def test_the_provenance_record_must_describe_exactly_the_measured_cells(tmp_path):
    """A provenance list that does not line up with the measurements is a table whose story about
    where its numbers came from is not checkable — it does not load."""
    context = _context()
    cells = [probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 16, 2)]
    measurements = [_measurement(context, cell) for cell in cells]
    with pytest.raises(ValueError, match="provenance"):
        probe.ProbeEvidence(
            context=context,
            measurements=tuple(measurements),
            projection_inputs=(("max_train_steps", 10), ("eval_every", 5), ("checkpoint_every", 5)),
            provenance=((cells[0], probe.PROVENANCE_MEASURED),),
        ).as_payload()


def test_the_authorization_protocol_names_the_shape_that_carries_provenance(tmp_path):
    """A consumer reading a table with no provenance record cannot tell "nothing was adopted" from
    "written by code that could not adopt". The version is what tells it — and the version is pinned
    once, by :func:`test_the_protocol_names_the_shape_that_carries_exclusions`; what this asserts is
    that the SHAPE the version promises is actually there, which is the part a bump could break."""
    table = _run(_config(pos_fit_authorization=_attempt(tmp_path, "att-1")), _CountingMeasurer())
    assert table["protocol"] == probe.AUTHORIZATION_PROTOCOL
    for field in ("cell_provenance", "excluded_cells", "skipped_cells", "exclusion_reason"):
        assert field in table, f"the declared protocol must carry {field}"


def test_adoption_cannot_reach_a_cell_the_ladder_did_not_ask_for(tmp_path):
    """Adoption fills cells; it never adds them. A published cell outside this run's ladder is not
    measured, not adopted and not authorized."""
    _publish_one(tmp_path, cell=probe.FitCell("one_step", 64, 4))
    _, table = _resume(tmp_path)
    assert list(table["measured_cells"]) == [{"arm": "rollout", "microbatch": 8, "k_b": 2}]


# =============================================================================================
# 7. Round F5b — the Codex review. Three findings, and the honest statement of what is left.
#
# B1: a cell artifact proved CONSISTENCY, not LEGALITY. Every hashed value was supplied by the
#     writer, so a forger could copy the published context, set favourable peaks, recompute both
#     digests, and be adopted. The committed battery probe scored that attack REFUSED while the
#     forgery was in fact ADOPTED — it watched the final-table digest move (propagation) and called
#     that a refusal (legality). Corrected below and in the harness.
# B2: `code_sha` read git HEAD, which is not the running bytes: a dirty tree measures uncommitted
#     code under a committed SHA, and a tarball trusted a caller-supplied `COMMIT`.
# MAJOR: two publishers could leave `content-B + marker-A` and tear the pair permanently.
# =============================================================================================


def _deployed_py_files() -> list[Path]:
    package = Path(probe.__file__).resolve().parent
    return sorted(p for p in package.rglob("*.py") if "tests" not in p.relative_to(package).parts)


def test_the_manifest_is_the_running_bytes_of_the_deployed_measurement_code(tmp_path):
    """B2. The identity adoption trusts must be CONTENT, not a label a caller can type.

    `git rev-parse HEAD` names a commit; it does not name what is on disk. The F5 delta was itself
    the proof — its running bytes were uncommitted while its derived SHA read `a3ba5c0`."""
    digest = probe.deployed_manifest_digest()
    assert probe._DIGEST_RE.match(digest)
    assert probe.deployed_manifest_digest() == digest, "the same tree must hash the same way twice"

    files = _deployed_py_files()
    assert files, "the manifest must cover something"
    assert all("tests" not in str(path) for path in files)
    assert any(path.name == "pos_rollout_fit_probe.py" for path in files)


def test_editing_a_deployed_module_moves_the_manifest(tmp_path, monkeypatch):
    """The property the whole binding rests on: a byte of measurement code changes the identity."""
    package = tmp_path / "maxdiffusion"
    (package / "sub").mkdir(parents=True)
    (package / "pos_rollout_thing.py").write_text("x = 1\n")
    (package / "sub" / "other.py").write_text("y = 2\n")
    (package / "tests").mkdir()
    (package / "tests" / "test_x.py").write_text("# changes every round\n")

    before = probe.deployed_manifest_digest(root=str(package))
    (package / "tests" / "test_x.py").write_text("# edited: tests do not produce measurements\n")
    assert probe.deployed_manifest_digest(root=str(package)) == before, "tests are excluded, with reason"

    (package / "sub" / "other.py").write_text("y = 3\n")
    assert probe.deployed_manifest_digest(root=str(package)) != before, "a deployed byte moves the manifest"


def test_the_context_carries_the_manifest_and_adoption_compares_it(tmp_path):
    """The manifest is inside the context digest, so every existing binding carries it for free."""
    context = _context()
    assert context.manifest_digest == probe.deployed_manifest_digest()
    assert context.as_payload()["manifest_digest"] == context.manifest_digest
    assert probe.ProbeContext.from_payload(context.as_payload()).digest() == context.digest()
    assert "manifest_digest" in dataclasses.replace(context, manifest_digest="0" * 64).differences(context)


def test_a_cell_measured_by_other_running_bytes_is_re_measured(tmp_path, capsys):
    """B1/B2 together, as the review demanded: a rehashed FAVOURABLE-peak artifact re-measures.

    The forger here does what the reviewer's construction does — copies the published context, sets
    both trials to a peak that trivially fits, recomputes the payload digest and the marker — but is
    not running the deployed bytes. The manifest refuses it."""
    published = _publish_one(tmp_path)

    def forge(payload):
        for trial in payload["trials"]:
            # F10: a cheap cell is now one whose BOUND is cheap, and whose watermark does not
            # contradict it. A forger sets all three, because all three are inside the artifact.
            trial["peak_bytes"] = trial["analysis_bytes"] = trial["watermark_bytes"] = 1
        payload["context"]["manifest_digest"] = "0" * 64

    _damage(published, forge)
    measurer, table = _resume(tmp_path)
    assert len(measurer.calls) == 2, "a fabricated cell from other bytes must be MEASURED, not adopted"
    assert "manifest_digest" in capsys.readouterr().out
    assert all(entry["peak_bytes"] != 1 for entry in table["measurements"]), "the fabrication never landed"


def test_a_process_declaring_a_commit_refuses_to_publish_from_a_dirty_tree(tmp_path, monkeypatch):
    """B2's other half. `COMMIT` is the launcher's declaration "this process is running commit X".

    A process that makes that declaration while its tree carries uncommitted changes is publishing a
    provenance claim it cannot support, and it is refused loudly rather than warned about. A process
    that declares nothing is identified by its manifest alone, which is what adoption compares — so
    the developer's dirty checkout still runs, and still cannot masquerade as a commit."""
    monkeypatch.setattr(probe, "_git_head", lambda start: "a" * 40)
    monkeypatch.setattr(probe, "_git_dirty_paths", lambda start: ("src/maxdiffusion/pos_rollout_fit_probe.py",))

    monkeypatch.delenv("COMMIT", raising=False)
    probe.derive_code_sha(environ={})  # no declaration, no refusal

    with pytest.raises(ValueError, match="uncommitted"):
        probe.derive_code_sha(environ={"COMMIT": "a" * 40})


def test_a_deployment_without_git_is_identified_by_content_not_by_a_declared_commit(monkeypatch):
    """A tarball has no git objects, and `COMMIT` is an environment variable anybody can set. It may
    LABEL the artifact; it may not be the only thing standing behind it, so the derivation refuses
    unless a content manifest is bound alongside it."""
    monkeypatch.setattr(probe, "_git_head", lambda start: "")
    with pytest.raises(ValueError, match="content"):
        probe.derive_code_sha(environ={"COMMIT": "b" * 40}, manifest="")
    assert probe.derive_code_sha(environ={"COMMIT": "b" * 40}, manifest="c" * 64) == "b" * 40


def test_the_module_declares_what_adoption_does_not_prove():
    """The trust boundary, stated in the module rather than assumed by its readers.

    Codex asked for an authenticated publication authority. This round did not build one: the
    artifacts are integrity-checked and content-bound, NOT authenticated, and the trust anchor is the
    bucket ACL — the same anchor the final authorization table has always rested on. That is a real
    limitation and it is written down, because the alternative on offer was an in-repo shared secret
    that the same bucket writers could read."""
    text = probe.__doc__ + probe.publish_cell.__doc__ + probe.adopt_published_cell.__doc__
    for claim in ("not authenticated", "bucket", "integrity"):
        assert claim in text.lower(), f"the module must state {claim!r} plainly"


# --- MAJOR: content-addressed objects, one commit marker, verify-and-repair -------------------


def test_a_cell_content_object_is_named_by_its_own_digest(tmp_path):
    """Immutable by construction: two payloads cannot contend for one name, so the pair cannot tear."""
    path = _attempt(tmp_path, "att-1")
    cell = probe.FitCell("rollout", 8, 2)
    context = _context(_config(pos_fit_authorization=path))
    artifact = probe.CellArtifact(
        cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=1.0),)
    )
    record = probe.publish_cell(probe.cell_marker_path(path, cell), artifact)

    marker = Path(probe.cell_marker_path(path, cell))
    assert marker.exists() and marker.read_text().strip() == record["sha256"]
    content = Path(probe.cell_content_path(path, cell, record["sha256"]))
    assert content.exists() and record["sha256"] in content.name, "the whole digest names the object"
    assert probe.load_cell_artifact(str(marker)).trials[0].step_seconds == 1.0


def test_two_interleaved_publishers_never_leave_a_torn_pair(tmp_path):
    """The MAJOR, with DISTINCT payloads because production timings differ between attempts.

    The steps are interleaved in the worst order a real race produces: both contents land, then the
    markers commit in the opposite order. Whatever the marker ends up naming must be a WHOLE artifact
    that verifies — never A's marker over B's content."""
    path = _attempt(tmp_path, "att-1")
    cell = probe.FitCell("rollout", 8, 2)
    context = _context(_config(pos_fit_authorization=path))
    marker = probe.cell_marker_path(path, cell)
    a, b = (
        probe.CellArtifact(
            cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=step),)
        )
        for step in (25.347, 25.356)
    )

    for first, second in ((a, b), (b, a)):
        for stale in Path(probe.cell_publication_dir(path)).glob("*"):
            stale.unlink()
        probe.publish_cell_content(marker, first)
        probe.publish_cell_content(marker, second)
        probe.commit_cell_marker(marker, second)
        probe.commit_cell_marker(marker, first)
        loaded = probe.load_cell_artifact(marker)
        assert loaded.trials[0].step_seconds == first.trials[0].step_seconds
        assert len(loaded.trials) == 1


def test_a_marker_pointing_at_a_missing_content_object_is_repaired_not_returned_from(tmp_path, capsys):
    """The review's second half of the MAJOR: a later publisher that finds a broken pair must repair
    it. Returning early leaves the cell permanently unadoptable — the poisoned cache again."""
    path = _attempt(tmp_path, "att-1")
    cell = probe.FitCell("rollout", 8, 2)
    context = _context(_config(pos_fit_authorization=path))
    artifact = probe.CellArtifact(
        cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=1.0),)
    )
    marker = probe.cell_marker_path(path, cell)
    record = probe.publish_cell(marker, artifact)
    Path(probe.cell_content_path(path, cell, record["sha256"])).unlink()  # the content object is gone

    probe.publish_cell(marker, artifact)
    assert "repair" in capsys.readouterr().out.lower()
    assert probe.load_cell_artifact(marker).trials[0].step_seconds == 1.0


def test_a_marker_naming_content_that_does_not_hash_to_it_is_refused(tmp_path):
    """The marker is the commit; content that does not hash to what the marker names is not it."""
    path = _attempt(tmp_path, "att-1")
    cell = probe.FitCell("rollout", 8, 2)
    context = _context(_config(pos_fit_authorization=path))
    artifact = probe.CellArtifact(
        cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=1.0),)
    )
    marker = probe.cell_marker_path(path, cell)
    record = probe.publish_cell(marker, artifact)
    content = Path(probe.cell_content_path(path, cell, record["sha256"]))
    stored = json.loads(content.read_text())
    stored["payload"]["trials"][0]["step_seconds"] = 99.0
    content.write_text(json.dumps(stored, sort_keys=True))

    with pytest.raises(ValueError, match="not the one that was committed"):
        probe.load_cell_artifact(marker)


def test_the_retired_two_object_scheme_really_did_tear(tmp_path):
    """The MAJOR's red side, re-derived in-test rather than argued from the diff (F4's technique).

    "Content objects cannot tear" is only interesting if the thing they replaced could. The retired
    scheme is reproduced here — a fixed-name content file, then a digest sidecar, each overwritten
    independently — and driven through the same interleaving the green test uses. It ends as
    ``content-B + digest-A``: a pair that no reader can accept, for a cell that can then never be
    adopted. That is the state the reviewer found, and it is why the marker had to stop naming a
    fixed path and start naming a digest."""
    path = _attempt(tmp_path, "att-1")
    cell = probe.FitCell("rollout", 8, 2)
    context = _context(_config(pos_fit_authorization=path))
    directory = Path(probe.cell_publication_dir(path))
    directory.mkdir(parents=True, exist_ok=True)
    retired_content, retired_sidecar = directory / "rollout_m8_k2.json", directory / "rollout_m8_k2.json.old"

    def retired_publish(artifact):
        payload = artifact.as_payload()
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload, digest

    a, b = (
        probe.CellArtifact(
            cell=cell, context=context, job_identity=_RUN, trials=(_measurement(context, cell, step=step),)
        )
        for step in (25.347, 25.356)
    )
    payload_a, digest_a = retired_publish(a)
    payload_b, digest_b = retired_publish(b)
    assert digest_a != digest_b, "production payloads differ between attempts; that is the whole problem"

    # The interleaving: A writes content, B writes content, B commits, A commits.
    retired_content.write_text(json.dumps({"payload": payload_a, "sha256": digest_a}, sort_keys=True))
    retired_content.write_text(json.dumps({"payload": payload_b, "sha256": digest_b}, sort_keys=True))
    retired_sidecar.write_text(digest_b)
    retired_sidecar.write_text(digest_a)

    stored = json.loads(retired_content.read_text())
    assert stored["sha256"] == digest_b and retired_sidecar.read_text() == digest_a, "content-B + digest-A"
    assert digest_a != stored["sha256"], "no reader can accept this pair, and no publisher repaired it"

    # The deployed scheme, same interleaving, same distinct payloads: whole and adoptable.
    marker = probe.cell_marker_path(path, cell)
    probe.publish_cell_content(marker, a)
    probe.publish_cell_content(marker, b)
    probe.commit_cell_marker(marker, b)
    probe.commit_cell_marker(marker, a)
    assert probe.load_cell_artifact(marker).trials[0].step_seconds == 25.347


def test_a_banked_cell_whose_trials_DISAGREE_is_QUARANTINED_and_re_measured(tmp_path, capsys):
    """**F10c, review MODERATE 2 — issue #10's wedge, made concrete and then closed.**

    F10b refuses to publish a table whose trials disagree about the cell's compiled analysis (one
    executable measured twice has one account of its footprint). The reviewer then banked exactly
    that artifact — two individually valid trials reporting 10 and 20 GiB — and watched two
    consecutive retries ADOPT it and die at publication with ``analysis_disagreement``: the poison
    lives in the cache the retry reads, so every attempt after it fails identically and forever.

    The table-wide raise stays. What changes is that a CACHED artifact which cannot aggregate is
    refused for adoption, which converts a permanent outage into one extra measurement.
    """
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload["trials"][1].update(analysis_bytes=10 * 1024**3))
    assert probe.load_cell_artifact(published), "the artifact itself is still well-formed and digest-valid"

    measurer, table = _resume(tmp_path)
    assert len(measurer.calls) == 2, "the poisoned cell is MEASURED, not adopted"
    printed = capsys.readouterr().out
    assert "not adopting" in printed and "do not aggregate" in printed, printed
    assert len(table["authorized_cells"]) == 1, "and the retry publishes a table instead of dying"

    # ...and the cost is ONE extra measurement, not one per attempt forever: the retry banked a
    # clean artifact of its own, and the attempt after it adopts that one rather than the poison.
    again, table = _resume(tmp_path, attempt="att-3")
    assert again.calls == [], "the third attempt adopts the clean cell the second one banked"
    assert len(table["authorized_cells"]) == 1
    assert any(row["provenance"].startswith(probe.ADOPTED_PREFIX) for row in table["cell_provenance"])


def test_banking_REFUSES_to_write_a_cell_whose_trials_do_not_aggregate(tmp_path):
    """The other half of the quarantine: this probe never puts such an artifact in the bucket.

    The run that measured it is going to stop at ``build_evidence`` with the same message either
    way, so stopping at banking costs nothing and keeps the next attempt's adoption root clean.
    """
    context = _context()
    cell = probe.FitCell("rollout", 8, 2)
    marker = probe.cell_marker_path(_attempt(tmp_path, "att-bank"), cell)
    artifact = probe.CellArtifact(
        cell=cell,
        context=context,
        job_identity=_RUN,
        trials=(
            _measurement(context, cell),
            dataclasses.replace(_measurement(context, cell), analysis_bytes=10 * 1024**3),
        ),
    )
    with pytest.raises(ValueError, match="analysis_disagreement"):
        probe.publish_cell(marker, artifact)
    assert not Path(marker).exists(), "nothing was banked, so nothing can be adopted from it"


@pytest.mark.parametrize(
    "field, value",
    [
        ("peak_bytes", 9.9),
        ("peak_bytes", True),
        ("capacity_bytes", 32.5 * 1024**3 + 0.5),
        ("reservation_failures", 0.9),
        ("analysis_bytes", 20 * 1024**3 + 0.5),
    ],
)
def test_a_banked_cell_carrying_a_MALFORMED_count_is_not_adopted(tmp_path, capsys, field, value):
    """**F10c, review MODERATE (b), as the ROUND TRIP the reviewer asked for.**

    ``CellMeasurement.from_payload`` used bare ``int()`` on the three counts, which runs before
    ``_checked`` ever sees the record — so a digest-valid banked artifact carrying ``peak_bytes:
    9.9`` or ``true`` was rounded into a well-formed measurement and travelled load -> adopt ->
    republish -> ``assert_cell_authorized``. Parsing is exact now, so the artifact does not load at
    all, adoption logs it and the cell is measured instead. (A negative count was already refused
    and still is — the case below the parametrisation.)
    """
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload["trials"][0].update({field: value}))
    with pytest.raises(ValueError):
        probe.load_cell_artifact(published)

    measurer, table = _resume(tmp_path)
    assert len(measurer.calls) == 2, "a malformed banked count is re-measured, never rounded"
    assert "not adopting" in capsys.readouterr().out
    assert len(table["authorized_cells"]) == 1
    for recorded in table["measurements"]:
        assert isinstance(recorded["peak_bytes"], int) and not isinstance(recorded["peak_bytes"], bool)


def test_a_banked_cell_whose_BINDING_count_is_fractional_is_not_adopted(tmp_path, capsys):
    """**F10d, review MODERATE.** A digest-valid artifact recording ``device_count: 8.5`` loaded as
    ``8`` and was ADOPTED by an eight-device context: a topology nobody ran, binding to one that
    did, through a truncation in the BINDING field itself. The count is parsed exactly on both sides
    now, so the artifact does not load and the cell is measured instead."""
    published = _publish_one(tmp_path)
    # A COMPETENT forger: every derived field agrees with the 8 a truncating reader would rebuild,
    # so the parse is the only thing standing between this artifact and adoption.
    _damage(published, lambda payload: payload["context"].update(device_count=8.5))
    with pytest.raises(ValueError):
        probe.load_cell_artifact(published)

    measurer, table = _resume(tmp_path)
    assert len(measurer.calls) == 2, "a fractional device count is re-measured, never rounded into a match"
    assert "not adopting" in capsys.readouterr().out
    assert len(table["authorized_cells"]) == 1
    assert table["context"]["device_count"] == 8


def test_a_banked_cell_carrying_a_NEGATIVE_count_is_still_not_adopted(tmp_path):
    """The case F10b already refused, kept as a test because F10c rewrote the parser around it."""
    published = _publish_one(tmp_path)
    _damage(published, lambda payload: payload["trials"][0].update(analysis_bytes=-1))
    with pytest.raises(ValueError, match="non-negative"):
        probe.load_cell_artifact(published)


def test_the_accepted_residual_a_bucket_writer_can_forge_a_cell(tmp_path):
    """**This test asserts a WEAKNESS, deliberately, so that it cannot be forgotten or overstated.**

    Review F5c, BLOCKER 1. A forger who can read one current artifact copies its context verbatim —
    the manifest digest is public *in the payload* — swaps the trials for one-byte peaks, recomputes
    the payload digest and the marker, and **is adopted**. No deployed source tree is needed. The
    manifest is recomputed locally but only equality-compared against a value the forger controls.

    This is the declared trust boundary, not a bug to be fixed here: the anchor is the bucket ACL,
    and authentication (workload identity / KMS) is escalated to Yixun as a policy decision. The test
    exists so that (a) the residual is measured rather than asserted in prose, and (b) if anyone ever
    adds a publication authority, **this test fails** and forces the docstring, the worklog and probe
    `F5-8` to be updated together. Its failure would be good news."""
    published = _publish_one(tmp_path)

    def forge_in_boundary(payload):
        for trial in payload["trials"]:
            # F10: the cheap cell is a cheap BOUND with a watermark that agrees with it. Nothing in
            # the artifact contradicts anything else, which is exactly why it is adopted.
            trial["peak_bytes"] = trial["analysis_bytes"] = trial["watermark_bytes"] = 1
        # The context -- manifest digest included -- is copied EXACTLY as published. Nothing foreign.

    _damage(published, forge_in_boundary)
    artifact = probe.load_cell_artifact(published)
    assert artifact.context.manifest_digest == probe.deployed_manifest_digest(), "the forgery is in-boundary"

    measurer, table = _resume(tmp_path)
    assert len(measurer.calls) == 0, "ACCEPTED RESIDUAL: the in-boundary forgery is adopted, not measured"
    assert [entry["peak_bytes"] for entry in table["measurements"]] == [1], "and its fabricated peak is authorized"


# =============================================================================================
# Round F5d — the harness's own verdict accounting is now under test.
#
# F5c introduced DECLARED for the one attack this campaign accepts. The re-review found that
# `_report` accepted the WORD from any probe: a probe drifting into "DECLARED: accidental drift"
# was counted as an accepted residual. An accepted residual is a decision somebody made and wrote
# down, so the set of probes allowed to say it is enumerated, and anything else is a loud harness
# failure. The battery is review-package evidence, so its classifier is production for this purpose.
# =============================================================================================


@functools.lru_cache(maxsize=1)
def _harness():
    """Load `reviewer_attacks.py` as a module. It lives under docs/ and is not importable by name."""
    import importlib.util

    path = (
        Path(probe.__file__).resolve().parents[1]
        / "docs"
        / "worklogs_yixun"
        / "exp_06_rollout_adapter_claude"
        / "harness"
        / "reviewer_attacks.py"
    )
    if not path.exists():  # the worktree layout puts docs/ at the repo root, not under src/
        path = Path(probe.__file__).resolve().parents[2] / path.relative_to(Path(probe.__file__).resolve().parents[1])
    spec = importlib.util.spec_from_file_location("exp06_reviewer_attacks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_an_allowlisted_probe_may_report_an_accepted_residual(capsys):
    """Both directions, because only one of them was ever checked."""
    harness = _harness()
    harness._VERDICTS.clear()

    harness._report("F5-8   forge w/ CURRENT manifest", lambda: "DECLARED: the accepted residual")
    assert harness._VERDICTS == ["DECLARED"], "the allowlisted probe still declares"

    harness._VERDICTS.clear()
    harness._report("P9-1   some other probe", lambda: "DECLARED: accidental drift")
    assert harness._VERDICTS == ["UNPARSED"], "a non-allowlisted DECLARED is a harness failure, not a residual"
    out = capsys.readouterr().out
    assert "HARNESS FAILURE" in out and "allowlist" in out
    assert "accidental drift" in out, "the original verdict is still shown, so the drift is diagnosable"


def test_the_summary_fails_the_run_when_a_verdict_is_unclassifiable(capsys):
    harness = _harness()
    harness._VERDICTS.clear()
    harness._VERDICTS.extend(["REFUSED", "DECLARED"])
    assert harness._summarize() is True
    harness._VERDICTS.append("UNPARSED")
    assert harness._summarize() is False
    assert "FAILED" in capsys.readouterr().out

    harness._VERDICTS.clear()
    harness._VERDICTS.extend(["REFUSED", "SUCCEEDED"])
    assert harness._summarize() is False


def test_the_declared_allowlist_names_exactly_the_probes_that_earned_it():
    """One entry today. Adding a second is a decision that has to be made here, in the diff a
    reviewer reads — which is the same denylist/allowlist discipline `FINGERPRINT_EXCLUSIONS` uses."""
    harness = _harness()
    assert harness._MAY_DECLARE == frozenset({"F5-8"})
    assert harness._probe_id("F5-8   forge w/ CURRENT manifest") == "F5-8"
    assert harness._probe_id("A-B1(a) module issue token   :") == "A-B1(a)"


# =============================================================================================
# Round F6 `cell-exclusion` — a cell the hardware cannot compile must be DECLARED, not discovered.
#
# M1-4 attempt 1 banked 12 of 16 cells (F5 working end to end in production) and then died at
# `one_step microbatch=32 k=2` with `bad_smem_address` — the identical cell and chip fault as M1-3
# attempt 2, on a different VM on a different day. **2/2 on the same cell is deterministic**: an XLA
# codegen fault under the F4 scan at chunk width 32 for the one_step loss on v6e-8 (width 32 is fine
# on the rollout arm; one_step is fine at widths 8 and 16). The four tail cells are unreachable, and
# because the ladder cannot finish, the table cannot publish — 12 good cells held hostage by 4.
#
# The mechanism below lets a run DECLARE those cells excluded, with a reason, so the table publishes
# and accounts for every cell. Populating the list is a plan deviation for Yixun; the YAML default is
# empty and stays empty here.
# =============================================================================================


_TAIL = ("one_step:32:2", "one_step:32:4", "one_step:64:2", "one_step:64:4")
_WHY = "bad_smem_address: deterministic XLA codegen fault, one_step scan chunk width>=32 on v6e-8 (issue #18)"


def _excluding(*cells, reason=_WHY, **overrides):
    return _config(
        pos_fit_excluded_cells=",".join(cells),
        pos_fit_exclusion_reason=reason if cells else "",
        **overrides,
    )


def test_the_exclusion_list_parses_strictly_or_says_why(tmp_path):
    """A typo in an exclusion is a cell that silently still runs — or worse, one that silently does
    not. Every malformed entry is a loud config error, and an empty list parses to nothing."""
    assert probe.parse_excluded_cells(_config()) == ()
    assert probe.parse_excluded_cells(_excluding(*_TAIL)) == tuple(
        probe.FitCell("one_step", mb, k) for mb in (32, 64) for k in (2, 4)
    )
    for bad, why in (
        ("one_step:32", "three fields"),
        ("one_step:32:2:9", "three fields"),
        ("sideways:32:2", "arm"),
        ("one_step:33:2", "ladder"),
        ("one_step:32:3", "ladder"),
        ("one_step:x:2", "whole number"),
        ("one_step:32:2,one_step:32:2", "twice"),
    ):
        with pytest.raises(ValueError, match=why):
            probe.parse_excluded_cells(_excluding(bad))


def test_an_exclusion_without_a_declared_reason_is_refused(tmp_path):
    """An undocumented exclusion is a cell that quietly stopped being measured. The reason is
    required, and it lands in the table where a reader of the authorization will see it."""
    with pytest.raises(ValueError, match="pos_fit_exclusion_reason"):
        probe.parse_excluded_cells(_excluding(*_TAIL, reason=""))
    assert probe.declared_exclusion_reason(_excluding(*_TAIL)) == _WHY
    assert probe.declared_exclusion_reason(_config()) == ""


def test_an_excluded_cell_is_never_built_never_measured_and_never_banked(tmp_path):
    """The call-count proof. An excluded cell must not reach the measurer at all — not compiled, not
    timed, not published — because the whole point is that compiling it kills the VM."""
    path = _attempt(tmp_path, "att-1")
    excluded = probe.FitCell("one_step", 32, 2)
    measurer = _CountingMeasurer()
    table = _run(
        _excluding("one_step:32:2", pos_fit_authorization=path),
        measurer,
        cells=[probe.FitCell("rollout", 8, 2), excluded],
    )
    assert excluded not in measurer.calls, "the excluded cell reached the measurer"
    assert len(measurer.calls) == 2, "only the surviving cell was measured, twice"
    assert not Path(probe.cell_marker_path(path, excluded)).exists(), "an excluded cell banks nothing"
    assert table["excluded_cells"] == [{"arm": "one_step", "microbatch": 32, "k_b": 2, "reason": _WHY}]


def test_the_table_accounts_for_every_cell_of_the_ladder_it_was_asked_to_run(tmp_path):
    """Never silently absent. Authorized + refused + excluded (+ skipped) must cover the request."""
    path = _attempt(tmp_path, "att-1")
    cells = [probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 16, 2), probe.FitCell("one_step", 32, 2)]
    table = _run(
        _excluding("one_step:32:2", pos_fit_authorization=path),
        _CountingMeasurer(peaks={("rollout", 16, 2): 31 * 1024**3}),
        cells=cells,
    )
    accounted = (
        [tuple(c.values()) for c in table["authorized_cells"]]
        + [(c["arm"], c["microbatch"], c["k_b"]) for c in table["refused_cells"]]
        + [(c["arm"], c["microbatch"], c["k_b"]) for c in table["excluded_cells"]]
        + [(c["arm"], c["microbatch"], c["k_b"]) for c in table["skipped_cells"]]
    )
    assert sorted(accounted) == sorted((c.arm, c.microbatch, c.k_b) for c in cells)
    assert len(table["authorized_cells"]) == 1 and len(table["refused_cells"]) == 1


def test_a_cell_the_microbatch_cannot_divide_is_recorded_rather_than_dropped(tmp_path):
    """The other silent absence, closed by the same invariant: a cell whose microbatch does not
    divide the logical batch was printed to the log and then vanished from the table."""
    path = _attempt(tmp_path, "att-1")
    table = _run(
        _config(pos_fit_authorization=path, pos_logical_batch=24),
        _CountingMeasurer(),
        cells=[probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 16, 2)],
    )
    assert [(c["arm"], c["microbatch"], c["k_b"]) for c in table["skipped_cells"]] == [("rollout", 16, 2)]
    assert "divide" in table["skipped_cells"][0]["reason"]


def test_quoting_an_excluded_cell_is_refused_with_its_declared_reason(tmp_path):
    """Fail-loud consumption: an excluded cell must be as unconstructible as a refused one, and the
    refusal must say it was excluded ON PURPOSE rather than never measured by accident."""
    path = _attempt(tmp_path, "att-1")
    excluded = probe.FitCell("one_step", 32, 2)
    config = _excluding("one_step:32:2", pos_fit_authorization=path)
    _run(config, _CountingMeasurer(), cells=[probe.FitCell("rollout", 8, 2), excluded])
    authorization = probe.load_authorization(path)

    with pytest.raises(ValueError, match="EXCLUDED"):
        probe.assert_cell_authorized(authorization, excluded, context=_context(config))
    try:
        probe.assert_cell_authorized(authorization, excluded, context=_context(config))
    except ValueError as error:
        assert _WHY in str(error), "the refusal quotes the declared reason"
        assert "never measured" not in str(error), "an excluded cell is not an unmeasured one"


def test_an_empty_exclusion_list_is_todays_behaviour_bit_for_bit(tmp_path):
    """Defect (d) of F5, restated for F6: the mechanism must be inert until it is declared."""
    plain = _run(_config(pos_fit_authorization=_attempt(tmp_path, "a")), _CountingMeasurer())
    declared_empty = _run(_excluding(pos_fit_authorization=_attempt(tmp_path, "b")), _CountingMeasurer())
    assert json.dumps(_without_provenance(plain), sort_keys=True) == json.dumps(
        _without_provenance(declared_empty), sort_keys=True
    )
    assert plain["excluded_cells"] == [] and plain["exclusion_reason"] == ""


def test_the_exclusion_declaration_is_outside_the_per_cell_recipe_and_inside_the_run_digest(tmp_path):
    """The decision recorded in the F6 brief, asserted in both directions.

    Excluding cell X must not invalidate the banked artifacts of cells Y — otherwise declaring an
    exclusion would throw away every cell F5 exists to keep. So the exclusion declaration is NOT in
    the per-cell recipe fingerprint. It IS inside the run-level table digest, because a reader of the
    authorization must be able to tell that the table was decided under a declared exclusion."""
    assert probe.recipe_fingerprint(_excluding(*_TAIL)) == probe.recipe_fingerprint(_config())
    assert probe.recipe_fingerprint(_excluding("one_step:32:2")) == probe.recipe_fingerprint(_config())

    plain = _run(_config(pos_fit_authorization=_attempt(tmp_path, "a")), _CountingMeasurer())
    excluded = _run(
        _excluding("one_step:8:2", pos_fit_authorization=_attempt(tmp_path, "b")),
        _CountingMeasurer(),
        cells=[probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 16, 2)],
    )
    assert excluded["sha256"] != plain["sha256"], "the run-level digest covers the exclusion"
    assert excluded["context"]["recipe_fingerprint"] == plain["context"]["recipe_fingerprint"]


def test_declaring_an_exclusion_does_not_invalidate_cells_already_banked(tmp_path):
    """The reason the fingerprint decision matters, executed: M1-4 banked 12 cells before the fault.
    Declaring the 4 unreachable ones must let those 12 be ADOPTED, not re-measured — otherwise the
    exclusion costs exactly what it was introduced to save."""
    _first_attempt(tmp_path, die_after=4)
    measurer = _CountingMeasurer()
    table = _run(
        _excluding(
            "rollout:8:2",  # the cell att-1 did NOT reach; the two it banked must still adopt
            pos_fit_authorization=_attempt(tmp_path, "att-2"),
            pos_fit_adoption_root=_adoption_root(tmp_path),
        ),
        measurer,
    )
    assert len(measurer.calls) == 0, "the banked cells were adopted although an exclusion was declared"
    assert len(table["authorized_cells"]) == 2 and len(table["excluded_cells"]) == 1


def test_the_protocol_names_the_shape_that_carries_exclusions():
    """v3 tables cannot express an excluded cell, so a v3 loader reading a v4 table would silently
    treat an excluded cell as never-measured. Fail closed on the version, as for v3.

    v6 (F9): the measurements now carry the peak's ATTRIBUTION and the runtime/analysis readings it
    was decided from. A v5 reader would take a ``peak_bytes`` whose provenance it cannot see.

    v7 (F10): the same fields, decided by a different rule — the analysis is the authorization bound
    and the watermark is a cross-check — so a v6 reader would re-decide a v7 table under a rule it
    was not published under. The CELL protocol deliberately stays at v2: what a trial measures did
    not change, and bumping it would strand every banked cell for no measurement reason."""
    assert probe.AUTHORIZATION_PROTOCOL == "exp06.fit_authorization.v7"
    assert probe.CELL_PROTOCOL == "exp06.fit_cell.v2"


def test_excluding_the_whole_ladder_is_refused_rather_than_publishing_nothing(tmp_path):
    with pytest.raises(ValueError, match="every cell"):
        _run(
            _excluding("rollout:8:2", pos_fit_authorization=_attempt(tmp_path, "att-1")),
            _CountingMeasurer(),
            cells=[probe.FitCell("rollout", 8, 2)],
        )


# =============================================================================================
# Round F6b — the F6 review. Two of these correct tests of mine that proved less than they said.
# =============================================================================================


def _doctor(path, mutate):
    """Edit a published table and re-hash it, the way a competent editor would."""
    stored = json.loads(Path(path).read_text())
    mutate(stored["payload"])
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode()).hexdigest()
    Path(path).write_text(json.dumps(stored, sort_keys=True))


def test_a_code_change_between_attempts_refuses_the_cells_banked_before_it(tmp_path, monkeypatch, capsys):
    """The F5 -> F6 production transition, modelled across TWO manifests rather than inside one.

    Review F6b, BLOCKER. M1-4 banked 12 cells at `6eda654` (manifest `4bbdbb28…`); the F6 delta edits
    `pos_rollout_fit_probe.py`, which the manifest covers, so M1-5 runs at a different manifest
    (`64f92825…`) and **cannot adopt them**. The Planner's ruling is that this is correct and there
    will be NO migration rule: grandfathering cells measured by different code is exactly the hole
    F5b/F5c closed. The existing adoption tests bank and adopt inside one unchanged process and cannot
    see this at all, which is why the reviewer could not take them as evidence.

    So the manifest is moved between attempts here, and both halves of the ruling are asserted: the
    transition re-measures, and attempts at the SAME new manifest converge on each other's work."""
    cells = [probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 16, 2)]
    root = _adoption_root(tmp_path)

    monkeypatch.setattr(probe, "deployed_manifest_digest", lambda root=None: "a" * 64)
    first = _CountingMeasurer()
    _run(_config(pos_fit_authorization=_attempt(tmp_path, "att-1"), pos_fit_adoption_root=root), first, cells=cells)
    assert len(first.calls) == 4, "attempt 1 measured and banked both cells"

    # The deployed code changes — a new commit, a new manifest. This is F5 -> F6.
    monkeypatch.setattr(probe, "deployed_manifest_digest", lambda root=None: "b" * 64)
    after = _CountingMeasurer()
    _run(_config(pos_fit_authorization=_attempt(tmp_path, "att-2"), pos_fit_adoption_root=root), after, cells=cells)
    assert len(after.calls) == 4, "cells measured by the OLD code must be re-measured, not grandfathered"
    assert "manifest_digest" in capsys.readouterr().out, "and the refusal must name the field that moved"

    # ...and the work is not lost twice: a later attempt at the SAME new manifest adopts.
    converged = _CountingMeasurer()
    _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "att-3"), pos_fit_adoption_root=root), converged, cells=cells
    )
    assert len(converged.calls) == 0, "attempt 3 adopts attempt 2's cells: convergence at the new SHA"


@pytest.mark.parametrize(
    "doctor, message",
    [
        (lambda p: p["excluded_cells"].append({**p["authorized_cells"][0], "reason": "smuggled"}), "both"),
        (lambda p: p["skipped_cells"].append({**p["authorized_cells"][0], "reason": "smuggled"}), "both"),
        (lambda p: p["excluded_cells"].append(dict(p["excluded_cells"][0])), "twice"),
        (lambda p: p["excluded_cells"][0].update(reason=""), "reason"),
    ],
    ids=["authorized-and-excluded", "authorized-and-skipped", "duplicate-exclusion", "exclusion-without-reason"],
)
def test_a_doctored_table_whose_lists_overlap_does_not_load(tmp_path, doctor, message):
    """Review F6b, MAJOR 1. The loader type-checked the new lists and nothing else, so an
    edited-and-rehashed table could list one cell as BOTH authorized and excluded — and
    `assert_cell_authorized` returns on the authorized list before it ever looks at exclusions, so
    the smuggled cell would run. A cell has ONE status; a table that gives it two settles nothing."""
    path = _attempt(tmp_path, "att-1")
    _run(
        _excluding("one_step:32:2", pos_fit_authorization=path),
        _CountingMeasurer(),
        cells=[probe.FitCell("rollout", 8, 2), probe.FitCell("one_step", 32, 2)],
    )
    probe.load_authorization(path)  # sound before doctoring
    _doctor(path, doctor)
    with pytest.raises(ValueError, match=message):
        probe.load_authorization(path)


def test_the_exclusion_declaration_moves_the_run_digest_with_the_measured_set_held_fixed(tmp_path):
    """Review F6b, MAJOR 2(a) — my earlier test observed the right outcome for the wrong reason.

    It declared an exclusion that was FILTERED OUT (the cell was not in the requested list), so the
    digest change it saw came from measuring a different number of cells, not from the declaration.
    Here the MEASURED set is identical in every run — the same two cells, the same trials — and only
    the declaration varies: present vs absent, and then reason vs reason. The run digest must move for
    each independently, and no per-cell recipe fingerprint may move at all."""
    measured = [probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 16, 2)]
    third = probe.FitCell("one_step", 8, 2)

    plain = _run(_config(pos_fit_authorization=_attempt(tmp_path, "a")), _CountingMeasurer(), cells=measured)
    declared = _run(
        _excluding("one_step:8:2", pos_fit_authorization=_attempt(tmp_path, "b")),
        _CountingMeasurer(),
        cells=measured + [third],
    )
    other_reason = _run(
        _excluding(
            "one_step:8:2", reason="a different declared reason", pos_fit_authorization=_attempt(tmp_path, "c")
        ),
        _CountingMeasurer(),
        cells=measured + [third],
    )

    assert plain["measured_cells"] == declared["measured_cells"] == other_reason["measured_cells"]
    assert plain["measurements"] == declared["measurements"] == other_reason["measurements"]
    assert declared["sha256"] != plain["sha256"], "declaring an exclusion must move the run digest"
    assert other_reason["sha256"] != declared["sha256"], "so must changing only the declared REASON"
    fingerprints = {t["context"]["recipe_fingerprint"] for t in (plain, declared, other_reason)}
    assert len(fingerprints) == 1, "and no per-cell recipe fingerprint may move with the declaration"


#: The fields a table may legitimately differ in without the RUN being different, and why each is
#: allowed to vary. Everything outside this projection is compared literally (review F6b, MAJOR 2b).
_UNSTABLE_FIELDS = {
    "cell_provenance": "records WHERE each cell came from; adoption paths are attempt-scoped and differ by design",
    "sha256": "a function of the payload, so comparing it as well as the payload would be comparing twice",
}


def test_an_empty_exclusion_declaration_is_behaviourally_inert(tmp_path):
    """Review F6b, MAJOR 2(b): the claim is narrowed to what is actually provable here.

    This does NOT prove byte-identity with the pre-F6 table, and the earlier "bit-for-bit with today"
    wording was wrong: F6 bumped the protocol v3 -> v4 and added `excluded_cells`, `skipped_cells` and
    `exclusion_reason`, so literal identity with HEAD is impossible by construction. What IS provable,
    and what matters, is that DECLARING the keys empty changes nothing about the run: same cells
    measured, same numbers, same verdicts, and the new fields present and empty."""
    plain = _run(_config(pos_fit_authorization=_attempt(tmp_path, "a")), _CountingMeasurer())
    declared_empty = _run(_excluding(pos_fit_authorization=_attempt(tmp_path, "b")), _CountingMeasurer())
    stable = lambda table: {k: v for k, v in table.items() if k not in _UNSTABLE_FIELDS}  # noqa: E731
    assert json.dumps(stable(plain), sort_keys=True) == json.dumps(stable(declared_empty), sort_keys=True)
    assert plain["excluded_cells"] == [] and plain["exclusion_reason"] == "" and plain["skipped_cells"] == []
    assert set(_UNSTABLE_FIELDS) < set(plain), "the projection must name fields the table actually has"


@pytest.mark.parametrize("raw", [",", "one_step:32:2,", ",one_step:32:2", "one_step:32:2,,one_step:32:4", " , "])
def test_an_empty_token_in_the_declaration_is_a_loud_error(raw):
    """Review F6b, MINOR. `pos_fit_excluded_cells=","` parsed as NO exclusions, so a declaration meant
    to keep the probe away from the deterministic-fault cell could silently walk straight into it."""
    with pytest.raises(ValueError, match="empty"):
        probe.parse_excluded_cells(_config(pos_fit_excluded_cells=raw, pos_fit_exclusion_reason=_WHY))


# =============================================================================================
# Round F7 `manifest-identity` — measured in production, not argued.
#
# M1-6 attempt 2 adopted ZERO cells. The M1-5 bank was refused with
#
#     not adopting .../cells/rollout_m8_k2.json.digest: it was measured under a different program --
#     ['code_sha'] differ (measured on f51a8a6.../v6e x8, running 5631a36.../v6e x8)
#
# while `manifest_digest` MATCHED: the only difference between those two tips is the Planner's
# ledger commits, which are docs-only. F5b's "any commit invalidates every banked cell" meets the
# campaign's own discipline of recording every submission in the ledger, and the product of the two
# is guaranteed bank loss on every resubmission — the failure F5 exists to prevent, reintroduced by
# the binding that was supposed to protect it.
#
# F6b already wrote down the right principle: **the manifest is the identity; code_sha is the
# label.** F7 makes the code agree. The label stays in the artifact, the table and the run digest,
# where it has audit value; it stops being able to throw away a measurement it does not disagree with.
# =============================================================================================


def _at(monkeypatch, *, code_sha, manifest):
    monkeypatch.setattr(probe, "derive_code_sha", lambda **kwargs: code_sha)
    monkeypatch.setattr(probe, "deployed_manifest_digest", lambda root=None: manifest)


def test_a_docs_only_commit_does_not_throw_away_the_bank(tmp_path, monkeypatch, capsys):
    """(a) The exact production construction: bank at tip A, adopt at tip B, bytes identical.

    Two tips that differ only in committed documentation are the same program. Refusing the bank
    across them cost M1-6 its entire inherited ladder for no measurable reason at all."""
    cells = [probe.FitCell("rollout", 8, 2), probe.FitCell("rollout", 16, 2)]
    root = _adoption_root(tmp_path)
    manifest = "m" * 64

    _at(monkeypatch, code_sha="f51a8a6" + "0" * 33, manifest=manifest)
    first = _CountingMeasurer()
    _run(_config(pos_fit_authorization=_attempt(tmp_path, "att-1"), pos_fit_adoption_root=root), first, cells=cells)
    assert len(first.calls) == 4

    _at(monkeypatch, code_sha="5631a36" + "0" * 33, manifest=manifest)  # a docs-only descendant
    second = _CountingMeasurer()
    table = _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "att-2"), pos_fit_adoption_root=root), second, cells=cells
    )
    assert len(second.calls) == 0, "identical bytes under a different label must still be adopted"
    assert all(row["provenance"].startswith(probe.ADOPTED_PREFIX) for row in table["cell_provenance"])

    out = capsys.readouterr().out
    assert "label" in out.lower() and "f51a8a6" in out, "the drift is reported, not silently swallowed"
    assert table["context"]["code_sha"].startswith("5631a36"), "the running label is still what the table records"


def test_a_different_build_under_the_same_label_is_still_refused(tmp_path, monkeypatch, capsys):
    """(b) The direction that must survive: the label is not evidence, so it cannot be a free pass.

    Two deployments can carry the same `COMMIT` and different bytes — a dirty tree, a stale tarball,
    a hand-edited module on a worker. That is exactly what F5b's manifest was introduced for, and
    narrowing the binding must not weaken it by one field."""
    cells = [probe.FitCell("rollout", 8, 2)]
    root = _adoption_root(tmp_path)
    label = "a" * 40

    _at(monkeypatch, code_sha=label, manifest="1" * 64)
    _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "att-1"), pos_fit_adoption_root=root),
        _CountingMeasurer(),
        cells=cells,
    )

    _at(monkeypatch, code_sha=label, manifest="2" * 64)  # same label, different running bytes
    after = _CountingMeasurer()
    _run(_config(pos_fit_authorization=_attempt(tmp_path, "att-2"), pos_fit_adoption_root=root), after, cells=cells)
    assert len(after.calls) == 2, "a different build must be re-measured however it is labelled"
    assert "manifest_digest" in capsys.readouterr().out


def test_the_binding_names_exactly_what_adoption_compares(tmp_path):
    """The binding is a declared set, so narrowing it is a diff a reviewer reads rather than a
    behaviour they have to infer. `code_sha` is deliberately absent; everything that decides what a
    measurement measured is present."""
    assert "code_sha" not in probe.ProbeContext.BINDING_FIELDS
    for field in (
        "manifest_digest",
        "model_revision",
        "device_kind",
        "device_count",
        "geometry",
        "recipe_fingerprint",
    ):
        assert field in probe.ProbeContext.BINDING_FIELDS, field

    context = _context()
    assert probe.ProbeContext.BINDING_FIELDS < tuple(context.as_payload()) or set(
        probe.ProbeContext.BINDING_FIELDS
    ) < set(context.as_payload())
    label_only = dataclasses.replace(context, code_sha="9" * 40)
    assert label_only.binding_digest() == context.binding_digest(), "the label is outside the binding"
    assert label_only.digest() != context.digest(), "...and inside the full context digest, for audit"
    moved = dataclasses.replace(context, manifest_digest="9" * 64)
    assert moved.binding_digest() != context.binding_digest()
    assert "manifest_digest" in moved.binding_differences(context)
    assert "code_sha" not in label_only.binding_differences(context)


def test_resume_binds_to_the_build_and_not_to_the_label(tmp_path):
    """The same narrowing where the same breakage would happen next: a training run resumed after a
    ledger commit would otherwise refuse its own predecessor's checkpoint.

    The F5c/F6b direction is asserted in the same test so it cannot be lost: two publications sharing
    a LABEL but built from different bytes must still refuse each other."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    parent = str(tmp_path / "attempts")
    build_a, build_b = "1" * 64, "2" * 64
    trainer.publish_attempt(
        parent,
        attempt="att-1",
        arm="rollout",
        code_sha="f51a8a6" + "0" * 33,
        context_digest="c" * 64,
        binding_digest=build_a,
        step=1000,
        checkpoint_dir=f"{parent}/att-1/checkpoints",
    )
    trainer.publish_attempt(
        parent,
        attempt="att-2",
        arm="rollout",
        code_sha="f51a8a6" + "0" * 33,
        context_digest="d" * 64,
        binding_digest=build_b,
        step=9000,
        checkpoint_dir=f"{parent}/att-2/checkpoints",
    )

    # A later attempt at a NEW label but the SAME build adopts att-1 -- and not the higher-step att-2,
    # which shares the label and was built from different bytes.
    chosen = trainer.select_resume_publication(
        parent, code_sha="5631a36" + "0" * 33, arm="rollout", binding_digest=build_a
    )
    assert chosen is not None and chosen["attempt"] == "att-1"
    assert (
        trainer.select_resume_publication(parent, code_sha="anything", arm="rollout", binding_digest="9" * 64) is None
    )


# =============================================================================================
# Round F7b — the same narrowing at the M2 GATE, which F7 flagged and left alone.
#
# F7 fixed adoption and resume and reported that `assert_cell_authorized` still compared the full
# context: an M2 launched after a ledger commit would be refused an authorization M1 published
# before it — the identical failure one step later, and the one that would have bitten at the most
# expensive moment (a 64-chip reservation, refused at startup). Ruled YES; done here.
# =============================================================================================


def _authorization(tmp_path, name, context, cells=(("rollout", 32, 2),)):
    """A real published table, measured under `context`."""
    measurements = [_measurement(context, probe.FitCell(*cell)) for cell in cells]
    evidence = probe.build_evidence(
        context, measurements, max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
    )
    return probe.publish_authorization(str(tmp_path / name), evidence)


def test_the_gate_accepts_an_authorization_across_a_docs_only_commit(tmp_path, capsys):
    """M1 publishes at tip A; the Planner records the launch in the ledger; M2 starts at tip B.

    Same bytes, same model, same topology, same recipe — and until F7b the gate refused, which would
    have killed an M2 launch at startup with a 64-chip reservation held. The label drifts; the
    program does not."""
    measured_at = _context()
    running_now = dataclasses.replace(measured_at, code_sha="5631a36" + "0" * 33)
    assert measured_at.code_sha != running_now.code_sha

    published = _authorization(tmp_path, "m1.json", measured_at)
    probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=running_now)

    out = capsys.readouterr().out
    assert "label" in out.lower(), "the drift is reported at the gate, not silently swallowed"
    assert measured_at.code_sha[:12] in out and running_now.code_sha[:12] in out


def test_the_gate_still_refuses_an_authorization_from_a_different_build(tmp_path):
    """The dangerous direction, unchanged: identical LABEL, different bytes. A dirty tree or a stale
    tarball can produce exactly this, and it is what the manifest was introduced for."""
    measured_at = _context()
    running_now = dataclasses.replace(measured_at, manifest_digest="9" * 64)
    published = _authorization(tmp_path, "m1.json", measured_at)

    with pytest.raises(ValueError, match="manifest_digest"):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=running_now)


@pytest.mark.parametrize(
    "field, value",
    [("model_revision", "Some-Other/Model@" + "0" * 40), ("device_count", 64), ("recipe_fingerprint", "7" * 64)],
)
def test_the_gate_refuses_every_field_that_decides_what_was_measured(tmp_path, field, value):
    """The binding is not just the manifest: a foreign model, another topology or another recipe all
    still refuse, so narrowing the gate to the build did not narrow it to one field."""
    measured_at = _context()
    published = _authorization(tmp_path, "m1.json", measured_at)
    running_now = dataclasses.replace(measured_at, **{field: value})
    with pytest.raises(ValueError, match=field):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=running_now)


def test_a_v4_table_is_refused_by_version_and_not_by_a_cryptic_field_error(tmp_path):
    """End-to-end version story for M2: M1 publishes v5, M2's loader takes v5, and a table from the
    protocol before it says SO — the operator needs "re-run M1", not a missing-key traceback."""
    context = _context()
    published = _authorization(tmp_path, "m1.json", context)
    path = str(tmp_path / "v4.json")
    payload = {**{k: v for k, v in published.items() if k != "sha256"}, "protocol": "exp06.fit_authorization.v4"}
    Path(path).write_text(
        json.dumps(
            {"payload": payload, "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()},
            sort_keys=True,
        )
    )
    with pytest.raises(ValueError) as raised:
        probe.load_authorization(path)
    message = str(raised.value)
    assert "exp06.fit_authorization.v4" in message and probe.AUTHORIZATION_PROTOCOL in message
    assert "measurements" not in message and "excluded_cells" not in message, "not a field-level error"


# =============================================================================================
# Round F7c — the binding did not cover the compiler.
#
# The reviewer executed it: two contexts differing in `LIBTPU_INIT_ARGS` and
# `XLA_PYTHON_CLIENT_MEM_FRACTION` compared binding-EQUAL. The manifest hashes `src/maxdiffusion`
# Python and nothing else, but the launcher sets the runtime and compiler policy through the
# environment — and a peak measured at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` is not a statement
# about a run at 0.5, nor is a step time measured under different `LIBTPU_INIT_ARGS` a statement
# about one measured without them. Same source, different compiler, different measurement.
# =============================================================================================


_POLICY_A = {
    "LIBTPU_INIT_ARGS": "--xla_tpu_enable_async_collective_fusion=true",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
}
_POLICY_B = {
    "LIBTPU_INIT_ARGS": "--xla_tpu_enable_async_collective_fusion=false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
}


#: Captured before any patching: `_under_policy` replaces the module attribute, so calling it through
#: the module inside the replacement would recurse forever.
_REAL_RUNTIME_POLICY = probe.derive_runtime_policy


def _under_policy(monkeypatch, policy):
    """Point the real derivation at a chosen environment, the way a launcher would."""
    monkeypatch.setattr(probe, "derive_runtime_policy", lambda environ=None: _REAL_RUNTIME_POLICY(policy))


def _canonical(mapping):
    return probe.canonical_runtime_policy(probe.derive_runtime_policy(mapping))


def test_the_runtime_policy_is_derived_raw_and_canonicalised_only_for_the_digest():
    """Two claims, and F7d corrected both.

    The RAW value is what lands in the artifact — the first version stored the normalized form and
    called it raw, so the audit value was the transformed one. And normalization happens BETWEEN
    flags, never INSIDE one: the reviewer collided two valid, materially different settings
    (`--xla_dump_hlo_module_re='foo  bar'` against `'foo bar'`) because whitespace was collapsed
    across the whole string."""
    declared = probe.RUNTIME_POLICY_KEYS
    for key in ("LIBTPU_INIT_ARGS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_PLATFORMS", "XLA_FLAGS"):
        assert key in declared, f"{key} reaches the compiler or the runtime and must be bound"

    raw = probe.derive_runtime_policy({"XLA_FLAGS": "  --a   --b  "})
    assert dict(raw)["XLA_FLAGS"] == "  --a   --b  ", "the artifact records what the environment said"

    assert _canonical({}) == _canonical(dict.fromkeys(declared, "")) == _canonical(dict.fromkeys(declared, "   "))
    assert _canonical({"XLA_FLAGS": "--a   --b"}) == _canonical({"XLA_FLAGS": "--a --b"}), "between flags"
    assert _canonical({"XLA_FLAGS": "--b --a"}) != _canonical({"XLA_FLAGS": "--a --b"}), "ORDER is policy"

    # The reviewer's executed collision: two valid, DISTINCT dump filters.
    wide = _canonical({"XLA_FLAGS": "--xla_dump_hlo_module_re='foo  bar'"})
    narrow = _canonical({"XLA_FLAGS": "--xla_dump_hlo_module_re='foo bar'"})
    assert wide != narrow, "whitespace INSIDE a quoted flag value is part of the value, not formatting"

    unbalanced = _canonical({"XLA_FLAGS": "--re='unterminated"})
    assert unbalanced["XLA_FLAGS"] == ["--re='unterminated"], "an unparseable policy is not simplified"


def test_two_launches_differing_only_in_compiler_policy_are_not_the_same_program(monkeypatch):
    """The reviewer's executed construction, as a binding assertion."""
    _under_policy(monkeypatch, _POLICY_A)
    here = _context()
    _under_policy(monkeypatch, _POLICY_B)
    there = _context()

    assert here.manifest_digest == there.manifest_digest, "same source -- this is the point"
    assert here.binding_digest() != there.binding_digest(), "different compiler policy is a different program"
    assert "runtime_policy_digest" in here.binding_differences(there)
    assert here.as_payload()["runtime_policy"] != there.as_payload()["runtime_policy"], "raw values are audited"


def test_a_cell_measured_under_other_compiler_policy_is_re_measured(tmp_path, monkeypatch, capsys):
    """Through ADOPTION, which is where a stale peak would be inherited."""
    cells = [probe.FitCell("rollout", 8, 2)]
    root = _adoption_root(tmp_path)

    _under_policy(monkeypatch, _POLICY_A)
    _run(
        _config(pos_fit_authorization=_attempt(tmp_path, "att-1"), pos_fit_adoption_root=root),
        _CountingMeasurer(),
        cells=cells,
    )
    _under_policy(monkeypatch, _POLICY_B)
    after = _CountingMeasurer()
    _run(_config(pos_fit_authorization=_attempt(tmp_path, "att-2"), pos_fit_adoption_root=root), after, cells=cells)
    assert len(after.calls) == 2, "a peak measured under other compiler flags is not evidence about this run"
    assert "runtime_policy_digest" in capsys.readouterr().out


def test_the_gate_refuses_an_authorization_measured_under_other_compiler_policy(tmp_path, monkeypatch):
    """...and through the M2 GATE, which is where it would cost a reservation."""
    _under_policy(monkeypatch, _POLICY_A)
    measured_at = _context()
    published = _authorization(tmp_path, "m1.json", measured_at)
    _under_policy(monkeypatch, _POLICY_B)
    running_now = _context()
    with pytest.raises(ValueError, match="runtime_policy_digest"):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=running_now)


def test_resume_refuses_state_built_under_other_compiler_policy(tmp_path, monkeypatch):
    """...and through RESUME — calling the real selector, not comparing two digests.

    F7d: the first version of this asserted `binding_digest() != binding_digest()`, which is a
    restatement of the previous test rather than evidence about resume. The F3c liveness lesson says
    a guard that inspects a value proves a value; this one publishes an attempt and runs the deployed
    selector against it."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer as trainer

    parent = str(tmp_path / "attempts")
    _under_policy(monkeypatch, _POLICY_A)
    built_under_a = _context()
    trainer.publish_attempt(
        parent,
        attempt="att-1",
        arm="rollout",
        code_sha=built_under_a.code_sha,
        context_digest=built_under_a.digest(),
        binding_digest=built_under_a.binding_digest(),
        step=1000,
        checkpoint_dir=f"{parent}/att-1/checkpoints",
    )
    assert (
        trainer.select_resume_publication(
            parent, arm="rollout", binding_digest=built_under_a.binding_digest(), code_sha=built_under_a.code_sha
        )["attempt"]
        == "att-1"
    ), "its own policy adopts it"

    _under_policy(monkeypatch, _POLICY_B)
    built_under_b = _context()
    assert (
        trainer.select_resume_publication(
            parent, arm="rollout", binding_digest=built_under_b.binding_digest(), code_sha=built_under_b.code_sha
        )
        is None
    ), "state built under other compiler flags is not this run's to resume"


def test_a_docs_only_commit_still_costs_nothing_under_the_wider_binding(monkeypatch):
    """F7 must survive F7c: the LABEL is still outside the binding, policy or no policy."""
    _under_policy(monkeypatch, _POLICY_A)
    here = _context()
    relabelled = dataclasses.replace(here, code_sha="5631a36" + "0" * 33)
    assert relabelled.binding_digest() == here.binding_digest()

"""exp_04 R6 — ``null_adapter_runner_core``: arm execution, metric tables, records, adequacy probe.

This round is orchestration, so almost every test here is a **composition test**: it rebuilds one arm
out of the R1–R4 primitives the runner is supposed to be calling, and demands the runner's own output
bit for bit. Wiring bugs -- A2 replayed from the wrong start, the probe seeded from the global draw,
the pivots recomputed per arm -- are exactly the class of error that produces plausible numbers, so
"plausible" is never the standard: ``f32_bits`` equality is.

What carries the round:

- **The writer-order contract (the R4c deliverable).** ``verify_replay``'s docstring states it and
  R4c's review deferred the pin to this round: ``expected_final_latent`` must be computed from the
  values the record will *store*. ``z_i0`` is always kept at the source fp16, so a writer that
  replays the pre-cast fp32 tensors emits records that cannot verify.
  ``test_writer_order_...`` builds both orders through the same public codec and shows the mutant's
  records failing ``verify_replay`` while the correct ones pass. The header declares ``fp32`` there
  on purpose: it isolates the ``z_i0`` cast, which happens under *every* dtype policy, so the test is
  about the order rather than about fp16 rounding.
- **One inversion per batch.** The pivots are the expensive part and every arm shares them (A2 keeps
  the same targets and only replaces the starting pivot with eps_0). Pinned by a call count under
  ``jax.disable_jit()`` -- both of ``invert_trajectory`` itself and of the velocity seam, whose exact
  per-arm budget is spelled out below.
- **Frame-0 exclusion.** The primary metric excludes the pinned latent frame. The fixture's
  ``z_video`` carries a deliberately wild frame 0 (never seen by any arm, since inversion pins it to
  ``z_i0`` immediately), so a metric that included it would be off by four orders of magnitude.
- **The tables are the gates' input.** They round-trip through ``null_adapter_gates.parse_table`` and
  are fed to the real gate. ``future_ssim`` is absent until R7 decodes, and the gate therefore reports
  every observation invalid -- asserted here, so the R7 dependency is a test result rather than a
  comment.

Geometry is production throughout (48/9/12/20 latents, 512x4096 context, 25 sampler steps): the toy
velocity makes a full six-arm run cost about a second, which is cheaper than the honesty lost to a
tiny-shape seam. The private-seam allowance in the round brief was therefore not needed.
"""

from __future__ import annotations

import dataclasses
import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bit_test_helpers import f32_bits
from maxdiffusion.models.wan.null_inversion_wan import (
    LATENT_SHAPE,
    base_context_fingerprint,
    global_noise,
    invert_trajectory,
    keyed_noise,
    optimize_null_embeddings,
    replay_with_nulls,
)
from maxdiffusion.null_adapter_gates import NoiseConvention, gate_g1, parse_table, select_target
from maxdiffusion.null_adapter_records import ProvenanceHeader, make_record
from maxdiffusion.null_adapter_runner_core import (
    ADEQUACY_GRID,
    ADOPTION_FACTOR,
    DEFAULT_RECIPE,
    METHODS,
    PLATEAU_BOUNDARY_ATOL,
    PLATEAU_MIN_IMPROVEMENT,
    PROBE_K_SET,
    RECIPE_LIMITED,
    RECONSTRUCTION_LIMITED,
    UNDETERMINED,
    AdequacyReport,
    CapacityBatch,
    CapacityParams,
    RecipeScore,
    adopt_recipe,
    batch_fingerprint,
    build_capacity_records,
    emit_metric_tables,
    run_adequacy_probe,
    run_capacity_example_batch,
)
from maxdiffusion.null_adapter_verify import canonical_sigmas, verify_replay


_L, _S, _D = 16, 512, 4096
_Z_I0_SHAPE = (48, 1, 12, 20)
_SIGMAS = jnp.asarray(canonical_sigmas())
_STEPS = _SIGMAS.shape[0] - 1
_GUIDE_SCALE = 5.0
_REVISION = "Wan2.2-TI2V-5B@f370228"
# The R1 golden names, so the keyed draws these tests compare against are the ones pinned by
# ``test_null_adapter_noise.py`` rather than freshly invented strings.
_NAMES = ("droid_ep_000001/w0", "droid_ep_000042/w3")
_PROBE_NAMES = (*_NAMES, "droid_ep_000123/w7")  # three, so a median is not also a mean

_RNG = np.random.default_rng(20260805)
_BASE_CONTEXT = jnp.asarray(_RNG.standard_normal((_S, _D), dtype=np.float32) * 0.02)
_WEIGHTS = jnp.asarray(_RNG.standard_normal((_L, _D), dtype=np.float32) * 0.02)
_PATTERN = jnp.asarray(_RNG.standard_normal(LATENT_SHAPE, dtype=np.float32) * 0.05)
# The latent-driven terms deliberately dominate the context- and timestep-driven ones: with a
# context-dominated oracle every example follows almost the same trajectory and its tracking loss
# comes out equal to five digits, which would make "median over examples" indistinguishable from
# "mean over examples" in the adequacy test. All four couplings stay non-zero.
_C_PATTERN, _C_Z, _C_MIX, _C_T = 0.2, 0.5, 0.05, 1e-5


def _velocity_fn(z, timestep_2d, context):
    """Coupled oracle: reads the context (what is optimized), the latent, and the per-token timestep.

    The cross-frame mean is deliberate. Latent frame 0 is pinned to ``z_i0`` at every step, so without
    a term that mixes across frames an fp16 round-trip of ``z_i0`` would move frame 0 alone; with it,
    the storage cast reaches the whole tensor -- which is what the writer-order test measures.
    """
    scale = jnp.sum(context[:, :_L] * _WEIGHTS, axis=(1, 2))
    per_example_t = jnp.mean(timestep_2d, axis=1)
    return (
        _C_PATTERN * scale[:, None, None, None, None] * _PATTERN
        + _C_Z * z
        + _C_MIX * jnp.mean(z, axis=2, keepdims=True)
        + _C_T * per_example_t[:, None, None, None, None]
    )


def _forbidden_velocity(*args, **kwargs):
    """Every argument rejection must fire before a single model forward."""
    raise AssertionError("arguments must be validated before any compute")


def _batch(names=_NAMES, seed=7):
    """Per-example scales 1, 3, 5, ... -- without them the toy dynamics make every example score
    almost identically, and statistics like "median over examples" would be indistinguishable from
    "mean over examples" (the adequacy test asserts the difference)."""
    rng = np.random.default_rng(seed)
    scale = (1.0 + 2.0 * np.arange(len(names), dtype=np.float32))[:, None, None, None, None]
    z_video = rng.standard_normal((len(names), *LATENT_SHAPE), dtype=np.float32) * scale
    # Latent frame 0 is pinned to z_i0 before the first forward, so no arm ever sees this value: it
    # exists purely to make "did the primary metric drop frame 0?" a four-order-of-magnitude question.
    z_video[:, :, 0] = 100.0
    z_i0 = rng.standard_normal((len(names), *_Z_I0_SHAPE), dtype=np.float32) * scale
    return CapacityBatch(names=tuple(names), z_i0=jnp.asarray(z_i0), z_video=jnp.asarray(z_video))


def _params(**overrides):
    return CapacityParams(**{"inner_iters": 1, "lr": 1e-2, "guide_scale": _GUIDE_SCALE, "l_null": _L, **overrides})


@functools.lru_cache(maxsize=None)
def _cached_run():
    """One six-arm run, reused by the wiring/metric/record tests (about a second of toy compute)."""
    batch, params = _batch(), _params()
    return batch, params, run_capacity_example_batch(_velocity_fn, batch, _BASE_CONTEXT, params)


@functools.lru_cache(maxsize=None)
def _cached_probe():
    """One two-cell adequacy run, shared by the score and the trace-retention tests."""
    return run_adequacy_probe(
        _velocity_fn, _batch(_PROBE_NAMES), _BASE_CONTEXT, ((1, 1e-2), (2, 3e-2)), guide_scale=_GUIDE_SCALE, l_null=_L
    )


@functools.lru_cache(maxsize=None)
def _cached_inversion(names=_NAMES):
    """The pivots, rebuilt independently of the runner: w=1 means one context and no CFG mixing."""
    batch = _batch(names)
    cond = jnp.broadcast_to(_BASE_CONTEXT, (len(names), _S, _D))
    return invert_trajectory(lambda z, t: _velocity_fn(z, t, cond), batch.z_video, batch.z_i0, _SIGMAS)


def _replay(z_start, nulls, batch):
    return replay_with_nulls(
        _velocity_fn, z_start, batch.z_i0, _SIGMAS, nulls, _BASE_CONTEXT, guide_scale=_GUIDE_SCALE
    )


def _frozen_nulls():
    """The A0/A2-0 control: the context's own leading rows, so v_unc == v_cond at every step."""
    return jnp.broadcast_to(_BASE_CONTEXT[:_L], (_STEPS, _L, _D))


def _eps0(batch):
    return jnp.broadcast_to(global_noise(0), (len(batch.names), *LATENT_SHAPE))


def _optimize(pivots, batch, inner_iters=1, lr=1e-2):
    return optimize_null_embeddings(
        _velocity_fn,
        pivots,
        batch.z_i0,
        _SIGMAS,
        _BASE_CONTEXT[:_L],
        _BASE_CONTEXT,
        inner_iters=inner_iters,
        lr=lr,
        guide_scale=_GUIDE_SCALE,
    )


def test_a0_is_a_frozen_replay_from_the_inversion_endpoint():
    batch, _, results = _cached_run()

    expected = _replay(_cached_inversion()[0], _frozen_nulls(), batch)

    np.testing.assert_array_equal(f32_bits(results.final_latents["a0"]), f32_bits(expected))


def test_a1_optimizes_and_replays_from_the_inversion_endpoint():
    batch, params, results = _cached_run()
    traj = _cached_inversion()

    nulls, _, _, _ = _optimize(traj, batch, inner_iters=params.inner_iters, lr=params.lr)

    np.testing.assert_array_equal(f32_bits(results.nulls["a1"]), f32_bits(nulls))
    np.testing.assert_array_equal(f32_bits(results.z_start["a1"]), f32_bits(traj[0]))
    np.testing.assert_array_equal(f32_bits(results.final_latents["a1"]), f32_bits(_replay(traj[0], nulls, batch)))


def test_a2_optimizes_from_eps0_against_the_same_pivots():
    """A2 replaces the *starting* pivot with eps_0 and keeps every target: same inversion, new basin."""
    batch, params, results = _cached_run()
    traj, eps0 = _cached_inversion(), _eps0(batch)

    nulls, _, _, _ = _optimize(traj.at[0].set(eps0), batch, inner_iters=params.inner_iters, lr=params.lr)

    np.testing.assert_array_equal(f32_bits(results.nulls["a2"]), f32_bits(nulls))
    np.testing.assert_array_equal(f32_bits(results.z_start["a2"]), f32_bits(eps0))
    np.testing.assert_array_equal(f32_bits(results.final_latents["a2"]), f32_bits(_replay(eps0, nulls, batch)))


def test_a2_0_is_the_frozen_control_from_the_same_eps0():
    batch, _, results = _cached_run()

    expected = _replay(_eps0(batch), _frozen_nulls(), batch)

    np.testing.assert_array_equal(f32_bits(results.final_latents["a2_0"]), f32_bits(expected))
    # ... and it is genuinely a different arm from A0, which starts at the inversion endpoint.
    assert not np.array_equal(np.asarray(results.final_latents["a2_0"]), np.asarray(results.final_latents["a0"]))


def test_eps0_is_the_broadcast_global_draw_not_a_batch_shaped_one():
    """R1's assembly rule: one canonical eps_0 shared by every example, broadcast rather than drawn."""
    batch, _, results = _cached_run()

    z_start = np.asarray(results.z_start["a2"])

    for row in range(len(batch.names)):
        np.testing.assert_array_equal(z_start[row], np.asarray(global_noise(0)))


@pytest.mark.parametrize("method, arm", [("a1_probe", "a1"), ("a2_probe", "a2")])
def test_probe_arms_replay_the_optimized_nulls_from_stacked_keyed_noise(method, arm):
    """The transfer probe's whole point is *fresh per-example* noise: keyed(name, k), stacked."""
    batch, _, results = _cached_run()

    for index, k in enumerate(PROBE_K_SET):
        z_start = jnp.stack([keyed_noise(name, k) for name in batch.names])
        expected = _replay(z_start, jnp.asarray(results.nulls[arm]), batch)

        np.testing.assert_array_equal(f32_bits(results.final_latents[method][index]), f32_bits(expected))


def test_the_probe_k_set_is_the_plan_s_fixed_three_seeds():
    _, _, results = _cached_run()

    assert PROBE_K_SET == (0, 1, 2)
    assert results.final_latents["a1_probe"].shape[0] == len(PROBE_K_SET)
    assert set(results.metrics["a1_probe"][_NAMES[0]]) == {"0", "1", "2"}
    assert set(results.metrics["a1"][_NAMES[0]]) == {"0"}


def test_inversion_runs_once_per_batch_and_the_velocity_budget_is_exact(monkeypatch):
    """Counted eagerly (``disable_jit``), where every scan iteration is a real call.

    Per outer step the optimizer spends 1 conditional + J inner + 1 locked-advance forward and a
    replay spends 2 (conditional and null branch), so the whole run is
    ``N * (1 inversion + 2 * (J + 2) optimizations + 20 replay forwards)``: A0, A1, A2, A2-0 and two
    three-seed probes make ten replays. Recomputing the pivots for A2 would add ``N`` calls on top of
    a second ``invert_trajectory`` entry, and both are visible here.
    """
    from maxdiffusion import null_adapter_runner_core as runner

    inversions, calls = [], []
    original = runner.invert_trajectory
    monkeypatch.setattr(runner, "invert_trajectory", lambda *a, **k: (inversions.append(1), original(*a, **k))[1])

    def counting_velocity(z, timestep_2d, context):
        calls.append(1)
        return _velocity_fn(z, timestep_2d, context)

    batch, params = _batch(_NAMES[:1]), _params(inner_iters=1)
    with jax.disable_jit():
        run_capacity_example_batch(counting_velocity, batch, _BASE_CONTEXT, params)

    assert len(inversions) == 1
    assert len(calls) == _STEPS * (1 + 2 * (params.inner_iters + 2) + 20)


def test_future_mse_excludes_latent_frame_zero_and_full_mse_does_not():
    batch, _, results = _cached_run()
    video = np.asarray(batch.z_video)

    for index, name in enumerate(batch.names):
        for method in ("a0", "a1", "a2", "a2_0"):
            latent = np.asarray(results.final_latents[method])[index]
            entry = results.metrics[method][name]["0"]

            assert entry["future_mse"] == pytest.approx(
                float(np.mean((latent[:, 1:] - video[index, :, 1:]) ** 2)), rel=1e-5
            )
            assert entry["full_mse"] == pytest.approx(float(np.mean((latent - video[index]) ** 2)), rel=1e-5)
            # The fixture's wild frame 0 lives only in the full-tensor number.
            assert entry["full_mse"] > 100.0 * entry["future_mse"]


def test_probe_metrics_are_reported_per_seed():
    batch, _, results = _cached_run()
    video = np.asarray(batch.z_video)

    for index, name in enumerate(batch.names):
        for position, k in enumerate(PROBE_K_SET):
            latent = np.asarray(results.final_latents["a1_probe"])[position, index]
            entry = results.metrics["a1_probe"][name][str(k)]

            assert entry["future_mse"] == pytest.approx(
                float(np.mean((latent[:, 1:] - video[index, :, 1:]) ** 2)), rel=1e-5
            )


def test_metric_tables_round_trip_and_reach_the_real_gates():
    """The emitted tables are the gates' input, so they are handed to the real gate, not inspected."""
    batch, _, results = _cached_run()

    tables = emit_metric_tables(results)
    parsed = {method: parse_table(json.dumps(table)) for method, table in tables.items()}

    assert set(tables) == set(METHODS)
    verdict = gate_g1(parsed["a1"], parsed["a0"], list(batch.names), NoiseConvention.GLOBAL)
    assert verdict.numbers["coverage_ok"], verdict.numbers
    assert verdict.numbers["missing_names"] == {"method": [], "control": []}
    selection = select_target(verdict, parsed["a1_probe"], list(batch.names), NoiseConvention.KEYED, verdict)
    assert selection.numbers["coverage_ok"], selection.numbers


def test_the_absent_ssim_makes_every_observation_invalid_until_r7_decodes():
    """Not a wart to work around: the gate must refuse to score tables that have no SSIM in them."""
    batch, _, results = _cached_run()

    tables = emit_metric_tables(results)

    assert all("future_ssim" not in entry for entry in tables["a1"][batch.names[0]].values())
    verdict = gate_g1(
        parse_table(json.dumps(tables["a1"])),
        parse_table(json.dumps(tables["a0"])),
        list(batch.names),
        NoiseConvention.GLOBAL,
    )
    assert not verdict.passed
    assert "invalid_fraction" in verdict.reasons and verdict.numbers["invalid_fraction"] == 1.0


def test_emit_metric_tables_refuses_a_table_that_does_not_cover_the_batch():
    _, _, results = _cached_run()
    broken = dataclasses.replace(
        results, metrics={**results.metrics, "a1": dict(list(results.metrics["a1"].items())[:1])}
    )

    with pytest.raises(ValueError, match="does not cover"):
        emit_metric_tables(broken)


def test_emit_metric_tables_refuses_a_missing_seed():
    _, _, results = _cached_run()
    dropped = {name: {"0": seeds["0"]} for name, seeds in results.metrics["a1_probe"].items()}
    broken = dataclasses.replace(results, metrics={**results.metrics, "a1_probe": dropped})

    with pytest.raises(ValueError, match="seed keys"):
        emit_metric_tables(broken)


def _header(dtype_policy="fp32", **overrides):
    fields = {
        "manifest_hash": "a" * 64,
        "code_sha": "b" * 40,
        "model_revision": _REVISION,
        "sigma_vector": canonical_sigmas(),
        "guide_scale": _GUIDE_SCALE,
        "base_context_fingerprint": base_context_fingerprint(_BASE_CONTEXT),
        "optimization_config": {"inner_iters": 1, "lr": 0.01},
        "dtype_policy": dtype_policy,
        "l_null": _L,
    }
    return ProvenanceHeader(**{**fields, **overrides})


def _example_fields(names):
    rng = np.random.default_rng(3)
    return {
        name: {
            "ordinal": index,
            "split": "dev",
            "episode": name.split("/")[0],
            "actions": rng.standard_normal((32, 7), dtype=np.float32),
        }
        for index, name in enumerate(names)
    }


def _verify(record, header, **overrides):
    kwargs = {
        "expected_model_revision": _REVISION,
        "expected_guide_scale": _GUIDE_SCALE,
        "expected_noise_convention": "keyed",
        "expected_arm": "A1",
        "atol": 1e-5,
        **overrides,
    }
    return verify_replay(record, header, _velocity_fn, _BASE_CONTEXT, **kwargs)


def _records_replaying_before_the_cast(results, batch, header, fields, arm="a1"):
    """The mutant writer, spelled out: replay the pre-cast fp32 tensors, cast only when recording.

    Identical to ``build_capacity_records`` in every other respect -- same codec, same header, same
    fields -- so the only thing the verification difference can be attributed to is the order.
    """
    nulls = np.asarray(results.nulls[arm])
    expected = np.asarray(
        replay_with_nulls(
            _velocity_fn,
            jnp.asarray(results.z_start[arm]),
            batch.z_i0,
            _SIGMAS,
            jnp.asarray(nulls),
            _BASE_CONTEXT,
            guide_scale=header.guide_scale,
        )
    )
    return [
        make_record(
            name=name,
            ordinal=fields[name]["ordinal"],
            split=fields[name]["split"],
            episode=fields[name]["episode"],
            z_i0=np.asarray(batch.z_i0)[index],
            actions=fields[name]["actions"],
            z_video=np.asarray(batch.z_video)[index],
            latent_dtype=header.dtype_policy,
            nulls=nulls[:, index],
            z_start=np.asarray(results.z_start[arm])[index],
            expected_final_latent=expected[index],
            noise_convention="keyed",
            arm="A1",
            per_step_final_losses=np.asarray(results.per_step_final_losses[arm][index], np.float32),
            final_future_mse=results.metrics[arm][name]["0"]["future_mse"],
        )
        for index, name in enumerate(batch.names)
    ]


def test_writer_order_is_cast_then_replay_then_record():
    """The R4c deliverable. ``z_i0`` is stored fp16 under every dtype policy, so a writer that
    replays before casting claims an endpoint its own stored inputs do not reach.

    Measured on this fixture: correctly-built fp32 records reproduce their endpoint *exactly*
    (max|delta| = 0, which incidentally re-confirms R4a's batched-equals-singleton replay property at
    production geometry), while the wrong-order records miss by 1.3e-3 and 3.9e-3 -- two orders above
    the ``atol`` used here. Under the fp16 policy the same two orders come out at 1.6e-3/3.9e-3 and
    2.3e-3/8.5e-3, i.e. **the storage tolerance an fp16 record needs would hide the mutant**, which is
    why this test declares fp32 and the fp16 case is only asked to verify, not to discriminate.
    """
    batch, _, results = _cached_run()
    header, fields = _header(), _example_fields(batch.names)

    correct = build_capacity_records(_velocity_fn, results, batch, _BASE_CONTEXT, header, fields, arm="a1")
    mutant = _records_replaying_before_the_cast(results, batch, header, fields)

    for record in correct:
        _verify(record, header)
    for record in mutant:
        with pytest.raises(ValueError, match="replay does not reproduce"):
            _verify(record, header)


def test_records_carry_the_arm_s_labels_and_the_header_s_policy():
    batch, _, results = _cached_run()
    header, fields = _header(), _example_fields(batch.names)

    records = build_capacity_records(_velocity_fn, results, batch, _BASE_CONTEXT, header, fields, arm="a2")

    for index, record in enumerate(records):
        assert (record.name, record.arm, record.noise_convention) == (batch.names[index], "A2", "global")
        assert record.latent_dtype == "fp32" and record.per_step_final_losses.shape == (_STEPS,)
        assert record.final_future_mse == pytest.approx(results.metrics["a2"][record.name]["0"]["future_mse"])
        np.testing.assert_array_equal(record.z_start, np.asarray(results.z_start["a2"])[index].astype(np.float32))
        np.testing.assert_allclose(
            record.per_step_final_losses, np.asarray(results.per_step_final_losses["a2"][index]), rtol=1e-6
        )
        _verify(record, header, expected_arm="A2", expected_noise_convention="global")


def test_fp16_records_verify_within_the_storage_tolerance():
    """The other dtype policy still goes through cast -> replay -> record; only the slack changes."""
    batch, _, results = _cached_run()
    header, fields = _header(dtype_policy="fp16"), _example_fields(batch.names)

    records = build_capacity_records(_velocity_fn, results, batch, _BASE_CONTEXT, header, fields, arm="a1")

    for record in records:
        assert record.latent_dtype == "fp16" and record.nulls.dtype == np.float16
        _verify(record, header, atol=1e-2)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"sigma_vector": np.linspace(1.0, 0.0, 26, dtype=np.float32)}, "sigma_vector"),
        ({"l_null": _L + 1}, "l_null"),
        ({"base_context_fingerprint": "d" * 64}, "base_context_fingerprint"),
        ({"dtype_policy": "fp8"}, "dtype_policy"),
    ],
)
def test_record_building_refuses_a_header_the_records_could_never_verify_under(overrides, message):
    batch, _, results = _cached_run()

    with pytest.raises(ValueError, match=message):
        build_capacity_records(
            _forbidden_velocity,
            results,
            batch,
            _BASE_CONTEXT,
            _header(**overrides),
            _example_fields(batch.names),
            arm="a1",
        )


def test_record_building_refuses_an_unknown_arm_and_incomplete_example_fields():
    batch, _, results = _cached_run()
    fields = _example_fields(batch.names)

    with pytest.raises(ValueError, match="arm must be one of"):
        build_capacity_records(_forbidden_velocity, results, batch, _BASE_CONTEXT, _header(), fields, arm="a0")
    without_actions = {name: {k: v for k, v in row.items() if k != "actions"} for name, row in fields.items()}
    with pytest.raises(ValueError, match="example_fields"):
        build_capacity_records(
            _forbidden_velocity, results, batch, _BASE_CONTEXT, _header(), without_actions, arm="a1"
        )


def test_record_building_refuses_results_from_a_different_batch():
    batch, _, results = _cached_run()
    other = _batch(names=("droid_ep_000999/w0", "droid_ep_000998/w0"))

    with pytest.raises(ValueError, match="different examples"):
        build_capacity_records(
            _forbidden_velocity, results, other, _BASE_CONTEXT, _header(), _example_fields(other.names), arm="a1"
        )


def test_arm_results_bind_the_run_that_produced_them():
    """Without this bind, ``verify_replay`` -- which checks a record against itself -- cannot tell a
    correctly-labelled artifact from a correctly-*formed* one (Codex R6 review, finding 1)."""
    batch, params, results = _cached_run()

    assert results.params == params
    assert results.base_context_fingerprint == base_context_fingerprint(_BASE_CONTEXT)
    assert results.batch_fingerprint == batch_fingerprint(batch.names, batch.z_i0, batch.z_video)


def test_the_batch_fingerprint_is_content_addressed_and_order_free():
    batch = _batch()
    fingerprint = batch_fingerprint(batch.names, batch.z_i0, batch.z_video)

    shuffled = batch_fingerprint(batch.names[::-1], batch.z_i0[::-1], batch.z_video[::-1])
    perturbed = batch_fingerprint(batch.names, batch.z_i0, batch.z_video.at[1, 0, 5, 0, 0].add(1e-3))
    renamed = batch_fingerprint((batch.names[0], "droid_ep_000999/w0"), batch.z_i0, batch.z_video)

    assert shuffled == fingerprint  # the same examples in a different batch order are the same batch
    assert perturbed != fingerprint and renamed != fingerprint


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"guide_scale": 7.0}, "header guide_scale"),
        ({"optimization_config": {"inner_iters": 50, "lr": 0.01}}, "optimization_config"),
        ({"optimization_config": {"inner_iters": 1, "lr": 0.03}}, "optimization_config"),
        ({"optimization_config": {"lr": 0.01}}, "optimization_config"),  # J not declared at all
    ],
)
def test_record_building_refuses_a_header_that_misdeclares_the_run(overrides, message):
    """The reviewer's mutant: nulls optimized at (J=1, lr=1e-2, w=5) published as (J=50, lr=3e-2, w=7).

    Every one of these produced records that passed ``verify_replay`` before this round's revision --
    the replay is self-consistent; it is the *labels* that lie.
    """
    batch, _, results = _cached_run()

    with pytest.raises(ValueError, match=message):
        build_capacity_records(
            _forbidden_velocity,
            results,
            batch,
            _BASE_CONTEXT,
            _header(**overrides),
            _example_fields(batch.names),
            arm="a1",
        )


def test_record_building_refuses_results_whose_bound_l_null_disagrees_with_the_header():
    """Two checks cover ``l_null``: the nulls' actual width, and the *bound recipe*. They coincide for
    a well-formed result and come apart exactly here -- where the number that would be published as
    the run's provenance is not the number the run was configured with. Without this case the recipe
    binding is invisible, because the width check answers first for every honest input.
    """
    batch, params, results = _cached_run()
    mislabelled = dataclasses.replace(results, params=dataclasses.replace(params, l_null=8))

    with pytest.raises(ValueError, match="header l_null 16 does not match the run's l_null 8"):
        build_capacity_records(
            _forbidden_velocity,
            mislabelled,
            batch,
            _BASE_CONTEXT,
            _header(),
            _example_fields(batch.names),
            arm="a1",
        )


def test_record_building_refuses_a_batch_whose_tensors_changed_under_the_same_names():
    """The reviewer's second mutant: same names, different data. The stale ``final_future_mse`` and
    ``z_video`` would otherwise have been cached against the new tensors without a murmur."""
    batch, _, results = _cached_run()
    impostor = _batch(names=batch.names, seed=99)

    assert impostor.names == batch.names
    with pytest.raises(ValueError, match="batch fingerprint"):
        build_capacity_records(
            _forbidden_velocity,
            results,
            impostor,
            _BASE_CONTEXT,
            _header(),
            _example_fields(batch.names),
            arm="a1",
        )


def test_record_building_refuses_results_optimized_against_a_different_context():
    """The header agrees with the context passed in; it is the *run* that used a different T5("")."""
    batch, _, results = _cached_run()
    other_context = _BASE_CONTEXT + 0.01

    with pytest.raises(ValueError, match="arm_results were produced against a different base_context"):
        build_capacity_records(
            _forbidden_velocity,
            results,
            batch,
            other_context,
            _header(base_context_fingerprint=base_context_fingerprint(other_context)),
            _example_fields(batch.names),
            arm="a1",
        )


@pytest.mark.parametrize("arm", ["a1", "a2"])
def test_the_optimizer_s_diagnostic_traces_are_retained(arm):
    """Plan §4-P1 asks for per-step final losses *and* per-inner-iteration grad-norm traces."""
    batch, params, results = _cached_run()
    traces = results.diagnostics[arm]

    assert set(traces) == {"tracking_losses", "grad_norms"}
    for label, trace in traces.items():
        assert trace.shape == (_STEPS, params.inner_iters, len(batch.names)), label
        assert np.all(np.isfinite(trace))
    assert np.all(traces["grad_norms"] > 0.0)  # at w=5 the null branch really does receive gradient
    assert results.per_step_final_losses[arm].shape == (len(batch.names), _STEPS)


def test_the_retained_traces_are_the_optimizer_s_own():
    batch, params, results = _cached_run()

    _, _, losses, norms = _optimize(_cached_inversion(), batch, inner_iters=params.inner_iters, lr=params.lr)

    np.testing.assert_array_equal(f32_bits(results.diagnostics["a1"]["tracking_losses"]), f32_bits(losses))
    np.testing.assert_array_equal(f32_bits(results.diagnostics["a1"]["grad_norms"]), f32_bits(norms))


def _poison(monkeypatch, field):
    """Replace one of the optimizer's returned tensors with NaN, leaving the rest of the run intact."""
    from maxdiffusion import null_adapter_runner_core as runner

    original = runner.optimize_null_embeddings

    def poisoned(*args, **kwargs):
        returned = list(original(*args, **kwargs))  # (nulls, z_bar, losses, grad_norms)
        index = {"z_bar": 1, "losses": 2, "norms": 3}[field]
        returned[index] = jnp.full_like(returned[index], jnp.nan)
        return tuple(returned)

    monkeypatch.setattr(runner, "optimize_null_embeddings", poisoned)


@pytest.mark.parametrize(
    "field, message",
    [("losses", "tracking_losses"), ("norms", "grad_norms"), ("z_bar", "per_step_final_losses")],
)
def test_nan_diagnostics_fail_the_capacity_run_instead_of_riding_along(monkeypatch, field, message):
    """A trace full of NaN is not a diagnostic, and a report built on one is not evidence."""
    _poison(monkeypatch, field)

    with pytest.raises(ValueError, match=f"{message} must be finite"):
        run_capacity_example_batch(_velocity_fn, _batch(_NAMES[:1]), _BASE_CONTEXT, _params())


def test_nan_diagnostics_fail_the_adequacy_probe(monkeypatch):
    _poison(monkeypatch, "norms")

    with pytest.raises(ValueError, match="grad_norms must be finite"):
        run_adequacy_probe(
            _velocity_fn, _batch(_NAMES[:1]), _BASE_CONTEXT, ((1, 0.01),), guide_scale=_GUIDE_SCALE, l_null=_L
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"inner_iters": 0}, "inner_iters must be an integer >= 1"),
        ({"inner_iters": 1.5}, "inner_iters must be an integer >= 1"),
        ({"inner_iters": True}, "inner_iters must be an integer >= 1"),  # bools are not iteration counts
        ({"lr": -1e-3}, "lr must be finite and non-negative"),
        ({"lr": float("nan")}, "lr must be finite and non-negative"),
    ],
)
def test_an_unusable_recipe_is_rejected_before_the_inversion(overrides, message):
    """R3 rejects these too -- but only after the batch's inversion has already been paid for."""
    with pytest.raises(ValueError, match=message):
        run_capacity_example_batch(_forbidden_velocity, _batch(), _BASE_CONTEXT, _params(**overrides))


@pytest.mark.parametrize(
    "grid, message",
    [
        (((0, 0.01),), "inner_iters must be an integer >= 1"),
        (((1.5, 0.01),), "inner_iters must be an integer >= 1"),
        (((1, -0.01),), "lr must be finite and non-negative"),
        (((1, float("nan")),), "lr must be finite and non-negative"),
        (((1, 0.01), (2, float("inf"))), "lr must be finite and non-negative"),  # every cell, not just the first
    ],
)
def test_an_unusable_grid_cell_is_rejected_before_the_inversion(grid, message):
    with pytest.raises(ValueError, match=message):
        run_adequacy_probe(_forbidden_velocity, _batch(), _BASE_CONTEXT, grid, guide_scale=_GUIDE_SCALE, l_null=_L)


def test_adequacy_scores_are_the_median_of_the_mean_final_tracking_loss():
    """Plan M3, computed independently: the *post*-inner-loop loss, which is not ``losses[:, -1]``.

    ``optimize_null_embeddings`` logs the loss *before* each update, so the value the plan asks for
    lives in the returned pivot trajectory instead: ``mean((z_bar[i+1] - traj[i+1])**2)``, the
    objective evaluated at the locked nulls.
    """
    batch, report = _batch(_PROBE_NAMES), _cached_probe()

    traj = _cached_inversion(_PROBE_NAMES)
    assert report.names == _PROBE_NAMES
    assert tuple((score.inner_iters, score.lr) for score in report.scores) == ((1, 0.01), (2, 0.03))
    for score in report.scores:
        _, z_bar, _, _ = _optimize(traj, batch, inner_iters=score.inner_iters, lr=score.lr)
        per_example = np.asarray(jnp.mean((z_bar[1:] - traj[1:]) ** 2, axis=(2, 3, 4, 5))).mean(axis=0)

        np.testing.assert_allclose(np.asarray(score.per_example), per_example, rtol=1e-5)
        assert score.score == pytest.approx(float(np.median(per_example)), rel=1e-6)
        # The fixture separates the two statistics, so "median" above is load-bearing.
        assert score.score != pytest.approx(float(np.mean(per_example)), rel=1e-3)


def test_every_adequacy_recipe_keeps_its_own_traces():
    """Each grid cell is a diagnostic in its own right: the curves are why the probe exists."""
    report = _cached_probe()

    for score in report.scores:
        assert score.tracking_losses.shape == (_STEPS, score.inner_iters, len(_PROBE_NAMES))
        assert score.grad_norms.shape == score.tracking_losses.shape
        assert score.final_losses.shape == (len(_PROBE_NAMES), _STEPS)
        np.testing.assert_allclose(np.asarray(score.per_example), score.final_losses.mean(axis=1), rtol=1e-6)
        assert np.all(np.isfinite(score.tracking_losses)) and np.all(score.grad_norms > 0.0)


def test_the_adequacy_probe_inverts_once_for_the_whole_grid(monkeypatch):
    from maxdiffusion import null_adapter_runner_core as runner

    inversions = []
    original = runner.invert_trajectory
    monkeypatch.setattr(runner, "invert_trajectory", lambda *a, **k: (inversions.append(1), original(*a, **k))[1])

    run_adequacy_probe(
        _velocity_fn, _batch(_NAMES[:1]), _BASE_CONTEXT, ((1, 1e-2), (2, 1e-2)), guide_scale=_GUIDE_SCALE, l_null=_L
    )

    assert len(inversions) == 1


def test_the_default_adequacy_grid_is_the_plan_s():
    assert DEFAULT_RECIPE == (10, 0.01)
    assert set(ADEQUACY_GRID) == {(j, lr) for j in (10, 25, 50) for lr in (0.01, 0.03)}
    assert (ADOPTION_FACTOR, PLATEAU_MIN_IMPROVEMENT) == (0.5, 0.10)
    assert PLATEAU_BOUNDARY_ATOL == 1e-9  # declared, not incidental -- see the plateau tests


def _report(*entries, names=("a", "b")):
    """A synthetic report: adoption reads only the scores, so the traces are shaped stubs."""
    trace = np.zeros((2, 1, len(names)), np.float32)
    return AdequacyReport(
        names=names,
        scores=tuple(
            RecipeScore(
                inner_iters=j,
                lr=lr,
                score=score,
                per_example=(score, score),
                tracking_losses=trace,
                grad_norms=trace,
                final_losses=np.zeros((len(names), 2), np.float32),
            )
            for j, lr, score in entries
        ),
    )


def test_no_recipe_beating_half_the_default_keeps_the_default():
    report = _report((10, 0.01, 1.0), (25, 0.01, 0.6), (50, 0.01, 0.51), (10, 0.03, 0.9))

    adoption = adopt_recipe(report)

    assert (adoption.inner_iters, adoption.lr, adoption.adopted) == (10, 0.01, False)
    assert adoption.numbers["threshold"] == pytest.approx(0.5)


def test_the_lowest_qualifying_score_is_adopted():
    report = _report((10, 0.01, 1.0), (25, 0.01, 0.5), (50, 0.01, 0.2), (10, 0.03, 0.45))

    adoption = adopt_recipe(report)

    assert (adoption.inner_iters, adoption.lr, adoption.adopted) == (50, 0.01, True)
    assert adoption.numbers["adopted_score"] == pytest.approx(0.2)


def test_the_threshold_is_inclusive_at_exactly_half_the_default():
    report = _report((10, 0.01, 1.0), (25, 0.01, 0.5))

    assert adopt_recipe(report).inner_iters == 25
    assert not adopt_recipe(_report((10, 0.01, 1.0), (25, 0.01, 0.5 + 1e-9))).adopted


@pytest.mark.parametrize(
    "entries, expected",
    [
        ([(10, 0.01, 1.0), (25, 0.01, 0.3), (50, 0.01, 0.3)], (25, 0.01)),  # tie -> the cheaper J
        ([(10, 0.01, 1.0), (25, 0.03, 0.3), (25, 0.01, 0.3)], (25, 0.01)),  # tie at equal J -> lower lr
        ([(10, 0.01, 1.0), (10, 0.03, 0.3), (25, 0.01, 0.3)], (10, 0.03)),  # J dominates lr
    ],
)
def test_ties_break_towards_the_cheaper_recipe(entries, expected):
    adoption = adopt_recipe(_report(*entries))

    assert (adoption.inner_iters, adoption.lr) == expected


@pytest.mark.parametrize(
    "coarse, fine, expected",
    [
        (1.0, 0.999, RECONSTRUCTION_LIMITED),  # 0.1% -- more J buys nothing
        (1.0, 0.95, RECONSTRUCTION_LIMITED),  # 5%
        (1.0, 0.90001, RECONSTRUCTION_LIMITED),  # 9.999%: just below, and the epsilon must not eat it
        (1.0, 0.9, RECIPE_LIMITED),  # exactly 10%: the plan gives "below 10%" to reconstruction only,
        (0.5, 0.45, RECIPE_LIMITED),  # ... and both of these evaluate to 0.09999999999999998 in binary
        (1.0, 0.89999, RECIPE_LIMITED),  # 10.001%: just above
        (1.0, 0.85, RECIPE_LIMITED),  # 15%
        (1.0, 0.5, RECIPE_LIMITED),  # 50% -- the recipe, not the reconstruction, was the limit
    ],
)
def test_plateau_is_classified_from_the_j25_to_j50_improvement(coarse, fine, expected):
    report = _report((10, 0.01, 1.0), (25, 0.01, coarse), (50, 0.01, fine))

    adoption = adopt_recipe(report)

    assert adoption.plateau == expected
    assert adoption.numbers["plateau_improvement"] == pytest.approx((coarse - fine) / coarse)


def test_plateau_is_read_from_the_adopted_recipe_s_learning_rate_column():
    """The 3e-2 column plateaus (0.2 -> 0.199) while the 1e-2 column does not (0.9 -> 0.1).

    Both reports below carry both columns and differ in one entry, so the classification can only be
    coming from which column the adopted recipe points at.
    """
    columns = ((10, 0.01, 1.0), (25, 0.01, 0.9), (50, 0.01, 0.1), (25, 0.03, 0.2), (50, 0.03, 0.199))

    coarse_lr_loses = adopt_recipe(_report(*columns, (10, 0.03, 0.2)))
    coarse_lr_wins = adopt_recipe(_report(*columns, (10, 0.03, 0.05)))

    assert (coarse_lr_loses.inner_iters, coarse_lr_loses.lr, coarse_lr_loses.plateau) == (50, 0.01, RECIPE_LIMITED)
    assert (coarse_lr_wins.inner_iters, coarse_lr_wins.lr, coarse_lr_wins.plateau) == (
        10,
        0.03,
        RECONSTRUCTION_LIMITED,
    )


def test_plateau_is_undetermined_without_both_scan_points():
    report = _report((10, 0.01, 1.0), (25, 0.01, 0.2))

    adoption = adopt_recipe(report)

    assert adoption.plateau == UNDETERMINED and adoption.numbers["plateau_improvement"] is None


@pytest.mark.parametrize(
    "entries, message",
    [
        ([(25, 0.01, 0.5)], "default recipe"),
        ([(10, 0.01, 1.0), (10, 0.01, 0.2)], "duplicate"),
        ([(10, 0.01, float("nan"))], "must be finite"),
    ],
)
def test_adopt_recipe_refuses_an_unusable_report(entries, message):
    with pytest.raises(ValueError, match=message):
        adopt_recipe(_report(*entries))


def test_a_nonfinite_recipe_score_is_never_adopted():
    report = _report((10, 0.01, 1.0), (25, 0.01, float("nan")), (50, 0.01, float("-inf")))

    adoption = adopt_recipe(report)

    assert (adoption.inner_iters, adoption.adopted) == (10, False)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda b: dataclasses.replace(b, z_video=b.z_video[:, :, :4]), "z_video must have the production shape"),
        (lambda b: dataclasses.replace(b, z_i0=b.z_i0[:, :, :, :6]), "z_i0 must have the production shape"),
        (lambda b: dataclasses.replace(b, z_video=b.z_video[:1]), "z_video must have the production shape"),
        (lambda b: dataclasses.replace(b, names=(b.names[0], b.names[0])), "unique"),
        (lambda b: dataclasses.replace(b, names=()), "at least one example"),
        (lambda b: dataclasses.replace(b, z_video=b.z_video.at[0, 0, 0, 0, 0].set(np.nan)), "z_video must be finite"),
    ],
)
def test_the_production_geometry_is_enforced_before_any_compute(mutate, message):
    batch = mutate(_batch())

    with pytest.raises(ValueError, match=message):
        run_capacity_example_batch(_forbidden_velocity, batch, _BASE_CONTEXT, _params())


@pytest.mark.parametrize(
    "context, message",
    [
        (jnp.zeros((256, _D), jnp.float32), "base_context must have the production shape"),
        (jnp.zeros((_S, 2048), jnp.float32), "base_context must have the production shape"),
        (jnp.zeros((1, _S, _D), jnp.float32), "base_context must have the production shape"),
    ],
)
def test_the_base_context_geometry_is_enforced(context, message):
    with pytest.raises(ValueError, match=message):
        run_capacity_example_batch(_forbidden_velocity, _batch(), context, _params())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"l_null": 0}, "l_null must be an integer"),
        ({"l_null": _S + 1}, "l_null must be an integer"),
        ({"l_null": 16.0}, "l_null must be an integer"),
        ({"guide_scale": float("nan")}, "guide_scale must be finite"),
    ],
)
def test_the_parameters_are_validated_before_any_compute(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_capacity_example_batch(_forbidden_velocity, _batch(), _BASE_CONTEXT, _params(**overrides))


@pytest.mark.parametrize(
    "grid, message",
    [((), "adequacy grid"), (((10, 0.01), (10, 0.01)), "adequacy grid")],
)
def test_the_adequacy_grid_is_validated_before_any_compute(grid, message):
    with pytest.raises(ValueError, match=message):
        run_adequacy_probe(_forbidden_velocity, _batch(), _BASE_CONTEXT, grid, guide_scale=_GUIDE_SCALE, l_null=_L)


def test_the_adequacy_probe_enforces_the_same_geometry():
    batch = dataclasses.replace(_batch(), z_video=jnp.zeros((2, 48, 9, 12, 21), jnp.float32))

    with pytest.raises(ValueError, match="z_video must have the production shape"):
        run_adequacy_probe(
            _forbidden_velocity, batch, _BASE_CONTEXT, ((1, 0.01),), guide_scale=_GUIDE_SCALE, l_null=_L
        )

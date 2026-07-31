"""CPU-only tests for the exp_02 OVERFIT100 eval context modes (plan §1 / D11).

The three rollout context modes and the value-level derangement that makes the "shuffled"
ablation meaningful:

  * ``correct``  -- the episode's OWN context-table row (``context_table[episode_index]``).
  * ``null``     -- the exp_01 empty-prompt embedding, BIT-IDENTICAL to what
    ``_build_full_ft_validation_state`` computes (characterization test: both builders run
    the same real ``_compute_null_context`` against the same stub pipeline and the arrays
    must compare equal bit-for-bit).
  * ``shuffled`` -- a seeded derangement of instruction VALUES. Plan §1: an index-level
    derangement is TOO WEAK here because 6 duplicate-instruction groups cover 22 of the 100
    episodes, so a permutation with no fixed point can still hand an episode a string equal
    to its own. The property pinned here is at the VALUE level: for every episode, the text
    it receives (post-strip) differs from its own text -- proven on a WORST-CASE fixture
    (7 duplicates out of 15 episodes, i.e. exactly the ``2 * n_max <= N`` boundary) and on
    the real committed 100-episode cohort, for several seeds.

Also pinned: mode/seed spec parsing, determinism from ``context_shuffle_seed``, that the
rollout body really feeds the selected context to the transformer, that the overfit100
rollout is exp_01's sampler (bitwise-equal output when every table row equals the null
embedding), and -- statistic-side -- that ablation rows can never enter ``m_corr``.

Stub nnx transformer + fake pipeline; no weights, no GCS, one CPU device.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.overfit100_success_statistic as stat
import maxdiffusion.trainers.wan_ti2v_full_ft_trainer as full_ft
import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

_REPO = Path(gen.__file__).parents[2]
_MANIFEST = _REPO / "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json"

MESH_AXES = ("data", "fsdp", "context", "tensor")


# --------------------------------------------------------------------------------------
# (1) Spec parsing.
# --------------------------------------------------------------------------------------


def test_parse_context_modes():
    assert gen.parse_context_modes("correct") == ("correct",)
    assert gen.parse_context_modes(" correct , null , shuffled ") == ("correct", "null", "shuffled")
    assert gen.parse_context_modes("shuffled,correct") == ("shuffled", "correct")  # order preserved
    for bad, needle in (("", "context_modes"), ("   ", "context_modes"), ("correct,bogus", "bogus")):
        with pytest.raises(ValueError) as ei:
            gen.parse_context_modes(bad)
        assert needle in str(ei.value)
    with pytest.raises(ValueError) as ei:
        gen.parse_context_modes("correct,correct")
    assert "duplicate" in str(ei.value).lower()


def test_parse_rollout_seeds():
    assert gen.parse_rollout_seeds("0") == (0,)
    assert gen.parse_rollout_seeds(" 0 , 1 , 2 ") == (0, 1, 2)
    assert gen.parse_rollout_seeds("2,0") == (2, 0)  # order preserved (it is the seed list)
    for bad in ("", "  ", "0,0", "-1", "x"):
        with pytest.raises(ValueError):
            gen.parse_rollout_seeds(bad)


# --------------------------------------------------------------------------------------
# (2) The value-level derangement.
# --------------------------------------------------------------------------------------


def _assert_value_derangement(texts, sigma):
    n = len(texts)
    assert sorted(sigma) == list(range(n)), "sigma must be a permutation of the episode indices"
    for i in range(n):
        assert texts[sigma[i]].strip() != texts[i].strip(), f"episode {i} received its own instruction"


def test_derangement_on_distinct_texts():
    texts = [f"text {i}" for i in range(5)]
    sigma = gen.value_derangement(texts, seed=0)
    _assert_value_derangement(texts, sigma)


def test_derangement_worst_case_duplicate_group_7_of_15():
    # The BOUNDARY case: the largest duplicate group is exactly half the cohort (2*7 <= 15).
    texts = ["fold cloth"] * 7 + [f"unique {i}" for i in range(8)]
    for seed in range(6):
        sigma = gen.value_derangement(texts, seed=seed)
        _assert_value_derangement(texts, sigma)
        # Every member of the big group must be sent OUTSIDE the group.
        for i in range(7):
            assert sigma[i] >= 7


def test_derangement_handles_several_duplicate_groups():
    texts = ["a"] * 4 + ["b"] * 3 + ["c"] * 2 + [f"u{i}" for i in range(5)]
    for seed in (0, 1, 7, 99):
        _assert_value_derangement(texts, gen.value_derangement(texts, seed=seed))


def test_derangement_is_deterministic_in_the_seed():
    texts = ["a"] * 4 + [f"u{i}" for i in range(6)]
    assert gen.value_derangement(texts, seed=3) == gen.value_derangement(texts, seed=3)
    # Different seeds give different assignments (with 10 texts a collision is not expected),
    # and every one of them is still a valid value derangement.
    variants = {gen.value_derangement(texts, seed=s) for s in range(8)}
    assert len(variants) > 1
    for sigma in variants:
        _assert_value_derangement(texts, sigma)


def test_derangement_ignores_only_surrounding_whitespace_when_comparing():
    # "a" and " a " are the SAME instruction: a value derangement must not pair them.
    texts = ["a", " a ", "b", "c"]
    for seed in range(5):
        sigma = gen.value_derangement(texts, seed=seed)
        _assert_value_derangement(texts, sigma)
        assert texts[sigma[0]].strip() != "a" and texts[sigma[1]].strip() != "a"


def test_derangement_impossible_cases_are_refused():
    with pytest.raises(ValueError) as ei:
        gen.value_derangement(["only one"], seed=0)
    assert "1" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        gen.value_derangement(["same", "same"], seed=0)
    assert "same" in str(ei.value) or "derangement" in str(ei.value).lower()
    # 8 of 15 duplicates: 2*8 > 15, so no value derangement exists -> loud failure.
    with pytest.raises(ValueError) as ei:
        gen.value_derangement(["dup"] * 8 + [f"u{i}" for i in range(7)], seed=0)
    assert "8" in str(ei.value) and "15" in str(ei.value)


def test_derangement_two_distinct_texts_is_a_swap():
    assert gen.value_derangement(["a", "b"], seed=0) == (1, 0)


def test_derangement_on_the_real_committed_cohort():
    episodes = json.loads(_MANIFEST.read_text())["episodes"]
    texts = [e["used_text"] for e in sorted(episodes, key=lambda e: e["episode_index"])]
    assert len(texts) == 100
    assert len(set(texts)) == 84  # 6 duplicate groups over 22 episodes (plan §1)
    from collections import Counter

    assert max(Counter(texts).values()) == 7
    for seed in range(4):
        _assert_value_derangement(texts, gen.value_derangement(texts, seed=seed))


def test_index_level_derangement_would_be_too_weak_here():
    # The plan's reason for a VALUE-level rule, made executable: a fixed-point-free
    # permutation over a duplicate group still hands an episode its own instruction.
    texts = ["dup", "dup", "x", "y"]
    index_only = (1, 0, 3, 2)  # no fixed point, yet episodes 0/1 keep "dup"
    assert all(index_only[i] != i for i in range(4))
    assert texts[index_only[0]] == texts[0]
    # The real helper never produces that.
    _assert_value_derangement(texts, gen.value_derangement(texts, seed=0))


# --------------------------------------------------------------------------------------
# (3) Mode -> context source.
# --------------------------------------------------------------------------------------


def test_context_source_index_per_mode():
    sigma = (2, 0, 1)
    assert gen.context_source_index("correct", 1, sigma) == 1
    assert gen.context_source_index("shuffled", 1, sigma) == 0
    assert gen.context_source_index("null", 1, sigma) is None
    with pytest.raises(ValueError):
        gen.context_source_index("bogus", 0, sigma)
    with pytest.raises(ValueError):
        gen.context_source_index("shuffled", 5, sigma)  # index outside the derangement


def test_context_for_mode_selects_the_right_row():
    table = jnp.stack([jnp.full((2, 4), float(i)) for i in range(3)])
    null = jnp.full((1, 2, 4), -1.0)
    sigma = (2, 0, 1)
    correct = gen.overfit100_context_for_mode(table, null, "correct", 1, sigma)
    shuffled = gen.overfit100_context_for_mode(table, null, "shuffled", 1, sigma)
    nulled = gen.overfit100_context_for_mode(table, null, "null", 1, sigma)
    assert correct.shape == shuffled.shape == nulled.shape == (1, 2, 4)
    assert float(correct[0, 0, 0]) == 1.0
    assert float(shuffled[0, 0, 0]) == 0.0  # sigma[1] = 0
    assert float(nulled[0, 0, 0]) == -1.0
    assert not np.array_equal(np.asarray(correct), np.asarray(shuffled))


# --------------------------------------------------------------------------------------
# (4) The rollout really feeds the selected context; and it IS exp_01's sampler.
# --------------------------------------------------------------------------------------

_CALLS: list[dict] = []


@pytest.fixture(autouse=True)
def _clear_calls():
    _CALLS.clear()
    yield
    _CALLS.clear()


class _RecordingStub(nnx.Module):
    def __init__(self, gain=0.3):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        _CALLS.append(kwargs)
        return self.gain[...] * kwargs["hidden_states"].astype(jnp.float32)


def _rollout_config():
    return SimpleNamespace(
        model_type="OVERFIT100_TI2V",
        weights_dtype="float32",
        activations_dtype="float32",
        side_adapter_sampling_steps=2,
        flow_shift=5.0,
        side_adapter_guide_scale=1.0,
    )


def _rollout_data(context):
    b, c, f, h, w = 1, 2, 3, 4, 4
    k1, k2 = jax.random.split(jax.random.key(7))
    return {
        "z_i0": jax.random.normal(k1, (b, c, 1, h, w), dtype=jnp.float32),
        "z_video": jax.random.normal(k2, (b, c, f, h, w), dtype=jnp.float32),
        "context": context,
    }


def _scheduler():
    return FlaxFlowMatchScheduler(dtype=jnp.float32, shift=5.0, sigma_min=0.0, sigma_max=1.0)


def _overfit100_state(stub, table):
    graphdef, params, rest = nnx.split(stub, nnx.Param, ...)
    return overfit100.Overfit100TrainState(
        step=0,
        apply_fn=None,
        params=params,
        tx=None,
        opt_state=None,
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=table,
    )


def test_overfit100_rollout_feeds_the_supplied_context_and_no_actions():
    table = jnp.stack([jnp.full((2, 8), float(i)) for i in range(3)])
    state = _overfit100_state(_RecordingStub(), table)
    context = table[1][None]
    z, metrics = gen._rollout_overfit100_sample(
        state, _rollout_data(context), jax.random.key(0), _scheduler(), _rollout_config()
    )
    assert len(_CALLS) == 1  # fori_loop traces the body once
    call = _CALLS[0]
    assert set(call) == {"hidden_states", "timestep", "encoder_hidden_states", "deterministic"}
    assert "actions" not in call
    assert call["deterministic"] is True
    np.testing.assert_array_equal(np.asarray(call["encoder_hidden_states"]), np.asarray(context))
    assert np.all(np.isfinite(np.asarray(z)))
    for key in ("latent_mse", "latent_mae", "z_pred_std", "z_target_std", "z_init_anchor_mse"):
        assert key in metrics


def test_overfit100_rollout_is_exp01s_sampler_bitwise_when_the_row_is_the_null_embedding():
    # PARITY: with the gathered context equal to exp_01's null embedding, the overfit100
    # rollout must reproduce the full-FT rollout EXACTLY (same sigmas, same per-token
    # timesteps, same frame-0 pin, same Euler update).
    null = jnp.full((1, 2, 8), 0.25, dtype=jnp.float32)
    table = jnp.concatenate([null, null, null], axis=0)
    stub_a, stub_b = _RecordingStub(0.31), _RecordingStub(0.31)
    config, scheduler, rng = _rollout_config(), _scheduler(), jax.random.key(3)
    data = _rollout_data(table[1][None])

    z_new, m_new = gen._rollout_overfit100_sample(_overfit100_state(stub_a, table), data, rng, scheduler, config)

    graphdef, params, rest = nnx.split(stub_b, nnx.Param, ...)
    old_state = full_ft.FullFTTrainState(
        step=0,
        apply_fn=None,
        params=params,
        tx=None,
        opt_state=None,
        graphdef=graphdef,
        rest_of_state=rest,
        null_context=null,
    )
    old_data = {
        "z_i0": data["z_i0"],
        "z_video": data["z_video"],
        "actions": jnp.zeros((1, 32, 7), dtype=jnp.float32),
    }
    old_config = SimpleNamespace(**{**vars(config), "model_type": "FULL_FT_TI2V"})
    z_old, m_old = gen._rollout_sample(old_state, old_data, rng, scheduler, old_config)

    np.testing.assert_array_equal(np.asarray(z_new), np.asarray(z_old))
    for key in m_old:
        assert float(m_new[key]) == float(m_old[key])


# --------------------------------------------------------------------------------------
# (5) The validation-state builder: real table, real null context, exp_01-identical null.
# --------------------------------------------------------------------------------------


def _cpu_mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), MESH_AXES)


class _FakePipeline:
    """Stand-in for WanPipelineTI2V_2_2 with deterministic text embeddings."""

    def __init__(self, mesh, *, text_dim=8, max_len=2):
        self.mesh = mesh
        self.transformer = _RecordingStub(0.5)
        self.text_encoder = object()
        self.tokenizer = object()
        self.vae = object()
        self.vae_cache = object()
        self._text_dim = text_dim
        self._max_len = max_len
        self.encode_prompt_calls = []
        self.t5_calls = []

    def _embed(self, prompt):
        # A deterministic, prompt-dependent embedding: value = len(prompt) + 0.5.
        return np.stack(
            [np.full((self._max_len, self._text_dim), len(p) + 0.5, dtype=np.float32) for p in prompt],
        )

    def encode_prompt(self, *, prompt, negative_prompt, num_videos_per_prompt, max_sequence_length):
        self.encode_prompt_calls.append((tuple(prompt), tuple(negative_prompt), max_sequence_length))
        return jnp.asarray(self._embed(prompt)), None

    def _get_t5_prompt_embeds(self, *, prompt, num_videos_per_prompt, max_sequence_length):
        self.t5_calls.append((tuple(prompt), max_sequence_length))
        return self._embed(prompt)


def _episodes_dir(tmp_path, texts):
    directory = tmp_path / "train3"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": [{"episode_index": i, "episode_id": 100 + i, "used_text": text} for i, text in enumerate(texts)]
    }
    (directory / "episodes.json").write_text(json.dumps(payload, indent=2) + "\n")
    return str(directory)


def _builder_config(train_dir, **overrides):
    base = {
        "model_type": "OVERFIT100_TI2V",
        "logical_axis_rules": (),
        "weights_dtype": "float32",
        "activations_dtype": "float32",
        "wan_max_sequence_length": 2,
        "text_dim": 8,
        "num_text_slots": 3,
        "text_encode_batch": 2,
        "train_data_dir": train_dir,
        "eval_data_dir": train_dir,
        # The exp_02 config gates the builder enforces before any weights load.
        "expected_windows": 42,
        "model_manifest_path": str(_MANIFEST),
        "checkpoint_steps": (),
        "max_train_steps": 4,
        "warmup_steps": 0,
        "learning_rate": 1.0e-5,
        "learning_rate_schedule_steps": -1,
        "warmup_steps_fraction": 0.5,
        "adam_b1": 0.9,
        "adam_b2": 0.999,
        "adam_eps": 1.0e-8,
        "adam_weight_decay": 1.0e-2,
        "opt_enable_grad_global_norm_clipping": True,
        "max_grad_norm": 1.0,
        "opt_enable_grad_clipping": False,
        "max_grad_value": 1.0,
        "side_adapter_guide_scale": 1.0,
        "side_adapter_noise_mode": "fresh",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _install_builder_stubs(monkeypatch, pipeline, *, preflight=True):
    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_load_wan_pipeline", lambda self: pipeline)
    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_shard_state", lambda self, mesh, state: (state, "SH"))
    if preflight:
        monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_preflight_dataset", lambda self: {"records": 3})
        monkeypatch.setattr(
            overfit100.WanTI2VOverfit100Trainer, "_validate_pinned_snapshot", staticmethod(lambda c: "x" * 40)
        )


def test_build_overfit100_validation_state_builds_table_and_null_context(tmp_path, monkeypatch):
    texts = ["aa", "bbb", "cccc"]
    train_dir = _episodes_dir(tmp_path, texts)
    pipeline = _FakePipeline(_cpu_mesh())
    _install_builder_stubs(monkeypatch, pipeline)

    trainer, out_pipeline, mesh, state, shardings, null_context = gen._build_overfit100_validation_state(
        _builder_config(train_dir)
    )

    assert isinstance(trainer, overfit100.WanTI2VOverfit100Trainer)
    assert isinstance(state, overfit100.Overfit100TrainState)
    assert shardings == "SH"
    # The table is the REAL builder's output: one row per episode, in episode_index order,
    # each row the encoding of THAT episode's instruction.
    assert tuple(state.context_table.shape) == (3, 2, 8)
    for i, text in enumerate(texts):
        assert float(state.context_table[i, 0, 0]) == pytest.approx(len(text) + 0.5)
    # Bounded encode loop (text_encode_batch=2 -> chunks of 2 then 1).
    assert [len(call[0]) for call in pipeline.t5_calls] == [2, 1]
    # The null context comes from the EMPTY prompt through the inherited path.
    assert tuple(null_context.shape) == (1, 2, 8)
    assert float(null_context[0, 0, 0]) == pytest.approx(0.5)
    assert pipeline.encode_prompt_calls == [(("",), ("",), 2)]
    # Text modules freed; the VAE is KEPT (the eval decodes latents to video).
    assert not hasattr(out_pipeline, "text_encoder") and not hasattr(out_pipeline, "tokenizer")
    assert hasattr(out_pipeline, "vae") and hasattr(out_pipeline, "vae_cache")
    assert not hasattr(out_pipeline, "transformer")  # split into the state
    assert full_ft._adam_moment_trees(state.opt_state)[0] is not None  # a REAL optimizer state
    assert mesh is pipeline.mesh


def test_null_context_is_bit_identical_to_the_exp01_full_ft_path(tmp_path, monkeypatch):
    # CHARACTERIZATION: the exp_01 builder and the overfit100 builder must produce the
    # SAME null embedding from the same pipeline -- the "null" ablation is exp_01's null.
    train_dir = _episodes_dir(tmp_path, ["aa", "bbb", "cccc"])
    config = _builder_config(train_dir)

    pipe_new = _FakePipeline(_cpu_mesh())
    _install_builder_stubs(monkeypatch, pipe_new)
    *_, null_new = gen._build_overfit100_validation_state(config)

    pipe_old = _FakePipeline(_cpu_mesh())
    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_load_wan_pipeline", lambda self: pipe_old)
    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_shard_state", lambda self, mesh, state: (state, "SH"))
    _, _, _, old_state, _ = gen._build_full_ft_validation_state(config)

    np.testing.assert_array_equal(np.asarray(null_new), np.asarray(old_state.null_context))
    assert null_new.dtype == old_state.null_context.dtype
    assert pipe_new.encode_prompt_calls == pipe_old.encode_prompt_calls


def test_build_overfit100_validation_state_runs_the_preflight_before_loading_the_pipeline(tmp_path, monkeypatch):
    train_dir = _episodes_dir(tmp_path, ["aa", "bbb", "cccc"])
    order = []

    def _preflight(self):
        order.append("preflight")
        return {}

    def _snapshot(config):
        order.append("snapshot")
        return "x" * 40

    def _load(self):
        order.append("load")
        raise AssertionError("the pipeline loader must not run when the preflight fails")

    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_preflight_dataset", _preflight)
    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_validate_pinned_snapshot", staticmethod(_snapshot))
    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_load_wan_pipeline", _load)
    with pytest.raises(AssertionError):
        gen._build_overfit100_validation_state(_builder_config(train_dir))
    assert order == ["snapshot", "preflight", "load"]


def test_restore_validation_state_dispatches_to_the_overfit100_builder(monkeypatch):
    seen = {}

    def _overfit_builder(config):
        seen["called"] = True
        return ("TR", "PIPE", "MESH", "STATE", "SH", "NULL")

    def _boobytrap(config):
        raise AssertionError("the full-FT builder must not run for OVERFIT100_TI2V")

    monkeypatch.setattr(gen, "_build_overfit100_validation_state", _overfit_builder)
    monkeypatch.setattr(gen, "_build_full_ft_validation_state", _boobytrap)
    monkeypatch.setattr(gen, "_restore_checkpoint_state", lambda config, state, ckpt, **kw: (state, 2500))
    config = SimpleNamespace(
        model_type="OVERFIT100_TI2V",
        checkpoint_dir="gs://x/ck",
        output_dir="gs://x",
        run_name="r",
        checkpoint_step=2500,
    )
    out = gen._restore_overfit100_validation_state(config)
    assert seen["called"] is True
    assert out[-1] == 2500  # the restored step rides at the end, exp_01 convention
    assert out[5] == "NULL"


# --------------------------------------------------------------------------------------
# (6) The run_overfit100 DRIVER, end to end on CPU (no mesh sharding, no ffmpeg, no GCS).
# --------------------------------------------------------------------------------------


class _ContextSensitiveStub(nnx.Module):
    """Velocity that DEPENDS on the context, so a mis-routed context changes the metrics."""

    def __init__(self, gain=0.2):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        _CALLS.append(kwargs)
        context_term = jnp.mean(kwargs["encoder_hidden_states"].astype(jnp.float32))
        return self.gain[...] * kwargs["hidden_states"].astype(jnp.float32) + 0.1 * context_term


class _DecodingPipeline(_FakePipeline):
    """Adds the VAE decode seam: latents -> [1, T, H, W, 3] frames, value = mean(latent)."""

    def _denormalize_latents(self, latents):
        return latents

    def _decode_latents_to_video(self, latents):
        value = float(jnp.mean(latents.astype(jnp.float32)))
        return np.full((1, 5, 8, 8, 3), np.tanh(value) * 0.5 + 0.5, dtype=np.float32)


def _driver_manifest(tmp_path, episodes, *, with_video=False):
    """``with_video`` adds the per-episode ``video_fingerprint`` the auxiliary RGB path pulls."""
    payload = {
        "vae_fingerprint": {"hf_repo": "r", "revision": "a" * 40},
        "episodes": [
            {
                "episode_index": index,
                "episode_id": 100 + index,
                "used_text": text,
                "n_windows": n_windows,
                **(
                    {
                        "video_fingerprint": {
                            "uri": f"gs://bucket/videos/{100 + index}/0.mp4",
                            "generation": 17 + index,
                            "md5": "Szm9uNUI2AtjyRNTtpm9SA==",
                            "size": 4,
                        }
                    }
                    if with_video
                    else {}
                ),
            }
            for index, (text, n_windows) in enumerate(episodes)
        ],
    }
    path = tmp_path / "driver_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path)


def _driver_records(episodes, *, channels, frames, height, width):
    records = []
    for index, (_text, n_windows) in enumerate(episodes):
        for slot in range(n_windows):
            start = 4 * slot
            fill = float(index + 1) / 10.0
            records.append(
                {
                    "name": gen.overfit100_window_name(100 + index, start).encode(),
                    "episode_id": 100 + index,
                    "episode_index": index,
                    "window_start": start,
                    "z_i0": np.full((channels, 1, height, width), fill, dtype=np.float16).tobytes(),
                    "z_video": np.full((channels, frames, height, width), fill, dtype=np.float16).tobytes(),
                    "instruction": _text.encode(),
                }
            )
    return records


def test_run_overfit100_writes_the_aggregation_artifact_for_every_window_mode_seed(tmp_path, monkeypatch):
    # Drives the PRODUCTION driver with the jit seam replaced by the unjitted rollout: selection,
    # per-mode context, the per-window rng, the decode, the rows and all three artifacts.
    episodes = [("fold cloth", 3), ("press button", 1)]  # canonical starts: 4 and 0
    manifest = _driver_manifest(tmp_path, episodes)
    data_dir = tmp_path / "train2"
    data_dir.mkdir()
    (data_dir / "episodes.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_index": i, "episode_id": 100 + i, "used_text": text}
                    for i, (text, _n) in enumerate(episodes)
                ]
            },
            indent=2,
        )
        + "\n"
    )
    channels, frames, height, width = 2, 3, 4, 4
    records = _driver_records(episodes, channels=channels, frames=frames, height=height, width=width)
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(records))
    monkeypatch.setattr(gen, "assert_ssim_available", lambda: None)  # no scikit-image needed on CPU

    config = SimpleNamespace(
        model_type="OVERFIT100_TI2V",
        run_name="drv",
        eval_data_dir=str(data_dir),
        train_data_dir=str(data_dir),
        model_manifest_path=manifest,
        checkpoint_dir=str(tmp_path / "ck"),
        output_dir=str(tmp_path / "out"),
        validation_output_dir=str(tmp_path / "out" / "validation"),
        eval_pass_role="s3_segment_final",
        eval_windows="canonical",
        rollout_seeds="0,1,2",
        context_modes="correct,null,shuffled",
        context_shuffle_seed=0,
        write_videos=False,
        eval_aux_rgb=True,  # exercised, but the synthetic manifest has no MP4 -> aux stays null
        flagged_windows="",
        num_text_slots=2,
        latent_channels=channels,
        latent_frames=frames,
        latent_height=height,
        latent_width=width,
        weights_dtype="float32",
        activations_dtype="float32",
        side_adapter_sampling_steps=25,  # the role requires exactly D9's 25-step sampler
        side_adapter_guide_scale=1.0,
        flow_shift=5.0,
        logical_axis_rules=(),
        fps=8,
    )

    mesh = _cpu_mesh()
    pipeline = _DecodingPipeline(mesh)
    table = jnp.stack([jnp.full((2, 8), float(i) + 1.0, dtype=jnp.float32) for i in range(2)])
    state = _overfit100_state(_ContextSensitiveStub(0.2), table)
    null_context = jnp.full((1, 2, 8), -3.0, dtype=jnp.float32)
    trainer = SimpleNamespace(_create_scheduler=lambda: (_scheduler(), None))
    monkeypatch.setattr(
        gen,
        "_restore_overfit100_validation_state",
        lambda cfg: (trainer, pipeline, mesh, state, "SH", null_context, 2500),
    )
    # The jit seam: run the rollout unjitted so no state-sharding tree is needed on CPU.
    monkeypatch.setattr(
        gen,
        "_overfit100_rollout_fn",
        lambda state_shardings, data_shardings, replicated, scheduler, config: (
            lambda s, batch, rng: gen._rollout_overfit100_sample(s, batch, rng, scheduler, config)
        ),
    )
    monkeypatch.setattr(gen, "read_episode_texts", lambda directory, slots: [text for text, _n in episodes])

    gen.run_overfit100(config)

    step_root = tmp_path / "out" / "validation" / "step_002500_s3_segment_final"
    artifact = json.loads((step_root / "aggregation.json").read_text())
    assert artifact["schema"] == gen.OVERFIT100_AGGREGATION_SCHEMA
    assert artifact["checkpoint_step"] == 2500
    assert artifact["num_windows"] == 2
    assert artifact["eval_pass_role"] == "s3_segment_final"
    assert artifact["role_validation"]["ok"] is True
    # D1: the FIXED manifest-derived cohort, plus the explicit coverage sets.
    assert artifact["canonical_cohort"] == [[100, 4], [101, 0]]
    assert artifact["covered_canonical_windows"] == [[100, 4], [101, 0]]
    assert artifact["missing_canonical_windows"] == []
    assert artifact["manifest_sha256"] == hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    assert artifact["rollout_seeds"] == [0, 1, 2] and artifact["context_modes"] == ["correct", "null", "shuffled"]
    assert artifact["context_derangement"] == [1, 0]
    rows = artifact["rows"]
    assert len(rows) == 2 * 3 * 3  # windows x modes x seeds
    assert {row["name"] for row in rows} == {"ep100_v0_s00004", "ep101_v0_s00000"}
    # The context each row rolled out with is recorded and mode-correct.
    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["context_mode"], []).append(row)
    assert sorted(r["context_source_episode_index"] for r in by_mode["correct"]) == [0, 0, 0, 1, 1, 1]
    assert all(r["context_source_episode_index"] is None for r in by_mode["null"])
    assert sorted(r["context_source_episode_index"] for r in by_mode["shuffled"]) == [0, 0, 0, 1, 1, 1]
    for row in by_mode["shuffled"]:
        assert row["context_source_episode_index"] != row["episode_index"]
    # Every row carries the full column set and finite primary metrics. (The JSON artifact is
    # written with sort_keys=True, so field ORDER is pinned by summary.csv's header below.)
    for row in rows:
        assert set(row.keys()) == set(gen.OVERFIT100_ROW_FIELDS)
        assert np.isfinite(row["latent_mse"]) and np.isfinite(row["pixel_mse"])
        # The auxiliary block degraded gracefully (no MP4 in the synthetic manifest).
        assert row["ssim_vs_rgb"] is None and row["vae_ceiling_ssim"] is None
        assert isinstance(row["aux_status"], str) and row["aux_status"] != "ok"
    # Sibling artifacts, and NO videos (write_videos=False keeps the checkpoint light).
    csv_lines = (step_root / "summary.csv").read_text().splitlines()
    assert csv_lines[0].split(",") == list(gen.OVERFIT100_ROW_FIELDS)  # the column ORDER contract
    assert len(csv_lines) == 1 + len(rows)
    summary = json.loads((step_root / "summary.json").read_text())
    assert summary["num_windows"] == 2 and summary["num_rows"] == 18
    assert summary["eval_pass_role"] == "s3_segment_final"
    assert summary["missing_canonical_windows"] == []
    # D5: aux coverage is summarized even when every attempt failed (no MP4 in the fixture).
    coverage = summary["per_mode"]["aux_coverage"]
    assert coverage["requested"] == 18 and coverage["ok"] == 0 and coverage["coverage_fraction"] == 0.0
    assert coverage["failure_reason_counts"]
    assert {"correct", "null", "shuffled"} <= set(summary["per_mode"])
    assert not list(step_root.glob("mode_*"))
    # The rollout used the per-window key: the same window+seed appears once per mode, and the
    # correct-mode rows differ from the null-mode rows (different conditioning, same noise).
    correct = {(r["name"], r["seed"]): r["latent_mse"] for r in by_mode["correct"]}
    nulled = {(r["name"], r["seed"]): r["latent_mse"] for r in by_mode["null"]}
    assert set(correct) == set(nulled)
    assert any(correct[k] != nulled[k] for k in correct)


def test_run_overfit100_writes_videos_only_when_asked(tmp_path, monkeypatch):
    # The gate itself: with write_videos=True the driver calls the shared _save_video helper for
    # the three exp_01-convention filenames per (mode, seed, window), under mode_/seed_ dirs.
    saved = []
    monkeypatch.setattr(gen, "_save_video", lambda frames, path, fps: saved.append(path))
    episodes = [("fold cloth", 1), ("press button", 1)]
    manifest = _driver_manifest(tmp_path, episodes)
    data_dir = tmp_path / "train2"
    data_dir.mkdir()
    (data_dir / "episodes.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_index": i, "episode_id": 100 + i, "used_text": text}
                    for i, (text, _n) in enumerate(episodes)
                ]
            }
        )
    )
    records = _driver_records(episodes, channels=2, frames=3, height=4, width=4)
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(records))
    monkeypatch.setattr(gen, "assert_ssim_available", lambda: None)  # no scikit-image needed on CPU
    config = SimpleNamespace(
        model_type="OVERFIT100_TI2V",
        run_name="drv",
        eval_data_dir=str(data_dir),
        train_data_dir=str(data_dir),
        model_manifest_path=manifest,
        checkpoint_dir=str(tmp_path / "ck"),
        output_dir=str(tmp_path / "out"),
        validation_output_dir=str(tmp_path / "out" / "validation"),
        eval_pass_role="s3_intermediate",
        eval_windows="canonical",
        rollout_seeds="0",
        context_modes="correct",
        context_shuffle_seed=0,
        write_videos=True,
        eval_aux_rgb=False,
        flagged_windows="ep100_v0_s00000",
        num_text_slots=2,
        latent_channels=2,
        latent_frames=3,
        latent_height=4,
        latent_width=4,
        weights_dtype="float32",
        activations_dtype="float32",
        side_adapter_sampling_steps=25,
        side_adapter_guide_scale=1.0,
        flow_shift=5.0,
        logical_axis_rules=(),
        fps=8,
    )
    mesh = _cpu_mesh()
    pipeline = _DecodingPipeline(mesh)
    table = jnp.stack([jnp.full((2, 8), float(i) + 1.0, dtype=jnp.float32) for i in range(2)])
    state = _overfit100_state(_ContextSensitiveStub(0.2), table)
    trainer = SimpleNamespace(_create_scheduler=lambda: (_scheduler(), None))
    monkeypatch.setattr(
        gen,
        "_restore_overfit100_validation_state",
        lambda cfg: (trainer, pipeline, mesh, state, "SH", jnp.zeros((1, 2, 8)), 1000),
    )
    monkeypatch.setattr(
        gen,
        "_overfit100_rollout_fn",
        lambda state_shardings, data_shardings, replicated, scheduler, config: (
            lambda s, batch, rng: gen._rollout_overfit100_sample(s, batch, rng, scheduler, config)
        ),
    )
    gen.run_overfit100(config)

    step_root = tmp_path / "out" / "validation" / "step_001000_s3_intermediate"
    assert len(saved) == 2 * 3  # two windows x (gt, pred, comparison)
    for path in saved:
        assert "/mode_correct/seed_0/ep10" in path
        assert path.endswith(("ground_truth.mp4", "sample.mp4", "comparison_gt_top_pred_bottom.mp4"))
    # A per-window metrics.json rides with the videos, and the flag is recorded in the artifact.
    assert (step_root / "mode_correct" / "seed_0" / "ep100_v0_s00000" / "metrics.json").exists()
    artifact = json.loads((step_root / "aggregation.json").read_text())
    assert artifact["flagged_windows"] == [[100, 0]]
    assert artifact["num_windows"] == 2  # flagging never removes a window from the denominator
    assert artifact["context_derangement"] is None  # no shuffled mode -> no derangement built


# --------------------------------------------------------------------------------------
# (7) Statistic side: ablation rows can never enter m_corr.
# --------------------------------------------------------------------------------------


def test_ablation_rows_do_not_enter_the_success_statistic():
    windows = [(100, 8)]
    rows = []
    for seed in (0, 1, 2):
        rows.append(
            {
                "name": "ep100_v0_s00008",
                "episode_id": 100,
                "episode_index": 0,
                "window_start": 8,
                "checkpoint_step": 2500,
                "seed": seed,
                "context_mode": "correct",
                "ssim": 0.40,
            }
        )
        for mode in ("null", "shuffled"):
            rows.append({**rows[-1], "context_mode": mode, "ssim": 1.0})
    out = stat.evaluate_success(rows, canonical_windows=windows, segment_final_checkpoints=[2500])
    assert out["per_checkpoint"][0]["m_corr"]["[100, 8]"] == pytest.approx(0.40)
    assert out["verdict"] == "none"
    assert out["ablation_summary"]["null"]["mean_ssim"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# (6b) D2/D4 at the driver level: a mislabeled role is refused before the checkpoint, and a
# completed artifact is immutable evidence.
# --------------------------------------------------------------------------------------


def _driver_env(tmp_path, monkeypatch, episodes, *, role, seeds, modes, steps=25, boobytrap_restore=False):
    """Common driver fixture: manifest + set sidecar + records + stubs. Returns the config."""
    manifest = _driver_manifest(tmp_path, episodes)
    data_dir = tmp_path / "trainN"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "episodes.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_index": i, "episode_id": 100 + i, "used_text": text}
                    for i, (text, _n) in enumerate(episodes)
                ]
            }
        )
    )
    records = _driver_records(episodes, channels=2, frames=3, height=4, width=4)
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(records))
    monkeypatch.setattr(gen, "assert_ssim_available", lambda: None)
    monkeypatch.setattr(gen, "read_episode_texts", lambda directory, slots: [text for text, _n in episodes])
    mesh = _cpu_mesh()
    pipeline = _DecodingPipeline(mesh)
    table = jnp.stack([jnp.full((2, 8), float(i) + 1.0, dtype=jnp.float32) for i in range(len(episodes))])
    state = _overfit100_state(_ContextSensitiveStub(0.2), table)
    trainer = SimpleNamespace(_create_scheduler=lambda: (_scheduler(), None))

    def _restore(cfg):
        if boobytrap_restore:
            raise AssertionError("the checkpoint must not be restored when the pass fails its role")
        return (trainer, pipeline, mesh, state, "SH", jnp.zeros((1, 2, 8)), 2500)

    monkeypatch.setattr(gen, "_restore_overfit100_validation_state", _restore)
    monkeypatch.setattr(
        gen,
        "_overfit100_rollout_fn",
        lambda state_shardings, data_shardings, replicated, scheduler, config: (
            lambda s, batch, rng: gen._rollout_overfit100_sample(s, batch, rng, scheduler, config)
        ),
    )
    return SimpleNamespace(
        model_type="OVERFIT100_TI2V",
        run_name="drv",
        eval_data_dir=str(data_dir),
        train_data_dir=str(data_dir),
        model_manifest_path=manifest,
        checkpoint_dir=str(tmp_path / "ck"),
        output_dir=str(tmp_path / "out"),
        validation_output_dir=str(tmp_path / "out" / "validation"),
        eval_pass_role=role,
        eval_windows="canonical",
        rollout_seeds=seeds,
        context_modes=modes,
        context_shuffle_seed=0,
        write_videos=False,
        eval_aux_rgb=False,
        flagged_windows="",
        num_text_slots=len(episodes),
        latent_channels=2,
        latent_frames=3,
        latent_height=4,
        latent_width=4,
        weights_dtype="float32",
        activations_dtype="float32",
        side_adapter_sampling_steps=steps,
        side_adapter_guide_scale=1.0,
        flow_shift=5.0,
        logical_axis_rules=(),
        fps=8,
    )


def test_driver_refuses_a_segment_final_pass_that_only_runs_correct_mode(tmp_path, monkeypatch):
    # THE D2 hole: the shipped default is correct-mode only, but a segment final needs all three.
    # The checkpoint loader is booby-trapped, so the refusal must happen before any model work.
    config = _driver_env(
        tmp_path,
        monkeypatch,
        [("fold cloth", 1), ("press button", 1)],
        role="s3_segment_final",
        seeds="0,1,2",
        modes="correct",
        boobytrap_restore=True,
    )
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    msg = str(ei.value)
    assert "s3_segment_final" in msg and "mode" in msg


def test_driver_refuses_a_pass_whose_sampling_steps_are_not_25(tmp_path, monkeypatch):
    config = _driver_env(
        tmp_path,
        monkeypatch,
        [("fold cloth", 1)],
        role="s3_intermediate",
        seeds="0",
        modes="correct",
        steps=20,
        boobytrap_restore=True,
    )
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    assert "25" in str(ei.value)


def test_driver_refuses_a_full_set_role_that_only_covers_canonical_windows(tmp_path, monkeypatch):
    # eval_windows='canonical' with role s3_full_set covers 2 of 5 built windows -> refused.
    config = _driver_env(
        tmp_path,
        monkeypatch,
        [("fold cloth", 3), ("press button", 2)],
        role="s3_full_set",
        seeds="0",
        modes="correct",
        boobytrap_restore=True,
    )
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    msg = str(ei.value)
    assert "s3_full_set" in msg and "built window" in msg


def test_driver_all_spec_satisfies_the_full_set_role(tmp_path, monkeypatch):
    episodes = [("fold cloth", 3), ("press button", 2)]
    config = _driver_env(tmp_path, monkeypatch, episodes, role="s3_full_set", seeds="0", modes="correct")
    config.eval_windows = "all"
    gen.run_overfit100(config)
    artifact = json.loads(
        (tmp_path / "out" / "validation" / "step_002500_s3_full_set" / "aggregation.json").read_text()
    )
    assert artifact["eval_pass_role"] == "s3_full_set"
    assert len(artifact["covered_windows"]) == 5  # every built window of the two episodes
    assert artifact["all_windows_size"] == 5
    assert artifact["missing_canonical_windows"] == []
    assert len(artifact["rows"]) == 5


def test_driver_rerunning_the_same_pass_is_idempotent_but_a_changed_result_is_refused(tmp_path, monkeypatch):
    # D4: an eval artifact is immutable evidence. An infra retry that reproduces the same bytes is
    # a no-op; a DIFFERENT result at the same path must surface rather than overwrite.
    episodes = [("fold cloth", 1), ("press button", 1)]
    config = _driver_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    gen.run_overfit100(config)
    path = tmp_path / "out" / "validation" / "step_002500_s3_intermediate" / "aggregation.json"
    first = path.read_text()

    config2 = _driver_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    gen.run_overfit100(config2)  # byte-identical rerun -> tolerated
    assert path.read_text() == first

    # Now a pass that produces DIFFERENT numbers at the same path (a changed stub gain).
    config3 = _driver_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    table = jnp.stack([jnp.full((2, 8), float(i) + 9.0, dtype=jnp.float32) for i in range(len(episodes))])
    changed_state = _overfit100_state(_ContextSensitiveStub(0.9), table)
    trainer = SimpleNamespace(_create_scheduler=lambda: (_scheduler(), None))
    pipeline = _DecodingPipeline(_cpu_mesh())
    monkeypatch.setattr(
        gen,
        "_restore_overfit100_validation_state",
        lambda cfg: (trainer, pipeline, pipeline.mesh, changed_state, "SH", jnp.zeros((1, 2, 8)), 2500),
    )
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config3)
    assert "aggregation.json" in str(ei.value)
    assert path.read_text() == first  # the original evidence is intact


def test_driver_computes_real_aux_ceilings_end_to_end(tmp_path, monkeypatch):
    # The S2 finding, at the level it actually bit: the DRIVER passes config-derived strings, and
    # the auxiliary fetch used to receive one where a path-like was required -- so all 90 rows of
    # every S2 artifact carried "AttributeError: 'str' object has no attribute 'parent'" and the
    # run produced no VAE ceilings at all. The whole pass must now report ok coverage.
    from maxdiffusion.data_preprocessing import build_overfit100_dataset as builder

    episodes = [("fold cloth", 1), ("press button", 1)]
    destinations = []

    def _fetch(uri, fingerprint, destination):
        destinations.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)  # the real helper's first act
        destination.write_bytes(b"\x00\x00\x00\x00")
        return destination

    monkeypatch.setattr(builder, "fetch_pinned", _fetch)
    monkeypatch.setattr(builder, "decode_mp4_frames", lambda path, **kw: np.zeros((40, 8, 8, 3), dtype=np.uint8))

    config = _driver_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    config.model_manifest_path = _driver_manifest(tmp_path, episodes, with_video=True)
    config.eval_aux_rgb = True
    gen.run_overfit100(config)

    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    artifact = json.loads((step_root / "aggregation.json").read_text())
    assert [row["aux_status"] for row in artifact["rows"]] == ["ok", "ok"]
    for row in artifact["rows"]:
        assert row["vae_ceiling_ssim"] is not None and row["ssim_vs_rgb"] is not None
    # One pinned download per episode, each handed a real Path.
    assert len(destinations) == 2
    assert all(isinstance(dest, Path) and dest.name == "0.mp4" for dest in destinations)
    # D5's coverage block reports full coverage, so no WARNING/ERROR lines are emitted.
    coverage = json.loads((step_root / "summary.json").read_text())["per_mode"]["aux_coverage"]
    assert coverage == {
        "requested": 2,
        "ok": 2,
        "failed": 0,
        "coverage_fraction": 1.0,
        "failure_reason_counts": {},
    }
    assert gen.aux_coverage_log_lines(coverage) == []


# --------------------------------------------------------------------------------------
# (eval-ffmpeg strengthening, finding 2) The aux-degradation warning, OBSERVED THROUGH THE
# DRIVER.
#
# The first version of this test patched max_logging.log but never called run_overfit100 and
# never asserted the captured list, so deleting the production log call would still have
# passed. This drives the real driver, records log and restore events IN ORDER, and asserts
# exactly one warning that precedes the restore.
# --------------------------------------------------------------------------------------


def _run_driver_recording_events(tmp_path, monkeypatch, *, which, aux):
    """Drive run_overfit100, returning the ordered ("log", msg) / ("restore",) event list."""
    events: list[tuple] = []
    episodes = [("fold cloth", 1), ("press button", 1)]
    config = _driver_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    config.eval_aux_rgb = aux

    inner_restore = gen._restore_overfit100_validation_state

    def _recording_restore(cfg):
        events.append(("restore",))
        return inner_restore(cfg)

    monkeypatch.setattr(gen, "_restore_overfit100_validation_state", _recording_restore)
    monkeypatch.setattr(gen.max_logging, "log", lambda message: events.append(("log", str(message))))
    monkeypatch.setattr(gen.shutil, "which", which)
    gen.run_overfit100(config)
    return events


def test_driver_logs_exactly_one_aux_warning_before_the_restore(tmp_path, monkeypatch):
    events = _run_driver_recording_events(
        tmp_path,
        monkeypatch,
        which=lambda binary: None if binary == "ffmpeg" else "/usr/bin/" + binary,
        aux=True,
    )
    warnings = [
        i for i, event in enumerate(events) if event[0] == "log" and "ALL aux metrics will be null" in event[1]
    ]
    restores = [i for i, event in enumerate(events) if event[0] == "restore"]
    assert len(warnings) == 1, f"expected exactly one aux warning, got {len(warnings)}"
    assert len(restores) == 1
    assert warnings[0] < restores[0], "the warning must precede the checkpoint restore"
    message = events[warnings[0]][1]
    assert "ffmpeg" in message and "WARNING" in message
    assert "gsutil" not in message  # only the MISSING binary is named


@pytest.mark.parametrize(
    "which,aux,label",
    [
        (lambda binary: "/usr/bin/" + binary, True, "tools present"),
        (lambda binary: None, False, "aux disabled"),
    ],
)
def test_driver_is_silent_when_there_is_nothing_to_warn_about(tmp_path, monkeypatch, which, aux, label):
    events = _run_driver_recording_events(tmp_path, monkeypatch, which=which, aux=aux)
    warnings = [event for event in events if event[0] == "log" and "ALL aux metrics will be null" in event[1]]
    assert warnings == [], f"{label}: unexpected aux warning"
    assert any(event[0] == "restore" for event in events)  # the run still happened

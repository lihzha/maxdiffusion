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
import shutil
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
_CONFIG_YML = _REPO / "src/maxdiffusion/configs/base_wan_5b_overfit100.yml"

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
    """Adds the VAE decode seam: latents -> [1, T, H, W, 3] frames, value = mean(latent).

    ``decode_calls`` counts invocations: a resumed pass must skip the GROUND-TRUTH decode of any
    window whose tuples are all staged, which is where the wall-clock saving comes from.
    """

    def __init__(self, mesh, *, text_dim=8, max_len=2):
        super().__init__(mesh, text_dim=text_dim, max_len=max_len)
        self.decode_calls = 0

    def _denormalize_latents(self, latents):
        return latents

    def _decode_latents_to_video(self, latents):
        self.decode_calls += 1
        value = float(jnp.mean(latents.astype(jnp.float32)))
        return np.full((1, 5, 8, 8, 3), np.tanh(value) * 0.5 + 0.5, dtype=np.float32)


def _write_success_marker(data_dir: Path, *, summary_sha256="1" * 64):
    """The cycle-B publication marker a real eval set always carries (its summary_sha256 is what
    the run signature binds as the dataset's identity)."""
    (data_dir / "_SUCCESS").write_text(
        json.dumps(
            {
                "build_id": "20260729-000000",
                "build_commit": "0" * 40,
                "shards": 1,
                "records": 2,
                "summary_sha256": summary_sha256,
                "manifest_sha256": "f" * 64,
            },
            indent=2,
        )
        + "\n"
    )


def _driver_manifest(tmp_path, episodes, *, with_video=False, name=None):
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
    path = tmp_path / (name or ("driver_manifest_video.json" if with_video else "driver_manifest.json"))
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
    _write_success_marker(data_dir)
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
    _write_success_marker(data_dir)
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


_DRIVER_PIPELINE: dict = {}


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
    _write_success_marker(data_dir)
    records = _driver_records(episodes, channels=2, frames=3, height=4, width=4)
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(records))
    monkeypatch.setattr(gen, "assert_ssim_available", lambda: None)
    # A real worker HAS scikit-image (assert_ssim_available refuses to start otherwise), so the
    # fixture must produce finite, in-range SSIM -- this venv has no skimage and _frame_ssim would
    # return NaN, which staging correctly refuses to admit as evidence.
    monkeypatch.setattr(
        gen,
        "_frame_ssim",
        lambda pred, target: float(
            max(-1.0, min(1.0, 1.0 - float(np.mean(np.abs(np.asarray(pred) - np.asarray(target))))))
        ),
    )
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
    _DRIVER_PIPELINE["pipeline"] = pipeline
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
    #
    # Row STAGING is switched off here so this stays a test of the D4 guard alone. With staging on,
    # a rerun resumes every row (the envelope -- checkpoint/role/manifest/commit -- still matches)
    # and therefore reproduces identical bytes even when the in-memory state was swapped, which is
    # the intended resume semantics but would hide what this test is about. The staging-on
    # behaviour is covered by test_a_completed_pass_reruns_idempotently_through_staging and
    # test_the_immutability_guard_still_blocks_a_changed_rerun.
    monkeypatch.setenv("OVERFIT100_EVAL_RESUME", "0")
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


# ======================================================================================
# (eval-resume) Preemption-tolerant per-(window, mode, seed) staging.
#
# The step-2500 passes are 900 and 1,629 rollouts (~85 min / ~2.4 h), but spot uptime has been
# as short as 34 minutes: holding every row in memory and writing aggregation.json only at the
# end means each preemption restarts from zero. Each completed rollout is now staged as one
# JSON file under the role-keyed step_root, and a restart admits staged rows ONLY if their
# envelope carries a RUN SIGNATURE identical to this run's -- any mismatch, type error, domain
# error or corruption HARD FAILS the pass rather than silently recomputing, because a foreign
# staging directory means something an operator must resolve deliberately.
#
# PARITY SCOPE (review finding 4): resume is exact for the DETERMINISTIC primary/statistic
# fields -- every rollout's rng is window_fold_key(seed, episode_id, window_start), independent
# of visit order, and rows are appended in the same windows->modes->seeds order whether resumed
# or recomputed. The AUXILIARY block is not part of that guarantee: it depends on gsutil/ffmpeg
# and the network, so it can legitimately differ between attempts. Admission is by TUPLE
# IDENTITY, not content, so a staged aux failure persists into the resumed artifact -- which is
# safe precisely because the VAE ceiling is recoverable independently (the S2-ceiling-backfill
# path).
# ======================================================================================


def _resume_env(
    tmp_path,
    monkeypatch,
    episodes,
    *,
    role="s3_segment_final",
    seeds="0,1,2",
    modes="correct,null,shuffled",
    out="out",
    aux=False,
):
    config = _driver_env(tmp_path, monkeypatch, episodes, role=role, seeds=seeds, modes=modes)
    config.output_dir = str(tmp_path / out)
    config.validation_output_dir = str(tmp_path / out / "validation")
    config.eval_aux_rgb = aux
    monkeypatch.setenv("COMMIT", "c" * 40)
    monkeypatch.delenv("OVERFIT100_EVAL_RESUME", raising=False)
    return config


def _staging_files(step_root: Path) -> list[str]:
    root = step_root / gen.OVERFIT100_STAGING_DIR
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.json")) if root.exists() else []


def _dir_snapshot(root: Path) -> dict:
    """Every file under ``root`` -> its exact bytes (for whole-directory immutability checks)."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


_WINDOW = {
    "name": "ep100_v0_s00000",
    "episode_id": 100,
    "episode_index": 0,
    "window_start": 0,
    "canonical": True,
    "used_text": "fold cloth",
}


def _sample_row(name="ep100_v0_s00000", mode="correct", seed=0, **overrides):
    row = {
        "name": name,
        "episode_id": 100,
        "episode_index": 0,
        "window_start": 0,
        "canonical": True,
        "checkpoint_step": 2500,
        "seed": seed,
        "context_mode": mode,
        "context_source_episode_index": 0,
        "ssim": 0.91,
        "latent_mse": 0.01,
        "latent_mae": 0.1,
        "pixel_mse": 0.002,
        "pixel_mae": 0.02,
        "z_pred_std": 1.0,
        "z_target_std": 1.0,
        "z_init_anchor_mse": 0.0,
        "ssim_vs_rgb": None,
        "pixel_mse_vs_rgb": None,
        "vae_ceiling_ssim": None,
        "aux_status": "not_requested",
    }
    row.update(overrides)
    return row


def _signature(**overrides) -> dict:
    """A canonical run signature, built by the REAL builder so its key set is always exact."""
    signature = _build_signature()
    signature.update(overrides)
    return signature


def _stage(step_root, row, *, signature=None, windows=None):
    return gen.write_staged_row(str(step_root), row, run_signature=signature or _signature())


def _read(step_root, *, signature=None, windows=None, seeds=(0,), modes=("correct",), derangement=None):
    return gen.read_staged_rows(
        str(step_root),
        run_signature=signature or _signature(),
        windows=windows if windows is not None else [_WINDOW],
        seeds=list(seeds),
        modes=list(modes),
        derangement=derangement,
    )


# --------------------------------------------------------------------------------------
# Round-trip + the run signature (finding 1).
# --------------------------------------------------------------------------------------


def test_staged_row_round_trips_through_its_envelope(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    row = _sample_row()
    path = Path(_stage(step_root, row))
    assert (
        path.relative_to(step_root) == Path(gen.OVERFIT100_STAGING_DIR) / "correct" / "seed_0" / "ep100_v0_s00000.json"
    )
    payload = json.loads(path.read_text())
    assert payload["schema"] == gen.OVERFIT100_STAGED_ROW_SCHEMA
    assert payload["run_signature"] == _signature()
    assert payload["run_signature_sha256"] == gen.run_signature_sha256(_signature())
    assert payload["row"] == row
    assert _read(step_root) == {("ep100_v0_s00000", "correct", 0): row}


def test_read_staged_rows_is_empty_when_nothing_is_staged(tmp_path):
    assert _read(tmp_path / "step_002500_s3_segment_final", windows=[]) == {}


def test_run_signature_is_built_from_the_run_and_strictly_typed():
    config = SimpleNamespace(
        checkpoint_dir="gs://b/ck",
        train_data_dir="gs://b/train100",
        eval_data_dir="gs://b/train100",
        eval_windows="canonical",
        context_shuffle_seed=3,
        num_text_slots=100,
        side_adapter_sampling_steps=25,
        side_adapter_guide_scale=1.0,
        flow_shift=5.0,
        weights_dtype="bfloat16",
        activations_dtype="bfloat16",
        eval_aux_rgb=True,
        write_videos=False,
        pretrained_model_name_or_path="/cache/snapshots/" + "b" * 40,
        expected_model_revision="b" * 40,
    )
    signature = _build_signature(
        config,
        derangement=(1, 0),
        seeds=(0, 1, 2),
        modes=("correct", "null", "shuffled"),
        eval_summary_sha256="2" * 64,
    )
    # Every field the review named is bound...
    for field in (
        "checkpoint_step",
        "resolved_checkpoint_dir",
        "pass_role",
        "manifest_sha256",
        "code_commit",
        "model_snapshot",
        "model_revision",
        "train_data_dir",
        "eval_data_dir",
        "train_summary_sha256",
        "eval_summary_sha256",
        "eval_windows_spec",
        "num_windows",
        "rollout_seeds",
        "context_modes",
        "context_shuffle_seed",
        "context_derangement_sha256",
        "num_text_slots",
        "sampling_steps",
        "guide_scale",
        "flow_shift",
        "weights_dtype",
        "activations_dtype",
        "eval_aux_rgb",
        "write_videos",
    ):
        assert field in signature, field
    # ...with exact JSON types (bools are bools, not ints).
    assert type(signature["checkpoint_step"]) is int
    assert type(signature["eval_aux_rgb"]) is bool and type(signature["write_videos"]) is bool
    assert type(signature["guide_scale"]) is float and type(signature["flow_shift"]) is float
    assert signature["rollout_seeds"] == [0, 1, 2] and all(type(s) is int for s in signature["rollout_seeds"])
    # The derangement is bound by identity, not by presence.
    assert signature["context_derangement_sha256"] != "none"
    other = _build_signature(
        config,
        derangement=(0, 1),
        seeds=(0, 1, 2),
        modes=("correct", "null", "shuffled"),
        eval_summary_sha256="2" * 64,
    )
    assert other["context_derangement_sha256"] != signature["context_derangement_sha256"]
    assert gen.run_signature_sha256(other) != gen.run_signature_sha256(signature)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("checkpoint_step", 1000),
        ("pass_role", "s3_intermediate"),
        ("manifest_sha256", "b" * 64),
        ("code_commit", "d" * 40),
        ("model_snapshot", "/cache/snapshots/" + "e" * 40),
        ("model_revision", "e" * 40),
        ("train_data_dir", "gs://bucket/train10"),
        ("eval_data_dir", "gs://bucket/train10"),
        ("train_summary_sha256", "9" * 64),
        ("eval_summary_sha256", "9" * 64),
        ("eval_windows_spec", "all"),
        ("num_windows", 2),
        ("rollout_seeds", [0, 1]),
        ("context_modes", ["correct", "null"]),
        ("context_shuffle_seed", 7),
        ("context_derangement_sha256", "f" * 64),
        ("num_text_slots", 100),
        ("sampling_steps", 20),
        ("guide_scale", 5.0),
        ("flow_shift", 3.0),
        ("weights_dtype", "bfloat16"),
        ("activations_dtype", "bfloat16"),
        ("eval_aux_rgb", True),
        ("write_videos", True),
    ],
)
def test_every_bound_signature_field_is_enforced(tmp_path, field, bad):
    # One case per bound input: a staged row produced under ANY different setting is refused.
    step_root = tmp_path / "step_002500_s3_segment_final"
    _stage(step_root, _sample_row(), signature=_signature(**{field: bad}))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    message = str(ei.value)
    assert field in message and "run signature" in message.lower()


def test_a_tampered_signature_hash_is_refused(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    path = Path(_stage(step_root, _sample_row()))
    payload = json.loads(path.read_text())
    payload["run_signature_sha256"] = "0" * 64
    path.write_text(json.dumps(payload, indent=2))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    assert "run_signature_sha256" in str(ei.value)


def test_staging_requires_a_real_commit_and_hard_disables_on_unknown(tmp_path, monkeypatch):
    # "unknown" provenance must never admit or write: it cannot distinguish two code states.
    monkeypatch.delenv("COMMIT", raising=False)
    enabled, reason = gen.resume_state(SimpleNamespace(overfit100_eval_resume=True))
    assert enabled is False
    assert "COMMIT" in reason and "40" in reason
    monkeypatch.setenv("COMMIT", "not-a-sha")
    assert gen.resume_state(SimpleNamespace(overfit100_eval_resume=True))[0] is False
    monkeypatch.setenv("COMMIT", "c" * 40)
    assert gen.resume_state(SimpleNamespace(overfit100_eval_resume=True))[0] is True


def test_driver_disables_staging_entirely_without_a_valid_commit(tmp_path, monkeypatch):
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    monkeypatch.setenv("COMMIT", "unknown")
    logged: list[str] = []
    monkeypatch.setattr(gen.max_logging, "log", lambda message: logged.append(str(message)))
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    assert _staging_files(step_root) == []  # neither read NOR written
    assert any("COMMIT" in line for line in logged)


# --------------------------------------------------------------------------------------
# Strict-type / domain admission (finding 2).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    ["{ not json", "[]", '"a string"', json.dumps({"row": {}})],
    ids=["corrupt", "list", "string", "no_envelope"],
)
def test_corrupt_or_foreign_staged_file_hard_fails(tmp_path, content):
    step_root = tmp_path / "step_002500_s3_segment_final"
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(ValueError):
        _read(step_root)


def test_a_foreign_schema_tag_hard_fails_even_when_everything_else_is_valid(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    payload = gen.staged_row_envelope(_sample_row(), run_signature=_signature())
    payload["schema"] = "some_other_writer_v9"
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, indent=2))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    assert "some_other_writer_v9" in str(ei.value) and gen.OVERFIT100_STAGED_ROW_SCHEMA in str(ei.value)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("checkpoint_step", "2500"),  # a STRING that int() would happily coerce
        ("checkpoint_step", 2500.0),  # a float that == 2500
        ("checkpoint_step", True),  # bool is an int subclass
        ("num_windows", "1"),
        ("guide_scale", "1.0"),
        ("eval_aux_rgb", 0),  # int where a bool is required
        ("rollout_seeds", ["0"]),
        ("context_modes", "correct"),  # a bare string, not a list
    ],
)
def test_signature_values_are_type_exact_not_coerced(tmp_path, field, bad):
    step_root = tmp_path / "step_002500_s3_segment_final"
    _stage(step_root, _sample_row(), signature=_signature(**{field: bad}))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    assert field in str(ei.value)


def test_a_bool_cannot_masquerade_as_an_int_in_the_signature(tmp_path):
    # THE case that makes `type(x) is int` load-bearing rather than decorative: `True == 1`, so an
    # isinstance check would admit a bool where an int is required and the value comparison would
    # not notice. Only exact typing refuses it.
    step_root = tmp_path / "step_002500_s3_segment_final"
    expected = _signature(checkpoint_step=1)
    staged = _signature(checkpoint_step=True)
    assert staged["checkpoint_step"] == expected["checkpoint_step"]  # True == 1
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(gen.staged_row_envelope(_sample_row(checkpoint_step=1), run_signature=staged)))
    with pytest.raises(ValueError) as ei:
        _read(step_root, signature=expected)
    message = str(ei.value)
    assert "checkpoint_step" in message and "bool" in message


@pytest.mark.parametrize(
    "field,bad",
    [
        ("seed", "0"),
        ("seed", 0.0),
        ("seed", False),
        ("episode_id", "100"),
        ("episode_index", 1.0),
        ("window_start", True),
        ("canonical", 1),  # int where a bool is required
        ("canonical", "true"),
        ("name", 100),
        ("context_mode", 0),
        ("aux_status", None),
        ("ssim", "0.91"),
        ("ssim", None),
        ("latent_mse", "0.01"),
        ("context_source_episode_index", "0"),
    ],
)
def test_row_fields_are_type_exact_not_coerced(tmp_path, field, bad):
    step_root = tmp_path / "step_002500_s3_segment_final"
    row = _sample_row(**{field: bad})
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(gen.staged_row_envelope(row, run_signature=_signature())))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    assert field in str(ei.value)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("ssim", float("nan")),
        ("ssim", float("inf")),
        ("ssim", 1.5),  # outside SSIM's [-1, 1]
        ("ssim", -2.0),
        ("latent_mse", -0.5),  # a squared error cannot be negative
        ("pixel_mse", float("nan")),
        ("z_pred_std", -1.0),
        ("ssim_vs_rgb", 2.0),
        ("vae_ceiling_ssim", float("nan")),
        ("pixel_mse_vs_rgb", -1.0),
    ],
)
def test_metric_domains_are_validated_before_admission(tmp_path, field, bad):
    step_root = tmp_path / "step_002500_s3_segment_final"
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(gen.staged_row_envelope(_sample_row(**{field: bad}), run_signature=_signature())))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    assert field in str(ei.value)


@pytest.mark.parametrize("field,bad", [("name", "ep999_v0_s00000"), ("context_mode", "null"), ("seed", 2)])
def test_staged_row_identity_must_match_its_path(tmp_path, field, bad):
    step_root = tmp_path / "step_002500_s3_segment_final"
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    payload = gen.staged_row_envelope(_sample_row(**{field: bad}), run_signature=_signature())
    path.write_text(json.dumps(payload, indent=2))
    with pytest.raises(ValueError) as ei:
        _read(step_root, seeds=(0, 2), modes=("correct", "null"))
    assert "ep100_v0_s00000" in str(ei.value)


@pytest.mark.parametrize(
    "field,bad", [("episode_id", 999), ("episode_index", 5), ("window_start", 8), ("canonical", False)]
)
def test_row_identity_is_checked_against_the_full_window_descriptor(tmp_path, field, bad):
    # Not just the name: every identity field must equal the SELECTED window's descriptor.
    step_root = tmp_path / "step_002500_s3_segment_final"
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(gen.staged_row_envelope(_sample_row(**{field: bad}), run_signature=_signature())))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    assert field in str(ei.value)


def test_context_source_index_must_match_the_mode_and_derangement(tmp_path):
    # A shuffled row claiming its own row (or the wrong deranged row) is refused.
    step_root = tmp_path / "step_002500_s3_segment_final"
    signature = _signature(context_modes=["shuffled"], context_derangement_sha256="f" * 64)
    path = step_root / gen.OVERFIT100_STAGING_DIR / "shuffled" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    bad = _sample_row(mode="shuffled", context_source_episode_index=0)  # correct-mode source
    path.write_text(json.dumps(gen.staged_row_envelope(bad, run_signature=signature)))
    with pytest.raises(ValueError) as ei:
        _read(step_root, signature=signature, modes=("shuffled",), derangement=(1, 0))
    assert "context_source_episode_index" in str(ei.value)
    # The right source index is admitted.
    good = _sample_row(mode="shuffled", context_source_episode_index=1)
    path.write_text(json.dumps(gen.staged_row_envelope(good, run_signature=signature)))
    assert _read(step_root, signature=signature, modes=("shuffled",), derangement=(1, 0))


def test_staged_row_with_wrong_columns_hard_fails(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    short = _sample_row()
    del short["ssim"]
    path = step_root / gen.OVERFIT100_STAGING_DIR / "correct" / "seed_0" / "ep100_v0_s00000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(gen.staged_row_envelope(short, run_signature=_signature())))
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    assert "ssim" in str(ei.value)


def test_staged_row_for_an_unselected_window_hard_fails(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    _stage(step_root, _sample_row())
    with pytest.raises(ValueError) as ei:
        _read(step_root, windows=[{**_WINDOW, "name": "ep200_v0_s00000", "episode_id": 200}])
    assert "ep100_v0_s00000" in str(ei.value)


# --------------------------------------------------------------------------------------
# Enumeration (finding 5).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "stray.json",
        "correct/stray.json",
        "correct/seed_0/nested/deep.json",
        "correct/seed_0/row.json.tmp",
        "correct/seed_0/notes.txt",
        "correct/seedX/ep100_v0_s00000.json",
    ],
)
def test_any_nonconforming_object_under_staging_is_refused(tmp_path, relative):
    step_root = tmp_path / "step_002500_s3_segment_final"
    _stage(step_root, _sample_row())
    stray = step_root / gen.OVERFIT100_STAGING_DIR / relative
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text("{}")
    with pytest.raises(ValueError) as ei:
        _read(step_root)
    message = str(ei.value)
    assert relative.rsplit("/", 1)[-1] in message
    assert str(step_root / gen.OVERFIT100_STAGING_DIR) in message  # the exact root to clear


def test_enumeration_error_lists_every_offender_with_a_count(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    root = step_root / gen.OVERFIT100_STAGING_DIR
    for i in range(7):
        stray = root / f"stray_{i}.txt"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("x")
    with pytest.raises(ValueError) as ei:
        _read(step_root, windows=[])
    message = str(ei.value)
    assert "7" in message  # the count, not just the first offender
    assert message.count("stray_") >= 3


def test_empty_directories_under_staging_are_tolerated(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    _stage(step_root, _sample_row())
    (step_root / gen.OVERFIT100_STAGING_DIR / "null" / "seed_1").mkdir(parents=True)
    assert _read(step_root) == {("ep100_v0_s00000", "correct", 0): _sample_row()}


# --------------------------------------------------------------------------------------
# Driver behaviour: staging, skip set, parity, D4 ordering.
# --------------------------------------------------------------------------------------


def test_driver_stages_every_row_as_it_completes(tmp_path, monkeypatch):
    episodes = [("fold cloth", 1), ("press button", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    assert _staging_files(step_root) == [
        "correct/seed_0/ep100_v0_s00000.json",
        "correct/seed_0/ep101_v0_s00000.json",
    ]
    staged = json.loads((step_root / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json").read_text())
    artifact = json.loads((step_root / "aggregation.json").read_text())
    assert staged["row"] in artifact["rows"]
    signature = staged["run_signature"]
    assert signature["code_commit"] == artifact["commit"]  # one provenance source, not two
    assert signature["manifest_sha256"] == artifact["manifest_sha256"]
    assert signature["pass_role"] == artifact["eval_pass_role"]
    assert signature["resolved_checkpoint_dir"] == artifact["checkpoint_dir"].rstrip("/")


def test_resume_recomputes_only_the_missing_tuples(tmp_path, monkeypatch):
    episodes = [("fold cloth", 1), ("press button", 1)]
    first = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="a")
    gen.run_overfit100(first)
    root_a = tmp_path / "a" / "validation" / "step_002500_s3_intermediate" / gen.OVERFIT100_STAGING_DIR

    second = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="b")
    root_b = tmp_path / "b" / "validation" / "step_002500_s3_intermediate" / gen.OVERFIT100_STAGING_DIR
    (root_b / "correct" / "seed_0").mkdir(parents=True)
    shutil.copy(root_a / "correct/seed_0/ep100_v0_s00000.json", root_b / "correct/seed_0/ep100_v0_s00000.json")

    _CALLS.clear()
    gen.run_overfit100(second)
    assert len(_CALLS) == 1  # only the missing tuple was rolled out
    artifact = json.loads(
        (tmp_path / "b" / "validation" / "step_002500_s3_intermediate" / "aggregation.json").read_text()
    )
    assert [row["name"] for row in artifact["rows"]] == ["ep100_v0_s00000", "ep101_v0_s00000"]


def test_resumed_aggregation_is_byte_identical_to_straight_through(tmp_path, monkeypatch):
    # Parity for the DETERMINISTIC fields (aux off here; the aux-enabled contract is pinned
    # separately below): visit order cannot change a number, because every rollout's rng is keyed
    # on the window identity rather than on its position in the loop.
    episodes = [("fold cloth", 1), ("press button", 1)]
    first = _resume_env(tmp_path, monkeypatch, episodes, out="a")
    gen.run_overfit100(first)
    step_a = tmp_path / "a" / "validation" / "step_002500_s3_segment_final"
    straight_through = (step_a / "aggregation.json").read_bytes()

    second = _resume_env(tmp_path, monkeypatch, episodes, out="b")
    step_b = tmp_path / "b" / "validation" / "step_002500_s3_segment_final"
    relatives = [
        f"{mode}/seed_{seed}/ep100_v0_s00000.json" for mode in ("correct", "null", "shuffled") for seed in (0, 1, 2)
    ]
    relatives += [
        "null/seed_2/ep101_v0_s00000.json",
        "shuffled/seed_1/ep101_v0_s00000.json",
        "correct/seed_0/ep101_v0_s00000.json",
    ]
    for relative in relatives:
        target = step_b / gen.OVERFIT100_STAGING_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(step_a / gen.OVERFIT100_STAGING_DIR / relative, target)
    _CALLS.clear()
    gen.run_overfit100(second)
    # 18 tuples total, 12 staged -> exactly 6 rollouts recomputed...
    assert len(_CALLS) == 6
    # ...and ep100, whose every tuple was staged, is never decoded: the whole-window short-circuit
    # skips its GROUND-TRUTH decode too (1 gt decode for ep101 + 6 prediction decodes).
    assert _DRIVER_PIPELINE["pipeline"].decode_calls == 7
    assert (step_b / "aggregation.json").read_bytes() == straight_through
    assert (step_b / "summary.csv").read_bytes() == (step_a / "summary.csv").read_bytes()


def _aux_spies(monkeypatch, *, fail=False):
    from maxdiffusion.data_preprocessing import build_overfit100_dataset as builder

    def _fetch(uri, fingerprint, destination):
        if fail:
            raise builder.BuildError("gs://b/o.mp4: pinned download failed (gsutil exit 1)")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00\x00\x00\x00")
        return destination

    monkeypatch.setattr(builder, "fetch_pinned", _fetch)
    monkeypatch.setattr(builder, "decode_mp4_frames", lambda path, **kw: np.zeros((40, 8, 8, 3), dtype=np.uint8))


def test_parity_holds_with_aux_enabled_when_both_attempts_succeed(tmp_path, monkeypatch):
    # Finding 4 (a): with the auxiliary path working in both attempts, whole-file parity still holds.
    episodes = [("fold cloth", 1), ("press button", 1)]
    _aux_spies(monkeypatch)
    first = _resume_env(
        tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="a", aux=True
    )
    first.model_manifest_path = _driver_manifest(tmp_path, episodes, with_video=True)
    gen.run_overfit100(first)
    step_a = tmp_path / "a" / "validation" / "step_002500_s3_intermediate"
    assert all(row["aux_status"] == "ok" for row in json.loads((step_a / "aggregation.json").read_text())["rows"])

    second = _resume_env(
        tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="b", aux=True
    )
    second.model_manifest_path = first.model_manifest_path
    step_b = tmp_path / "b" / "validation" / "step_002500_s3_intermediate"
    target = step_b / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json"
    target.parent.mkdir(parents=True)
    shutil.copy(step_a / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json", target)
    gen.run_overfit100(second)
    assert (step_b / "aggregation.json").read_bytes() == (step_a / "aggregation.json").read_bytes()


def test_a_staged_aux_failure_is_admitted_as_is_even_when_a_fresh_attempt_would_succeed(tmp_path, monkeypatch):
    # Finding 4 (b), pinned semantics: admission is by TUPLE IDENTITY, not content. A row staged
    # while gsutil/ffmpeg were unavailable keeps its recorded aux failure on resume -- the pass is
    # not silently re-run for it -- which is acceptable precisely because the VAE ceiling is
    # recoverable independently (the S2-ceiling-backfill path), and the aux_status makes the gap
    # explicit in the artifact.
    episodes = [("fold cloth", 1), ("press button", 1)]
    _aux_spies(monkeypatch, fail=True)
    first = _resume_env(
        tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="a", aux=True
    )
    first.model_manifest_path = _driver_manifest(tmp_path, episodes, with_video=True)
    gen.run_overfit100(first)
    step_a = tmp_path / "a" / "validation" / "step_002500_s3_intermediate"

    second = _resume_env(
        tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="b", aux=True
    )
    second.model_manifest_path = first.model_manifest_path
    step_b = tmp_path / "b" / "validation" / "step_002500_s3_intermediate"
    target = step_b / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json"
    target.parent.mkdir(parents=True)
    shutil.copy(step_a / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json", target)
    _aux_spies(monkeypatch)  # the second attempt's aux WOULD succeed
    gen.run_overfit100(second)

    rows = {row["name"]: row for row in json.loads((step_b / "aggregation.json").read_text())["rows"]}
    assert rows["ep100_v0_s00000"]["aux_status"].startswith("BuildError:")  # staged failure kept
    assert rows["ep100_v0_s00000"]["vae_ceiling_ssim"] is None
    assert rows["ep101_v0_s00000"]["aux_status"] == "ok"  # recomputed tuple got its ceiling
    coverage = json.loads((step_b / "summary.json").read_text())["per_mode"]["aux_coverage"]
    assert coverage["requested"] == 2 and coverage["ok"] == 1 and coverage["failed"] == 1


def test_resume_logs_one_line_with_the_split(tmp_path, monkeypatch):
    episodes = [("fold cloth", 1), ("press button", 1)]
    first = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="a")
    gen.run_overfit100(first)
    root_a = tmp_path / "a" / "validation" / "step_002500_s3_intermediate" / gen.OVERFIT100_STAGING_DIR
    second = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="b")
    root_b = tmp_path / "b" / "validation" / "step_002500_s3_intermediate" / gen.OVERFIT100_STAGING_DIR
    (root_b / "correct" / "seed_0").mkdir(parents=True)
    shutil.copy(root_a / "correct/seed_0/ep100_v0_s00000.json", root_b / "correct/seed_0/ep100_v0_s00000.json")

    logged: list[str] = []
    monkeypatch.setattr(gen.max_logging, "log", lambda message: logged.append(str(message)))
    gen.run_overfit100(second)
    lines = [line for line in logged if "resumed n_rows=" in line]
    assert len(lines) == 1
    assert "resumed n_rows=1" in lines[0] and "recomputed=1" in lines[0]


@pytest.mark.parametrize("disable", ["env", "config"], ids=["env OVERFIT100_EVAL_RESUME=0", "config flag False"])
def test_resume_can_be_switched_off(tmp_path, monkeypatch, disable):
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    if disable == "env":
        monkeypatch.setenv("OVERFIT100_EVAL_RESUME", "0")
    else:
        config.overfit100_eval_resume = False
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    assert _staging_files(step_root) == []
    assert (step_root / "aggregation.json").exists()


def test_resume_env_switch_rejects_a_junk_value(tmp_path, monkeypatch):
    monkeypatch.setenv("OVERFIT100_EVAL_RESUME", "yes")
    monkeypatch.setenv("COMMIT", "c" * 40)
    with pytest.raises(ValueError) as ei:
        gen.resume_state(SimpleNamespace(overfit100_eval_resume=True))
    assert "OVERFIT100_EVAL_RESUME" in str(ei.value)


def test_resume_is_disabled_and_explained_under_multi_host(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMIT", "c" * 40)
    monkeypatch.setattr(gen.jax, "process_count", lambda: 2)
    enabled, reason = gen.resume_state(SimpleNamespace(overfit100_eval_resume=True))
    assert enabled is False
    assert "process_count=2" in reason and "host" in reason.lower()

    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    logged: list[str] = []
    monkeypatch.setattr(gen.max_logging, "log", lambda message: logged.append(str(message)))
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    assert _staging_files(step_root) == []
    assert any("process_count=2" in line for line in logged)


def test_resume_state_is_enabled_by_default_single_host(monkeypatch):
    monkeypatch.setenv("COMMIT", "c" * 40)
    enabled, reason = gen.resume_state(SimpleNamespace())
    assert enabled is True and "enabled" in reason


def test_a_completed_pass_reruns_idempotently_through_staging(tmp_path, monkeypatch):
    # D4-first (finding 3): with aggregation.json present the rerun takes the ORIGINAL
    # recompute-in-memory path -- staging is neither read nor written -- and the immutable writer
    # then sees identical bytes, so the completed role directory is untouched.
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    before = _dir_snapshot(step_root)
    _CALLS.clear()
    logged: list[str] = []
    monkeypatch.setattr(gen.max_logging, "log", lambda message: logged.append(str(message)))
    gen.run_overfit100(config)
    assert len(_CALLS) == 1  # recomputed in memory, NOT resumed
    assert _dir_snapshot(step_root) == before  # nothing in the role dir changed, staging included
    assert any("PUBLISHED" in line for line in logged)


def test_the_immutability_guard_still_blocks_a_changed_rerun_without_touching_the_role_dir(tmp_path, monkeypatch):
    # D4-first: a rerun that would produce DIFFERENT bytes is refused, and because staging is
    # disabled by the completed artifact, NOTHING in the role directory is mutated first --
    # including staging_rows, which the previous version of this test did not check.
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    before = _dir_snapshot(step_root)
    assert any(name.startswith(gen.OVERFIT100_STAGING_DIR) for name in before)  # staging IS present

    table = jnp.stack([jnp.full((2, 8), 9.0, dtype=jnp.float32) for _ in episodes])
    changed = _overfit100_state(_ContextSensitiveStub(0.9), table)
    trainer = SimpleNamespace(_create_scheduler=lambda: (_scheduler(), None))
    pipeline = _DecodingPipeline(_cpu_mesh())
    monkeypatch.setattr(
        gen,
        "_restore_overfit100_validation_state",
        lambda cfg: (trainer, pipeline, pipeline.mesh, changed, "SH", jnp.zeros((1, 2, 8)), 2500),
    )
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    assert "aggregation.json" in str(ei.value)
    assert _dir_snapshot(step_root) == before  # not even a staged row was rewritten


def test_a_partial_role_dir_still_resumes(tmp_path, monkeypatch):
    # The completed-artifact check must not disable resume for a PREEMPTED pass, whose role dir
    # holds staging but no aggregation.json -- that is exactly the case resume exists for.
    episodes = [("fold cloth", 1), ("press button", 1)]
    first = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="a")
    gen.run_overfit100(first)
    step_a = tmp_path / "a" / "validation" / "step_002500_s3_intermediate"
    second = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct", out="b")
    step_b = tmp_path / "b" / "validation" / "step_002500_s3_intermediate"
    target = step_b / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json"
    target.parent.mkdir(parents=True)
    shutil.copy(step_a / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json", target)
    assert not (step_b / "aggregation.json").exists()
    _CALLS.clear()
    gen.run_overfit100(second)
    assert len(_CALLS) == 1  # resumed, not recomputed from zero


def test_config_carries_the_resume_switch():
    import yaml

    cfg = yaml.safe_load(_CONFIG_YML.read_text())
    assert cfg["overfit100_eval_resume"] is True
    assert "OVERFIT100_EVAL_RESUME" in _CONFIG_YML.read_text()


# ======================================================================================
# (eval-resume pass 2) Complete run signature, write-suppressed completed dirs, and
# MARKER-LAST transactional publication.
# ======================================================================================


def _published(step_root: Path) -> bool:
    return (step_root / gen.OVERFIT100_PUBLISHED_MARKER).exists()


# --------------------------------------------------------------------------------------
# Finding 1: the signature binds the rest of the rollout-affecting inputs.
# --------------------------------------------------------------------------------------


def _full_signature_config(**overrides):
    base = {
        "checkpoint_dir": "gs://b/ck",
        "output_dir": "gs://b",
        "run_name": "r",
        "train_data_dir": "gs://b/train100",
        "eval_data_dir": "gs://b/train100",
        "eval_windows": "canonical",
        "context_shuffle_seed": 0,
        "num_text_slots": 2,
        "side_adapter_sampling_steps": 25,
        "side_adapter_guide_scale": 1.0,
        "flow_shift": 5.0,
        "flow_sigma_min": 0.0,
        "flow_sigma_max": 1.0,
        "weights_dtype": "float32",
        "activations_dtype": "float32",
        "eval_aux_rgb": False,
        "write_videos": False,
        "pretrained_model_name_or_path": "/cache/snapshots/" + "b" * 40,
        "expected_model_revision": "b" * 40,
        "latent_channels": 48,
        "latent_frames": 9,
        "latent_height": 12,
        "latent_width": 20,
        "wan_max_sequence_length": 512,
        "attention": "flash",
        "precision": "DEFAULT",
        "flash_min_seq_length": 4096,
        "flash_block_sizes": {},
        "split_head_dim": True,
        "fps": 16,
        "use_qwix_quantization": False,
        "quantization": "",
        "qwix_module_path": ".*",
        "weight_quantization_calibration_method": "absmax",
        "act_quantization_calibration_method": "absmax",
        "bwd_quantization_calibration_method": "absmax",
        "wan_transformer_pretrained_model_name_or_path": "/cache/snapshots/" + "b" * 40,
        "from_pt": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_signature(config=None, **overrides):
    kwargs = {
        "checkpoint_step": 2500,
        "pass_role": "s3_segment_final",
        "manifest_sha256": "a" * 64,
        "code_commit": "c" * 40,
        "derangement": None,
        "windows": [_WINDOW],
        "seeds": (0,),
        "modes": ("correct",),
        "train_summary_sha256": "1" * 64,
        "eval_summary_sha256": "1" * 64,
        "resolved_checkpoint_dir": "gs://b/ck",
        "scheduler_sigma_min": 0.0,
        "scheduler_sigma_max": 1.0,
        "num_train_timesteps": 1000,
    }
    kwargs.update(overrides)
    return gen.overfit100_run_signature(config or _full_signature_config(), **kwargs)


def test_signature_binds_every_rollout_affecting_input():
    signature = _build_signature()
    for field in (
        "scheduler_sigma_min",
        "scheduler_sigma_max",
        "latent_channels",
        "latent_frames",
        "latent_height",
        "latent_width",
        "wan_max_sequence_length",
        "attention",
        "precision",
        "flash_min_seq_length",
        "flash_block_sizes",
        "split_head_dim",
        "fps",
        "resolved_checkpoint_dir",
    ):
        assert field in signature, field
    assert set(signature) == {"schema"} | {name for name, _ in gen.OVERFIT100_RUN_SIGNATURE_TYPES}
    assert type(signature["scheduler_sigma_min"]) is float
    assert type(signature["split_head_dim"]) is bool
    assert type(signature["fps"]) is int


def test_signature_uses_the_resolved_checkpoint_dir_not_the_config_string():
    # An empty checkpoint_dir resolves to <output_dir>/<run_name>/checkpoints -- the directory the
    # restore actually reads, which is what must be bound.
    config = _full_signature_config(checkpoint_dir="")
    resolved = gen._resolved_checkpoint_dir(config)
    assert resolved == "gs://b/r/checkpoints"
    signature = _build_signature(config, resolved_checkpoint_dir=resolved)
    assert signature["resolved_checkpoint_dir"] == resolved
    # The RAW config string is deliberately not a signature field: it is a spelling of the resolved
    # value, and carrying both is what let a trailing slash reject otherwise-identical rows.
    assert "checkpoint_dir" not in signature


@pytest.mark.parametrize(
    "field,bad",
    [
        ("scheduler_sigma_min", 0.1),
        ("scheduler_sigma_max", 0.9),
        ("latent_channels", 16),
        ("latent_frames", 5),
        ("latent_height", 24),
        ("latent_width", 40),
        ("wan_max_sequence_length", 256),
        ("attention", "dot_product"),
        ("precision", "HIGHEST"),
        ("flash_min_seq_length", 2048),
        ("flash_block_sizes", '{"block_q": 128}'),
        ("split_head_dim", False),
        ("fps", 8),
        ("resolved_checkpoint_dir", "gs://b/OTHER/checkpoints"),
    ],
)
def test_every_new_signature_field_is_enforced(tmp_path, field, bad):
    step_root = tmp_path / "step_002500_s3_segment_final"
    expected = _build_signature()
    _stage(step_root, _sample_row(), signature={**expected, field: bad})
    with pytest.raises(ValueError) as ei:
        _read(step_root, signature=expected)
    assert field in str(ei.value)


def test_an_extra_signature_key_is_refused(tmp_path):
    # Exact key set: a staged signature carrying a field this run does not know cannot be compared.
    step_root = tmp_path / "step_002500_s3_segment_final"
    expected = _build_signature()
    _stage(step_root, _sample_row(), signature={**expected, "future_knob": 1})
    with pytest.raises(ValueError) as ei:
        _read(step_root, signature=expected)
    message = str(ei.value)
    assert "future_knob" in message and "key set" in message.lower()


def test_signature_hash_equality_is_required_against_the_expected_signature(tmp_path):
    # Review-required belt and braces beside the field-by-field comparison: the staged signature
    # must hash to the SAME value as this run's independently computed one.
    #
    # Honest note: with the exact key set enforced and every typed field compared by value, this
    # check is currently UNREACHABLE as the sole failure -- no input can pass those and still hash
    # differently. It is kept because the review required it and because it keeps the guarantee
    # whole-object rather than field-enumerated (a future field added to the builder but forgotten
    # in the type table would be caught by the key-set check today, and by this one regardless).
    # Its presence is therefore asserted against the source as well as exercised behaviourally.
    import inspect

    step_root = tmp_path / "step_002500_s3_segment_final"
    expected = _build_signature()
    assert gen.run_signature_sha256(expected) == gen.run_signature_sha256(dict(expected))
    _stage(step_root, _sample_row(), signature=expected)
    assert _read(step_root, signature=expected)
    src = inspect.getsource(gen.read_staged_rows)
    assert "expected_hash = run_signature_sha256(expected_signature)" in src
    assert "if recorded_hash != expected_hash:" in src


# --------------------------------------------------------------------------------------
# Finding 2: a completed role directory takes NO filesystem writes.
# --------------------------------------------------------------------------------------


def test_a_completed_dir_writes_nothing_even_with_videos_enabled(tmp_path, monkeypatch):
    # The hole the previous snapshot tests missed: they ran write_videos=False, so the loop's
    # _save_video / metrics.json overwrites (save_video uses overwrite=True) went unchecked.
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    config.write_videos = True
    saved: list[str] = []
    monkeypatch.setattr(
        gen,
        "_save_video",
        lambda frames, path, fps: (
            saved.append(path),
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(b"v"),
        )[0],
    )
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    assert saved and _published(step_root)
    before = _dir_snapshot(step_root)

    saved.clear()
    monkeypatch.setattr(
        gen,
        "_save_video",
        lambda frames, path, fps: pytest.fail("a completed role directory must not write videos"),
    )
    monkeypatch.setattr(
        gen,
        "_write_json",
        lambda path, value: pytest.fail(f"a completed role directory must not write {path}"),
    )
    gen.run_overfit100(config)
    assert _dir_snapshot(step_root) == before


def test_a_completed_dir_suppresses_staging_writes_too(tmp_path, monkeypatch):
    # Today this is doubly guaranteed: a published directory turns resume OFF (so the staging write
    # is unreachable) AND the write itself is gated on the suppression flag. The behavioural half is
    # asserted here; the belt-and-braces half is asserted against the source, because no test can
    # reach it while resume_on is already False -- and it is what keeps the guarantee if a future
    # change ever re-enables resume in published mode.
    import inspect

    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    before = _dir_snapshot(step_root)
    monkeypatch.setattr(gen, "write_staged_row", lambda *a, **kw: pytest.fail("completed dirs must not stage rows"))
    gen.run_overfit100(config)
    assert _dir_snapshot(step_root) == before
    src = inspect.getsource(gen.run_overfit100)
    assert "if resume_on and not writes_suppressed:" in src


# --------------------------------------------------------------------------------------
# Finding 3: marker-last transactional publication.
# --------------------------------------------------------------------------------------


def test_the_published_marker_is_written_last_and_defines_completion(tmp_path, monkeypatch):
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    order: list[str] = []
    real_json, real_text = gen._write_json_immutable, gen._write_text_immutable
    monkeypatch.setattr(
        gen,
        "_write_json_immutable",
        lambda path, value, **kw: (order.append(path.rsplit("/", 1)[-1]), real_json(path, value))[1],
    )
    monkeypatch.setattr(
        gen,
        "_write_text_immutable",
        lambda path, text, **kw: (order.append(path.rsplit("/", 1)[-1]), real_text(path, text))[1],
    )
    gen.run_overfit100(config)
    assert order[-1] == gen.OVERFIT100_PUBLISHED_MARKER, order
    assert set(order[:-1]) == {"aggregation.json", "summary.csv", "summary.json"}

    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    state = gen.overfit100_publication_state(str(step_root))
    assert state["published"] is True and state["state"] == "published"
    marker = json.loads((step_root / gen.OVERFIT100_PUBLISHED_MARKER).read_text())
    assert marker["eval_pass_role"] == "s3_intermediate"
    assert marker["n_rows"] == 1
    assert marker["run_signature_sha256"] == gen.run_signature_sha256(
        json.loads((step_root / gen.OVERFIT100_STAGING_DIR / "correct/seed_0/ep100_v0_s00000.json").read_text())[
            "run_signature"
        ]
    )


def test_an_artifact_without_the_marker_is_not_completion(tmp_path):
    step_root = tmp_path / "step_002500_s3_intermediate"
    step_root.mkdir(parents=True)
    (step_root / "aggregation.json").write_text("{}")
    state = gen.overfit100_publication_state(str(step_root))
    assert state["published"] is False
    assert state["state"] == "partial_publication"
    assert state["artifacts_present"] == ["aggregation.json"]
    assert gen.overfit100_publication_state(str(tmp_path / "nothing_here"))["state"] == "fresh"


def test_a_mid_publication_preemption_is_repaired_from_staging(tmp_path, monkeypatch):
    # Publication only starts once the grid is complete, so staging holds every row. A crash after
    # aggregation.json (the exact case that used to brick a role dir) must republish the missing
    # artifacts from those same staged rows and then place the marker.
    episodes = [("fold cloth", 1), ("press button", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"

    boom = RuntimeError("preempted mid-publication")
    real_text = gen._write_text_immutable

    def _die_on_csv(path, text, **kw):
        if path.endswith("summary.csv"):
            raise boom
        return real_text(path, text, **kw)

    monkeypatch.setattr(gen, "_write_text_immutable", _die_on_csv)
    with pytest.raises(RuntimeError):
        gen.run_overfit100(config)
    assert (step_root / "aggregation.json").exists()
    assert not (step_root / "summary.csv").exists()
    assert not _published(step_root)
    aggregation_before = (step_root / "aggregation.json").read_bytes()

    monkeypatch.setattr(gen, "_write_text_immutable", real_text)
    _CALLS.clear()
    logged: list[str] = []
    monkeypatch.setattr(gen.max_logging, "log", lambda message: logged.append(str(message)))
    gen.run_overfit100(config)
    assert _CALLS == []  # rebuilt from staging; no rollout re-run
    assert (step_root / "aggregation.json").read_bytes() == aggregation_before  # unchanged
    assert (step_root / "summary.csv").exists() and (step_root / "summary.json").exists()
    assert _published(step_root)
    assert any("publication" in line.lower() for line in logged)


def test_publication_resume_refuses_when_staging_cannot_rebuild_the_grid(tmp_path, monkeypatch):
    # Fail-closed: a half-published directory whose staging was cleared cannot be repaired, and
    # recomputing would risk a different aux block silently replacing published evidence.
    episodes = [("fold cloth", 1), ("press button", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    real_text = gen._write_text_immutable
    monkeypatch.setattr(
        gen,
        "_write_text_immutable",
        lambda path, text, **kw: (
            (_ for _ in ()).throw(RuntimeError("boom"))
            if path.endswith("summary.csv")
            else real_text(path, text, **kw)
        ),
    )
    with pytest.raises(RuntimeError):
        gen.run_overfit100(config)
    monkeypatch.setattr(gen, "_write_text_immutable", real_text)
    shutil.rmtree(step_root / gen.OVERFIT100_STAGING_DIR)  # staging cleared by an operator

    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    message = str(ei.value)
    assert "publication" in message.lower()
    assert str(step_root) in message  # names exactly what to deal with
    assert not _published(step_root)


def test_a_foreign_writer_publishing_different_bytes_is_refused(tmp_path, monkeypatch):
    # Single-writer is enforced operationally (old jobs cancelled and confirmed dead before a
    # relaunch), but a marker that appears mid-run with DIFFERENT content is still refused by the
    # immutable writer rather than silently overwritten.
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    real_json = gen._write_json_immutable

    def _foreign_marker_appears(path, value, **kw):
        if path.endswith("aggregation.json"):
            marker = Path(step_root) / gen.OVERFIT100_PUBLISHED_MARKER
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"eval_pass_role": "s3_intermediate", "n_rows": 999}) + "\n")
        return real_json(path, value, **kw)

    monkeypatch.setattr(gen, "_write_json_immutable", _foreign_marker_appears)
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    assert gen.OVERFIT100_PUBLISHED_MARKER in str(ei.value)


def test_publication_state_is_probed_before_any_staging_interaction():
    import inspect

    src = inspect.getsource(gen.run_overfit100)
    assert src.index("overfit100_publication_state") < src.index("read_staged_rows")
    assert src.index("overfit100_publication_state") < src.index("write_staged_row")


# --------------------------------------------------------------------------------------
# Finding 4: enumeration must fail closed on listing errors; GCS marker semantics.
# --------------------------------------------------------------------------------------


def test_a_listing_error_during_enumeration_is_fatal(tmp_path, monkeypatch):
    root = tmp_path / "step_002500_s3_segment_final"
    _stage(root, _sample_row())

    def _walk_with_error(top, onerror=None, **kwargs):
        if onerror is not None:
            onerror(OSError("permission denied listing gs://bucket/staging_rows/correct"))
        return iter(())

    monkeypatch.setattr(gen.tf.io.gfile, "walk", _walk_with_error)
    with pytest.raises(ValueError) as ei:
        _read(root)
    assert "permission denied" in str(ei.value)


def test_gcs_zero_byte_directory_markers_are_tolerated_but_other_zero_byte_objects_are_not(tmp_path, monkeypatch):
    root = tmp_path / "step_002500_s3_segment_final"
    staging = str(root / gen.OVERFIT100_STAGING_DIR)
    real_path = _stage(root, _sample_row())

    def _walk_with_marker(top, onerror=None, **kwargs):
        # GCS represents some folders as zero-byte objects whose name ends with "/".
        yield (staging, ["correct"], ["marker/"])
        yield (f"{staging}/correct", ["seed_0"], [""])
        yield (f"{staging}/correct/seed_0", [], ["ep100_v0_s00000.json"])

    monkeypatch.setattr(gen.tf.io.gfile, "walk", _walk_with_marker)
    # ...and it really is zero bytes (finding 3: the size, not just the name, decides).
    monkeypatch.setattr(gen.tf.io.gfile, "stat", lambda path: SimpleNamespace(length=0))
    assert _read(root)  # markers skipped, the real row admitted
    assert Path(real_path).exists()

    def _walk_with_stray_zero_byte(top, onerror=None, **kwargs):
        yield (staging, ["correct"], ["_leftover"])
        yield (f"{staging}/correct/seed_0", [], ["ep100_v0_s00000.json"])

    monkeypatch.setattr(gen.tf.io.gfile, "walk", _walk_with_stray_zero_byte)
    with pytest.raises(ValueError) as ei:
        _read(root)
    assert "_leftover" in str(ei.value)


def test_enumeration_does_not_require_the_prefix_to_exist(tmp_path):
    # On GCS a prefix is not an object: an absent staging root must read as "nothing staged"
    # (the walk's NotFound on the root is tolerated), never as an error and never via an
    # exists() probe on a prefix.
    import inspect

    assert _read(tmp_path / "step_002500_s3_segment_final", windows=[]) == {}
    src = inspect.getsource(gen._enumerate_staging_files)
    assert "onerror=_rethrow" in src
    assert "tf.io.gfile.exists(root)" not in src


# --------------------------------------------------------------------------------------
# Finding 5: the aux-recovery guidance must describe the real procedure.
# --------------------------------------------------------------------------------------


def test_staging_error_text_does_not_promise_ceiling_recovery_by_clearing_staging(tmp_path):
    step_root = tmp_path / "step_002500_s3_segment_final"
    _stage(step_root, _sample_row(), signature=_build_signature(checkpoint_step=1000))
    with pytest.raises(ValueError) as ei:
        _read(step_root, signature=_build_signature())
    message = str(ei.value)
    assert str(step_root / gen.OVERFIT100_STAGING_DIR) in message
    assert "ceiling" not in message.lower()  # no false recovery promise in the staging error


def test_the_aux_recovery_note_points_at_the_backfill_not_at_staging():
    import inspect

    src = inspect.getsource(gen.run_overfit100)
    note = src[src.index("SCOPE OF THE PARITY CLAIM") : src.index("staged_rows: dict")]
    assert "backfill" in note
    # It must NOT tell the operator that clearing staging recovers a published row's ceiling.
    assert "clears the staging root and re-runs" not in note


# ======================================================================================
# (eval-resume pass 3) Complete runtime identity, an AUTHENTICATED marker, and a
# compare-only published mode.
# ======================================================================================


def test_signature_binds_the_effective_runtime_identity():
    signature = _build_signature()
    for field in (
        "num_train_timesteps",
        "use_qwix_quantization",
        "quantization",
        "qwix_module_path",
        "weight_quantization_calibration_method",
        "act_quantization_calibration_method",
        "bwd_quantization_calibration_method",
        "transformer_weight_source",
        "from_pt",
    ):
        assert field in signature, field
    assert set(signature) == {"schema"} | {name for name, _ in gen.OVERFIT100_RUN_SIGNATURE_TYPES}
    assert type(signature["use_qwix_quantization"]) is bool
    assert type(signature["num_train_timesteps"]) is int


@pytest.mark.parametrize(
    "field,bad",
    [
        ("num_train_timesteps", 500),  # scales every rollout timestep
        ("use_qwix_quantization", True),  # can replace the rollout graph entirely
        ("quantization", "fp8_full"),
        ("qwix_module_path", ".*transformer.*"),
        ("weight_quantization_calibration_method", "minmax"),
        ("act_quantization_calibration_method", "minmax"),
        ("bwd_quantization_calibration_method", "minmax"),
        ("transformer_weight_source", "/cache/snapshots/" + "e" * 40),
        ("from_pt", False),
    ],
)
def test_every_runtime_identity_field_is_enforced(tmp_path, field, bad):
    step_root = tmp_path / "step_002500_s3_segment_final"
    expected = _build_signature()
    _stage(step_root, _sample_row(), signature={**expected, field: bad})
    with pytest.raises(ValueError) as ei:
        _read(step_root, signature=expected)
    assert field in str(ei.value)


def test_resolved_checkpoint_dir_normalizes_trailing_slashes():
    # The comment promises that equivalent spellings agree; make that true rather than merely
    # claimed, so a trailing slash in the launcher cannot force a needless full recompute.
    plain = _full_signature_config(checkpoint_dir="gs://b/ck")
    slashed = _full_signature_config(checkpoint_dir="gs://b/ck/")
    assert gen._resolved_checkpoint_dir(plain) == gen._resolved_checkpoint_dir(slashed) == "gs://b/ck"
    fallback = _full_signature_config(checkpoint_dir="", output_dir="gs://b/", run_name="r")
    assert gen._resolved_checkpoint_dir(fallback) == "gs://b/r/checkpoints"


# --------------------------------------------------------------------------------------
# Finding 2: the marker must be authenticated, and published mode compare-only.
# --------------------------------------------------------------------------------------


def _publish_then(tmp_path, monkeypatch, *, episodes=None, role="s3_intermediate", seeds="0", modes="correct"):
    episodes = episodes or [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role=role, seeds=seeds, modes=modes)
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / f"step_002500_{role}"
    return config, step_root


def test_the_marker_records_what_it_authenticates(tmp_path, monkeypatch):
    _, step_root = _publish_then(tmp_path, monkeypatch)
    marker = json.loads((step_root / gen.OVERFIT100_PUBLISHED_MARKER).read_text())
    assert marker["schema"] == gen.OVERFIT100_PUBLISHED_MARKER_SCHEMA
    assert marker["eval_pass_role"] == "s3_intermediate"
    assert marker["checkpoint_step"] == 2500
    assert marker["n_rows"] == 1
    assert marker["manifest_sha256"] == json.loads((step_root / "aggregation.json").read_text())["manifest_sha256"]
    assert marker["aggregation_sha256"] == hashlib.sha256((step_root / "aggregation.json").read_bytes()).hexdigest()
    assert sorted(marker["artifacts"]) == sorted(gen.OVERFIT100_FINAL_ARTIFACTS)


@pytest.mark.parametrize(
    "content",
    ["", "   ", "{ not json", "[]", json.dumps({"schema": "foreign_v1"}), json.dumps({})],
    ids=["empty", "blank", "corrupt", "list", "foreign_schema", "no_fields"],
)
def test_an_unparseable_or_foreign_marker_hard_fails_at_entry(tmp_path, monkeypatch, content):
    config, step_root = _publish_then(tmp_path, monkeypatch)
    (step_root / gen.OVERFIT100_PUBLISHED_MARKER).write_text(content)
    before = _dir_snapshot(step_root)
    monkeypatch.setattr(gen, "_write_json", lambda path, value: pytest.fail(f"wrote {path} before validating"))
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    message = str(ei.value)
    assert gen.OVERFIT100_PUBLISHED_MARKER in message
    assert str(step_root) in message  # operator guidance names the directory
    assert _dir_snapshot(step_root) == before


def test_a_marker_valid_in_every_way_except_its_schema_tag_hard_fails(tmp_path, monkeypatch):
    # Isolates the SCHEMA check: everything else about this marker is correct, so nothing but the
    # tag can reject it (without this case the key-set check masks the schema check).
    config, step_root = _publish_then(tmp_path, monkeypatch)
    marker_path = step_root / gen.OVERFIT100_PUBLISHED_MARKER
    marker = json.loads(marker_path.read_text())
    marker["schema"] = "overfit100_eval_published_v9"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    before = _dir_snapshot(step_root)
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    assert "overfit100_eval_published_v9" in str(ei.value)
    assert _dir_snapshot(step_root) == before


def test_a_marker_with_an_extra_key_hard_fails(tmp_path, monkeypatch):
    # Isolates the KEY-SET check: schema, types and every bound value are correct, so only the
    # exact key set can reject this one.
    config, step_root = _publish_then(tmp_path, monkeypatch)
    marker_path = step_root / gen.OVERFIT100_PUBLISHED_MARKER
    marker = json.loads(marker_path.read_text())
    marker["published_by"] = "someone else"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    before = _dir_snapshot(step_root)
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    assert "published_by" in str(ei.value)
    assert _dir_snapshot(step_root) == before


@pytest.mark.parametrize(
    "field,bad",
    [
        ("eval_pass_role", "s3_full_set"),
        ("checkpoint_step", 1000),
        ("manifest_sha256", "b" * 64),
        ("n_rows", 99),
        ("aggregation_sha256", "c" * 64),
    ],
)
def test_a_marker_that_does_not_bind_this_run_hard_fails_at_entry(tmp_path, monkeypatch, field, bad):
    config, step_root = _publish_then(tmp_path, monkeypatch)
    marker_path = step_root / gen.OVERFIT100_PUBLISHED_MARKER
    marker = json.loads(marker_path.read_text())
    marker[field] = bad
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    before = _dir_snapshot(step_root)
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    assert field in str(ei.value)
    assert _dir_snapshot(step_root) == before


def test_a_marker_with_a_missing_final_artifact_hard_fails_before_any_write(tmp_path, monkeypatch):
    # A copied marker must not let an INCOMPLETE directory pass as published and then be silently
    # completed by the create-if-absent writers.
    config, step_root = _publish_then(tmp_path, monkeypatch)
    (step_root / "summary.csv").unlink()
    before = _dir_snapshot(step_root)
    monkeypatch.setattr(gen, "_write_json", lambda path, value: pytest.fail(f"wrote {path}"))
    monkeypatch.setattr(gen, "_write_text_immutable", lambda path, text: pytest.fail(f"wrote {path}"))
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    message = str(ei.value)
    assert "summary.csv" in message and gen.OVERFIT100_PUBLISHED_MARKER in message
    assert _dir_snapshot(step_root) == before


def test_published_mode_is_compare_only_and_never_creates(tmp_path, monkeypatch):
    # Even with every artifact present, published mode must COMPARE: the create-if-absent path of
    # the immutable writers has to be unreachable there.
    config, step_root = _publish_then(tmp_path, monkeypatch)
    before = _dir_snapshot(step_root)
    created: list[str] = []
    monkeypatch.setattr(gen, "_write_json", lambda path, value: created.append(path))
    gen.run_overfit100(config)
    assert created == []  # nothing written at all
    assert _dir_snapshot(step_root) == before

    # The compare_only WIRING is asserted against the source: with entry authentication now
    # rejecting a marker beside a missing artifact (pass-3 finding 2a), the driver can no longer
    # reach publication with anything absent, so no behavioural test can distinguish
    # compare_only=True from False there. It stays as defence in depth -- if entry validation ever
    # missed a state, compare-only is what still refuses to CREATE published evidence.
    import inspect

    src = inspect.getsource(gen.run_overfit100)
    assert "compare_only = writes_suppressed" in src
    assert "compare_only=compare_only" in src


def test_compare_only_writers_refuse_a_missing_artifact(tmp_path):
    path = str(tmp_path / "aggregation.json")
    with pytest.raises(ValueError) as ei:
        gen._write_json_immutable(path, {"a": 1}, compare_only=True)
    assert "aggregation.json" in str(ei.value)
    assert not Path(path).exists()
    gen._write_json_immutable(path, {"a": 1})  # normal mode creates
    gen._write_json_immutable(path, {"a": 1}, compare_only=True)  # identical bytes compare fine
    with pytest.raises(ValueError):
        gen._write_json_immutable(path, {"a": 2}, compare_only=True)

    text_path = str(tmp_path / "summary.csv")
    with pytest.raises(ValueError):
        gen._write_text_immutable(text_path, "a,b\n", compare_only=True)
    gen._write_text_immutable(text_path, "a,b\n")
    gen._write_text_immutable(text_path, "a,b\n", compare_only=True)


def test_a_partial_publication_is_still_repairable_after_the_marker_rules(tmp_path, monkeypatch):
    # Regression guard for the pass-2 behaviour: no marker + some artifacts is STILL the
    # publication-resume path, not a hard fail.
    episodes = [("fold cloth", 1), ("press button", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    real_text = gen._write_text_immutable
    monkeypatch.setattr(
        gen,
        "_write_text_immutable",
        lambda path, text, **kw: (
            (_ for _ in ()).throw(RuntimeError("boom"))
            if path.endswith("summary.csv")
            else real_text(path, text, **kw)
        ),
    )
    with pytest.raises(RuntimeError):
        gen.run_overfit100(config)
    monkeypatch.setattr(gen, "_write_text_immutable", real_text)
    _CALLS.clear()
    gen.run_overfit100(config)
    assert _CALLS == []
    assert _published(step_root)


# --------------------------------------------------------------------------------------
# Finding 3: trailing-slash objects are markers only when they are ZERO BYTES.
# --------------------------------------------------------------------------------------


def _walk_yielding(staging, entries):
    def _walk(top, onerror=None, **kwargs):
        for dirpath, dirnames, filenames in entries:
            yield (dirpath, dirnames, filenames)

    return _walk


def test_a_zero_byte_trailing_slash_object_is_a_directory_marker(tmp_path, monkeypatch):
    root = tmp_path / "step_002500_s3_segment_final"
    _stage(root, _sample_row())
    staging = str(root / gen.OVERFIT100_STAGING_DIR)
    monkeypatch.setattr(
        gen.tf.io.gfile,
        "walk",
        _walk_yielding(
            staging, [(staging, ["correct"], ["marker/"]), (f"{staging}/correct/seed_0", [], ["ep100_v0_s00000.json"])]
        ),
    )
    monkeypatch.setattr(gen.tf.io.gfile, "stat", lambda path: SimpleNamespace(length=0))
    assert _read(root)  # tolerated: a genuinely zero-byte placeholder


def test_a_nonzero_trailing_slash_object_is_refused(tmp_path, monkeypatch):
    # A foreign object whose name merely ends in "/" is NOT a GCS folder placeholder.
    root = tmp_path / "step_002500_s3_segment_final"
    _stage(root, _sample_row())
    staging = str(root / gen.OVERFIT100_STAGING_DIR)
    monkeypatch.setattr(
        gen.tf.io.gfile,
        "walk",
        _walk_yielding(
            staging,
            [(staging, ["correct"], ["not_a_marker/"]), (f"{staging}/correct/seed_0", [], ["ep100_v0_s00000.json"])],
        ),
    )
    monkeypatch.setattr(gen.tf.io.gfile, "stat", lambda path: SimpleNamespace(length=17))
    with pytest.raises(ValueError) as ei:
        _read(root)
    assert "not_a_marker/" in str(ei.value)


def test_a_stat_failure_on_a_candidate_marker_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "step_002500_s3_segment_final"
    _stage(root, _sample_row())
    staging = str(root / gen.OVERFIT100_STAGING_DIR)
    monkeypatch.setattr(
        gen.tf.io.gfile,
        "walk",
        _walk_yielding(
            staging, [(staging, ["correct"], ["marker/"]), (f"{staging}/correct/seed_0", [], ["ep100_v0_s00000.json"])]
        ),
    )

    def _boom(path):
        raise OSError("transient stat failure")

    monkeypatch.setattr(gen.tf.io.gfile, "stat", _boom)
    with pytest.raises(ValueError) as ei:
        _read(root)
    assert "transient stat failure" in str(ei.value)


# ======================================================================================
# (eval-resume pass 4) The two residuals: a canonical checkpoint identity in the FULL
# signature, and the true same-commit re-verification contract.
# ======================================================================================


def test_checkpoint_dir_slash_variants_produce_identical_signatures_end_to_end(tmp_path):
    # The BLOCKER, end to end: normalizing only `_resolved_checkpoint_dir` was not enough while the
    # raw string rode along as its own field -- exact admission then rejected a trailing-slash
    # retry. Whole signatures (and their hashes) must be equal, and each variant must admit the
    # other's staged rows.
    plain = _full_signature_config(checkpoint_dir="gs://b/ck")
    slashed = _full_signature_config(checkpoint_dir="gs://b/ck/")
    a = _build_signature(plain, resolved_checkpoint_dir=gen._resolved_checkpoint_dir(plain))
    b = _build_signature(slashed, resolved_checkpoint_dir=gen._resolved_checkpoint_dir(slashed))
    assert a == b
    assert gen.run_signature_sha256(a) == gen.run_signature_sha256(b)

    step_root = tmp_path / "step_002500_s3_segment_final"
    _stage(step_root, _sample_row(), signature=a)
    assert _read(step_root, signature=b) == {("ep100_v0_s00000", "correct", 0): _sample_row()}
    # ...and the reverse direction too.
    other_root = tmp_path / "other" / "step_002500_s3_segment_final"
    _stage(other_root, _sample_row(), signature=b)
    assert _read(other_root, signature=a)


def test_the_driver_admits_a_slash_variant_retrys_staged_rows(tmp_path, monkeypatch):
    # The same property through the real driver: a retry whose checkpoint_dir gained a trailing
    # slash must RESUME, not recompute.
    episodes = [("fold cloth", 1), ("press button", 1)]
    first = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    first.checkpoint_dir = str(tmp_path / "ck")
    gen.run_overfit100(first)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    shutil.rmtree(step_root / "aggregation.json", ignore_errors=True)
    for name in gen.OVERFIT100_FINAL_ARTIFACTS + (gen.OVERFIT100_PUBLISHED_MARKER,):
        (step_root / name).unlink()  # keep only staging: the preempted-retry shape

    second = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    second.checkpoint_dir = str(tmp_path / "ck") + "/"  # the only difference
    _CALLS.clear()
    gen.run_overfit100(second)
    assert _CALLS == []  # every row resumed despite the slash


def test_published_re_verification_is_same_commit_only(tmp_path, monkeypatch):
    # The MINOR, pinned as the REAL contract (not the false "cross-commit works" claim): published
    # mode regenerates aggregation.json, which embeds COMMIT, and byte-compares it -- so a newer
    # commit re-verifying a published directory REFUSES, and that refusal is the fail-closed intent.
    episodes = [("fold cloth", 1)]
    config = _resume_env(tmp_path, monkeypatch, episodes, role="s3_intermediate", seeds="0", modes="correct")
    gen.run_overfit100(config)
    step_root = tmp_path / "out" / "validation" / "step_002500_s3_intermediate"
    before = _dir_snapshot(step_root)

    monkeypatch.setenv("COMMIT", "d" * 40)  # a newer build re-verifying the same directory
    with pytest.raises(ValueError) as ei:
        gen.run_overfit100(config)
    assert "aggregation.json" in str(ei.value)
    assert _dir_snapshot(step_root) == before  # refused before mutating anything

    monkeypatch.setenv("COMMIT", "c" * 40)  # the SAME commit re-verifies cleanly
    gen.run_overfit100(config)
    assert _dir_snapshot(step_root) == before


def test_the_contract_text_states_same_commit_re_verification():
    import inspect

    marker_doc = inspect.getdoc(gen.overfit100_published_marker) or ""
    assert "same-signature" in marker_doc or "same commit" in marker_doc or "same-commit" in marker_doc
    # The retracted claim must be gone.
    assert "must still be able to verify" not in marker_doc
    assert "newer commit" in marker_doc  # it says plainly what a newer commit does: refuse

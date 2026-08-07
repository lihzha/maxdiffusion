"""exp_06 `rollout_adapter` — T1 `exp03-imports`: the pinned exp_03 import, characterized.

exp_06 needs exp_03's sampler and its RNG/support primitives, and plan-review F2 rejected the way
that would normally get them: a branch merge measured ~44k insertions with real conflicts. The
approved path (plan §5-1) is a **pinned-SHA blob import plus kernel extraction with equivalence
tests**, taking exp_03 @ ``2ef9b8a`` as a read-only source. This round imports and pins; the loss
kernel is T2's round and is deliberately absent.

**Oracle discipline: characterization, not artificial red** (plan §6, review F12). Nothing new is
being invented here, so there is no behaviour that *should* fail first. What these tests do instead
is nail down what arrived, so that anything which later moves — an edit to the copy, a re-pin
without a decision, a "harmless" tweak to the key arithmetic — fails loudly and specifically:

1. **Provenance.** The imported sampler's body hashes to the recorded pinned-blob hash under two
   independent digests (git's blob object id and sha256), the file's own header agrees with the
   module constants, and the recorded source SHA is the plan's ``2ef9b8a``.
2. **The sampler.** Its sigma grid, its per-step sigma consumption, its shape/dtype contract, its
   frame-0 pin, its determinism, and — the headline — **bitwise parity with exp_06's own deployed
   evaluator step** at both CFG branches in float32 and bfloat16. Plus the two properties T3a's
   gradient contract will stand on: the step is differentiable/rematable and free of host effects.
3. **The helpers.** The key derivation re-derived from first principles, its determinism, purpose
   scoping and **resume stability** (the key at step N is independent of how the run reached N),
   and the §3b support policy: one scalar draw per batch, legal range only, terminal ``sigma = 0``
   never reached.

**What deliberately does NOT transfer.** exp_03's "one sampler, one definition" AST guard asserts
its evaluator has no inline copy of the step. exp_06 has NOT rewired its evaluator, so that guard
would be false here. The property is replaced by :func:`test_the_deployed_evaluator_body_has_not_
drifted_from_the_copy_pinned_here` — a drift tripwire on the verbatim copy below, which is what
makes the parity claim self-verifying without importing the (heavy, unimportable) eval module.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import pos_rollout_support as support
from maxdiffusion.models.wan import overfit100_sampling as sampling
from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    apply_first_frame_pin,
)

_SAMPLER_PATH = Path(sampling.__file__).resolve()
_PACKAGE_ROOT = _SAMPLER_PATH.parents[2]  # .../src/maxdiffusion
_EVALUATOR_PATH = _PACKAGE_ROOT / "generate_wan_side_adapter.py"
_THIS_PATH = Path(__file__).resolve()

# The PRODUCTION rollout grid: 25 sampler steps, flow shift 5.0, sigma in [0, 1], 1000 train
# timesteps. Pinned rather than parametrized because the support's legal range and the "terminal
# sigma is never reached" claim are facts about THIS grid (plan §3b).
_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX, _NUM_TRAIN_TIMESTEPS = 25, 5.0, 0.0, 1.0, 1000
# Toy latent geometry. The sampler takes its shapes from ``z``, so nothing about the batch is baked
# in; what parity needs is the op sequence and the dtypes, not the sizes.
_B, _C, _F, _H, _W = 1, 4, 3, 4, 6
_GUIDE_SCALE = 5.0  # the deployed CFG scale (base_wan_5b_side_adapter.yml)


# =============================================================================================
# Fixtures: a deterministic stand-in for the frozen 5B and for the adapter forward.
# =============================================================================================


class _StubTransformer:
    """Reads every input the real velocity model reads, so a mis-built argument is observable."""

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, deterministic=True):
        b = hidden_states.shape[0]
        tokens = timestep.reshape(b, -1).astype(jnp.float32)
        t_scalar = jnp.mean(jnp.sin(tokens / 137.0), axis=-1)[:, None, None, None, None]
        ctx = jnp.mean(encoder_hidden_states.astype(jnp.float32))
        v = jnp.tanh(hidden_states.astype(jnp.float32) * 0.5) + t_scalar + 0.01 * ctx
        del deterministic
        return v.astype(hidden_states.dtype)


class _StubAdapters:
    def __init__(self, tag: float):
        self.tag = tag


def wan_action_adapter_forward(
    transformer, adapters, *, hidden_states, timestep, encoder_hidden_states, actions, deterministic
):
    """Stand-in for the real adapter forward, resolved by name inside the verbatim copy below."""
    base = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        deterministic=deterministic,
    )
    action_term = jnp.mean(actions.astype(jnp.float32)) * adapters.tag
    return (base.astype(jnp.float32) + 0.05 * action_term).astype(hidden_states.dtype)


class _Config:
    def __init__(self, guide_scale):
        self.side_adapter_guide_scale = guide_scale


def _grid():
    return sampling.overfit100_sampler_grid(
        num_inference_steps=_STEPS,
        flow_shift=_SHIFT,
        sigma_min=_SIGMA_MIN,
        sigma_max=_SIGMA_MAX,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
    )


def _inputs(dtype):
    key = jax.random.key(0)
    k_z, k_i0, k_ctx, k_act = jax.random.split(key, 4)
    z = jax.random.normal(k_z, (_B, _C, _F, _H, _W), dtype=jnp.float32).astype(dtype)
    z_i0 = jax.random.normal(k_i0, (_B, _C, 1, _H, _W), dtype=jnp.float32).astype(dtype)
    null_context = jax.random.normal(k_ctx, (_B, 7, 8), dtype=jnp.float32).astype(dtype)
    actions = jax.random.normal(k_act, (_B, 32, 7), dtype=jnp.float32).astype(dtype)
    return apply_first_frame_pin(z, z_i0), z_i0, null_context, actions


# ---------------------------------------------------------------------------------------------
# The reference: the loop body of ``generate_wan_side_adapter._rollout_sample``, copied VERBATIM
# from exp_06's own tree (src/maxdiffusion/generate_wan_side_adapter.py:136-158 at c07dc3f). Do not
# refactor it -- its value is that it is the deployed code, not that it is pretty. The eval module
# itself cannot be imported in a test venv (it pulls `transformers`), so the copy is kept honest by
# an AST drift tripwire against the file it was copied from.
# ---------------------------------------------------------------------------------------------


def _deployed_rollout_reference(*, transformer, adapters, sigmas, timesteps, null_context, z_i0, actions, config, z):
    b, _, f_lat, h_lat, w_lat = z.shape

    def _body(i, current):
        step_t = jnp.broadcast_to(timesteps[i], (b,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
        v_cond = wan_action_adapter_forward(
            transformer,
            adapters,
            hidden_states=current,
            timestep=timestep_2d,
            encoder_hidden_states=null_context,
            actions=actions,
            deterministic=True,
        )
        if abs(config.side_adapter_guide_scale - 1.0) > 1e-6:
            v_uncond = transformer(
                hidden_states=current,
                timestep=timestep_2d,
                encoder_hidden_states=null_context,
                deterministic=True,
            )
            v = v_uncond + config.side_adapter_guide_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return apply_first_frame_pin(current + (sigmas[i + 1] - sigmas[i]).astype(current.dtype) * v, z_i0)

    return jax.lax.fori_loop(0, _STEPS, _body, z)


def _imported_rollout(*, transformer, adapters, sigmas, timesteps, null_context, z_i0, actions, config, z):
    """The same rollout expressed through the IMPORTED step -- the thing under characterization."""

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        v_cond = wan_action_adapter_forward(
            transformer,
            adapters,
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            actions=actions,
            deterministic=True,
        )
        if abs(config.side_adapter_guide_scale - 1.0) > 1e-6:
            v_uncond = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=True,
            )
            return v_uncond + config.side_adapter_guide_scale * (v_cond - v_uncond)
        return v_cond

    return jax.lax.fori_loop(
        0,
        _STEPS,
        lambda i, current: sampling.overfit100_sampler_step(
            current,
            i,
            velocity_fn=velocity_fn,
            sigmas=sigmas,
            timesteps=timesteps,
            context=null_context,
            z_i0=z_i0,
        ),
        z,
    )


def _function_node(source: str, name: str, *, within: str | None = None) -> ast.FunctionDef:
    scope = ast.parse(source)
    if within is not None:
        scope = _function_node(source, within)
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name and node is not scope:
            return node
    raise AssertionError(f"{name} not found{'' if within is None else f' inside {within}'}")


# =============================================================================================
# 1. Provenance -- the pin is machine-checkable and fails loudly if the copy is edited.
# =============================================================================================


def _git_blob_sha1(body: str) -> str:
    raw = body.encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _imported_body() -> str:
    record = support.EXP03_IMPORTED_BLOBS["src/maxdiffusion/models/wan/overfit100_sampling.py"]
    content = _SAMPLER_PATH.read_text(encoding="utf-8")
    sentinel = record["header_sentinel"]
    assert content.count(sentinel) == 1, "the provenance header sentinel must appear exactly once"
    return content.split(sentinel, 1)[1]


def test_the_imported_sampler_body_is_byte_identical_to_the_pinned_blob():
    # Two independent digests over the same bytes: sha256, and git's own object id for the pinned
    # blob. Changing one byte of the copy breaks both; faking one still leaves the other.
    record = support.EXP03_IMPORTED_BLOBS["src/maxdiffusion/models/wan/overfit100_sampling.py"]
    body = _imported_body()
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == record["body_sha256"]
    assert _git_blob_sha1(body) == record["git_blob_sha1"]
    # ...and the copy really is the whole module, not a stub: the symbols the trainer rounds import.
    for name in ("overfit100_sampler_grid", "overfit100_sampler_step", "overfit100_euler_update"):
        assert f"def {name}(" in body


def test_the_recorded_source_commit_is_the_plans_pin():
    # The plan pins exp_03 @ 2ef9b8a (its last APPROVE+GO reviewed commit). Advancing it is a
    # RECORDED DECISION (plan §5-1), so the literal short SHA is asserted here, not read from code.
    assert support.EXP03_PLAN_PIN_SHORT == "2ef9b8a"
    assert support.EXP03_SOURCE_COMMIT.startswith(support.EXP03_PLAN_PIN_SHORT)
    assert len(support.EXP03_SOURCE_COMMIT) == 40
    assert set(support.EXP03_SOURCE_COMMIT) <= set("0123456789abcdef")
    assert support.EXP03_SOURCE_BRANCH == "claude-exp_03_rollout_objective-20260802"


def test_the_imported_file_carries_the_same_pin_in_its_own_header():
    # Three-way agreement: the computed hash, the module constant, and the header the file wears.
    # Editing the blob and "fixing" only the constant still fails here.
    record = support.EXP03_IMPORTED_BLOBS["src/maxdiffusion/models/wan/overfit100_sampling.py"]
    content = _SAMPLER_PATH.read_text(encoding="utf-8")
    header = content.split(record["header_sentinel"], 1)[0]
    assert support.EXP03_SOURCE_COMMIT in header
    assert record["git_blob_sha1"] in header
    assert record["body_sha256"] in header
    assert "src/maxdiffusion/models/wan/overfit100_sampling.py" in header
    assert "DO NOT EDIT" in header.upper()
    # The header is comments only -- it must not perturb the module it precedes.
    assert all(not line.strip() or line.lstrip().startswith("#") for line in header.splitlines())


def test_every_extracted_symbol_declares_where_it_came_from():
    # The helpers could NOT be blob-imported (they live inside exp_03's 935-line full-FT trainer),
    # so their provenance is a per-symbol record instead of a file hash.
    assert set(support.EXP03_EXTRACTED_SYMBOLS) == {
        "EXP03_AUX_PURPOSES",
        "EXP03_AUX_SEED_OFFSET",
        "_purpose_id",
        "exp03_aux_key",
        "rollout_support",
    }
    for name, record in support.EXP03_EXTRACTED_SYMBOLS.items():
        assert hasattr(support, name), f"{name} is declared imported but is not defined"
        assert record["source_path"] == "src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py"
        start, end = record["source_lines"]
        assert isinstance(start, int) and isinstance(end, int) and 0 < start <= end
        assert len(record["sha256"]) == 64 and set(record["sha256"]) <= set("0123456789abcdef")


def test_the_loss_kernel_is_not_in_this_round():
    # T2 owns `_rollout_loss`. Importing it here would silently widen the round and skip its
    # equivalence oracles; this pins the boundary the plan drew (plan §6).
    assert not hasattr(support, "_rollout_loss")
    assert "_rollout_loss" not in support.EXP03_EXTRACTED_SYMBOLS
    assert "_rollout_loss" not in _SAMPLER_PATH.read_text(encoding="utf-8")


# =============================================================================================
# 2. The sampler, characterized.
# =============================================================================================


def test_the_sigma_grid_is_the_pinned_production_rollout_grid():
    sigmas, timesteps = _grid()
    values = np.asarray(sigmas, dtype=np.float64)
    assert values.shape == (_STEPS + 1,)  # N+1 sigmas for N steps
    assert values[0] == 1.0 and values[-1] == 0.0  # starts at sigma_max, ends at exactly 0
    assert np.all(np.diff(values) < 0.0)  # strictly descending
    assert abs(values[-2] - 0.1724137931) < 1e-6  # the smallest POSITIVE sigma; no clamp needed
    assert np.asarray(timesteps).shape == (_STEPS,)
    assert np.array_equal(np.asarray(timesteps), np.asarray(sigmas[:-1] * _NUM_TRAIN_TIMESTEPS))


def test_the_step_consumes_exactly_one_sigma_interval_at_its_index():
    # The sigma-grid consumption contract: step i reads sigmas[i] and sigmas[i+1] and nothing else,
    # and the scale is cast to the LATENT's dtype before multiplying (bf16 is where a reordered
    # cast shows up first).
    sigmas, _ = _grid()
    z, z_i0, _, _ = _inputs(jnp.float32)
    v = jnp.full_like(z, 0.25)
    for index in range(_STEPS):
        got = sampling.overfit100_euler_update(z, v, sigmas, index, z_i0)
        want = apply_first_frame_pin(z + (sigmas[index + 1] - sigmas[index]).astype(z.dtype) * v, z_i0)
        assert np.array_equal(np.asarray(got), np.asarray(want)), index


def test_a_full_rollout_runs_exactly_n_steps_and_lands_on_the_terminal_sigma():
    # The step count contract: indices 0..N-1 are consumed, so the last interval ends at
    # sigmas[N] == 0 -- the boundary every support draw excludes.
    sigmas, _ = _grid()
    consumed = []
    z, z_i0, null_context, actions = _inputs(jnp.float32)
    transformer, adapters = _StubTransformer(), _StubAdapters(tag=0.7)

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        del timestep, encoder_hidden_states
        return jnp.zeros_like(hidden_states)

    _, timesteps = _grid()
    for index in range(_STEPS):
        consumed.append((index, index + 1))
        z = sampling.overfit100_sampler_step(
            z, index, velocity_fn=velocity_fn, sigmas=sigmas, timesteps=timesteps, context=null_context, z_i0=z_i0
        )
    del transformer, adapters
    assert consumed[0] == (0, 1) and consumed[-1] == (_STEPS - 1, _STEPS)
    assert float(sigmas[consumed[-1][1]]) == 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("guide_scale", [1.0, _GUIDE_SCALE])
def test_the_imported_step_reproduces_the_deployed_evaluator_rollout_bit_for_bit(dtype, guide_scale):
    """THE characterization: exp_06's deployed rollout, re-expressed through the imported step.

    25 chained steps, both CFG branches (1.0 takes the no-CFG shortcut; 5.0 is the deployed
    two-forward combination), float32 and bfloat16 (the production ``weights_dtype``). Exact
    equality, not ``allclose``: bitwise reproducibility of this path is the asset T3a's parity
    obligation is built on.
    """
    sigmas, timesteps = _grid()
    z, z_i0, null_context, actions = _inputs(dtype)
    kwargs = {
        "transformer": _StubTransformer(),
        "adapters": _StubAdapters(tag=0.7),
        "sigmas": sigmas,
        "timesteps": timesteps,
        "null_context": null_context,
        "z_i0": z_i0,
        "actions": actions,
        "config": _Config(guide_scale),
        "z": z,
    }
    want = _deployed_rollout_reference(**kwargs)
    got = _imported_rollout(**kwargs)
    assert got.dtype == want.dtype == dtype
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_the_deployed_evaluator_body_has_not_drifted_from_the_copy_pinned_here():
    # The parity test above compares against a VERBATIM copy of the evaluator's loop body. That copy
    # is only evidence while it still matches the file it came from, and the eval module cannot be
    # imported in a test venv -- so the tripwire is structural: unparse both bodies and compare.
    deployed = _function_node(_EVALUATOR_PATH.read_text(encoding="utf-8"), "_body", within="_rollout_sample")
    reference = _function_node(_THIS_PATH.read_text(encoding="utf-8"), "_body", within="_deployed_rollout_reference")
    assert ast.unparse(deployed) == ast.unparse(reference)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_step_preserves_the_shape_and_dtype_contract_and_pins_frame_zero(dtype):
    sigmas, timesteps = _grid()
    z, z_i0, null_context, _ = _inputs(dtype)

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        del timestep, encoder_hidden_states
        return jnp.ones_like(hidden_states)

    got = sampling.overfit100_sampler_step(
        z, 3, velocity_fn=velocity_fn, sigmas=sigmas, timesteps=timesteps, context=null_context, z_i0=z_i0
    )
    assert got.shape == z.shape
    assert got.dtype == dtype
    # Latent frame 0 is the image condition -- pinned after the update, exactly.
    assert np.array_equal(np.asarray(got[:, :, :1]), np.asarray(z_i0))


def test_the_step_is_deterministic_under_a_fixed_key():
    # It draws no randomness of its own: the same inputs give bitwise the same output, and no
    # `jax.random` call exists anywhere in the module.
    sigmas, timesteps = _grid()
    z, z_i0, null_context, actions = _inputs(jnp.float32)
    transformer, adapters = _StubTransformer(), _StubAdapters(tag=0.7)
    kwargs = {
        "transformer": transformer,
        "adapters": adapters,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "null_context": null_context,
        "z_i0": z_i0,
        "actions": actions,
        "config": _Config(_GUIDE_SCALE),
        "z": z,
    }
    first = _imported_rollout(**kwargs)
    again = _imported_rollout(**kwargs)
    assert np.array_equal(np.asarray(first), np.asarray(again))
    assert "jax.random" not in _SAMPLER_PATH.read_text(encoding="utf-8")


def test_the_euler_update_implements_the_contraction_the_loss_kernels_will_use():
    # z_next - z_gt = (sigma_next / sigma) * (z - z_gt) under v* = (z - z_gt)/sigma. T2's kernels are
    # derived from this identity, so it is pinned before they are written.
    sigmas, _ = _grid()
    z, z_i0, _, _ = _inputs(jnp.float32)
    z_gt = apply_first_frame_pin(jnp.zeros_like(z), z_i0)
    index = 3
    sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
    z_next = sampling.overfit100_euler_update(z, (z - z_gt) / sigma, sigmas, index, z_i0)
    assert np.allclose(np.asarray(z_next), np.asarray(z_gt + (sigma_next / sigma) * (z - z_gt)), atol=1e-6)


def test_the_step_is_differentiable_and_rematable():
    # T3a unrolls this under jax.grad with jax.remat; without this the CFG gradient contract has no
    # substrate. Gradients must flow through CHAINED steps, not just one.
    sigmas, timesteps = _grid()
    z, z_i0, null_context, _ = _inputs(jnp.float32)

    def loss(scale_value):
        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            del timestep, encoder_hidden_states
            return hidden_states * scale_value

        z1 = sampling.overfit100_sampler_step(
            z, 0, velocity_fn=velocity_fn, sigmas=sigmas, timesteps=timesteps, context=null_context, z_i0=z_i0
        )
        z2 = sampling.overfit100_sampler_step(
            z1, 1, velocity_fn=velocity_fn, sigmas=sigmas, timesteps=timesteps, context=null_context, z_i0=z_i0
        )
        return jnp.sum(z2**2)

    grad = jax.grad(loss)(jnp.asarray(0.3, dtype=jnp.float32))
    assert np.isfinite(float(grad)) and float(grad) != 0.0
    assert np.allclose(float(jax.grad(jax.remat(loss))(jnp.asarray(0.3, dtype=jnp.float32))), float(grad), rtol=1e-5)


def test_the_sampler_module_is_side_effect_free_and_independent_of_the_evaluator():
    # It will be scanned and rematted inside a compiled train step, so nothing may talk to the host;
    # and a trainer must be able to take the sampler without dragging in the eval stack.
    source = _SAMPLER_PATH.read_text(encoding="utf-8")
    for forbidden in ("print(", "max_logging", "jax.debug", "open(", "np.save", "logging."):
        assert forbidden not in source, forbidden
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported, "no imports parsed -- the AST walk is not looking at the right file"
    for name in imported:
        assert "generate_wan_side_adapter" not in name, name
        assert not name.startswith(("tensorflow", "orbax")), name


def test_the_step_keeps_its_state_explicit():
    # "explicit args in, state out": no module state, no config object -- what makes it scannable.
    signature = inspect.signature(sampling.overfit100_sampler_step)
    assert list(signature.parameters)[:2] == ["z", "index"]
    for required in ("velocity_fn", "sigmas", "timesteps", "context", "z_i0"):
        assert signature.parameters[required].kind is inspect.Parameter.KEYWORD_ONLY
    assert "config" not in signature.parameters and "state" not in signature.parameters
    assert sampling.OVERFIT100_N_HIST == 1  # frame 0 is the condition and carries timestep 0


# =============================================================================================
# 3. The RNG helper, characterized (plan §3b).
# =============================================================================================


def _key_bits(key) -> np.ndarray:
    return np.asarray(jax.random.key_data(key))


def test_the_key_derivation_is_exactly_the_pinned_arithmetic():
    # Re-derived from first principles, not copied: key(seed + 1_000_003) folded with the step, then
    # with sha256(purpose_name)[:4] big-endian. Any change to the offset, the fold order, the digest
    # or its byte width lands here.
    for seed, step, purpose in ((0, 0, "index_support_rollout"), (7, 250, "p_ss_coin"), (3, 12_499, "index_support")):
        expected = jax.random.key(seed + 1_000_003)
        expected = jax.random.fold_in(expected, jnp.asarray(step, dtype=jnp.uint32))
        expected = jax.random.fold_in(
            expected, int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:4], "big")
        )
        got = support.exp03_aux_key(seed=seed, global_step=step, purpose=purpose)
        assert np.array_equal(_key_bits(got), _key_bits(expected)), (seed, step, purpose)
    assert support.EXP03_AUX_SEED_OFFSET == 1_000_003


def test_the_key_is_deterministic_purpose_scoped_and_step_keyed():
    first = support.exp03_aux_key(seed=0, global_step=10, purpose="index_support_rollout")
    assert np.array_equal(
        _key_bits(first), _key_bits(support.exp03_aux_key(seed=0, global_step=10, purpose="index_support_rollout"))
    )
    variants = {
        "step": support.exp03_aux_key(seed=0, global_step=11, purpose="index_support_rollout"),
        "purpose": support.exp03_aux_key(seed=0, global_step=10, purpose="index_support"),
        "seed": support.exp03_aux_key(seed=1, global_step=10, purpose="index_support_rollout"),
    }
    for label, key in variants.items():
        assert not np.array_equal(_key_bits(first), _key_bits(key)), label


def test_the_key_at_a_step_is_independent_of_how_the_run_reached_it():
    """Resume stability (plan §3b) -- the property a preempted-and-restarted arm depends on."""
    target = 250
    fresh = _key_bits(support.exp03_aux_key(seed=0, global_step=target, purpose="index_support_rollout"))

    # (a) after walking every earlier step in order, consuming each key
    for step in range(target):
        jax.random.normal(support.exp03_aux_key(seed=0, global_step=step, purpose="index_support_rollout"), (2,))
    walked = _key_bits(support.exp03_aux_key(seed=0, global_step=target, purpose="index_support_rollout"))
    # (b) after an out-of-order path (a resume replays a different prefix)
    for step in (900, 3, 41, 7, 12):
        jax.random.normal(support.exp03_aux_key(seed=0, global_step=step, purpose="p_ss_coin"), (2,))
    jumped = _key_bits(support.exp03_aux_key(seed=0, global_step=target, purpose="index_support_rollout"))
    assert np.array_equal(fresh, walked) and np.array_equal(fresh, jumped)

    # ...and structurally: a split-based derivation would be path-dependent, so no split may appear,
    # and the function may hold no process state at all -- module-level mutable state (a call
    # counter, a memoized "first step this process saw") is the classic resume-unsafe pattern and is
    # invisible to any single-process check.
    node = _function_node(Path(support.__file__).read_text(encoding="utf-8"), "exp03_aux_key")
    splits = [
        child for child in ast.walk(node) if isinstance(child, ast.Call) and getattr(child.func, "attr", "") == "split"
    ]
    assert not splits, "exp03_aux_key must derive, never split -- a split is path-dependent"
    assert not [child for child in ast.walk(node) if isinstance(child, ast.Global)]
    assert not [child for child in ast.walk(node) if isinstance(child, ast.Nonlocal)]


def _freshly_loaded_support():
    """The module re-executed into a BRAND-NEW namespace -- a process restart, minus the process.

    ``importlib.reload`` will not do: it re-executes into the existing module dict, so any runtime
    state the module body does not itself reassign survives the "restart". This builds a new module
    object each call, so module-level state really is gone.
    """
    spec = importlib.util.spec_from_file_location("_restarted_pos_rollout_support", Path(support.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("prefix", ["cold", "walk", "resume"])
def test_the_key_survives_a_restart_that_reached_the_step_differently(prefix):
    """The resume property proper: a RESTARTED run must reproduce the key at global step 250.

    The single-process check above cannot see the failure that matters. A derivation which memoizes
    the first step it ever saw (``base = global_step`` on the first call, then folds
    ``global_step - base``) is perfectly stable inside one process and silently wrong after a
    restart -- exactly the situation a preempted TPU job is in, and exactly the state plan §3b's
    "resume-stable" clause exists to forbid. So: three freshly loaded modules -- one cold, one that
    walked 0..249, one that resumed at 200 -- must all agree bit for bit with each other and with
    the long-lived import.
    """
    restarted = _freshly_loaded_support()
    if prefix == "walk":
        walk = range(250)
    elif prefix == "resume":
        walk = range(200, 250)
    else:
        walk = ()
    for step in walk:
        restarted.exp03_aux_key(seed=0, global_step=step, purpose="index_support_rollout")
    got = restarted.exp03_aux_key(seed=0, global_step=250, purpose="index_support_rollout")
    expected = support.exp03_aux_key(seed=0, global_step=250, purpose="index_support_rollout")
    assert np.array_equal(_key_bits(got), _key_bits(expected)), prefix


def test_the_key_is_tracer_safe_and_agrees_with_eager():
    # It is computed INSIDE the compiled train step, where the global step is a tracer.
    jitted = jax.jit(lambda value: support.exp03_aux_key(seed=0, global_step=value, purpose="index_support_rollout"))
    for step in (0, 250, 12_499):
        eager = support.exp03_aux_key(seed=0, global_step=step, purpose="index_support_rollout")
        assert np.array_equal(_key_bits(eager), _key_bits(jitted(jnp.asarray(step, dtype=jnp.int32)))), step


def test_purpose_ids_are_name_hashed_and_undeclared_purposes_are_rejected():
    ids = {purpose: support._purpose_id(purpose) for purpose in support.EXP03_AUX_PURPOSES}
    assert len(set(ids.values())) == len(ids)
    for purpose, value in ids.items():
        assert value == int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:4], "big")
    # Name-hashed ids are what makes a future purpose (e.g. exp_06's epsilon at T3b) purely additive.
    assert "index_support_rollout" in support.EXP03_AUX_PURPOSES
    with pytest.raises(ValueError):
        support._purpose_id("undeclared_purpose")
    with pytest.raises(ValueError):
        support.exp03_aux_key(seed=0, global_step=0, purpose="epsilon")


def test_the_auxiliary_root_is_not_the_training_stream_root():
    # The shared stream is key(seed + 1); the auxiliary root must be a different key entirely, or an
    # arm's extra draws would perturb the batch stream matched-C0 shares with R-B (plan §3b).
    assert support.EXP03_AUX_SEED_OFFSET != 1
    assert not np.array_equal(
        _key_bits(jax.random.key(0 + 1)), _key_bits(jax.random.key(0 + support.EXP03_AUX_SEED_OFFSET))
    )


def test_a_concrete_negative_global_step_is_rejected():
    with pytest.raises(ValueError):
        support.exp03_aux_key(seed=0, global_step=-1, purpose="index_support_rollout")


def test_the_salt_moves_only_the_salted_purpose_and_zero_is_the_unsalted_draw():
    unsalted = support.exp03_aux_key(seed=0, global_step=5, purpose="index_support_rollout")
    assert np.array_equal(
        _key_bits(unsalted),
        _key_bits(support.exp03_aux_key(seed=0, global_step=5, purpose="index_support_rollout", salt=0)),
    )
    assert not np.array_equal(
        _key_bits(unsalted),
        _key_bits(support.exp03_aux_key(seed=0, global_step=5, purpose="index_support_rollout", salt=9)),
    )
    other = support.exp03_aux_key(seed=0, global_step=5, purpose="p_ss_coin")
    assert np.array_equal(
        _key_bits(other), _key_bits(support.exp03_aux_key(seed=0, global_step=5, purpose="p_ss_coin", salt=0))
    )


# =============================================================================================
# 4. The support draw, characterized against plan §3b.
# =============================================================================================


def test_the_rollout_support_is_exactly_the_named_keyed_draw():
    # Not a frequency window: bit-for-bit the randint an independent caller gets from the NAMED key
    # with the pinned bounds. A biased mapping, a reused purpose or a shifted range fails here.
    for step in (0, 1, 7, 250, 12_499):
        expected = jax.random.randint(
            support.exp03_aux_key(seed=0, global_step=step, purpose="index_support_rollout"), (), 0, _STEPS - 2
        )
        start, end = support.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=2)
        assert int(start) == int(expected), step
        assert int(end) == int(expected) + 2, step
    # ...and it uses ITS OWN purpose: sharing A's start key would silently correlate the arms.
    assert not np.array_equal(
        _key_bits(support.exp03_aux_key(seed=0, global_step=7, purpose="index_support")),
        _key_bits(support.exp03_aux_key(seed=0, global_step=7, purpose="index_support_rollout")),
    )


@pytest.mark.parametrize("k_b", [1, 2, 4])
def test_the_rollout_support_stays_inside_the_legal_grid_range(k_b):
    # plan §3b, 1-based: sigma_hi uniform over {1 .. N-k}. 0-based (the pinned construction): start
    # in {0 .. N-1-k}, walking consecutively to end = start + k <= N-1.
    draws = [
        tuple(int(v) for v in support.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=k_b))
        for step in range(3000)
    ]
    for start, end in draws:
        assert end == start + k_b
        assert 0 <= start <= _STEPS - 1 - k_b
        assert end <= _STEPS - 1
    assert {start for start, _ in draws} == set(range(_STEPS - k_b))  # every legal start, no other


def test_the_rollout_support_never_starts_or_ends_on_the_terminal_zero_sigma():
    # The terminal boundary is excluded by CONSTRUCTION, not by a clamp: index N carries sigma 0,
    # and dividing an endpoint loss by (sigma_hi - sigma_lo)^2 at a zero sigma is what T2 must never
    # be handed.
    sigmas, _ = _grid()
    assert float(sigmas[_STEPS]) == 0.0
    assert float(sigmas[_STEPS - 1]) > 0.0
    for step in range(1000):
        start, end = support.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=2)
        assert int(start) != _STEPS and int(end) != _STEPS
        assert float(sigmas[int(start)]) > 0.0 and float(sigmas[int(end)]) > 0.0


def test_the_rollout_support_is_one_scalar_draw_per_batch():
    # plan §3b pins per-BATCH granularity. Structurally: the draw is a scalar and the signature has
    # no batch or per-example argument, so every example of a batch shares one support by
    # construction -- a per-example variant could not be expressed without changing this signature.
    start, end = support.rollout_support(seed=0, global_step=11, num_steps=_STEPS, k_b=2)
    assert jnp.ndim(start) == 0 and jnp.ndim(end) == 0
    assert jnp.shape(start) == () and jnp.shape(end) == ()
    parameters = inspect.signature(support.rollout_support).parameters
    assert set(parameters) == {"seed", "global_step", "num_steps", "k_b", "support_salt"}
    for name in parameters:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert not {"batch", "batch_size", "b", "per_example", "shape"} & set(parameters)


def test_the_rollout_support_is_resume_and_accumulation_stable():
    # Resume: the support at a step does not depend on the draws that preceded it. Accumulation:
    # every microbatch of one optimizer step shares that step's global_step, hence one support.
    fresh = tuple(int(v) for v in support.rollout_support(seed=0, global_step=41, num_steps=_STEPS, k_b=2))
    for step in range(200):
        support.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=2)
    after = tuple(int(v) for v in support.rollout_support(seed=0, global_step=41, num_steps=_STEPS, k_b=2))
    assert after == fresh
    microbatches = {
        tuple(int(v) for v in support.rollout_support(seed=0, global_step=41, num_steps=_STEPS, k_b=2))
        for _ in range(8)
    }
    assert microbatches == {fresh}


def test_the_rollout_support_is_tracer_safe():
    jitted = jax.jit(lambda value: support.rollout_support(seed=0, global_step=value, num_steps=_STEPS, k_b=2))
    for step in (0, 250, 9999):
        eager = support.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=2)
        traced = jitted(jnp.asarray(step, dtype=jnp.int32))
        assert tuple(int(v) for v in traced) == tuple(int(v) for v in eager), step


def test_the_support_salt_redraws_only_the_support():
    base = tuple(int(v) for v in support.rollout_support(seed=0, global_step=17, num_steps=_STEPS, k_b=2))
    assert base == tuple(
        int(v) for v in support.rollout_support(seed=0, global_step=17, num_steps=_STEPS, k_b=2, support_salt=0)
    )
    salted = [
        tuple(int(v) for v in support.rollout_support(seed=0, global_step=17, num_steps=_STEPS, k_b=2, support_salt=s))
        for s in range(1, 12)
    ]
    assert any(draw != base for draw in salted), "a non-zero salt must be able to move the support"

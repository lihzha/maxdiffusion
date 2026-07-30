"""CPU-only tests for the exp_02 overfit100 text table (plan D8 / F5).

D8's contract: encode the ``num_text_slots`` POSITIVE prompts only, in a bounded loop
(``text_encode_batch``, default 8), replicating ``WanPipeline.encode_prompt``'s positive
branch EXACTLY (``_get_t5_prompt_embeds``: tokenize to ``max_sequence_length``, mask
through UMT5, truncate to the true length, zero-pad; then the f32 conversion), with **no
negative prompts encoded**, the resulting ``[N, L, 4096]`` table in weights dtype,
replicated, and its byte total in the startup memory audit (the 400 MiB is predeclared).

Pinned here:
  (A) ``episodes.json`` contract -- exactly ``num_text_slots`` entries with
      index-CONTIGUOUS ``episode_index`` 0..N-1; count / gap / duplicate all raise with
      the directory named.
  (B) bounded-loop batching -- chunk sizes never exceed ``text_encode_batch``, the loop
      result is byte-identical to a one-shot encode, and the table is invariant to the
      ORDER entries appear in ``episodes.json`` (row i is always episode_index i).
  (C) no negatives -- ``encode_prompt`` is booby-trapped and the recorded prompt lists
      contain exactly the instructions (never the ``""`` negative the pipeline's
      negative branch would encode).
  (D) dtype/shape/replication + the emitted bytes-audit line.
  (E) the torch -> f32 conversion path is the pipeline's own
      (``.detach().float().numpy()``), proven with a real torch tensor.

CPU-only: a stub pipeline whose ``_get_t5_prompt_embeds`` is a deterministic function of
each prompt string -- no T5, no weights, no tokenizer. The darwin grain import stub lives
in ``conftest.py``.
"""

from __future__ import annotations

import json
import zlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100
from maxdiffusion import max_logging

_LEN, _DIM = 4, 8  # stub T5 geometry (real run: 512 x 4096)


def _stub_embedding(prompt: str) -> np.ndarray:
    """Deterministic, prompt-dependent embedding: row content depends ONLY on the text."""
    seed = zlib.crc32(prompt.encode("utf-8"))
    return np.random.default_rng(seed).standard_normal((_LEN, _DIM)).astype(np.float32)


class _StubPipeline:
    """Records every positive-branch call; ``encode_prompt`` is a booby trap."""

    def __init__(self, *, as_torch=False):
        self.calls: list[list[str]] = []
        self.as_torch = as_torch

    def _get_t5_prompt_embeds(self, prompt=None, num_videos_per_prompt=1, max_sequence_length=None):
        assert num_videos_per_prompt == 1
        assert max_sequence_length == _LEN
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        self.calls.append(prompts)
        stacked = np.stack([_stub_embedding(p) for p in prompts], axis=0)
        if self.as_torch:
            import torch

            return torch.from_numpy(stacked.copy()).to(dtype=torch.bfloat16)
        return stacked

    def encode_prompt(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("encode_prompt() encodes a NEGATIVE prompt; D8 forbids it")


def _episodes_json(directory, texts, *, indices=None, order=None):
    directory.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(texts))) if indices is None else list(indices)
    episodes = [
        {"episode_index": index, "episode_id": 25189 + index, "used_text": text, "n_windows": 16}
        for index, text in zip(indices, texts)
    ]
    if order is not None:
        episodes = [episodes[i] for i in order]
    (directory / "episodes.json").write_text(json.dumps({"episodes": episodes}, indent=2) + "\n")
    return episodes


def _texts(n):
    return [f"pick up the block number {i}" for i in range(n)]


def _cpu_mesh():
    device = jax.devices()[0]
    return jax.sharding.Mesh(np.array([device]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))


def _config(data_dir, *, num_text_slots, text_encode_batch=8, weights_dtype="float32"):
    return SimpleNamespace(
        train_data_dir=str(data_dir),
        num_text_slots=num_text_slots,
        text_encode_batch=text_encode_batch,
        wan_max_sequence_length=_LEN,
        text_dim=_DIM,
        weights_dtype=weights_dtype,
        logical_axis_rules=(),
    )


def _build(directory, *, num_text_slots, text_encode_batch=8, weights_dtype="float32", as_torch=False):
    pipeline = _StubPipeline(as_torch=as_torch)
    trainer = overfit100.WanTI2VOverfit100Trainer(
        _config(
            directory,
            num_text_slots=num_text_slots,
            text_encode_batch=text_encode_batch,
            weights_dtype=weights_dtype,
        )
    )
    table = trainer._build_context_table(pipeline, _cpu_mesh())
    return table, pipeline


# =======================================================================================
# (A) episodes.json contract
# =======================================================================================


def test_reads_texts_in_episode_index_order(tmp_path):
    texts = _texts(4)
    _episodes_json(tmp_path, texts)
    assert overfit100.read_episode_texts(str(tmp_path), 4) == texts


def test_order_in_the_file_does_not_matter(tmp_path):
    texts = _texts(4)
    _episodes_json(tmp_path, texts, order=[2, 0, 3, 1])
    assert overfit100.read_episode_texts(str(tmp_path), 4) == texts


def test_count_mismatch_raises_naming_the_dir(tmp_path):
    _episodes_json(tmp_path, _texts(4))
    with pytest.raises(ValueError) as ei:
        overfit100.read_episode_texts(str(tmp_path), 10)
    msg = str(ei.value)
    assert str(tmp_path) in msg and "10" in msg and "4" in msg


def test_non_contiguous_indices_raise(tmp_path):
    _episodes_json(tmp_path, _texts(3), indices=[0, 1, 3])  # gap at 2
    with pytest.raises(ValueError) as ei:
        overfit100.read_episode_texts(str(tmp_path), 3)
    assert str(tmp_path) in str(ei.value)


def test_duplicate_indices_raise(tmp_path):
    _episodes_json(tmp_path, _texts(3), indices=[0, 1, 1])
    with pytest.raises(ValueError):
        overfit100.read_episode_texts(str(tmp_path), 3)


def test_missing_episodes_json_raises_naming_the_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError) as ei:
        overfit100.read_episode_texts(str(tmp_path / "empty"), 3)
    assert "episodes.json" in str(ei.value)
    assert str(tmp_path / "empty") in str(ei.value)


def test_empty_instruction_raises(tmp_path):
    _episodes_json(tmp_path, ["a good instruction", ""])
    with pytest.raises(ValueError):
        overfit100.read_episode_texts(str(tmp_path), 2)


# =======================================================================================
# (B) bounded-loop batching + parity + order invariance
# =======================================================================================


def test_bounded_loop_chunks_never_exceed_text_encode_batch(tmp_path):
    _episodes_json(tmp_path, _texts(20))
    table, pipeline = _build(tmp_path, num_text_slots=20, text_encode_batch=8)
    assert [len(call) for call in pipeline.calls] == [8, 8, 4]
    assert table.shape == (20, _LEN, _DIM)


def test_loop_result_is_byte_identical_to_one_shot(tmp_path):
    _episodes_json(tmp_path, _texts(20))
    looped, looped_pipeline = _build(tmp_path, num_text_slots=20, text_encode_batch=8)
    one_shot, one_pipeline = _build(tmp_path, num_text_slots=20, text_encode_batch=20)
    assert len(looped_pipeline.calls) == 3 and len(one_pipeline.calls) == 1  # non-vacuity
    np.testing.assert_array_equal(np.asarray(looped), np.asarray(one_shot))


def test_table_is_invariant_to_episodes_json_order(tmp_path):
    texts = _texts(6)
    _episodes_json(tmp_path / "a", texts)
    _episodes_json(tmp_path / "b", texts, order=[5, 3, 1, 0, 4, 2])
    table_a, _ = _build(tmp_path / "a", num_text_slots=6, text_encode_batch=4)
    table_b, _ = _build(tmp_path / "b", num_text_slots=6, text_encode_batch=4)
    np.testing.assert_array_equal(np.asarray(table_a), np.asarray(table_b))


def test_row_i_is_the_embedding_of_episode_index_i(tmp_path):
    texts = _texts(6)
    _episodes_json(tmp_path, texts, order=[4, 2, 0, 5, 1, 3])
    table, _ = _build(tmp_path, num_text_slots=6, text_encode_batch=4)
    for index, text in enumerate(texts):
        np.testing.assert_array_equal(np.asarray(table[index]), _stub_embedding(text))
    # Rows genuinely differ (the fixture is not degenerate).
    assert not np.allclose(np.asarray(table[0]), np.asarray(table[1]))


def test_text_encode_batch_of_one_still_produces_the_same_table(tmp_path):
    _episodes_json(tmp_path, _texts(5))
    per_one, pipeline_one = _build(tmp_path, num_text_slots=5, text_encode_batch=1)
    batched, _ = _build(tmp_path, num_text_slots=5, text_encode_batch=5)
    assert [len(c) for c in pipeline_one.calls] == [1, 1, 1, 1, 1]
    np.testing.assert_array_equal(np.asarray(per_one), np.asarray(batched))


# =======================================================================================
# (C) no negative prompts encoded
# =======================================================================================


def test_only_positive_prompts_are_encoded(tmp_path):
    texts = _texts(9)
    _episodes_json(tmp_path, texts)
    _, pipeline = _build(tmp_path, num_text_slots=9, text_encode_batch=4)
    encoded = [prompt for call in pipeline.calls for prompt in call]
    assert encoded == texts  # exactly the instructions, in index order
    assert "" not in encoded  # the negative branch's empty prompt never appears
    assert len(encoded) == 9


def test_encode_prompt_is_never_called(tmp_path):
    # The booby trap in _StubPipeline.encode_prompt would raise; reaching a table proves
    # the positive branch was replicated rather than encode_prompt (which ALSO encodes a
    # negative prompt and would double the T5 work).
    _episodes_json(tmp_path, _texts(3))
    table, _ = _build(tmp_path, num_text_slots=3, text_encode_batch=2)
    assert table.shape == (3, _LEN, _DIM)


# =======================================================================================
# (D) dtype / shape / replication / audit line
# =======================================================================================


def test_table_dtype_follows_weights_dtype_and_is_replicated(tmp_path):
    _episodes_json(tmp_path, _texts(4))
    table, _ = _build(tmp_path, num_text_slots=4, text_encode_batch=2, weights_dtype="bfloat16")
    assert table.dtype == jnp.bfloat16
    assert table.shape == (4, _LEN, _DIM)
    assert table.sharding.is_fully_replicated


def test_shape_mismatch_against_config_geometry_raises(tmp_path):
    _episodes_json(tmp_path, _texts(3))
    pipeline = _StubPipeline()
    config = _config(tmp_path, num_text_slots=3)
    config.text_dim = _DIM + 1  # the config promises a different embedding width
    with pytest.raises(ValueError, match="text_dim|shape"):
        overfit100.WanTI2VOverfit100Trainer(config)._build_context_table(pipeline, _cpu_mesh())


def test_context_table_audit_line_reports_bytes_and_geometry():
    table = jnp.zeros((100, 512, 4096), dtype=jnp.bfloat16)
    line = overfit100.context_table_audit_line(table)
    assert "400.0" in line  # 100 x 512 x 4096 x 2 B = 400 MiB, predeclared by D8/F5
    assert "MiB" in line
    assert "bfloat16" in line
    assert "(100, 512, 4096)" in line


def test_build_context_table_emits_the_audit_line(tmp_path, monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr(max_logging, "log", lambda msg, *a, **k: logged.append(str(msg)))
    _episodes_json(tmp_path, _texts(4))
    table, _ = _build(tmp_path, num_text_slots=4, text_encode_batch=2, weights_dtype="bfloat16")
    audit = [line for line in logged if "context table" in line]
    assert audit, logged
    assert overfit100.context_table_audit_line(table) in audit


# =======================================================================================
# (E) the pipeline's own torch -> f32 conversion
# =======================================================================================


def test_torch_prompt_embeds_are_converted_like_encode_prompt(tmp_path):
    # encode_prompt does `jnp.array(embeds.detach().float().numpy(), dtype=jnp.float32)`;
    # the bf16 stub tensor must arrive as f32 values equal to the bf16-rounded reference.
    texts = _texts(3)
    _episodes_json(tmp_path, texts)
    table, _ = _build(tmp_path, num_text_slots=3, text_encode_batch=2, as_torch=True)
    assert table.dtype == jnp.float32  # weights_dtype float32 in this fixture
    for index, text in enumerate(texts):
        expected = jnp.asarray(_stub_embedding(text)).astype(jnp.bfloat16).astype(jnp.float32)
        np.testing.assert_array_equal(np.asarray(table[index]), np.asarray(expected))

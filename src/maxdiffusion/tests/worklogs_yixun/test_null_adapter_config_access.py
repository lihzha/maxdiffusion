"""exp_04 fix round — reading optional config keys off the REAL ``HyperParameters`` class.

J1's smoke got further than any run before it: the pipeline loaded, the revision resolved on first
contact, and then ``main`` died at ``getattr(config, "code_sha", "")``. The three-argument
``getattr`` is a lie on this class. ``HyperParameters.__getattr__`` raises **ValueError** for an
unknown key (``pyconfig.py:316-319``), and ``getattr``'s default only swallows ``AttributeError``,
so the fallback never runs -- it propagates, after the 5B model is already on device.

These tests run against the real class rather than a stand-in, because a stand-in is exactly what
hid the bug: every fake config in this suite is a ``SimpleNamespace``, where three-argument
``getattr`` behaves the way everybody assumed. The class is lifted out of ``pyconfig.py`` by AST --
importing that module drags in the whole TPU stack -- and executed against a fake key store, so the
``__getattr__``/``get_keys`` semantics under test are byte-for-byte the deployed ones.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest

from maxdiffusion.run_wan_null_inversion import mode_kwargs, optional_config_value, plan_run


_PYCONFIG = pathlib.Path("src/maxdiffusion/pyconfig.py")
# Every optional read the audit found: a key the YAML does NOT declare, so it may legitimately be
# absent and must resolve to its default instead of raising.
OPTIONAL_SITES = ("code_sha",)


def _real_hyperparameters(keys: dict):
    """The deployed ``HyperParameters``, extracted by AST and bound to a fake key store."""
    tree = ast.parse(_PYCONFIG.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HyperParameters")
    namespace = {"_config": types.SimpleNamespace(keys=dict(keys))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_PYCONFIG), "exec"), namespace)  # noqa: S102
    return namespace["HyperParameters"]()


def _yaml_keys() -> dict:
    import yaml

    with open("src/maxdiffusion/configs/base_wan_5b_null_inversion.yml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _names(count=10):
    return tuple(f"ep{index}_v0_s{index * 4:05d}" for index in range(count))


def _manifests(names=None, cohort="dev64"):
    names = names or _names()
    rows = [
        {
            "split": cohort,
            "name": name,
            "episode": name.split("_")[0][2:],
            "ordinal": index,
            "shard_path": "gs://bucket/data/val-00000.tfrecord",
            "shard_generation": "17000000000000",
            "shard_size": 4096,
        }
        for index, name in enumerate(names)
    ]
    return {
        "header": {"schema_version": 1, "shard_listing_checksum": "z" * 64},
        cohort: {"schema_version": 1, "cohort": cohort, "rows": rows},
    }


# --------------------------------------------------------------------------- the bug itself


def test_the_real_class_raises_value_error_where_getattr_promises_a_default():
    """The J1 traceback, reproduced. This is the property the fix is defined against."""
    config = _real_hyperparameters({"null_mode": "capacity"})

    assert config.null_mode == "capacity"  # present keys work fine, which is why this hid so long
    with pytest.raises(ValueError, match="Requested key code_sha, not in config"):
        getattr(config, "code_sha", "")


@pytest.mark.parametrize("key", OPTIONAL_SITES)
def test_an_optional_key_resolves_to_its_default_on_the_real_class(key):
    config = _real_hyperparameters({"null_mode": "capacity"})

    assert optional_config_value(config, key, "fallback") == "fallback"
    assert optional_config_value(config, key) is None


def test_a_present_key_is_returned_from_the_real_class():
    config = _real_hyperparameters({"code_sha": "a" * 40, "null_mode": "capacity"})

    assert optional_config_value(config, "code_sha", "fallback") == "a" * 40


def test_the_helper_reads_through_get_keys_rather_than_attribute_lookup():
    """``get_keys()`` is what ``HyperParameters`` actually exposes; attribute access is the fallback."""
    config = _real_hyperparameters({"code_sha": "b" * 40})

    assert "code_sha" in config.get_keys()
    assert optional_config_value(config, "code_sha", "") == "b" * 40


def test_the_declared_key_mapping_wins_over_attribute_lookup():
    """The two paths agree on ``HyperParameters``, so only an object where they *disagree* can show
    which one the helper actually took. ``get_keys()`` is the declared interface and must win."""

    class _Disagreeing:
        def get_keys(self):
            return {"code_sha": "from-get-keys"}

        def __getattr__(self, attr):
            return "from-attribute-lookup"

    assert optional_config_value(_Disagreeing(), "code_sha", "fallback") == "from-get-keys"


def test_a_key_absent_from_the_mapping_does_not_fall_through_to_attributes():
    """``get_keys()`` is the whole truth about what a config carries: a key missing from it is
    missing, even if attribute lookup would happily invent one."""

    class _Inventive:
        def get_keys(self):
            return {"null_mode": "capacity"}

        def __getattr__(self, attr):
            return "invented"

    assert optional_config_value(_Inventive(), "code_sha", "fallback") == "fallback"


# --------------------------------------------------------------------------- the other config shapes


def test_the_helper_works_on_a_plain_namespace():
    config = types.SimpleNamespace(code_sha="c" * 40)

    assert optional_config_value(config, "code_sha", "") == "c" * 40
    assert optional_config_value(config, "absent", "fallback") == "fallback"


def test_the_helper_works_on_a_mapping():
    assert optional_config_value({"code_sha": "d" * 40}, "code_sha", "") == "d" * 40
    assert optional_config_value({}, "code_sha", "fallback") == "fallback"


def test_the_helper_survives_an_object_that_raises_from_get_keys():
    class _Hostile:
        def get_keys(self):
            raise RuntimeError("config not initialized")

        def __getattr__(self, attr):
            raise ValueError(f"Requested key {attr}, not in config")

    assert optional_config_value(_Hostile(), "code_sha", "fallback") == "fallback"


def test_the_helper_guards_both_exception_types():
    class _AttributeStyle:
        def __getattr__(self, attr):
            raise AttributeError(attr)

    class _ValueStyle:
        def __getattr__(self, attr):
            raise ValueError(attr)

    assert optional_config_value(_AttributeStyle(), "code_sha", "x") == "x"
    assert optional_config_value(_ValueStyle(), "code_sha", "x") == "x"


# --------------------------------------------------------------------------- the composed J1 path


def _publishing_config(**overrides):
    """A real ``HyperParameters`` carrying exactly the YAML's keys -- i.e. no ``code_sha``."""
    keys = {**_yaml_keys(), "null_artifact_dir": "gs://b/run", "null_staging_dir": "", **overrides}
    keys.pop("code_sha", None)
    return _real_hyperparameters(keys)


def test_a_publishing_mode_resolves_its_commit_from_the_environment(monkeypatch):
    """The exact J1 failure, end to end: capacity must reach its provenance without a config key."""
    monkeypatch.setenv("COMMIT", "e" * 40)
    config = _publishing_config()
    manifests = _manifests()

    kwargs = mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())

    assert kwargs["code_sha"] == "e" * 40


def test_a_publishing_mode_still_refuses_when_neither_source_has_a_commit(monkeypatch):
    monkeypatch.delenv("COMMIT", raising=False)
    config = _publishing_config()
    manifests = _manifests()

    with pytest.raises(ValueError, match="40-character hex commit"):
        mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())


def test_a_config_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("COMMIT", "e" * 40)
    keys = {**_yaml_keys(), "null_artifact_dir": "gs://b/run", "null_staging_dir": "", "code_sha": "f" * 40}
    config = _real_hyperparameters(keys)
    manifests = _manifests()

    kwargs = mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())

    assert kwargs["code_sha"] == "f" * 40


def test_the_direct_opt_mode_resolves_its_commit_the_same_way(monkeypatch):
    monkeypatch.setenv("COMMIT", "e" * 40)
    config = _publishing_config(null_mode="direct_opt")
    manifests = _manifests()

    kwargs = mode_kwargs(config, plan_run(config, manifests), manifests, shards_for=lambda root: ())

    assert kwargs["code_sha"] == "e" * 40 and kwargs["iters"] == 300


def test_the_whole_plan_is_decided_from_a_real_config_object():
    """Every required key is read by direct attribute access, so this walks all of them at once."""
    config = _publishing_config(null_batch_size=4)
    manifests = _manifests()

    plan = plan_run(config, manifests)

    assert plan["mode"] == "capacity" and plan["cohort"] == "dev64"
    assert plan["batches"] == (_names()[0:4], _names()[4:8], _names()[8:10])
    assert plan["smoke_examples"] == 0 and plan["decode_batch_size"] == 8


# --------------------------------------------------------------------------- the audit's own pin


def test_every_required_key_this_driver_reads_is_declared_in_the_yaml():
    """The audit's authority: a key the YAML declares may be read directly; anything else is optional
    and must go through the helper. This pins the split so a new direct read of an undeclared key
    fails here instead of on a TPU."""
    source = pathlib.Path("src/maxdiffusion/run_wan_null_inversion.py").read_text()
    tree = ast.parse(source)
    declared = set(_yaml_keys())

    # ``config.get_keys()`` and ``config.get(...)`` inside the helper are method calls, not key
    # reads, so anything in a call position is excluded.
    called = {
        node.func
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    direct = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
        and node not in called
    }

    undeclared = sorted(key for key in direct if key not in declared)
    assert undeclared == [], f"read directly off config but not declared in the YAML: {undeclared}"


def test_no_three_argument_getattr_on_config_survives():
    """The idiomatic form is the bug; the audit removed every instance."""
    source = pathlib.Path("src/maxdiffusion/run_wan_null_inversion.py").read_text()
    tree = ast.parse(source)

    offenders = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "config"
        and isinstance(node.args[1], ast.Constant)
    ]

    assert offenders == [], f"three-argument getattr on config cannot fall back on HyperParameters: {offenders}"

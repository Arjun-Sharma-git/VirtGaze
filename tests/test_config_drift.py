"""Config-drift regression tests.

``default_config.yaml`` is the contract for every tunable in the system.  These
tests fail when a documented key is silently ignored by the code — the failure
mode that made ``inference.fallback_to_geometric`` (and 26 other keys) no-ops:
the value could be set, but nothing read it.

The check is deliberately strict.  A key only counts as "read" when it is
accessed *from config* (``.get("key")``, ``["key"]``, or the full dotted path),
so a same-named constructor parameter or local variable does not hide a key
that is never actually loaded from YAML.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from gaze_estimation.config.config import Config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "src" / "gaze_estimation" / "config" / "default_config.yaml"
)
CONFIG_DIR = REPO_ROOT / "configs"

# Source trees that are expected to read configuration.  The loader itself
# (config.py) is excluded: it defines the access API, it does not consume keys.
SOURCE_FILES = sorted(
    p
    for p in list((REPO_ROOT / "src" / "gaze_estimation").rglob("*.py"))
    + list((REPO_ROOT / "scripts").glob("*.py"))
    if p.name != "config.py"
)


def _source_texts() -> dict:
    return {p: p.read_text(encoding="utf-8") for p in SOURCE_FILES}


def _leaf_paths(node, prefix: tuple = ()) -> list:
    """Flatten nested YAML into a list of dotted leaf-key tuples."""
    paths = []
    for key, value in node.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return paths


def _is_read_from_config(dotted: str, texts: dict) -> bool:
    """True if *dotted* is accessed from config anywhere in the source."""
    leaf = dotted.rsplit(".", 1)[-1]
    patterns = (
        rf'\.get\(\s*["\']{re.escape(leaf)}["\']',
        rf'\[\s*["\']{re.escape(leaf)}["\']\s*\]',
        rf'["\']{re.escape(dotted)}["\']',
    )
    return any(
        re.search(pattern, text)
        for text in texts.values()
        for pattern in patterns
    )


def _load_yaml(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_every_default_config_key_is_read_by_the_code():
    """No documented key may be dead — changing it must have an effect."""
    default = _load_yaml(DEFAULT_CONFIG)
    texts = _source_texts()

    unread = [
        ".".join(path)
        for path in _leaf_paths(default)
        if not _is_read_from_config(".".join(path), texts)
    ]

    assert not unread, (
        "default_config.yaml documents keys that nothing reads, so setting them "
        f"has no effect: {unread}"
    )


def test_default_config_leaf_count_is_stable():
    """Guard against accidentally deleting the config's keys wholesale."""
    default = _load_yaml(DEFAULT_CONFIG)
    assert len(_leaf_paths(default)) == 71


def test_config_default_matches_yaml_top_level_sections():
    """``Config.default()`` must expose exactly the YAML's sections."""
    default = _load_yaml(DEFAULT_CONFIG)
    assert set(Config.default().to_dict()) == set(default)


@pytest.mark.parametrize(
    "config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name
)
def test_example_configs_only_override_known_keys(config_path: pathlib.Path):
    """Example configs must not reference keys that no longer exist."""
    known = {".".join(p) for p in _leaf_paths(_load_yaml(DEFAULT_CONFIG))}
    example = {".".join(p) for p in _leaf_paths(_load_yaml(config_path))}

    unknown = sorted(example - known)
    assert not unknown, (
        f"{config_path.name} overrides keys that are not in "
        f"default_config.yaml: {unknown}"
    )


def test_hierarchical_fallback_options_are_all_consumed():
    """Spot-check the options that were previously silently ignored.

    These four belong to the geometric fallback path; a regression here means
    the fallback can no longer be turned off, which is user-visible behaviour.
    """
    cfg = Config.default()
    for key in (
        "inference.fallback_to_geometric",
        "inference.fallback_distance_mm",
        "inference.geometric_min_confidence",
        "inference.confidence_threshold",
    ):
        assert cfg.get(key) is not None, f"{key} disappeared from the config"

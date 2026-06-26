"""Configuration loader: YAML -> Config object with dot-key access."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


@dataclass
class Config:
    """Top-level configuration loaded from YAML.

    Access nested keys with dot notation::

        cfg = Config.from_yaml("configs/my_config.yaml")
        width = cfg.get("camera.width", 1280)
    """

    raw: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        """Load the built-in default configuration."""
        return cls.from_yaml("")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a YAML file, merged on top of defaults."""
        default_raw: dict = {}
        if _DEFAULT_CONFIG_PATH.exists():
            with open(_DEFAULT_CONFIG_PATH) as f:
                default_raw = yaml.safe_load(f) or {}

        user_raw: dict = {}
        if path and os.path.exists(path):
            with open(path) as f:
                user_raw = yaml.safe_load(f) or {}

        merged = _deep_merge(default_raw, user_raw)
        return cls(raw=merged)

    def get(self, dot_key: str, default: Any = None) -> Any:
        """Access a nested key using dot notation.

        Examples::

            cfg.get("camera.width")     -> 1280
            cfg.get("mlp.hidden_dims")  -> [64, 128, 64]
            cfg.get("missing.key", 42)  -> 42
        """
        keys = dot_key.split(".")
        val: Any = self.raw
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, dot_key: str, value: Any) -> None:
        """Set a nested key using dot notation (mutates in-place)."""
        keys = dot_key.split(".")
        d = self.raw
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def section(self, key: str) -> dict:
        """Return a top-level section as a dict (empty dict if missing)."""
        return self.raw.get(key, {})

    def to_dict(self) -> dict:
        """Return the full config as a plain dict."""
        return dict(self.raw)

    def save(self, path: str) -> None:
        """Persist the current config to YAML."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.raw, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"Config(keys={list(self.raw.keys())})"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*. Override wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

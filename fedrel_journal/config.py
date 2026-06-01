from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Small YAML subset reader for the default config before PyYAML is installed."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" "):
            key = line.rstrip(":")
            root[key] = {}
            current = root[key]
            continue
        if current is None:
            raise ValueError("Invalid config: nested key before section")
        key, value = line.strip().split(":", 1)
        current[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

"""Load and validate cv.json / section_config.json.

Fails fast (raises DataLoadError) on malformed JSON, schema violations,
duplicate ids/section names, or CV entries referencing an unconfigured section.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .models import CVEntry, SectionConfig

_cv_entry_list_adapter: TypeAdapter[list[CVEntry]] = TypeAdapter(list[CVEntry])
_section_config_list_adapter: TypeAdapter[list[SectionConfig]] = TypeAdapter(
    list[SectionConfig]
)


class DataLoadError(Exception):
    """Raised when cv.json or section_config.json fail to load or cross-validate."""


def _read_json(path: Path) -> object:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataLoadError(f"Could not read {path}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Malformed JSON in {path}: {exc}") from exc


def load_cv(path: Path) -> list[CVEntry]:
    """Load and validate cv.json. Raises DataLoadError on any malformed entry."""
    data = _read_json(path)
    try:
        entries = _cv_entry_list_adapter.validate_python(data)
    except ValidationError as exc:
        raise DataLoadError(f"cv.json failed validation:\n{exc}") from exc

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for entry in entries:
        if entry.id in seen_ids:
            duplicate_ids.add(entry.id)
        seen_ids.add(entry.id)
    if duplicate_ids:
        raise DataLoadError(f"cv.json has duplicate ids: {sorted(duplicate_ids)}")

    return entries


def load_section_config(path: Path) -> list[SectionConfig]:
    """Load and validate section_config.json. Raises DataLoadError on any malformed entry."""
    data = _read_json(path)
    try:
        configs = _section_config_list_adapter.validate_python(data)
    except ValidationError as exc:
        raise DataLoadError(f"section_config.json failed validation:\n{exc}") from exc

    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    for config in configs:
        if config.name in seen_names:
            duplicate_names.add(config.name)
        seen_names.add(config.name)
    if duplicate_names:
        raise DataLoadError(
            f"section_config.json has duplicate section names: {sorted(duplicate_names)}"
        )

    return configs


def load_and_cross_validate(
    cv_path: Path, section_config_path: Path
) -> tuple[list[CVEntry], list[SectionConfig]]:
    """Load both files and enforce that every CV entry's section is configured."""
    entries = load_cv(cv_path)
    configs = load_section_config(section_config_path)

    known_sections = {config.name for config in configs}
    unknown_sections = {entry.section for entry in entries} - known_sections
    if unknown_sections:
        raise DataLoadError(
            "cv.json references sections not in section_config.json: "
            f"{sorted(unknown_sections)}"
        )

    return entries, configs

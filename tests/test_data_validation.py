"""Phase 1 data-validation tests: loading and cross-validating cv.json and
section_config.json.

Invalid variants are materialized in tmp_path so the committed sample fixtures
stay untouched. All imports go through the package public API, which doubles as
a regression guard for the package-import bug.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cv_to_resume_cli import (
    CVEntry,
    DataLoadError,
    SectionConfig,
    load_and_cross_validate,
    load_cv,
    load_section_config,
)


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _load_sample(sample_cv_path: Path, sample_section_config_path: Path) -> tuple[list[CVEntry], list[SectionConfig]]:
    return load_and_cross_validate(sample_cv_path, sample_section_config_path)


def test_happy_path_loads_both_fixtures(
    sample_cv_path: Path,
    sample_section_config_path: Path,
) -> None:
    entries, configs = _load_sample(sample_cv_path, sample_section_config_path)

    assert len(entries) == 15
    assert len(configs) == 3

    section_counts: dict[str, int] = {}
    for entry in entries:
        section_counts[entry.section] = section_counts.get(entry.section, 0) + 1
    assert section_counts == {"experience": 6, "education": 3, "skills": 6}

    by_name = {config.name: config for config in configs}
    assert [config.name for config in configs] == ["experience", "education", "skills"]
    assert by_name["experience"].max_entries == 5
    assert by_name["education"].max_entries == 2
    assert by_name["skills"].max_entries == 8

    null_tag_ids = {entry.id for entry in entries if entry.tags is None}
    assert null_tag_ids == {"exp-006", "skill-006"}


def test_happy_path_round_trips_section_config(sample_section_config_path: Path) -> None:
    configs = load_section_config(sample_section_config_path)

    assert [config.model_dump() for config in configs] == [
        {"name": "experience", "max_entries": 5},
        {"name": "education", "max_entries": 2},
        {"name": "skills", "max_entries": 8},
    ]


def test_duplicate_cv_ids_raise(
    tmp_path: Path,
    sample_section_config_path: Path,
) -> None:
    cv = [
        {"id": "exp-001", "section": "experience", "text": "First bullet"},
        {"id": "exp-001", "section": "experience", "text": "Duplicate id bullet"},
    ]
    cv_path = _write_json(tmp_path / "cv.json", cv)

    with pytest.raises(DataLoadError, match=r"duplicate ids.*exp-001"):
        load_and_cross_validate(cv_path, sample_section_config_path)


def test_duplicate_section_names_raise(
    tmp_path: Path,
    sample_section_config_path: Path,
) -> None:
    sections = [
        {"name": "experience", "max_entries": 5},
        {"name": "experience", "max_entries": 3},
    ]
    config_path = _write_json(tmp_path / "section_config.json", sections)

    cv_path = _write_json(tmp_path / "cv.json", [])

    with pytest.raises(DataLoadError, match=r"duplicate section names.*experience"):
        load_and_cross_validate(cv_path, config_path)


def test_unknown_section_reference_raises(
    tmp_path: Path,
    sample_section_config_path: Path,
) -> None:
    cv = [
        {"id": "exp-001", "section": "experience", "text": "Valid bullet"},
        {"id": "misc-001", "section": "publications", "text": "Bullet in unconfigured section"},
    ]
    cv_path = _write_json(tmp_path / "cv.json", cv)

    with pytest.raises(DataLoadError, match=r"sections not in section_config.json.*publications"):
        load_and_cross_validate(cv_path, sample_section_config_path)


def test_malformed_json_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "cv.json"
    bad_path.write_text('[{"id": "exp-001", "section": "experience", ', encoding="utf-8")

    with pytest.raises(DataLoadError, match="Malformed JSON"):
        load_cv(bad_path)


def test_multiline_text_raises(tmp_path: Path, sample_section_config_path: Path) -> None:
    cv = [
        {
            "id": "exp-001",
            "section": "experience",
            "text": "line one\nline two",
        }
    ]
    cv_path = _write_json(tmp_path / "cv.json", cv)

    with pytest.raises(DataLoadError, match="single line"):
        load_and_cross_validate(cv_path, sample_section_config_path)


def test_extra_key_on_entry_raises(
    tmp_path: Path,
    sample_section_config_path: Path,
) -> None:
    cv = [
        {
            "id": "exp-001",
            "section": "experience",
            "text": "Valid bullet",
            "foo": 1,
        }
    ]
    cv_path = _write_json(tmp_path / "cv.json", cv)

    with pytest.raises(DataLoadError, match="foo"):
        load_and_cross_validate(cv_path, sample_section_config_path)


def test_bad_id_format_raises(
    tmp_path: Path,
    sample_section_config_path: Path,
) -> None:
    cv = [
        {"id": "Bad ID!", "section": "experience", "text": "Valid bullet"},
    ]
    cv_path = _write_json(tmp_path / "cv.json", cv)

    with pytest.raises(DataLoadError, match="Bad ID!"):
        load_and_cross_validate(cv_path, sample_section_config_path)

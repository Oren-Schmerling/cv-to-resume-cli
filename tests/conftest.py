"""Shared fixtures: paths to the sample data used by the data-validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_cv_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_cv.json"


@pytest.fixture
def sample_section_config_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_section_config.json"

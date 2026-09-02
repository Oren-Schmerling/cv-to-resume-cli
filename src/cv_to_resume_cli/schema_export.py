"""Regenerate schemas/*.schema.json from the pydantic models.

The pydantic models in core/models.py are the source of truth. Run this
module directly whenever a model changes:

    uv run python -m cv_to_resume_cli.schema_export
"""

from __future__ import annotations

import json
from pathlib import Path

from .core.models import CVEntry, SectionConfig

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"


def export_schemas() -> None:
    _SCHEMAS_DIR.mkdir(exist_ok=True)

    cv_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "cv.json",
        "type": "array",
        "items": CVEntry.model_json_schema(),
    }
    section_config_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "section_config.json",
        "type": "array",
        "items": SectionConfig.model_json_schema(),
    }

    (_SCHEMAS_DIR / "cv.schema.json").write_text(
        json.dumps(cv_schema, indent=2) + "\n", encoding="utf-8"
    )
    (_SCHEMAS_DIR / "section_config.schema.json").write_text(
        json.dumps(section_config_schema, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    export_schemas()

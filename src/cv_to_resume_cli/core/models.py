"""Pydantic models for CV entries and section configuration.

These models are the single source of truth for data shape. schemas/*.schema.json
is generated from them (see core/schema_export.py) — do not hand-edit those files.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class CVEntry(BaseModel):
    """A single bullet-point entry from the CV.

    Each entry renders as exactly one line in the LaTeX template — there is no
    in-app line wrapping or reflow logic downstream, so `text` must already be
    one line when authored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        ..., description="Unique identifier, referenced by LLM selection output."
    )
    section: str = Field(
        ..., description="Section name; must match a name in section_config.json."
    )
    text: str = Field(
        ..., min_length=1, description="Single rendered line. No embedded newlines."
    )
    tags: list[str] | None = Field(
        default=None, description="Optional keywords to aid LLM matching."
    )

    @field_validator("id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(
                f"id {value!r} must be lowercase alphanumeric with '-'/'_' separators"
            )
        return value

    @field_validator("section")
    @classmethod
    def _validate_section_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("section must not be blank")
        return stripped

    @field_validator("text")
    @classmethod
    def _validate_single_line(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("text must be a single line (no embedded newlines)")
        if value != value.strip():
            raise ValueError("text must not have leading/trailing whitespace")
        return value

    @field_validator("tags")
    @classmethod
    def _validate_tags_not_blank(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if any(not tag.strip() for tag in value):
            raise ValueError("tags must not contain blank entries")
        return value


class SectionConfig(BaseModel):
    """Config for a single resume section: name + entry cap.

    List order in section_config.json is the section rendering order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        ..., min_length=1, description="Section name; matches CVEntry.section."
    )
    max_entries: int = Field(
        ..., gt=0, description="Hard cap on selected/rendered entries for this section."
    )

    @field_validator("name")
    @classmethod
    def _validate_name_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped

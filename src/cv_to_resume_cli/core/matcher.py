"""Phase 2 matcher: select CV entries per section via a vLLM-hosted model.

Uses the OpenAI-compatible chat-completions endpoint with structured
outputs (response_format json_schema). The decoding schema only allows valid
CV ids per section, with maxItems/uniqueItems as hard constraints. The call
layer still re-validates every response (defense in depth): JSON parse,
expected keys, id existence, no duplicates, and per-section caps.

Connection/HTTP errors are not retried — they surface immediately as
MatcherError. Malformed/invalid output is retried up to 3 attempts total.
Sections with no CV entries are omitted from schema/prompt and backfilled
with [] in the result.
"""

from __future__ import annotations

import json
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    OpenAIError,
)

from .models import CVEntry, SectionConfig
from .settings import Settings

_ATTEMPTS = 3
_TIMEOUT = 120

SectionSelection = dict[str, list[str]]


class MatcherError(Exception):
    """Raised when the matcher cannot produce a valid selection."""


class _ValidationError(Exception):
    """Internal: malformed or schema-invalid model output (retryable)."""


def _populated_sections(
    entries: list[CVEntry], config: list[SectionConfig]
) -> dict[str, list[CVEntry]]:
    grouped: dict[str, list[CVEntry]] = {section.name: [] for section in config}
    for entry in entries:
        if entry.section not in grouped:
            raise MatcherError(
                f"CV entry {entry.id} references unconfigured section "
                f"{entry.section!r}"
            )
        grouped[entry.section].append(entry)
    return {name: group for name, group in grouped.items() if group}


def build_selection_schema(
    entries: list[CVEntry], config: list[SectionConfig]
) -> dict[str, Any]:
    """Build the json_schema the model must satisfy while decoding."""
    grouped = _populated_sections(entries, config)

    properties: dict[str, Any] = {}
    for section in config:
        if section.name not in grouped:
            continue
        ids = [entry.id for entry in grouped[section.name]]
        properties[section.name] = {
            "type": "array",
            "items": {"type": "string", "enum": ids},
            "maxItems": section.max_entries,
            "uniqueItems": True,
        }

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_prompt(
    job_description: str,
    entries: list[CVEntry],
    config: list[SectionConfig],
) -> str:
    """Build the single flat user message sent to the model."""
    grouped = _populated_sections(entries, config)
    caps = ", ".join(
        f"{section.name}: {section.max_entries}"
        for section in config
        if section.name in grouped
    )

    lines: list[str] = [
        "You are selecting resume entries for a specific job.",
        "Return JSON with one key per section, each mapping to a list of entry ids.",
        "Select only entries that are relevant to the job description.",
        "Do not invent ids; use only the ids given below.",
        "An empty selection for a section is allowed.",
        "",
        "SECTION CAPS (maximum selected entries per section):",
        caps,
        "",
        "CV ENTRIES (id | section | text | tags):",
    ]
    for section in config:
        for entry in grouped.get(section.name, []):
            tags = ",".join(entry.tags) if entry.tags else ""
            lines.append(f"{entry.id} | {entry.section} | {entry.text} | {tags}")

    lines += [
        "",
        "JOB DESCRIPTION:",
        job_description,
        "",
        "OUTPUT FORMAT:",
        "Respond with exactly one JSON object of the shape",
        json.dumps({name: [f"<id from {name} entries>"] for name in grouped}),
        "containing only ids from the CV entries above.",
    ]
    return "\n".join(lines)


def _discover_model(exc: APIStatusError, settings: Settings, client: OpenAI) -> str:
    try:
        models = client.models.list()
    except OpenAIError as discovery_exc:
        raise MatcherError(
            f"Model discovery after 404 failed: {discovery_exc}"
        ) from discovery_exc
    names = [model.id for model in models]
    if len(names) != 1:
        raise MatcherError(
            f"Model {settings.hf_model_local_dir!r} not found (404) and "
            f"model discovery returned {len(names)} models: {names}"
        ) from exc
    return names[0]


def _validate(
    raw: object, entries: list[CVEntry], config: list[SectionConfig]
) -> SectionSelection:
    if not isinstance(raw, str):
        raise _ValidationError("response content is not a string")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _ValidationError(f"malformed JSON: {exc}") from exc
    grouped = _populated_sections(entries, config)

    if not isinstance(decoded, dict):
        raise _ValidationError("top-level JSON is not an object")
    if set(decoded) != set(grouped):
        raise _ValidationError(
            f"expected keys {sorted(grouped)}, got {sorted(decoded)}"
        )

    selection: SectionSelection = {}
    for name, ids in decoded.items():
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise _ValidationError(
                f"section {name!r} does not map to a list of strings"
            )
        valid = {entry.id for entry in grouped[name]}
        unknown = [i for i in ids if i not in valid]
        if unknown:
            raise _ValidationError(f"section {name!r} has unknown ids: {unknown}")
        if len(set(ids)) != len(ids):
            raise _ValidationError(f"section {name!r} has duplicate ids: {ids}")
        cap = next(s.max_entries for s in config if s.name == name)
        if len(ids) > cap:
            raise _ValidationError(
                f"section {name!r} exceeds cap {cap}: {len(ids)} ids"
            )
        selection[name] = list(ids)
    return selection


def _extract_and_validate(
    response: object, entries: list[CVEntry], config: list[SectionConfig]
) -> SectionSelection:
    try:
        content: object = response.choices[0].message.content  # type: ignore[attr-defined]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise _ValidationError(f"malformed response shape: {exc}") from exc
    return _validate(content, entries, config)


def select_entries(
    job_description: str,
    entries: list[CVEntry],
    config: list[SectionConfig],
    client: OpenAI | None = None,
    settings: Settings | None = None,
) -> SectionSelection:
    """Select CV entry ids per section via the vLLM model.

    Raises MatcherError on connection/HTTP errors and after exhausting
    retries on malformed/invalid output.
    """
    resolved_settings = settings if settings is not None else Settings()
    schema = build_selection_schema(entries, config)
    if not schema["properties"]:
        return {section.name: [] for section in config}

    resolved_client = (
        client
        if client is not None
        else OpenAI(
            base_url=f"http://{resolved_settings.vllm_host}:{resolved_settings.vllm_port}/v1",
            api_key="local",
            timeout=_TIMEOUT,
            max_retries=0,
        )
    )

    request: dict[str, Any] = {
        "model": resolved_settings.hf_model_local_dir,
        "messages": [
            {"role": "user", "content": build_prompt(job_description, entries, config)}
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "selection", "schema": schema, "strict": True},
        },
    }

    def chat() -> object:
        model: str = request["model"]
        try:
            response = resolved_client.chat.completions.create(
                model=model,
                messages=request["messages"],
                temperature=request["temperature"],
                response_format=request["response_format"],
            )
        except (APIConnectionError, APIStatusError) as exc:
            if not isinstance(exc, APIStatusError) or exc.status_code != 404:
                raise MatcherError(f"vLLM request failed: {exc}") from exc
            request["model"] = _discover_model(exc, resolved_settings, resolved_client)
            try:
                response = resolved_client.chat.completions.create(
                    model=request["model"],
                    messages=request["messages"],
                    temperature=request["temperature"],
                    response_format=request["response_format"],
                )
            except OpenAIError as fallback_exc:
                raise MatcherError(
                    f"vLLM request failed: {fallback_exc}"
                ) from fallback_exc
        except OpenAIError as exc:
            raise MatcherError(f"vLLM request failed: {exc}") from exc
        return response

    failures: list[str] = []
    for attempt in range(_ATTEMPTS):
        try:
            response = chat()
            validated = _extract_and_validate(response, entries, config)
        except _ValidationError as exc:
            failures.append(f"attempt {attempt + 1}: {exc}")
            continue
        return {section.name: validated.get(section.name, []) for section in config}

    raise MatcherError(
        f"no valid selection after {len(failures)} attempts: " + "; ".join(failures)
    )

"""Phase 2 matcher tests: schema builder, prompt builder, and select_entries.

All HTTP is mocked via a fake client injected into select_entries (the
package public API doubles as the export regression guard, per Phase 1
pattern). The 404 fallback path uses real OpenAI/NotFoundError instances
raised from a monkeypatched create, since we assert on actual SDK behavior
(but no network is involved). The live test at the bottom is opt-in via
LIVE_VLLM_TESTS=1 and skips otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import pytest
from openai import APIStatusError, NotFoundError, OpenAI

from cv_to_resume_cli import (
    CVEntry,
    MatcherError,
    SectionConfig,
    Settings,
    build_prompt,
    build_selection_schema,
    load_and_cross_validate,
    select_entries,
)

JD = (
    "Senior platform engineer role. Deep Kubernetes, Terraform, and Python "
    "backend experience required. Kafka and cost optimization a plus."
)


class _Resp:
    """ChatCompletion-shaped object built from plain data (no pydantic needed)."""

    def __init__(self, content: object) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(role="assistant", content=content))
        ]


class FakeClient:
    """OpenAI-shaped fake: records requests, replays scripted responses/errors."""

    def __init__(
        self,
        responses: list[object],
        errors: list[BaseException] | None = None,
        model_names: list[str] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._errors = list(errors or [])
        self._models = model_names
        self.calls: list[dict[str, Any]] = []

        client = self

        class _ChatCompletions:
            def create(self, **kwargs: Any) -> Any:
                client.calls.append(kwargs)
                if client._errors:
                    raise client._errors.pop(0)
                if not client._responses:
                    raise AssertionError("FakeClient got an unexpected extra call")
                return client._responses.pop(0)

        class _Models:
            def list(self) -> Any:
                page = SimpleNamespace(
                    data=[SimpleNamespace(id=name) for name in (client._models or [])]
                )
                return page

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_ChatCompletions().create))
        self.models = SimpleNamespace(list=_Models().list)


def _status_error(status_code: int, message: str) -> APIStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"message": message})
    if status_code == 404:
        return NotFoundError(message, response=response, body={"message": message})
    return APIStatusError(message, response=response, body={"message": message})


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "vllm_host": "127.0.0.1",
        "vllm_port": 8000,
        "hf_model_local_dir": "/models/slm",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def sample(
    sample_cv_path: Path, sample_section_config_path: Path
) -> tuple[list[CVEntry], list[SectionConfig]]:
    return load_and_cross_validate(sample_cv_path, sample_section_config_path)


@pytest.fixture
def settings() -> Settings:
    return _settings()


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------


def test_schema_builder_structure(sample: Any, settings: Settings) -> None:
    entries, config = sample
    schema = build_selection_schema(entries, config)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"experience", "education", "skills"}
    assert set(schema["properties"]) == {"experience", "education", "skills"}

    by_name = {c.name: c for c in config}
    for name, prop in schema["properties"].items():
        assert prop["type"] == "array"
        assert prop["uniqueItems"] is True
        assert prop["maxItems"] == by_name[name].max_entries
        assert set(prop["items"]["enum"]) == {
            e.id for e in entries if e.section == name
        }
        assert prop["items"]["type"] == "string"
    assert "minItems" not in schema["properties"]["experience"]


def test_schema_builder_omits_zero_entry_sections() -> None:
    entries = [
        CVEntry(id="exp-001", section="experience", text="One bullet", tags=["a"]),
    ]
    configs = [
        SectionConfig(name="experience", max_entries=3),
        SectionConfig(name="skills", max_entries=8),
    ]
    schema = build_selection_schema(entries, configs)

    assert set(schema["properties"]) == {"experience"}
    assert schema["required"] == ["experience"]
    assert schema["properties"]["experience"]["items"]["enum"] == ["exp-001"]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_prompt_contains_jd_ids_caps_and_sections(sample: Any) -> None:
    entries, config = sample
    prompt = build_prompt(JD, entries, config)

    assert JD in prompt
    assert "experience: 5" in prompt
    assert "education: 2" in prompt
    assert "skills: 8" in prompt
    for entry in entries:
        assert entry.id in prompt
        assert entry.text in prompt
    assert "exp-001 | experience |" in prompt
    assert "leadership,payments,performance" in prompt
    assert "<id from skills entries>" in prompt


def test_prompt_omits_zero_entry_sections() -> None:
    entries = [CVEntry(id="exp-001", section="experience", text="One bullet", tags=None)]
    configs = [
        SectionConfig(name="experience", max_entries=3),
        SectionConfig(name="skills", max_entries=8),
    ]
    prompt = build_prompt(JD, entries, configs)

    assert "skills: 8" not in prompt
    assert "<id from skills entries>" not in prompt


# ---------------------------------------------------------------------------
# select_entries: happy paths
# ---------------------------------------------------------------------------


def test_select_entries_happy_path_including_empty_backfill(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    payload = {
        "experience": ["exp-001", "exp-003"],
        "education": ["edu-001"],
        "skills": [],
    }
    client = FakeClient(responses=[_Resp(json.dumps(payload))])

    result = select_entries(JD, entries, config, client=client, settings=settings)

    assert result == payload
    assert isinstance(result, dict)
    assert isinstance(result["skills"], list)


def test_select_entries_returns_all_configured_sections(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    entries = [e for e in entries if e.section != "skills"]
    client = FakeClient(
        responses=[_Resp(json.dumps({"experience": [], "education": []}))]
    )

    result = select_entries(JD, entries, config, client=client, settings=settings)

    assert set(result) == {"experience", "education", "skills"}
    assert result["skills"] == []


def test_select_entries_wire_format(sample: Any, settings: Settings) -> None:
    entries, config = sample
    payload = {"experience": ["exp-001"], "education": ["edu-001"], "skills": ["skill-001"]}
    client = FakeClient(responses=[_Resp(json.dumps(payload))])

    select_entries(JD, entries, config, client=client, settings=settings)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "/models/slm"
    assert call["temperature"] == 0
    assert "max_tokens" not in call and "max_completion_tokens" not in call
    assert call["messages"] == [
        {"role": "user", "content": build_prompt(JD, entries, config)}
    ]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"]
    assert call["response_format"]["json_schema"]["schema"] == build_selection_schema(
        entries, config
    )
    assert call["response_format"]["json_schema"]["strict"] is True


def test_select_entries_empty_cv_short_circuits(settings: Settings) -> None:
    configs = [SectionConfig(name="experience", max_entries=3)]
    client = FakeClient(responses=[])

    result = select_entries(JD, [], configs, client=client, settings=settings)

    assert result == {"experience": []}
    assert client.calls == []


# ---------------------------------------------------------------------------
# select_entries: retries on malformed/invalid output
# ---------------------------------------------------------------------------


def test_select_entries_malformed_json_three_attempts_then_error(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    client = FakeClient(responses=[_Resp("not json")] * 3)

    with pytest.raises(MatcherError, match="attempt 3"):
        select_entries(JD, entries, config, client=client, settings=settings)

    assert len(client.calls) == 3


def test_select_entries_malformed_then_valid_succeeds(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    payload = {"experience": ["exp-002"], "education": [], "skills": ["skill-002"]}
    client = FakeClient(
        responses=[_Resp("{oops"), _Resp(json.dumps(payload))]
    )

    result = select_entries(JD, entries, config, client=client, settings=settings)

    assert result == payload
    assert len(client.calls) == 2


def test_select_entries_unknown_id_retries_then_fails(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    bad = {
        "experience": ["exp-999"],
        "education": ["edu-001"],
        "skills": [],
    }
    client = FakeClient(responses=[_Resp(json.dumps(bad))] * 3)

    with pytest.raises(MatcherError, match="unknown ids"):
        select_entries(JD, entries, config, client=client, settings=settings)

    assert len(client.calls) == 3


def test_select_entries_duplicate_id_retries_then_fails(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    bad = {
        "experience": ["exp-001", "exp-001"],
        "education": [],
        "skills": [],
    }
    client = FakeClient(responses=[_Resp(json.dumps(bad))] * 3)

    with pytest.raises(MatcherError, match="duplicate ids"):
        select_entries(JD, entries, config, client=client, settings=settings)


def test_select_entries_over_cap_retries_then_fails(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    bad = {
        "experience": ["exp-001", "exp-002", "exp-003", "exp-004", "exp-005", "exp-006"],
        "education": ["edu-001", "edu-002"],
        "skills": [],
    }
    client = FakeClient(responses=[_Resp(json.dumps(bad))] * 3)

    with pytest.raises(MatcherError, match="exceeds cap 5"):
        select_entries(JD, entries, config, client=client, settings=settings)


def test_select_entries_missing_key_retries_then_fails(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    bad = {"experience": ["exp-001"], "education": ["edu-001"]}
    client = FakeClient(responses=[_Resp(json.dumps(bad))] * 3)

    with pytest.raises(MatcherError, match="expected keys"):
        select_entries(JD, entries, config, client=client, settings=settings)


def test_select_entries_extra_key_retries_then_fails(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    bad = {
        "experience": ["exp-001"],
        "education": ["edu-001"],
        "skills": [],
        "publications": ["exp-002"],
    }
    client = FakeClient(responses=[_Resp(json.dumps(bad))] * 3)

    with pytest.raises(MatcherError, match="publications"):
        select_entries(JD, entries, config, client=client, settings=settings)


def test_select_entries_non_string_ids_retry_then_fail(
    sample: Any, settings: Settings
) -> None:
    entries, config = sample
    bad = {"experience": [1], "education": [], "skills": []}
    client = FakeClient(responses=[_Resp(json.dumps(bad))] * 3)

    with pytest.raises(MatcherError, match="does not map to a list of strings"):
        select_entries(JD, entries, config, client=client, settings=settings)


# ---------------------------------------------------------------------------
# select_entries: HTTP errors are not retried
# ---------------------------------------------------------------------------


def test_select_entries_connection_error_not_retried(
    sample: Any, settings: Settings
) -> None:
    import openai

    entries, config = sample
    request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
    conn_error = openai.APIConnectionError(message="Connection refused", request=request)
    client = FakeClient(responses=[], errors=[conn_error])

    with pytest.raises(MatcherError) as excinfo:
        select_entries(JD, entries, config, client=client, settings=settings)

    assert "vLLM request failed" in str(excinfo.value)
    assert len(client.calls) == 1
    assert excinfo.value.__cause__ is conn_error


def test_select_entries_http_error_not_retried(sample: Any, settings: Settings) -> None:
    entries, config = sample
    error = _status_error(500, "Internal Server Error")
    client = FakeClient(responses=[], errors=[error])

    with pytest.raises(MatcherError) as excinfo:
        select_entries(JD, entries, config, client=client, settings=settings)

    assert "vLLM request failed" in str(excinfo.value)
    assert len(client.calls) == 1
    assert excinfo.value.__cause__ is error


# ---------------------------------------------------------------------------
# select_entries: 404 model-not-found fallback
# ---------------------------------------------------------------------------


def _real_openai_with_behavior(
    responses: list[object],
    model_names: list[str] | None = None,
    models_status: int | None = None,
) -> Any:
    """Real OpenAI client over a mock transport (validates real SDK behavior)."""
    queue = list(responses)
    names = list(model_names or [])
    calls: list[dict[str, Any]] = []

    def chat_handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode()))
        if not queue:
            return httpx.Response(500, request=request, json={"message": "boom"})
        item = queue.pop(0)
        if isinstance(item, int):
            return httpx.Response(item, request=request, json={"message": "not found"})
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": item},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(200, json=payload, request=request)

    def models_handler(request: httpx.Request) -> httpx.Response:
        if models_status is not None:
            return httpx.Response(
                models_status, request=request, json={"message": "discovery failed"}
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": n, "object": "model"} for n in names],
            },
            request=request,
        )

    def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return models_handler(request)
        return chat_handler(request)

    client = OpenAI(
        base_url="http://127.0.0.1:1/v1",
        api_key="local",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(dispatch)),
    )
    client._calls = calls  # type: ignore[attr-defined]
    return client


def test_404_fallback_single_model_succeeds(sample: Any, settings: Settings) -> None:
    entries, config = sample
    payload = json.dumps(
        {"experience": ["exp-004"], "education": ["edu-003"], "skills": ["skill-004"]}
    )
    client = _real_openai_with_behavior([404, payload], model_names=["qwen3-1b"])

    result = select_entries(JD, entries, config, client=client, settings=settings)

    assert result == {
        "experience": ["exp-004"],
        "education": ["edu-003"],
        "skills": ["skill-004"],
    }
    calls = client._calls  # type: ignore[attr-defined]
    assert len(calls) == 2
    assert calls[0]["model"] == "/models/slm"
    assert calls[1]["model"] == "qwen3-1b"


def test_404_fallback_ambiguous_models_fail(sample: Any, settings: Settings) -> None:
    entries, config = sample
    client = _real_openai_with_behavior([404], model_names=["m-a", "m-b"])

    with pytest.raises(MatcherError, match="2 models"):
        select_entries(JD, entries, config, client=client, settings=settings)


def test_404_fallback_zero_models_fail(sample: Any, settings: Settings) -> None:
    entries, config = sample
    client = _real_openai_with_behavior([404], model_names=[])

    with pytest.raises(MatcherError, match="0 models"):
        select_entries(JD, entries, config, client=client, settings=settings)


def test_404_fallback_discovery_error_fails(sample: Any, settings: Settings) -> None:
    entries, config = sample
    client = _real_openai_with_behavior(
        [404], model_names=["m-a"], models_status=500
    )

    with pytest.raises(MatcherError, match="Model discovery"):
        select_entries(JD, entries, config, client=client, settings=settings)


# ---------------------------------------------------------------------------
# Live test (opt-in)
# ---------------------------------------------------------------------------

LIVE = os.environ.get("LIVE_VLLM_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="requires a live vLLM server (LIVE_VLLM_TESTS=1)")
def test_live_vllm_selection_structural(
    sample_cv_path: Path, sample_section_config_path: Path
) -> None:
    entries, config = load_and_cross_validate(sample_cv_path, sample_section_config_path)

    result = select_entries(
        JD,
        entries,
        config,
        client=OpenAI(base_url=f"http://{Settings().vllm_host}:{Settings().vllm_port}/v1", api_key="local", timeout=120, max_retries=0),
        settings=Settings(),
    )

    valid_ids = {e.id: e.section for e in entries}
    caps = {c.name: c.max_entries for c in config}
    assert set(result) == set(caps)
    for name, ids in result.items():
        assert len(ids) <= caps[name]
        assert len(set(ids)) == len(ids)
        for entry_id in ids:
            assert entry_id in valid_ids
            assert valid_ids[entry_id] == name

# Resume Generator — Roadmap

## Phase 1: Data Layer

- Define `cv.json` schema: `id`, `section`, `single-line text`, optional `tags`.
- Enforce 1-line constraint per entry (manual authoring discipline, no auto-wrap).
- Add `section_config.json`: `max_entries` per section (replaces `max_lines`).
- Validate JSON schema on load (fail fast on malformed entries).

## Phase 2: LLM Matching

- Host model via vLLM, enforce output with structured outputs (`response_format` `json_schema`) — `guided_json` / `--guided-decoding-backend` were removed in vLLM v0.12.
- Prompt: job description + full CV entries + per-section `max_entries` → strict JSON output `{section: [ids]}`.
- Constraints:
  - Decoding schema hard-constrains output at decode time: per-section `enum` of valid CV ids, `maxItems` = cap, `uniqueItems`. Zero-entry sections are omitted from the schema and backfilled as `[]`.
  - Call layer re-validates post-hoc (defense in depth): IDs exist, no duplicates, K ≤ max_entries, exact keys.
  - Retry (3 attempts) on malformed/invalid output; connection/HTTP errors fail fast as `MatcherError`.

## Phase 3: Selection Validation (Python)

- Cross-check returned IDs exist and aren't duplicated.
- Enforce hard cap per section (truncate if LLM over-selects).
- Fallback: if section returns empty/invalid, default to N most recent entries.

## Phase 4: LaTeX Templating

- Jinja2 with custom delimiters (avoid `{{ }}` conflict with LaTeX).
- Escape special chars: `& % _ # $ { }`.
- Template has fixed slots per section (matches `max_entries`).
- Missing/unused slots render blank gracefully (no broken layout).

## Phase 5: Compilation

- Use `tectonic` subprocess (self-contained, no TeX Live dependency).
- Capture stderr/stdout, surface compile errors clearly (don't silently fail).
- Output PDF to CWD with sensible filename (e.g. `resume_<company>_<date>.pdf`).

## Phase 6: CLI Wiring

- Prompt user to paste job description (stdin or file).
- Chain: load CV → call vLLM → validate → render → compile → save.
- Add flags: `--dry-run` (skip compile, show selected IDs), `--model`, `--k-override`.

## Phase 7: Testing / Hardening

- Unit tests: JSON validation, LaTeX escaping, entry-count enforcement.
- Golden-file test: known job description → expected entry IDs (regression check on prompt changes).
- Test malformed LLM output handling (structured-outputs failure path).

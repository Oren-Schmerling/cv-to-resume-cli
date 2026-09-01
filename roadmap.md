# Resume Generator — Roadmap

## Phase 1: Data Layer

- Define `cv.json` schema: `id`, `section`, `single-line text`, optional `tags`.
- Enforce 1-line constraint per entry (manual authoring discipline, no auto-wrap).
- Add `section_config.json`: `max_entries` per section (replaces `max_lines`).
- Validate JSON schema on load (fail fast on malformed entries).

## Phase 2: LLM Matching

- Host model via vLLM, use guided/structured decoding (outlines or vLLM's `guided_json`).
- Prompt: job description + full CV JSON + per-section `max_entries` → strict JSON output `{section: [ids]}`.
- Constraints:
  - Output must only contain IDs that exist in `cv.json`.
  - Enforce K ≤ max_entries per section at prompt level AND validate post-hoc.
  - Reject/retry on malformed JSON (guided decoding should make this rare).

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
- Test malformed LLM output handling (guided decoding failure path).

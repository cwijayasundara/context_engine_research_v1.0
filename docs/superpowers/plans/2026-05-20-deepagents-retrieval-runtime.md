# Deep Agents Retrieval Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Deep Agents 0.6 retrieval runtime using Gemini 3.5 Flash behind `AGENT_RUNTIME=deepagents`, while preserving the current `src/api/agent_runner.py` execution path as the default.

**Architecture:** Build a new `src/deep_retrieval` package with schema grounding, Cypher validation, LangChain/Deep Agents tool definitions, a Gemini harness profile hook, and a stream adapter that emits the existing frontend SSE event vocabulary. Route selection happens in `src/api/routes/agent.py`; all existing API/session/decision behavior remains intact.

**Tech Stack:** Python 3.12+, FastAPI, pytest, Deep Agents 0.6, LangChain Google GenAI, Neo4j, SSE, Gemini 3.5 Flash.

---

### Task 1: Cypher Guard And Schema Context

**Files:**
- Create: `src/deep_retrieval/__init__.py`
- Create: `src/deep_retrieval/cypher_guard.py`
- Create: `src/deep_retrieval/schema.py`
- Test: `tests/test_deep_cypher_guard.py`

- [ ] Write failing tests for read-only Cypher validation, unknown labels/properties/relationships, wrong relationship direction, and missing `LIMIT`.
- [ ] Implement `CypherGuard`, `CypherValidationError`, `build_schema_context()`, and `finance_cypher_examples()`.
- [ ] Run `FINCTX_LAZY=1 pytest -q tests/test_deep_cypher_guard.py`.

### Task 2: Stream Adapter

**Files:**
- Create: `src/deep_retrieval/stream_adapter.py`
- Test: `tests/test_deep_stream_adapter.py`

- [ ] Write failing tests for adapting Deep Agents-like event dictionaries into `token`, `tool_call`, `tool_result`, `subagent_start`, `subagent_end`, `result`, `done`, and `error`.
- [ ] Implement a defensive adapter that ignores unknown events and preserves already-normalized event tuples.
- [ ] Run `FINCTX_LAZY=1 pytest -q tests/test_deep_stream_adapter.py`.

### Task 3: Runtime Selection

**Files:**
- Create: `src/deep_retrieval/runtime.py`
- Modify: `src/api/routes/agent.py`
- Test: `tests/test_agent_runtime_selection.py`

- [ ] Write failing tests proving default routing uses `src.api.agent_runner.run_agent_stream` and `AGENT_RUNTIME=deepagents` uses `src.deep_retrieval.runtime.run_deep_agent_stream`.
- [ ] Implement route selection through a small `_select_agent_stream()` helper.
- [ ] Run `FINCTX_LAZY=1 pytest -q tests/test_agent_runtime_selection.py`.

### Task 4: Deep Agents Builder, Tools, Prompts, Profile

**Files:**
- Create: `src/deep_retrieval/prompts.py`
- Create: `src/deep_retrieval/profiles.py`
- Create: `src/deep_retrieval/tools.py`
- Create: `src/deep_retrieval/builder.py`
- Modify: `src/deep_retrieval/runtime.py`
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] Add lazy imports so tests pass without Deep Agents installed.
- [ ] Implement `build_deep_agent()` with `create_deep_agent`, Gemini model default `google_genai:gemini-3.5-flash`, `REPLMiddleware`, optional `ContextHubBackend`, subagent specs, and tools.
- [ ] Implement graph/wiki/fraud tools around existing Neo4j/wiki code and `CypherGuard`.
- [ ] Update requirements to Deep Agents 0.6 + QuickJS and Google GenAI support.
- [ ] Update `.env.example` with `AGENT_RUNTIME`, `DEEP_AGENT_MODEL`, and `GOOGLE_API_KEY`.
- [ ] Run `FINCTX_LAZY=1 python -m compileall -q src tests examples`.

### Task 5: Docs And Final Verification

**Files:**
- Modify: `README.md`
- Create: `scripts/smoke_deep_retrieval.py`

- [ ] Document the feature flag and Gemini setup without changing the default runtime.
- [ ] Add a smoke script that requires `AGENT_RUNTIME=deepagents` and `GOOGLE_API_KEY`.
- [ ] Run `FINCTX_LAZY=1 pytest -q`.
- [ ] Run `cd web && npm run build`.
- [ ] Commit only the Deep Agents runtime changes; leave unrelated dead-code plan changes unstaged.


# Deep Agents Retrieval Runtime Design

## Goal

Build a Deep Agents 0.6 retrieval runtime for the finance context engine using `gemini-3.5-flash`, without disturbing the existing `src/api/agent_runner.py` execution path.

## Current State

The production workbench route is `POST /api/agent/ask` in `src/api/routes/agent.py`. It currently calls `src/api/agent_runner.py`, which implements a custom OpenAI-compatible planner, sequential subagent loop, Neo4j/wiki tools, and SSE event stream.

There is an older DeepAgents path under `src/agent` and `src/retriever`, but it is not used by the FastAPI workbench. This code should not be deleted. It should either be replaced by or migrated into a new Deep Agents 0.6 runtime.

## External References

LangChain Deep Agents 0.6:

- `deepagents[quickjs]` and `REPLMiddleware` enable interpreter-backed Programmatic Tool Calling (PTC).
- Harness profiles tune prompt assembly, tool visibility, middleware, and subagent behavior per provider/model.
- `stream_events(..., version="v3")` emits typed events for messages, tools, subagents, state updates, and final output.
- `ContextHubBackend` is optional and requires LangSmith.

LangChain Neo4j Cypher guidance:

- Generate Cypher from explicit graph schema.
- Add examples to the Cypher-generation prompt.
- Limit result count.
- Return intermediate Cypher/results for observability.
- Optionally separate Cypher-generation and answer-generation LLMs.
- Validate generated Cypher, especially relationship directions.
- Provide database rows as tool/function context to improve grounded answers.

## Design Decision

Add a parallel Deep Agents runtime behind a feature flag. Do not replace the current runner until the new runtime passes tests and smoke checks.

Configuration:

- `AGENT_RUNTIME=current` keeps today’s behavior.
- `AGENT_RUNTIME=deepagents` switches `POST /api/agent/ask` to the new runtime.
- `DEEP_AGENT_MODEL=google_genai:gemini-3.5-flash` is the default Deep Agents model.
- `GOOGLE_API_KEY` is required for the Google GenAI LangChain provider.
- `LANGSMITH_API_KEY` enables optional `ContextHubBackend`; without it, use state/local backend only.

## Package Layout

Create `src/deep_retrieval/`:

- `runtime.py`: public `run_deep_agent_stream(question, history)` async generator that returns the same event vocabulary as the current SSE route.
- `builder.py`: constructs the Deep Agents graph with tools, subagents, middleware, backend, and model.
- `profiles.py`: registers a Gemini 3.5 Flash harness profile.
- `tools.py`: exposes graph, wiki, fraud, and context tools to Deep Agents.
- `cypher_guard.py`: validates read-only Cypher, schema references, relationship directions, limits, and parameters.
- `schema.py`: compiles ontology and optional Neo4j schema metadata into prompt/tool context.
- `stream_adapter.py`: converts Deep Agents v3 events into existing frontend event names.
- `prompts.py`: system prompt, subagent prompts, and Cypher-generation examples.

Keep `src/api/agent_runner.py` unchanged except for route selection.

## Runtime Flow

`POST /api/agent/ask`:

1. Reads session history as it does today.
2. If `AGENT_RUNTIME != deepagents`, calls `src.api.agent_runner.run_agent_stream`.
3. If `AGENT_RUNTIME == deepagents`, calls `src.deep_retrieval.runtime.run_deep_agent_stream`.
4. Stores session events and writes `Decision` nodes using the same code path.

Deep Agents flow:

1. Main agent receives the question and recent conversational turns.
2. `REPLMiddleware` lets the agent plan and run PTC code against tools.
3. Main agent decomposes work into recursive subagent tasks when needed.
4. Retrieval subagents gather graph/wiki/fraud evidence.
5. Tool outputs are compact and structured so Gemini receives rows and citations, not raw noisy dumps.
6. Final answer is synthesized from tool evidence only.
7. Stream adapter emits current UI-compatible events: `started`, `plan`, `subagent_start`, `tool_call`, `tool_result`, `graph_highlight`, `graph_update`, `token`, `result`, `done`, and `error`.

## Subagents

Main agent:

- Plans retrieval.
- Manages recursive follow-up questions in interpreter state.
- Calls subagents via `tools.task(...)`.
- Decides when evidence is sufficient.

`graph_analyst`:

- Generates parameterized Cypher using only supplied schema.
- Uses `cypher_generate` and `graph_query` tools.
- Never writes to Neo4j.

`wiki_retriever`:

- Lists and reads compiled wiki artifacts.
- Pulls page-level evidence and outbound links.

`fraud_investigator`:

- Retrieves anomalies, fraud scores, and alert context.
- Explains rule/GDS signals from stored graph data.

`advisor`:

- Produces user-facing recommendations from retrieved evidence.
- Does not call write tools.

## Gemini 3.5 Flash Harness Profile

Register a model-level profile for `google_genai:gemini-3.5-flash`.

Profile intent:

- Prefer short, structured tool inputs.
- Use tools for factual finance answers.
- Do not guess labels, properties, merchant names, or dates.
- When generating Cypher, emit only the Cypher and parameter dict through tools.
- Keep intermediate reasoning out of user-visible answer text.
- Prefer PTC for multi-step workflows: run parallel retrieval, filter rows, and return compact summaries.
- Disable or hide filesystem/write tools unless explicitly needed.

## Cypher Generation And Safety

The Cypher layer must not be a raw string execution surface.

Tool design:

```python
def cypher_generate(question: str, schema: str, examples: list[dict]) -> dict:
    return {
        "cypher": "...",
        "params": {...},
        "expected_columns": ["..."],
        "purpose": "...",
    }
```

```python
def graph_query(cypher: str, params: dict, purpose: str, expected_columns: list[str]) -> dict:
    return {
        "columns": [...],
        "rows": [...],
        "row_count": 0,
        "node_ids": [...],
        "cypher": "...",
    }
```

Guardrails:

- Reject writes: `CREATE`, `MERGE`, `DELETE`, `DETACH`, `SET`, `REMOVE`, `DROP`, `LOAD CSV`, unsafe `CALL`, APOC write procedures.
- Require parameterized values for user terms; reject obvious quoted user values in `WHERE`.
- Require `LIMIT` for row-returning queries unless the query is aggregate-only.
- Cap returned rows.
- Check labels, relationship types, and properties against ontology/schema.
- Check relationship directions against ontology triples.
- Allow only read-safe procedures if any are used.
- Return intermediate Cypher and rows as structured tool output for observability.

Prompting:

- Include ontology-derived schema.
- Include query examples for common finance questions:
  - merchant spend by year/month
  - top categories by month
  - recurring bills
  - settlement trace between credit card and current account
  - fraud alerts by severity/month
- Include the instruction from LangChain’s Neo4j guide: use only provided relationship types and properties.

## Streaming Contract

The new runtime must preserve the current frontend contract. The first implementation should adapt Deep Agents v3 events into existing events instead of changing React components.

Mapping:

- Deep Agents message text -> `token`
- tool call start -> `tool_call`
- tool result -> `tool_result`
- subagent lifecycle -> `subagent_start` / `subagent_end`
- graph tool rows with node ids -> `graph_highlight`
- graph context payloads -> `graph_update`
- final answer -> `result`
- run finish -> `done`
- exceptions -> `error`

## Error Handling

- If `AGENT_RUNTIME=deepagents` but `GOOGLE_API_KEY` is missing, emit `error` and do not fall back silently.
- If Deep Agents imports fail, emit `error` with the missing package name.
- If Cypher validation fails, return a structured tool error to the agent and allow one corrected retry.
- If Neo4j is unavailable, emit a tool error and synthesize a concise user-facing failure.
- If streaming adapter sees an unknown Deep Agents event, log it and ignore it unless it carries final text or error data.

## Testing

Unit tests:

- `tests/test_deep_cypher_guard.py`
  - accepts parameterized read queries
  - rejects write keywords
  - rejects unknown labels/properties/relationships
  - rejects wrong relationship direction
  - requires `LIMIT` for non-aggregate row-returning queries

- `tests/test_deep_stream_adapter.py`
  - maps message deltas to `token`
  - maps tool calls/results
  - maps subagent lifecycle
  - preserves `result` and `done`

- `tests/test_agent_runtime_selection.py`
  - default runtime calls current runner
  - `AGENT_RUNTIME=deepagents` calls new runtime

Smoke tests:

- `scripts/smoke_deep_retrieval.py`
  - requires `GOOGLE_API_KEY`
  - asks a known graph question
  - asserts at least one graph tool call and a final answer

Regression:

- Existing `FINCTX_LAZY=1 pytest -q` must pass.
- Existing `npm run build` must pass.

## Migration Strategy

Phase 1 adds the Deep Agents runtime behind a feature flag. Phase 2 compares answers and event streams for known questions. Phase 3 can switch the default only after the new runtime matches current frontend behavior and improves retrieval quality.

The prior dead-code cleanup plan must be revised before use. `src/agent` and `src/retriever` are no longer deletion targets; they are migration inputs.


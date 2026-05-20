# Personal Finance Context Engine

A local personal-finance workbench built around a Neo4j context graph, generated
wiki pages, a streaming analyst agent, and a fraud/anomaly layer.

The app ingests normalized bank and credit-card transactions, canonicalizes
merchants, writes a finance graph, compiles markdown wiki artifacts, and exposes
the result through a FastAPI backend and a Next.js workbench.

## Current Capabilities

- Neo4j finance graph: `Account`, `Statement`, `Transaction`, `Merchant`,
  `Category`, `Month`, `Day`, `Location`, `Alert`, and `Decision`.
- Generated wiki vault under `data/wiki/` for merchants, categories, months,
  annual summaries, home, and fraud-alert pages.
- Next.js workbench with graph, wiki browser, chat, timeline scrubber, compare
  view, money clock, alerts panel, and canonicalizer editor.
- Streaming agent endpoint: planner -> scoped subagents -> synthesizer over SSE.
- Query-focused graph slices: agent tool results emit graph node ids, and the
  canvas isolates the relevant one-hop subgraph.
- Click-to-inspect graph nodes and drag-to-pin node positions.
- Fraud/anomaly scoring with deterministic rules plus optional Neo4j GDS
  features.
- PII tokenization support for LLM-facing context.

## Repository Layout

```text
finance-context-engine/
├── README.md
├── docker-compose.yml              # Neo4j 5 + GDS
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── statements/                 # optional raw/parsed statement input
│   ├── statements_generated/        # generated sample normalized data
│   ├── ontology/finance.yaml
│   └── wiki/                       # generated markdown wiki vault
├── src/
│   ├── api/                        # FastAPI backend
│   │   ├── main.py
│   │   ├── agent_runner.py          # streaming planner/subagent/synthesizer loop
│   │   ├── decisions.py             # writes Decision trace nodes
│   │   └── routes/                  # graph, wiki, timeline, fraud, agent, canon, pii
│   ├── ingestion/                  # parse, normalize, graph load, wiki compile
│   ├── fraud/                      # rules, GDS projection, scoring, writeback
│   ├── pii/                        # vault and PII filtering
│   ├── ontology/schema.cypher
│   ├── agent/                      # older DeepAgents harness/prototype path
│   └── retriever/                  # graph/wiki tools and PII middleware
└── web/
    ├── app/                        # Next.js app routes
    ├── components/                 # GraphCanvas, ChatPanel, AlertsPanel, etc.
    └── lib/                        # typed API client and SSE consumer
```

## Architecture

```text
normalized transactions
        |
        v
merchant canonicalization + PII filtering
        |
        v
Neo4j context graph
        |
        +--> graph API / timeline API / fraud API
        |
        +--> wiki compiler -> data/wiki/*.md -> wiki API
        |
        v
FastAPI streaming agent
planner -> analyst / wiki_browser / advisor -> synthesizer
        |
        v
Next.js workbench
graph + wiki + chat + fraud alerts + canon editor
```

## Quick Start

### 1. Start Neo4j

```bash
docker compose up -d
```

Default development config expects Neo4j on the port configured in `.env`
or `.env.example`.

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Set at least:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `OPENAI_API_KEY` if you want the chat agent or LLM canonicalizer
- `OPENAI_MODEL` to pick the model (default `gpt-5.4-mini`)
- `OPENAI_BASE_URL` to route through an OpenAI-compatible provider

**Using Gemini (3.5 Flash) instead of OpenAI**

The agent uses the OpenAI Python SDK, but you can point it at Google's OpenAI-compatible Gemini endpoint without changing any code:

```bash
OPENAI_API_KEY=<your Google AI Studio API key>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-3.5-flash      # or gemini-2.5-flash / gemini-2.5-pro
```

This wires up `src/api/agent_runner.py` (the chat agent), `src/ingestion/llm_normalize.py` (merchant canonicalization), and `src/data_gen/generator.py` (synthetic data) — all three honour the same env vars. The OpenAI-compat layer supports tool/function calling, JSON mode, and streaming, which are everything the agent relies on.

To switch back to OpenAI, just unset `OPENAI_BASE_URL`.

**Experimental Deep Agents 0.6 retrieval runtime**

The default workbench chat still uses `src/api/agent_runner.py`. To try the
additive Deep Agents 0.6 runtime without changing the default path:

```bash
AGENT_RUNTIME=deepagents
DEEP_AGENT_MODEL=google_genai:gemini-3.5-flash
GOOGLE_API_KEY=<your Google AI Studio API key>
```

This path uses the new `src/deep_retrieval/` package: Gemini-specific harness
profile setup, QuickJS PTC middleware, schema-grounded Cypher guardrails, and
Deep Agents v3 stream-event adaptation back to the current frontend SSE shape.

### 4. Populate the graph, wiki, and fraud database

These four steps populate the system end-to-end: ontology → graph data → LLM wiki → fraud alerts. Run them in order, from the repo root, with the venv active.

```bash
# (a) Make sure Neo4j is up (GDS pre-installed in the docker image)
docker compose up -d neo4j
# wait ~15s on first boot; check http://localhost:7475 responds

# (b) OPTIONAL — wipe the DB for a clean re-populate. The loader is
# MERGE-based and idempotent, but some properties only set ON CREATE,
# so wiping is the safest reset when fields like t.year, t.amount, or
# canonicalization rules have changed.
docker exec finance-ctx-neo4j cypher-shell -u neo4j -p please-change-me \
  "MATCH (n) DETACH DELETE n;"

# (c) Ontology — `src/ontology/schema.cypher` is GENERATED from
# `data/ontology/finance.yaml`. Edit the YAML, never the cypher, then:
python -m src.ontology.compile --write       # regenerate schema.cypher
python -m src.ontology.compile --check       # CI-friendly drift check
# The constraints/indexes are auto-applied at the start of (d). Run this
# manually only if you want constraints without loading data:
docker exec -i finance-ctx-neo4j cypher-shell -u neo4j -p please-change-me \
  < src/ontology/schema.cypher

# (d) Load transactions into Neo4j (savings + credit card).
# Reads pre-normalized JSONL from data/statements_generated/*/_normalized
# and MERGEs Account / Statement / Month / Category / Merchant /
# Transaction / Day / Location / DescriptionTemplate nodes.
python -m src.ingestion.load_to_graph data/statements_generated/savings_stmt/_normalized
python -m src.ingestion.load_to_graph data/statements_generated/crdit_stmt/_normalized

# (e) Compile the LLM-facing wiki (Obsidian-style markdown vault).
# Reads from the now-populated graph, writes to
# data/wiki/{merchants,categories,months,annual}/*.md. Must run AFTER (d).
python -m src.ingestion.compile_wiki

# (f) Fraud analysis (writes Alert nodes + per-tx fraud_score)
# Rules only — fast, no GDS:
python -m src.fraud.run --skip-gds
# Or with the full graph-data-science pipeline (PageRank, Louvain,
# FastRP, KNN, node similarity, outlier marking):
python -m src.fraud.run

# (g) Verify
curl -s http://localhost:8000/api/graph/schema | python -m json.tool
# Transaction/Merchant/Category/Month/Statement/Account counts should
# all be populated (no nulls).
```

#### Notes

- **Idempotency**: `load_to_graph` is `MERGE`-based, so re-running won't duplicate data. Properties set `ON CREATE` (e.g. `t.year`, `t.amount`) won't update on subsequent runs — wipe (step b) if you need a clean reset.
- **Skip the parser**: the repo ships normalized JSONL in `data/statements_generated/*/_normalized/`, so you can go straight to step (d). The parser is only needed for raw `data/statements/*.md` ingest:
  ```bash
  python -m src.ingestion.parse_statements data/statements/
  ```
- **Restart the dev API** after re-populating only if you want the `/api/graph/schema` deprecation fix or other code changes — data updates are live.
- **OpenAI key**: `compile_wiki` and `src.fraud.run` don't need it. The chat agent (`POST /api/agent/ask`) does.

### 5. Run the backend

```bash
source .venv/bin/activate
uvicorn src.api.main:app --port 8000 --reload --reload-dir src
```

Use `--reload-dir src` so backend reloads do not thrash on `web/.next` or
generated wiki changes.

### 6. Run the frontend

```bash
cd web
npm install
npm run dev
```

Open <http://localhost:3000>.

## Workbench Guide

| Area | Purpose |
|---|---|
| Left wiki tree | Browse generated merchant/category/month/annual pages. |
| Center graph | Merchant -> Category and Merchant -> Month graph. Click nodes for details and wiki context. Double-click expands a one-hop neighborhood. |
| Graph focus | Chat tool results emit graph ids; focused turns isolate a relevant subgraph until reset. |
| Right alerts panel | Lists fraud/anomaly alerts for the current month or all months. |
| Right wiki panel | Shows the selected wiki page. Wikilinks are clickable. |
| Bottom chat | Streams planner, subagents, tool calls, and final answer. |
| Time scrubber | Selects a month and narrows graph/timeline/fraud alert context. |
| Money Flow | Emphasizes spend edges. |
| Decisions | Shows prior agent decisions linked to touched graph nodes. |
| Money Clock | Day-of-month spend rhythm view. |
| Compare | Side-by-side monthly mini-graph comparison. |
| Canon | Live editor for canonicalizer aliases, category locks, and cache eviction. |

### Graph Notes

The full graph is currently best treated as an overview, not as the primary
analysis surface. The more useful interaction pattern is:

1. Ask a specific question in chat.
2. Let the agent emit graph highlights from the query result.
3. Use the focused subgraph.
4. Click nodes for detail.
5. Double-click to expand around a node.
6. Reset focus when done.

This mirrors the context-graph pattern used by Neo4j examples: start with
retrieved context, show the local subgraph, and let the user expand from there.

## Fraud Transaction Design

The fraud layer is designed as a transaction triage surface, not as an
authoritative fraud decision engine. Use it to explain why a transaction is
worth reviewing.

### Data Model

Fraud scoring writes back to Neo4j:

- `Transaction.fraud_score`: float from `0.0` to `1.0`
- `Transaction.risk_flags`: list of rule names
- `(:Alert)-[:FLAGS]->(:Transaction)` for flagged transactions
- `Alert.kind`, `Alert.severity`, `Alert.rationale`, `Alert.created_at`

Additional graph features:

- `Day`: same-day grouping for duplicate, velocity, and card-testing rules
- `Location`: geographic signals when descriptions include location data
- `DescriptionTemplate`: normalized raw-description fingerprint
- `Merchant.pagerank`, `Merchant.community`, `Merchant.embedding`,
  `Merchant.is_outlier` from GDS

### Fraud Rules

Rules live in `src/fraud/rules.py`.

| Rule | What it flags |
|---|---|
| `duplicate_charge` | Same merchant, same day, same amount repeated. |
| `card_testing` | Small probe charge and larger same-day charge at different merchants. |
| `new_merchant_high_amount` | First transaction at a merchant is much larger than baseline spend. |
| `geo_mismatch` | Charge from an unusual country/location. |
| `velocity` | Many charges at the same merchant on the same day. |
| `round_fx` | Large round-number foreign charge. |

Scores are combined in `src/fraud/score.py`:

- rule signal weight: `0.6`
- GDS outlier/embedding-distance signal weight: `0.4`
- alert threshold: `0.50`

### Neo4j GDS Features

`src/fraud/gds.py` projects a merchant-coincidence graph where merchants are
linked when they occur on the same day. It then writes:

- PageRank
- Louvain community
- FastRP embeddings
- KNN similarity
- Node Similarity
- merchant outlier flag

Use `--skip-gds` when you want fast rule-only iteration.

### How To Use The Fraud UI

1. Load graph data and run `python -m src.fraud.run --skip-gds` or
   `python -m src.fraud.run`.
2. Start backend and frontend.
3. Open the workbench.
4. Use the time scrubber to select a month, or leave it on all months.
5. Review the `Alerts` panel in the right column.
6. Read each alert as: merchant, amount, date, location, score, and rationale.
7. Use the graph red rings to see merchants with high-risk alerts.
8. Click the merchant/node to open graph details and the wiki page.
9. Ask the chat agent questions like:
   - `Investigate the alerts for April 2026`
   - `Which alerts are highest confidence?`
   - `Explain the card testing alerts`
   - `Group suspicious transactions by merchant`

Recommended product direction:

- Treat the alert list as the primary fraud transaction UI.
- Keep the graph as supporting context, not the main fraud table.
- Add an alert detail drawer next: raw transaction fields, rule evidence,
  related same-day transactions, similar merchant history, and an analyst note.
- Add actions: `mark reviewed`, `false positive`, `needs follow-up`, and
  `export case`.
- Preserve every analyst decision as a `Decision` node linked to the alert,
  transaction, merchant, and month.

### Fraud API

```text
GET  /api/fraud/anomalies?month=YYYY-MM
GET  /api/fraud/score/{tx_id}
POST /api/fraud/recompute
POST /api/fraud/recompute?skip_gds=true
```

The frontend wrappers are in `web/lib/api.ts`:

- `fetchAnomalies(month?)`
- `recomputeFraud({ skipGds?: boolean })`

## Agent And Graph Focus

The workbench chat uses `src/api/agent_runner.py`, not the older
`src/agent/main.py` path. The runner:

1. Plans which subagents to run.
2. Runs scoped subagents:
   - `analyst`: Cypher against Neo4j
   - `wiki_browser`: wiki search/read
   - `advisor`: grounded recommendations and forecast ghosts
3. Streams events to the UI.
4. Emits `graph_highlight` ids from tool results.
5. Records a `Decision` node linked to touched graph entities.

An experimental Deep Agents 0.6 runtime now lives under `src/deep_retrieval/`
and is selected only with `AGENT_RUNTIME=deepagents`. It is designed to become
the retrieval-agent path after smoke tests and answer-quality comparisons.

For scalar queries, the backend also infers graph highlights from the Cypher
predicates. For example, a query about coffee in April 2026 can highlight:

```text
month:2026-04
merchant:Jamaica Blue
category:Coffee
```

## Important API Endpoints

```text
GET  /api/health
GET  /api/timeline
GET  /api/timeline/day_of_month
GET  /api/graph
GET  /api/graph?month=YYYY-MM
GET  /api/graph/expand?id=merchant:<name>
GET  /api/graph/trace?month=YYYY-MM
GET  /api/graph/decisions
GET  /api/wiki/tree
GET  /api/wiki/home
GET  /api/wiki/page?section=<section>&name=<name>
GET  /api/wiki/search?q=<term>
POST /api/agent/ask
GET  /api/agent/sessions/{id}
GET  /api/pii/preview
GET  /api/canon/cache
POST /api/canon/aliases
POST /api/canon/category-lock
DELETE /api/canon/cache/{raw}
GET  /api/fraud/anomalies
GET  /api/fraud/score/{tx_id}
POST /api/fraud/recompute
```

FastAPI OpenAPI docs are available at <http://localhost:8000/docs>.

## Stopping Servers

Use `Ctrl-C` in each terminal.

If a port is stuck:

```bash
lsof -ti :8000 | xargs kill -9
lsof -ti :3000 | xargs kill -9
```

## PII Handling

The PII layer uses:

- `src/pii/filter.py`: Presidio plus regex matching for finance-specific tokens
- `src/pii/vault.py`: process-local token vault
- `src/retriever/pii_middleware.py`: middleware for the older DeepAgents prototype path
- `src/deep_retrieval/cypher_guard.py`: read-only, schema-grounded guardrails for Deep Agents graph retrieval

The intended contract is that LLM-facing context receives stable tokens such
as `<PERSON_01>` or `<ACCT_02>` instead of raw names/account identifiers.

## Merchant Canonicalization

Merchant normalization is in `src/ingestion/normalize.py` and
`src/ingestion/llm_normalize.py`.

Backends:

- `NORMALIZER=llm`: cache-first LLM canonicalization with regex fallback
- `NORMALIZER=regex`: deterministic offline rules only

Useful files:

- `data/.canonicalize_cache.json`
- `/canon` UI
- `POST /api/canon/aliases`
- `POST /api/canon/category-lock`
- `DELETE /api/canon/cache/{raw}`

## Testing And Verification

Common checks:

```bash
python -m compileall src
cd web && ./node_modules/.bin/tsc --noEmit --incremental false
cd web && npm run build
```

If tests are present for your branch:

```bash
pytest tests/
```

Be careful with tests that touch Neo4j. Some fixtures can wipe the development
graph, so re-load sample data afterwards if needed.

## Known Limitations

- The full graph overview is usable but not a polished graph-exploration
  product. Query-focused subgraphs are the preferred interaction.
- Fraud rules are intentionally explainable and conservative, but they can
  produce false positives on synthetic or unusual data.
- `card_testing` in particular should be tuned per dataset.
- The sample transaction id strategy can merge truly identical rows. Add a
  source row number if duplicate-post detection must distinguish identical
  statement rows.
- Chat sessions are in-memory/localStorage oriented, not production storage.
- Secrets and API keys are development `.env` based.

## References

- https://github.com/johnymontana/context-graph-demo
- https://github.com/neo4j-labs/create-context-graph
- https://neo4j.com/docs/nvl/current/
- https://neo4j.com/blog/agentic-ai/hands-on-with-context-graphs-and-neo4j/
- https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

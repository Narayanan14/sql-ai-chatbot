# Snowflake SQL Chatbot

A Streamlit web app that lets you ask questions about your data in **plain English** and get back:

1. A validated **Snowflake SQL query**,
2. The **live query results** (executed directly against Snowflake), and
3. An **automatically chosen chart** (bar, line, heatmap, KPI cards, etc.) rendered with Altair.

Under the hood, [Google Gemini](https://ai.google.dev/) is given your live Snowflake table schema and asked to return a structured JSON object containing the SQL query *and* a recommendation for how to visualize the result. The app then executes the SQL against Snowflake and independently validates the visualization recommendation against the real result set before rendering it, so a bad or hallucinated chart suggestion never breaks the query results.

---

## Table of contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Data model](#data-model)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Using the chatbot](#using-the-chatbot)
- [Supported visualizations](#supported-visualizations)
- [Module reference](#module-reference)
- [Loading your own data](#loading-your-own-data)
- [Configuration reference](#configuration-reference)
- [Known limitations](#known-limitations)

---

## How it works

```
┌──────────────┐      1. question       ┌──────────────────┐
│   Streamlit   │ ─────────────────────▶ │   app.py          │
│   chat UI     │                        │  (build_prompt)    │
└──────────────┘                        └─────────┬─────────┘
                                                    │ 2. schema + question + prompt rules
                                                    ▼
                                         ┌────────────────────┐
                                         │  Gemini (google-genai)│
                                         │  structured JSON out  │
                                         │  -> QueryResult schema│
                                         └─────────┬──────────┘
                                                    │ 3. { sql, chart_type, x_axis, y_axis, ... }
                                                    ▼
                                         ┌────────────────────┐
                                         │ snowsqlexecute.py   │
                                         │ execute_query(sql)  │
                                         └─────────┬──────────┘
                                                    │ 4. columns + rows
                                                    ▼
                                         ┌────────────────────┐
                                         │ visualization.py     │
                                         │ validate + render    │
                                         │ chart with Altair     │
                                         └─────────┬──────────┘
                                                    │ 5. table + chart + CSV download
                                                    ▼
                                              Streamlit UI
```

Step by step, on every user question ([app.py](app.py)):

1. **Schema discovery** — [load_schema()](app.py) queries `INFORMATION_SCHEMA.COLUMNS` for the `ECOMMERCEMART` schema (via [snowsqlexecute.get_schema()](snowsqlexecute.py)) and caches the result for one hour (`st.cache_data(ttl=3600)`).
2. **Prompt construction** — [build_prompt()](app.py) builds a large instruction prompt containing the full table/column schema, the last 3 turns of conversation history (so follow-up questions like *"now break that down by region"* work), the user's question, strict SQL-generation rules, and a detailed rubric for choosing exactly one visualization type.
3. **Structured generation** — the prompt is sent to Gemini via `client.models.generate_content(...)` with `response_mime_type="application/json"` and a Pydantic `response_schema` (`QueryResult`), so Gemini is forced to return well-formed JSON containing the SQL plus visualization metadata (`chart_type`, `x_axis`, `y_axis`, `category_column`, `value_column`).
4. **Execution** — the generated SQL is executed as-is against Snowflake using SQLAlchemy ([snowsqlexecute.execute_query()](snowsqlexecute.py)). Timing for both the generation step and the execution step is captured and shown in the UI.
5. **Visualization validation & rendering** — [visualization.py](visualization.py) never trusts Gemini's chart recommendation blindly. `_validate_chart_spec()` re-derives/repairs the recommendation using the *actual* result DataFrame (correct column resolution, numeric-type detection, fallback heuristics, row-count/cardinality guards) before `render_visualization()` draws the chart. If validation fails, the app still shows the SQL and the results table — only the chart is skipped.
6. **Chat history** — results (SQL, DataFrame, chart metadata, timings) are stored in `st.session_state.messages` so the whole conversation persists across Streamlit reruns, and a condensed `question`/`sql` history is kept separately to give Gemini conversational context for follow-ups.

---

## Repository layout

```
sql_ai_chatbot/
├── app.py                      # Streamlit application (entry point)
├── config.py                   # Secret/env resolution helper
├── snowsqlexecute.py           # Snowflake connection, query execution, schema introspection
├── visualization.py            # Chart-spec validation + Altair chart rendering
├── requirements.txt            # Python dependencies
├── DataLoad_Script.ipynb       # One-time notebook that loads the CSV datasets into Snowflake
├── Datasets/                   # Sample e-commerce CSV data used to seed Snowflake
│   ├── customer_master.csv
│   ├── dataset_statistics.csv
│   ├── ecommerce_sales_customer_analytics_150k.csv
│   ├── order_items.csv
│   └── product_catalog.csv
├── .streamlit/
│   └── config.toml             # Streamlit theme (light theme, blue accent)
└── .env                        # Local secrets (git-ignored, you create this)
```

> `demo.py` may exist locally as a scratch CLI script for testing the Gemini/Snowflake round-trip outside of Streamlit, but it is intentionally excluded from version control (see `.gitignore`) and is not required to run the app.

---

## Data model

The app is wired to a Snowflake database/schema named **`DATA_ANALYTICS.ECOMMERCEMART`** (hard-coded in [snowsqlexecute.py](snowsqlexecute.py)). The sidebar's schema browser and Gemini's SQL generation are both driven entirely by whatever tables/columns actually exist in that schema — there is no hard-coded table list in the prompt.

The bundled sample dataset ([DataLoad_Script.ipynb](DataLoad_Script.ipynb)) loads five tables from [Datasets/](Datasets):

| Table | Rows (sample) | Description |
|---|---|---|
| `CUSTOMER_MASTER` | ~25,000 | One row per customer: demographics, segment, location, acquisition cost. |
| `ECOMMERCE_SALES_CUSTOMER_ANALYTICS` | ~138,000 | One row per order: order/payment/shipping status, customer snapshot, revenue/profit/margin, ratings, returns, marketing attribution, loyalty points. |
| `ORDER_ITEMS` | ~397,000 | One row per line item within an order: product, quantity, pricing, discounts, tax, shipping, profit. |
| `PRODUCT_CATALOG` | ~1,175 | One row per product: category/subcategory, brand, supplier, price, cost, rating. |
| `DATASET_STATISTICS` | 1 | Precomputed headline stats for the whole dataset (total revenue, profit, AOV, return rate, etc.). |

`ORDER_ITEMS.order_id` / `ECOMMERCE_SALES_CUSTOMER_ANALYTICS.order_id` and `*.product_id` / `PRODUCT_CATALOG.product_id` are the natural join keys between tables. Gemini is instructed to determine join paths itself from the live schema rather than relying on documented relationships, so keep column names self-descriptive if you load your own data.

---

## Prerequisites

- Python 3.12 (a `myenv/` virtual environment folder is already git-ignored; create your own instead of reusing it)
- A Snowflake account with a warehouse, database, and schema you can query
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

---

## Setup

1. **Clone and enter the repo, then create a virtual environment:**

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the project root (this file is git-ignored) with:

   ```env
   GEMINI_API_KEY=your-gemini-api-key
   SNOWFLAKE_USER_NAME=your-snowflake-username
   SNOWFLAKE_PASSWORD=your-snowflake-password
   SNOWFLAKE_ACCOUNT=your-snowflake-account-identifier
   SNOWFLAKE_WAREHOUSE=your-snowflake-warehouse
   ```

   Alternatively, if deploying on Streamlit Community Cloud, put the same keys in `.streamlit/secrets.toml` — [config.py](config.py)'s `get_secret()` checks `st.secrets` first and falls back to environment variables (loaded via `python-dotenv`), so either mechanism works locally or in the cloud.

3. **Point the app at your Snowflake data.** The database/schema (`DATA_ANALYTICS.ECOMMERCEMART`) is hard-coded in [snowsqlexecute.py](snowsqlexecute.py):

   ```python
   f'snowflake://{username}:{password}@{account}/DATA_ANALYTICS/ECOMMERCEMART'
   ```

   Either create a database named `DATA_ANALYTICS` with a schema named `ECOMMERCEMART` and load your tables into it, or edit this connection string to point at your own database/schema.

4. **(Optional) Load the sample dataset.** Open [DataLoad_Script.ipynb](DataLoad_Script.ipynb) in Jupyter and run all cells — it reads each CSV in [Datasets/](Datasets), connects to Snowflake using the same `.env` credentials, and appends the rows into `CUSTOMER_MASTER`, `DATASET_STATISTICS`, `ECOMMERCE_SALES_CUSTOMER_ANALYTICS`, `ORDER_ITEMS`, and `PRODUCT_CATALOG` (creating each table on first write via `to_sql(..., if_exists='append')`).

---

## Running the app

```powershell
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`). The theme (light mode, blue primary color) comes from [.streamlit/config.toml](.streamlit/config.toml).

---

## Using the chatbot

- **Model selector** (sidebar) — choose between `gemini-3.8-flash`, `gemini-2.5-flash`, and `gemini-2.5-pro`.
- **Schema browser** (sidebar) — lists every table and column currently visible in `ECOMMERCEMART`, with a text filter box. This is exactly the schema text Gemini receives, so it doubles as a way to sanity-check what the model can "see."
- **Example prompts** — shown as clickable chips when the chat is empty, e.g. *"Show the monthly revenue trend for 2025"*.
- **Chat input** — type any natural-language question about the data. Follow-up questions ("now split that by customer segment") work because the last 3 Q/SQL pairs are fed back into the prompt as context.
- **Per-response controls**:
  - Generated SQL is always shown in a code block.
  - Results are shown in a table, capped at the first **100 rows** on screen (a caption tells you if more rows were truncated; use the CSV download for the full result).
  - If a chart is available, a **"📊 Generate Visualization"** button appears — the chart is rendered lazily, only once requested, and then persists across reruns via `st.session_state`.
  - A **CSV download** button exports the full (untruncated) result set.
  - Row/column counts and generation/execution timings are shown under each response.
- **Clear chat** (sidebar) — wipes the conversation, the follow-up context, and any per-message visualization toggle state.

---

## Supported visualizations

Gemini is asked to recommend exactly one of the following chart types per query, based on rules embedded in [build_prompt()](app.py); [visualization.py](visualization.py) then validates/repairs that choice against the real result set:

| Chart type | When it's used | Notes |
|---|---|---|
| `bar` | Ranking/comparing one measure across categories | Auto-switches to horizontal bars for long labels or >8 categories; truncated to the top 20 by value with a caption if there are more. |
| `grouped_bar` | Comparing 2+ measures side-by-side per category | Falls back to `bar` if fewer than 2 valid numeric measures are resolved. |
| `stacked_bar` | Additive components summing to a meaningful total | Falls back to `bar` if fewer than 2 valid numeric measures are resolved. |
| `line` | Trends over an ordered/date dimension | Auto-detects datetime-like x-axis columns; requires ≥2 rows. |
| `area` | Cumulative/volume trends over time | Gradient fill; requires ≥2 rows. |
| `pie` | Small part-to-whole composition | Rendered as a donut chart; auto-converts to `bar` if there are more than 8 categories. |
| `scatter` | Relationship between two numeric measures | Auto-picks a low-cardinality categorical column (2–10 distinct values) to color points by, when available. |
| `histogram` | Distribution/spread of one continuous numeric measure | Uses Altair's automatic binning (max 30 bins). |
| `boxplot` | Comparing distributions/medians/outliers across categories | Requires both a categorical and a numeric column. |
| `heatmap` | Two-dimensional intensity pattern (e.g. day-of-week × hour) | Duplicate x/y combinations are summed; capped at 30×30 categories. |
| `kpi` | A handful of headline aggregate metrics | Rendered as `st.metric` cards (up to 6), with automatic `$`/`%`/thousand-separator formatting inferred from the column name (e.g. `REVENUE`, `RATE`, `MARGIN`). |
| `none` | Raw/lookup-style results | No chart is rendered. |

Any exception raised while validating or rendering a chart is swallowed — a broken visualization never takes down the SQL results.

---

## Module reference

### [app.py](app.py)
The Streamlit entry point. Responsible for: page config and CSS, the Gemini client (`get_client()`, cached with `st.cache_resource`), schema loading/caching (`load_schema()`, `schema_to_prompt_text()`), prompt construction (`build_prompt()`), the end-to-end query pipeline (`run_question()`), rendering each chat turn (`render_assistant_message()`), and all `st.session_state` management (`messages`, `history`, `pending_question`, per-message visualization toggles).

The Gemini response is parsed via the `QueryResult` Pydantic model:

```python
class QueryResult(BaseModel):
    sql: str
    question: str
    chart_type: ChartType = "none"       # one of the 12 types above
    x_axis: Optional[str] = None
    y_axis: List[str] = []
    category_column: Optional[str] = None
    value_column: Optional[str] = None
```

### [config.py](config.py)
`get_secret(key)` — reads a value from `st.secrets` first (for Streamlit Cloud deployments), falling back to `os.getenv(key)` (populated from `.env` via `python-dotenv`) if `st.secrets` isn't configured or the key is missing.

### [snowsqlexecute.py](snowsqlexecute.py)
- `execute_query(query)` — opens a fresh SQLAlchemy engine against Snowflake (`snowflake://user:password@account/DATA_ANALYTICS/ECOMMERCEMART?warehouse=...`), runs the query, returns `(columns, rows)`, and always disposes the engine in a `finally` block.
- `get_schema()` — runs `execute_query()` against `INFORMATION_SCHEMA.COLUMNS` filtered to `TABLE_SCHEMA = 'ECOMMERCEMART'`, returning every table/column/data-type triple currently in the schema.

### [visualization.py](visualization.py)
The largest module; kept intentionally decoupled from SQL generation so chart logic can evolve independently. Key pieces:
- **Helpers** — case-insensitive column resolution (`_resolve_column*`), numeric-type sniffing that tolerates Snowflake `DECIMAL` columns arriving as `object` dtype (`_numeric_candidates`), datetime detection (`_maybe_datetime`), axis number formatting (`_axis_format`).
- **`_validate_chart_spec()`** — the validation core. Takes Gemini's raw recommendation and the actual result DataFrame and returns either `None` (don't chart) or a normalized spec (`chart_type`, `x_axis`, `y_axis`, `category_column`, `value_column`, `color_col`) with per-chart-type fallback logic (e.g., inferring an axis Gemini omitted, demoting `pie`/`grouped_bar`/`stacked_bar` to `bar` when their preconditions aren't met).
- **Chart builders** — one function per chart type (`_bar_chart`, `_line_chart`, `_area_chart`, `_donut_chart`, `_scatter_chart`, `_histogram_chart`, `_boxplot_chart`, `_heatmap_chart`, `_render_kpis`), all built on **Altair**, sharing a consistent color palette and styling via `_finalize()`.
- **Public API** — `has_chart(...)` (cheap boolean check used by `app.py` to decide whether to show the "Generate Visualization" button) and `render_visualization(...)` (does the validation + rendering, called only after the user opts in).

### [DataLoad_Script.ipynb](DataLoad_Script.ipynb)
A one-time/idempotent-per-run ETL notebook: loads each CSV in [Datasets/](Datasets) with pandas, connects to Snowflake with the same `.env` credentials, and writes each DataFrame to Snowflake with `DataFrame.to_sql(..., schema='DATA_ANALYTICS.ECOMMERCEMART', if_exists='append')`, then verifies the load by re-querying `INFORMATION_SCHEMA.COLUMNS`.

---

## Loading your own data

The app makes no assumptions about table/column names beyond what's actually in `INFORMATION_SCHEMA.COLUMNS` for the `ECOMMERCEMART` schema — it re-reads the schema (cached for 1 hour) on every session. To point it at your own data:

1. Load your tables into a Snowflake schema (either reuse `DATA_ANALYTICS.ECOMMERCEMART` or edit the connection string in [snowsqlexecute.py](snowsqlexecute.py) and the `WHERE TABLE_SCHEMA = ...` filter in `get_schema()`).
2. Use descriptive table/column names — they are sent to Gemini verbatim as the only source of truth about your data.
3. Restart the app (or wait out the 1-hour schema cache) so the sidebar and prompt pick up the new schema.

---

## Configuration reference

| Variable | Where | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | `.env` / `st.secrets` | Auth for the Gemini API (`google-genai` client). |
| `SNOWFLAKE_USER_NAME` | `.env` | Snowflake login name. |
| `SNOWFLAKE_PASSWORD` | `.env` | Snowflake password. |
| `SNOWFLAKE_ACCOUNT` | `.env` | Snowflake account identifier (e.g. `xy12345.us-east-1`). |
| `SNOWFLAKE_WAREHOUSE` | `.env` | Snowflake warehouse used to run queries. |

`MAX_DISPLAY_ROWS` (100), `MODEL_OPTIONS`, and `EXAMPLE_QUESTIONS` are defined as constants near the top of [app.py](app.py) if you want to tune the on-screen row cap, offer different Gemini models, or change the example prompts.

---

## Known limitations

- **No SQL sandboxing** — the SQL Gemini generates is executed as-is against Snowflake with whatever privileges the configured Snowflake user has. Use a read-only role/warehouse for this user in any shared or production environment.
- **Single schema, single database** — the Snowflake connection string and schema filter (`DATA_ANALYTICS.ECOMMERCEMART`) are hard-coded rather than configurable via `.env`.
- **Conversation context is shallow** — only the last 3 Q/SQL pairs are replayed to Gemini for follow-up questions; longer multi-turn analytical threads may lose earlier context.
- **Chart selection is heuristic** — `_validate_chart_spec()` does its best to repair an invalid or missing recommendation from Gemini, but for unusual result shapes it may fall back to no chart at all (`chart_type = "none"`) rather than guessing wrong.

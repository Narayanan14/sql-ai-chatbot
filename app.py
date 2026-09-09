import json
import time
from datetime import datetime
from typing import List, Literal, Optional

import pandas as pd
import streamlit as st

from config import get_secret
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from snowsqlexecute import execute_query, get_schema
from visualization import has_chart, render_visualization


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

MAX_DISPLAY_ROWS = 100


st.set_page_config(
    page_title="Snowflake SQL Chatbot",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Structured Gemini response
# ---------------------------------------------------------------------------

ChartType = Literal[
    "bar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "pie",
    "scatter",
    "area",
    "histogram",
    "boxplot",
    "heatmap",
    "kpi",
    "none",
]


class QueryResult(BaseModel):
    sql: str
    question: str

    chart_type: ChartType = "none"

    # General chart axes.
    x_axis: Optional[str] = None
    y_axis: List[str] = Field(default_factory=list)

    # Additional metadata required for advanced visualizations.
    category_column: Optional[str] = None
    value_column: Optional[str] = None


# ---------------------------------------------------------------------------
# Models / examples
# ---------------------------------------------------------------------------

MODEL_OPTIONS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


EXAMPLE_QUESTIONS = [
    "Find the top 10 product categories by profit and show their revenue, quantity sold, and average customer rating",
    "Show the monthly revenue trend for 2025",
    "Show order volume by day of the week and hour of the day",
    "What are the total revenue, total profit, and average order value?",
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            180deg,
            #f7f9fc 0%,
            #eef2f9 100%
        );
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #2f80ed,
            #1c5fc4
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        color: #5b6472;
        font-size: 0.95rem;
        margin-top: 0.15rem;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.4rem 0.2rem;
    }

    .sql-card {
        background: #f4f6fb;
        border: 1px solid #dbe2ee;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .sql-label {
        color: #2f80ed;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .meta-row {
        display: flex;
        gap: 1.2rem;
        color: #6b7480;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    .example-chip button {
        border-radius: 999px !important;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e6eaf1;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

gemini_api_key = get_secret(
    "GEMINI_API_KEY"
)


@st.cache_resource(show_spinner=False)
def get_client():
    return genai.Client(
        api_key=gemini_api_key
    )


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

@st.cache_data(
    show_spinner=False,
    ttl=3600,
)
def load_schema():

    columns, rows = get_schema()

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    df.columns = [
        str(c).upper()
        for c in df.columns
    ]

    tables = {}

    for table_name, group in df.groupby(
        "TABLE_NAME"
    ):

        tables[table_name] = list(
            zip(
                group["COLUMN_NAME"],
                group["DATA_TYPE"],
            )
        )

    return df, tables


def schema_to_prompt_text(
    tables: dict,
) -> str:

    lines = []

    for table_name, cols in tables.items():

        lines.append(
            f"Table {table_name}:"
        )

        for col_name, data_type in cols:

            lines.append(
                f"  - {col_name} ({data_type})"
            )

    return "\n".join(
        lines
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    schema_text: str,
    history: list,
    user_prompt: str,
) -> str:

    context = ""

    if history:

        context_lines = [
            "Prior conversation "
            "(for resolving follow-up references):"
        ]

        for turn in history[-3:]:

            context_lines.append(
                f"Q: {turn['question']}"
            )

            context_lines.append(
                f"SQL: {turn['sql']}"
            )

        context = (
            "\n".join(context_lines)
            + "\n\n"
        )

    return f"""
You are an expert Snowflake SQL developer and data visualization analyst.

Schema:

{schema_text}

{context}

Question:

{user_prompt}

Generate a valid Snowflake SQL query using only the provided schema.

SQL Rules:

- Never invent tables, columns, or relationships.
- Verify all columns and JOIN keys exist in the schema.
- Determine the correct join path; use bridge tables when required.
- Handle table granularity carefully to avoid duplicate aggregations.
- Use valid Snowflake SQL.
- Always give every selected SQL expression an explicit alias.
- Ensure every visualization field references a column alias that will literally appear in the SQL result.
- Ensure the SQL result contains every field required by the recommended visualization.
- Do not select unnecessary columns merely to create a visualization.
- Preserve the analytical meaning of the user's question.

Also recommend exactly one primary visualization based on the semantic meaning of the question and the expected query result.

Supported visualization types:

bar:
Use for ranking or comparing one primary numeric measure across categories such as products, brands, regions, segments, or categories.

grouped_bar:
Use when two or more comparable numeric measures should be compared side-by-side across the same categories.

stacked_bar:
Use when multiple additive components contribute to a meaningful total across categories.

line:
Use for trends and changes across an ordered dimension, especially daily, monthly, quarterly, or yearly measures.

area:
Use for cumulative metrics or volume-oriented trends over time where magnitude is important.
Do not choose area simply because a date column exists.

pie:
Use only for meaningful part-to-whole composition with a small number of categories.
Do not use pie for ranking, trends, distributions, or large category counts.

scatter:
Use when the question asks about the relationship, correlation, or association between two meaningful numeric measures.
Each result row should represent an observation where possible.

histogram:
Use when the user asks about the distribution, frequency, spread, concentration, common range, or shape of one continuous numeric measure.
The SQL should normally return the raw numeric observations required to create the histogram rather than pre-aggregating them into arbitrary bins.

boxplot:
Use when comparing distributions, medians, quartiles, variability, spread, or outliers of a numeric measure across categories.
The SQL result should contain the category and underlying numeric observations needed to calculate the box plot.

heatmap:
Use for a two-dimensional matrix or intensity pattern such as:
- day of week vs hour of day
- month vs customer segment
- category vs region
- another pair of dimensions with a numeric measure represented by color intensity

kpi:
Use for one or a small number of aggregate headline business metrics such as:
- total revenue
- total profit
- average order value
- customer count
- order count
- return rate

none:
Use for raw detail results, lookup-style queries, record-level tables, or whenever visualization would add little analytical value.

Visualization selection rules:

- Choose the visualization that communicates the user's analytical intent most clearly.
- Do not default to bar merely because the result contains a category and numeric value.
- Prefer heatmap for two-dimensional intensity patterns.
- Prefer histogram when the user explicitly asks about the distribution of one continuous numeric variable.
- Prefer boxplot when the question concerns median, quartiles, spread, variability, distributions across groups, or outliers.
- Prefer KPI instead of none when the result consists of a few important aggregate metrics.
- Use none only when visualizing the result genuinely adds little value.
- Do not use scatter unless both axes represent meaningful numeric variables.
- Do not use pie when there are many categories.
- Do not plot unrelated metrics with dramatically different units/scales together simply because multiple measures exist.
- Recommend only one primary visualization.

Visualization metadata:

x_axis:
- Result alias used as the primary X dimension.
- May be null if the visualization does not naturally require one.

y_axis:
- List of result aliases representing numeric measures.
- May contain multiple measures for grouped_bar, stacked_bar, KPI, or selectable metric charts.

category_column:
- Additional categorical/dimensional result alias when needed.
- Required particularly for boxplot and heatmap when there is a second dimension.
- May be null when unnecessary.

value_column:
- Primary numeric result alias used by histogram, boxplot, or heatmap.
- May also be supplied for other visualizations where useful.

Chart-specific metadata requirements:

bar:
- x_axis = category alias
- y_axis = one or more numeric aliases

grouped_bar:
- x_axis = category alias
- y_axis = two or more comparable numeric measure aliases

stacked_bar:
- x_axis = category alias
- y_axis = two or more additive numeric component aliases

line:
- x_axis = ordered/date alias
- y_axis = numeric measure aliases

area:
- x_axis = ordered/date alias
- y_axis = numeric measure aliases

pie:
- x_axis = category alias
- y_axis = exactly one numeric measure alias

scatter:
- x_axis = first numeric measure alias
- y_axis = second numeric measure alias

histogram:
- value_column = numeric observation alias
- x_axis may also contain the same alias
- y_axis may be empty

boxplot:
- category_column = categorical grouping alias
- value_column = numeric observation alias
- x_axis may contain the category alias
- y_axis may contain the value alias

heatmap:
- x_axis = first dimension alias
- category_column = second dimension alias
- value_column = numeric intensity measure alias
- y_axis may contain the numeric intensity alias

kpi:
- y_axis = aggregate KPI aliases
- x_axis should normally be null

none:
- visualization fields may be null or empty

Critical requirements:

- Every visualization alias MUST exactly match an alias in the SQL SELECT result.
- Never invent visualization metadata.
- Do not reference underlying source-column names when the SQL result uses a different alias.
- The SQL and visualization recommendation must be mutually consistent.
"""


# ---------------------------------------------------------------------------
# Query generation + execution
# ---------------------------------------------------------------------------

def run_question(
    user_prompt: str,
    model: str,
):

    _, tables = load_schema()

    schema_text = schema_to_prompt_text(
        tables
    )

    final_prompt = build_prompt(
        schema_text,
        st.session_state.get(
            "history",
            [],
        ),
        user_prompt,
    )

    client = get_client()

    # ---------------------------------------------------------------
    # Gemini SQL generation
    # ---------------------------------------------------------------

    t0 = time.perf_counter()

    response = client.models.generate_content(
        model=model,
        contents=final_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QueryResult,
        ),
    )

    gen_seconds = (
        time.perf_counter()
        - t0
    )

    # ---------------------------------------------------------------
    # Parse structured response
    # ---------------------------------------------------------------

    json_response = json.loads(
        response.text
    )

    final_sql = json_response[
        "sql"
    ]

    chart_type = str(
        json_response.get(
            "chart_type"
        )
        or "none"
    ).strip().lower()

    x_axis = json_response.get(
        "x_axis"
    )

    y_axis = (
        json_response.get(
            "y_axis"
        )
        or []
    )

    if isinstance(
        y_axis,
        str,
    ):
        y_axis = [
            y_axis
        ]

    category_column = (
        json_response.get(
            "category_column"
        )
    )

    value_column = (
        json_response.get(
            "value_column"
        )
    )

    # ---------------------------------------------------------------
    # Snowflake execution
    # ---------------------------------------------------------------

    t1 = time.perf_counter()

    columns, rows = execute_query(
        final_sql
    )

    exec_seconds = (
        time.perf_counter()
        - t1
    )

    result_df = pd.DataFrame(
        rows,
        columns=columns,
    )

    return (
        final_sql,
        result_df,
        gen_seconds,
        exec_seconds,
        chart_type,
        x_axis,
        y_axis,
        category_column,
        value_column,
    )


# ---------------------------------------------------------------------------
# Assistant response rendering
# ---------------------------------------------------------------------------

def render_assistant_message(
    msg: dict,
):
    """
    Render a single assistant turn.

    Shared by:
        - historical responses
        - newest generated response

    Visualization metadata is retained in session state so charts can be
    reconstructed correctly after Streamlit reruns.
    """

    # ---------------------------------------------------------------
    # Error handling
    # ---------------------------------------------------------------

    if msg.get(
        "error"
    ):

        st.error(
            msg["error"]
        )

        if msg.get(
            "sql"
        ):

            st.markdown(
                '<div class="sql-label">'
                'Generated SQL'
                '</div>',
                unsafe_allow_html=True,
            )

            st.code(
                msg["sql"],
                language="sql",
            )

        return

    df = msg[
        "result_df"
    ]

    msg_key = id(
        msg
    )

    # ---------------------------------------------------------------
    # Generated SQL
    # ---------------------------------------------------------------

    st.markdown(
        '<div class="sql-label">'
        'Generated SQL'
        '</div>',
        unsafe_allow_html=True,
    )

    st.code(
        msg["sql"],
        language="sql",
    )

    # ---------------------------------------------------------------
    # Query result table
    # ---------------------------------------------------------------

    st.markdown(
        '<div class="sql-label">'
        'Query Results'
        '</div>',
        unsafe_allow_html=True,
    )

    display_df = df.head(
        MAX_DISPLAY_ROWS
    )

    st.dataframe(
        display_df,
        width="stretch",
        height=min(
            400,
            45 + 35 * len(display_df),
        ),
    )

    if len(df) > MAX_DISPLAY_ROWS:

        st.caption(
            f"Showing first {MAX_DISPLAY_ROWS} "
            f"of {len(df):,} rows — "
            "download the CSV for the full result."
        )

    # ---------------------------------------------------------------
    # Visualization availability check
    # ---------------------------------------------------------------

    chart_available = has_chart(
        df=df,
        chart_type=msg.get(
            "chart_type",
            "none",
        ),
        x_axis=msg.get(
            "x_axis"
        ),
        y_axis=msg.get(
            "y_axis"
        )
        or [],
        category_column=msg.get(
            "category_column"
        ),
        value_column=msg.get(
            "value_column"
        ),
    )

    if chart_available:

        show_viz_key = (
            f"show_viz_{msg_key}"
        )

        # -----------------------------------------------------------
        # Render visualization after it has been requested
        # -----------------------------------------------------------

        if st.session_state.get(
            show_viz_key
        ):

            render_visualization(
                df=df,
                chart_type=msg.get(
                    "chart_type",
                    "none",
                ),
                x_axis=msg.get(
                    "x_axis"
                ),
                y_axis=msg.get(
                    "y_axis"
                )
                or [],
                category_column=msg.get(
                    "category_column"
                ),
                value_column=msg.get(
                    "value_column"
                ),
                question=msg.get(
                    "question"
                ),
                key_prefix=f"viz_{msg_key}",
            )

        # -----------------------------------------------------------
        # Visualization trigger button
        # -----------------------------------------------------------

        elif st.button(
            "📊 Generate Visualization",
            key=f"genviz_{msg_key}",
        ):

            st.session_state[
                show_viz_key
            ] = True

            st.rerun()

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    st.markdown(
        f'<div class="meta-row">'
        f'<span>'
        f'📊 {len(df):,} rows × '
        f'{len(df.columns)} cols'
        f'</span>'
        f'<span>'
        f'⏱️ gen {msg["gen_seconds"]:.2f}s · '
        f'exec {msg["exec_seconds"]:.2f}s'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------
    # CSV download
    # ---------------------------------------------------------------

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),
        file_name=(
            "query_result_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ".csv"
        ),
        mime="text/csv",
        key=f"dl_{msg_key}",
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "history" not in st.session_state:
    st.session_state.history = []


if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        "### ⚙️ Settings"
    )

    model = st.selectbox(
        "Model",
        MODEL_OPTIONS,
        index=0,
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 📚 Schema"
    )

    schema_error = None

    try:

        schema_df, tables = load_schema()

    except Exception as exc:

        schema_error = str(
            exc
        )

        tables = {}

    if schema_error:

        st.error(
            "Could not load schema:"
            f"\n\n{schema_error}"
        )

    else:

        st.caption(
            f"{len(tables)} tables available"
        )

        search = st.text_input(
            "Filter tables",
            placeholder="e.g. orders",
        )

        for table_name, cols in tables.items():

            if (
                search
                and search.lower()
                not in table_name.lower()
            ):
                continue

            with st.expander(
                table_name,
                expanded=False,
            ):

                for col_name, data_type in cols:

                    st.markdown(
                        f"`{col_name}`  "
                        f"*{data_type}*"
                    )

    st.markdown(
        "---"
    )

    if st.button(
        "🗑️ Clear chat",
        width="stretch",
    ):

        st.session_state.messages = []
        st.session_state.history = []

        # Clear visualization state created by previous messages.
        for key in list(
            st.session_state.keys()
        ):

            if (
                str(key).startswith(
                    "show_viz_"
                )
                or str(key).startswith(
                    "viz_"
                )
            ):

                del st.session_state[
                    key
                ]

        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🗄️ Snowflake SQL Chatbot'
    '</div>',
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="subtitle">'
    'Ask questions in plain English — '
    'get SQL, live results, and intelligent visualizations '
    'from your warehouse.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

if not st.session_state.messages:

    st.markdown(
        "**Try asking:**"
    )

    for i in range(
        0,
        len(EXAMPLE_QUESTIONS),
        2,
    ):

        cols = st.columns(
            2
        )

        for col, question in zip(
            cols,
            EXAMPLE_QUESTIONS[
                i:
                i + 2
            ],
        ):

            with col:

                if st.button(
                    question,
                    width="stretch",
                    key=f"ex_{question}",
                ):

                    st.session_state.pending_question = (
                        question
                    )

                    st.rerun()


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:

    if msg[
        "role"
    ] == "user":

        with st.chat_message(
            "user"
        ):

            st.markdown(
                msg["content"]
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🗄️",
        ):

            render_assistant_message(
                msg
            )


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

chat_input = st.chat_input(
    "Ask a question about your data..."
)


user_question = (
    chat_input
    or st.session_state.pending_question
)


st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# Process question
# ---------------------------------------------------------------------------

if user_question:

    # ---------------------------------------------------------------
    # Save user message
    # ---------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )

    # ---------------------------------------------------------------
    # Assistant generation
    # ---------------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🗄️",
    ):

        with st.spinner(
            "Generating SQL and querying Snowflake..."
        ):

            try:

                (
                    sql,
                    result_df,
                    gen_seconds,
                    exec_seconds,
                    chart_type,
                    x_axis,
                    y_axis,
                    category_column,
                    value_column,
                ) = run_question(
                    user_question,
                    model,
                )

            except json.JSONDecodeError:

                error_msg = (
                    "The model returned an unexpected response. "
                    "Please try rephrasing your question."
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "error": error_msg,
                    }
                )

            except Exception as exc:

                error_msg = (
                    f"Query failed: {exc}"
                )

                sql_attempted = locals().get(
                    "sql"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "error": error_msg,
                        "sql": sql_attempted,
                    }
                )

            else:

                # ---------------------------------------------------
                # Save full assistant response including
                # visualization metadata.
                # ---------------------------------------------------

                assistant_message = {
                    "role": "assistant",
                    "sql": sql,
                    "result_df": result_df,
                    "gen_seconds": gen_seconds,
                    "exec_seconds": exec_seconds,

                    "chart_type": chart_type,
                    "x_axis": x_axis,
                    "y_axis": y_axis,

                    "category_column": category_column,
                    "value_column": value_column,

                    "question": user_question,
                }

                st.session_state.messages.append(
                    assistant_message
                )

                # ---------------------------------------------------
                # Save conversation context used by Gemini for
                # follow-up SQL questions.
                # ---------------------------------------------------

                st.session_state.history.append(
                    {
                        "question": user_question,
                        "sql": sql,
                    }
                )
    st.rerun()
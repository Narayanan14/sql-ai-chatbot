"""
Chart rendering for query results.

Gemini recommends visualization metadata alongside the SQL it generates.
This module validates that recommendation against the actual result DataFrame
and renders the visualization using Altair / Streamlit.

Supported visualizations:
    - bar
    - grouped_bar
    - stacked_bar
    - line
    - area
    - pie (rendered as donut)
    - scatter
    - histogram
    - boxplot
    - heatmap
    - kpi
    - none

The visualization layer is intentionally kept separate from SQL generation
so chart logic can evolve independently without affecting query execution.
"""

import textwrap
from numbers import Number

import altair as alt
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Altair configuration
# ---------------------------------------------------------------------

alt.data_transformers.disable_max_rows()


# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------

PRIMARY = "#2f80ed"

PALETTE = [
    "#2f80ed",
    "#56ccf2",
    "#1c5fc4",
    "#27ae60",
    "#f2994a",
    "#9b51e0",
    "#eb5757",
    "#219653",
    "#f2c94c",
    "#6b7480",
]

GRID_COLOR = "#e6eaf1"
TEXT_DARK = "#1a1e2a"
TEXT_MUTED = "#5b6472"


# ---------------------------------------------------------------------
# Supported visualizations
# ---------------------------------------------------------------------

ALLOWED_CHART_TYPES = {
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
}


# ---------------------------------------------------------------------
# Limits / display configuration
# ---------------------------------------------------------------------

MAX_DONUT_CATEGORIES = 8
MIN_ROWS_FOR_TREND = 2

DEFAULT_CHART_HEIGHT = 340

PER_BAR_PX = 26
MIN_HORIZONTAL_BAR_HEIGHT = 340
MAX_HORIZONTAL_BAR_HEIGHT = 6000

SCROLL_VIEWPORT_HEIGHT = 560

MAX_BAR_CATEGORIES = 20
MAX_HEATMAP_X_CATEGORIES = 30
MAX_HEATMAP_Y_CATEGORIES = 30
MAX_KPI_CARDS = 6


# =====================================================================
# GENERAL HELPERS
# =====================================================================


def _pretty(name: str) -> str:
    """
    Convert SQL-style aliases into user-friendly labels.

    Example:
        TOTAL_REVENUE -> Total Revenue
    """
    return str(name).replace("_", " ").strip().title()


def _resolve_column(name, columns):
    """
    Resolve a Gemini-provided column name against the actual DataFrame
    columns in a case-insensitive manner.
    """
    if not name:
        return None

    name = str(name).strip()

    if name in columns:
        return name

    lower_map = {
        str(c).lower(): c
        for c in columns
    }

    return lower_map.get(name.lower())


def _resolve_column_list(names, columns):
    """
    Resolve a list of Gemini-provided result aliases against actual
    DataFrame columns.
    """
    if not names:
        return []

    if not isinstance(names, list):
        names = [names]

    resolved = []

    for name in names:
        col = _resolve_column(name, columns)

        if col and col not in resolved:
            resolved.append(col)

    return resolved


def _numeric_candidates(df, cols):
    """
    Return columns where at least half of non-empty rows can be parsed
    as numeric.

    This handles Snowflake Decimal values which may arrive as object
    dtype in pandas.
    """
    ok = []

    n = len(df)

    if n == 0:
        return ok

    for c in cols:
        if c not in df.columns:
            continue

        coerced = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        valid_count = coerced.notna().sum()

        if valid_count >= max(
            1,
            int(n * 0.5),
        ):
            ok.append(c)

    return ok


def _categorical_candidates(df, columns, numeric_ok):
    """
    Return non-numeric columns that can reasonably act as dimensions.
    """
    return [
        c
        for c in columns
        if c not in numeric_ok
    ]


def _pick_color_col(df, exclude, numeric_ok):
    """
    Find a small-cardinality categorical column suitable for coloring
    scatter points.
    """
    for c in df.columns:

        if c in exclude:
            continue

        if c in numeric_ok:
            continue

        nunique = df[c].nunique(dropna=True)

        if 2 <= nunique <= 10:
            return c

    return None


def _axis_format(series):
    """
    Determine sensible numeric formatting for Altair axes/tooltips.
    """
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if s.empty:
        return ",.2f"

    if (s == s.round()).all():
        return ",.0f"

    return ",.2f"


def _maybe_datetime(series):
    """
    Detect whether a result column is primarily datetime-like.
    """
    converted = pd.to_datetime(
        series,
        errors="coerce",
    )

    threshold = max(
        1,
        int(len(series) * 0.7),
    )

    if converted.notna().sum() >= threshold:
        return converted

    return None


def _prepare_plot_df(df, x, y_cols):
    """
    Create a safe plotting DataFrame and convert specified measure
    columns to numeric.
    """
    cols = [x] + [
        c
        for c in y_cols
        if c != x
    ]

    cols = list(dict.fromkeys(cols))

    plot_df = df[cols].copy()

    for c in y_cols:
        if c in plot_df.columns:
            plot_df[c] = pd.to_numeric(
                plot_df[c],
                errors="coerce",
            )

    return plot_df


def _prepare_numeric_series(df, column):
    """
    Extract and numerically coerce a single column.
    """
    if column not in df.columns:
        return None

    plot_df = df[[column]].copy()

    plot_df[column] = pd.to_numeric(
        plot_df[column],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[column]
    )

    if plot_df.empty:
        return None

    return plot_df


# =====================================================================
# VISUALIZATION VALIDATION
# =====================================================================


def _validate_chart_spec(
    df,
    chart_type,
    x_axis=None,
    y_axis=None,
    category_column=None,
    value_column=None,
):
    """
    Validate Gemini's visualization recommendation against the real
    query result.

    Returns a normalized visualization specification or None.

    This validation layer is important because Gemini chooses the
    visualization before Python sees the actual result data.
    """

    # --------------------------------------------------------------
    # Basic chart type validation
    # --------------------------------------------------------------

    chart_type = (
        str(chart_type).strip().lower()
        if chart_type
        else "none"
    )

    # Gemini may occasionally say "donut".
    if chart_type == "donut":
        chart_type = "pie"

    # Common alternative spellings.
    aliases = {
        "grouped bar": "grouped_bar",
        "grouped-bar": "grouped_bar",
        "stacked bar": "stacked_bar",
        "stacked-bar": "stacked_bar",
        "box": "boxplot",
        "box_plot": "boxplot",
        "box plot": "boxplot",
        "heat_map": "heatmap",
        "heat map": "heatmap",
        "hist": "histogram",
        "metric": "kpi",
        "metrics": "kpi",
        "cards": "kpi",
    }

    chart_type = aliases.get(
        chart_type,
        chart_type,
    )

    if chart_type not in ALLOWED_CHART_TYPES:
        return None

    if chart_type == "none":
        return None

    if df is None or df.empty:
        return None

    columns = list(df.columns)

    numeric_ok = _numeric_candidates(
        df,
        columns,
    )

    categorical_ok = _categorical_candidates(
        df,
        columns,
        numeric_ok,
    )

    # --------------------------------------------------------------
    # Resolve Gemini aliases against actual result columns
    # --------------------------------------------------------------

    x = _resolve_column(
        x_axis,
        columns,
    )

    y_resolved = _resolve_column_list(
        y_axis,
        columns,
    )

    category = _resolve_column(
        category_column,
        columns,
    )

    value = _resolve_column(
        value_column,
        columns,
    )

    # ==============================================================
    # KPI
    # ==============================================================

    if chart_type == "kpi":

        metrics = [
            c
            for c in y_resolved
            if c in numeric_ok
        ]

        if value and value in numeric_ok:
            if value not in metrics:
                metrics.append(value)

        # If Gemini forgot metadata, infer numeric columns.
        if not metrics:
            metrics = numeric_ok.copy()

        if not metrics:
            return None

        metrics = metrics[:MAX_KPI_CARDS]

        return {
            "chart_type": "kpi",
            "x_axis": None,
            "y_axis": metrics,
            "category_column": None,
            "value_column": None,
            "color_col": None,
        }

    # ==============================================================
    # HISTOGRAM
    # ==============================================================

    if chart_type == "histogram":

        histogram_value = value

        if histogram_value not in numeric_ok:
            histogram_value = next(
                (
                    c
                    for c in y_resolved
                    if c in numeric_ok
                ),
                None,
            )

        if histogram_value is None and x in numeric_ok:
            histogram_value = x

        if histogram_value is None and numeric_ok:
            histogram_value = numeric_ok[0]

        if histogram_value is None:
            return None

        if len(df) < 2:
            return None

        return {
            "chart_type": "histogram",
            "x_axis": histogram_value,
            "y_axis": [],
            "category_column": None,
            "value_column": histogram_value,
            "color_col": None,
        }

    # ==============================================================
    # BOXPLOT
    # ==============================================================

    if chart_type == "boxplot":

        box_category = category

        if not box_category:
            if x and x not in numeric_ok:
                box_category = x
            elif categorical_ok:
                box_category = categorical_ok[0]

        box_value = value

        if box_value not in numeric_ok:
            box_value = next(
                (
                    c
                    for c in y_resolved
                    if c in numeric_ok
                ),
                None,
            )

        if box_value is None:
            box_value = next(
                (
                    c
                    for c in numeric_ok
                    if c != box_category
                ),
                None,
            )

        if not box_category or not box_value:
            return None

        if box_category == box_value:
            return None

        if len(df) < 2:
            return None

        return {
            "chart_type": "boxplot",
            "x_axis": box_category,
            "y_axis": [box_value],
            "category_column": box_category,
            "value_column": box_value,
            "color_col": None,
        }

    # ==============================================================
    # HEATMAP
    # ==============================================================

    if chart_type == "heatmap":

        heat_x = x

        # First dimension.
        if not heat_x:
            if categorical_ok:
                heat_x = categorical_ok[0]
            else:
                heat_x = columns[0]

        # Second dimension.
        heat_category = category

        if not heat_category:
            heat_category = next(
                (
                    c
                    for c in categorical_ok
                    if c != heat_x
                ),
                None,
            )

        # It is possible for hour/day dimensions to be numeric.
        # If no categorical dimension exists, allow another result
        # column to serve as the Y dimension.
        if not heat_category:
            heat_category = next(
                (
                    c
                    for c in columns
                    if c != heat_x
                    and c != value
                    and c not in y_resolved
                ),
                None,
            )

        heat_value = value

        if heat_value not in numeric_ok:
            heat_value = next(
                (
                    c
                    for c in y_resolved
                    if c in numeric_ok
                    and c != heat_x
                    and c != heat_category
                ),
                None,
            )

        if heat_value is None:
            heat_value = next(
                (
                    c
                    for c in numeric_ok
                    if c != heat_x
                    and c != heat_category
                ),
                None,
            )

        if not heat_x or not heat_category or not heat_value:
            return None

        if len({
            heat_x,
            heat_category,
            heat_value,
        }) < 3:
            return None

        return {
            "chart_type": "heatmap",
            "x_axis": heat_x,
            "y_axis": [heat_value],
            "category_column": heat_category,
            "value_column": heat_value,
            "color_col": None,
        }

    # ==============================================================
    # SCATTER
    # ==============================================================

    if chart_type == "scatter":

        y_numeric = [
            c
            for c in y_resolved
            if c in numeric_ok
            and c != x
        ]

        if value in numeric_ok and value != x:
            if value not in y_numeric:
                y_numeric.insert(
                    0,
                    value,
                )

        # If Gemini's metadata is invalid, try first two numeric
        # result columns.
        if x not in numeric_ok or not y_numeric:

            fallback_pair = numeric_ok[:2]

            if len(fallback_pair) < 2:
                return None

            x = fallback_pair[0]
            y_numeric = [
                fallback_pair[1]
            ]

        if len(df) < MIN_ROWS_FOR_TREND:
            return None

        y = y_numeric[0]

        color_col = _pick_color_col(
            df,
            exclude={x, y},
            numeric_ok=numeric_ok,
        )

        return {
            "chart_type": "scatter",
            "x_axis": x,
            "y_axis": [y],
            "category_column": None,
            "value_column": y,
            "color_col": color_col,
        }

    # ==============================================================
    # BAR / GROUPED BAR / STACKED BAR / LINE / AREA / PIE
    # ==============================================================

    y_valid = [
        c
        for c in y_resolved
        if c in numeric_ok
    ]

    if value and value in numeric_ok:
        if value not in y_valid:
            y_valid.append(value)

    if not y_valid:
        y_valid = [
            c
            for c in numeric_ok
            if c != x
        ]

    if not y_valid:
        return None

    # Infer an x-axis if Gemini didn't provide a valid one.
    if not x or x not in columns:

        non_numeric = [
            c
            for c in columns
            if c not in numeric_ok
        ]

        x = (
            non_numeric[0]
            if non_numeric
            else next(
                (
                    c
                    for c in columns
                    if c not in y_valid
                ),
                None,
            )
        )

    if not x:
        return None

    y_valid = [
        c
        for c in y_valid
        if c != x
    ]

    if not y_valid:
        return None

    # --------------------------------------------------------------
    # Trend validation
    # --------------------------------------------------------------

    if chart_type in (
        "line",
        "area",
    ):
        if len(df) < MIN_ROWS_FOR_TREND:
            return None

    # --------------------------------------------------------------
    # Pie validation
    # --------------------------------------------------------------

    if chart_type == "pie":

        if df[x].nunique(dropna=True) > MAX_DONUT_CATEGORIES:
            # Too many slices → bar is clearer.
            chart_type = "bar"
        else:
            y_valid = y_valid[:1]

    # --------------------------------------------------------------
    # Grouped / stacked bar validation
    # --------------------------------------------------------------

    if chart_type in (
        "grouped_bar",
        "stacked_bar",
    ):

        # These require at least two numeric measures when represented
        # in wide format.
        if len(y_valid) < 2:
            # A regular bar communicates a single measure better.
            chart_type = "bar"

    return {
        "chart_type": chart_type,
        "x_axis": x,
        "y_axis": y_valid,
        "category_column": category,
        "value_column": value,
        "color_col": None,
    }


# =====================================================================
# BASIC CHARTS
# =====================================================================


def _bar_chart(df, x, y):

    plot_df = _prepare_plot_df(
        df,
        x,
        [y],
    ).dropna(
        subset=[y]
    )

    if plot_df.empty:
        return None

    horizontal = (
        plot_df[x]
        .astype(str)
        .str.len()
        .max()
        > 14
        or len(plot_df) > 8
    )

    fmt = _axis_format(
        plot_df[y]
    )

    x_title = _pretty(x)
    y_title = _pretty(y)

    tooltip = [
        alt.Tooltip(
            x,
            type="nominal",
            title=x_title,
        ),
        alt.Tooltip(
            y,
            type="quantitative",
            title=y_title,
            format=fmt,
        ),
    ]

    if horizontal:

        height = int(
            min(
                MAX_HORIZONTAL_BAR_HEIGHT,
                max(
                    MIN_HORIZONTAL_BAR_HEIGHT,
                    PER_BAR_PX * len(plot_df),
                ),
            )
        )

        return (
            alt.Chart(plot_df)
            .mark_bar(
                cornerRadiusEnd=4,
                color=PRIMARY,
            )
            .encode(
                y=alt.Y(
                    x,
                    type="nominal",
                    title=x_title,
                    sort="-x",
                ),
                x=alt.X(
                    y,
                    type="quantitative",
                    title=y_title,
                    axis=alt.Axis(
                        format=fmt
                    ),
                ),
                tooltip=tooltip,
            )
            .properties(
                height=height
            )
        )

    return (
        alt.Chart(plot_df)
        .mark_bar(
            cornerRadiusEnd=4,
            color=PRIMARY,
            size=32,
        )
        .encode(
            x=alt.X(
                x,
                type="nominal",
                title=x_title,
                sort="-y",
            ),
            y=alt.Y(
                y,
                type="quantitative",
                title=y_title,
                axis=alt.Axis(
                    format=fmt
                ),
            ),
            tooltip=tooltip,
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
    )


# =====================================================================
# GROUPED BAR
# =====================================================================


def _grouped_bar_chart(
    df,
    x,
    y_cols,
):

    if not y_cols:
        return None

    plot_df = _prepare_plot_df(
        df,
        x,
        y_cols,
    )

    melted = plot_df.melt(
        id_vars=[x],
        value_vars=y_cols,
        var_name="_metric",
        value_name="_value",
    )

    melted["_value"] = pd.to_numeric(
        melted["_value"],
        errors="coerce",
    )

    melted = melted.dropna(
        subset=["_value"]
    )

    if melted.empty:
        return None

    return (
        alt.Chart(melted)
        .mark_bar(
            cornerRadiusEnd=3
        )
        .encode(
            x=alt.X(
                x,
                type="nominal",
                title=_pretty(x),
            ),
            xOffset=alt.XOffset(
                "_metric:N"
            ),
            y=alt.Y(
                "_value:Q",
                title="Value",
            ),
            color=alt.Color(
                "_metric:N",
                title=None,
                scale=alt.Scale(
                    range=PALETTE
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    x,
                    type="nominal",
                    title=_pretty(x),
                ),
                alt.Tooltip(
                    "_metric:N",
                    title="Metric",
                ),
                alt.Tooltip(
                    "_value:Q",
                    title="Value",
                    format=",.2f",
                ),
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
    )


# =====================================================================
# STACKED BAR
# =====================================================================


def _stacked_bar_chart(
    df,
    x,
    y_cols,
):

    if not y_cols:
        return None

    plot_df = _prepare_plot_df(
        df,
        x,
        y_cols,
    )

    melted = plot_df.melt(
        id_vars=[x],
        value_vars=y_cols,
        var_name="_metric",
        value_name="_value",
    )

    melted["_value"] = pd.to_numeric(
        melted["_value"],
        errors="coerce",
    )

    melted = melted.dropna(
        subset=["_value"]
    )

    if melted.empty:
        return None

    return (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X(
                x,
                type="nominal",
                title=_pretty(x),
            ),
            y=alt.Y(
                "_value:Q",
                title="Value",
                stack="zero",
            ),
            color=alt.Color(
                "_metric:N",
                title=None,
                scale=alt.Scale(
                    range=PALETTE
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    x,
                    type="nominal",
                    title=_pretty(x),
                ),
                alt.Tooltip(
                    "_metric:N",
                    title="Metric",
                ),
                alt.Tooltip(
                    "_value:Q",
                    title="Value",
                    format=",.2f",
                ),
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
    )


# =====================================================================
# LINE
# =====================================================================


def _line_chart(df, x, y):

    plot_df = _prepare_plot_df(
        df,
        x,
        [y],
    ).dropna(
        subset=[y]
    )

    if plot_df.empty:
        return None

    dt = _maybe_datetime(
        plot_df[x]
    )

    if dt is not None:
        plot_df[x] = dt
        x_type = "temporal"
    else:
        x_type = "nominal"

    fmt = _axis_format(
        plot_df[y]
    )

    x_title = _pretty(x)
    y_title = _pretty(y)

    show_points = (
        len(plot_df) <= 40
    )

    return (
        alt.Chart(plot_df)
        .mark_line(
            color=PRIMARY,
            point=show_points,
            strokeWidth=2.5,
        )
        .encode(
            x=alt.X(
                x,
                type=x_type,
                title=x_title,
                sort=None,
            ),
            y=alt.Y(
                y,
                type="quantitative",
                title=y_title,
                axis=alt.Axis(
                    format=fmt
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    x,
                    type=x_type,
                    title=x_title,
                ),
                alt.Tooltip(
                    y,
                    type="quantitative",
                    title=y_title,
                    format=fmt,
                ),
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
        .interactive()
    )


# =====================================================================
# AREA
# =====================================================================


def _area_chart(df, x, y):

    plot_df = _prepare_plot_df(
        df,
        x,
        [y],
    ).dropna(
        subset=[y]
    )

    if plot_df.empty:
        return None

    dt = _maybe_datetime(
        plot_df[x]
    )

    if dt is not None:
        plot_df[x] = dt
        x_type = "temporal"
    else:
        x_type = "nominal"

    fmt = _axis_format(
        plot_df[y]
    )

    x_title = _pretty(x)
    y_title = _pretty(y)

    return (
        alt.Chart(plot_df)
        .mark_area(
            line={
                "color": PRIMARY,
                "strokeWidth": 2,
            },
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(
                        color=PRIMARY,
                        offset=0,
                    ),
                    alt.GradientStop(
                        color="#ffffff",
                        offset=1,
                    ),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            opacity=0.55,
        )
        .encode(
            x=alt.X(
                x,
                type=x_type,
                title=x_title,
                sort=None,
            ),
            y=alt.Y(
                y,
                type="quantitative",
                title=y_title,
                axis=alt.Axis(
                    format=fmt
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    x,
                    type=x_type,
                    title=x_title,
                ),
                alt.Tooltip(
                    y,
                    type="quantitative",
                    title=y_title,
                    format=fmt,
                ),
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
        .interactive()
    )


# =====================================================================
# DONUT
# =====================================================================


def _donut_chart(df, x, y):

    plot_df = _prepare_plot_df(
        df,
        x,
        [y],
    ).dropna(
        subset=[y]
    )

    if plot_df.empty:
        return None

    plot_df = (
        plot_df
        .groupby(
            x,
            as_index=False,
        )[y]
        .sum()
    )

    total = plot_df[y].sum()

    if total == 0:
        return None

    plot_df["_share"] = (
        plot_df[y]
        / total
        * 100
    ).round(1)

    fmt = _axis_format(
        plot_df[y]
    )

    x_title = _pretty(x)
    y_title = _pretty(y)

    return (
        alt.Chart(plot_df)
        .mark_arc(
            innerRadius=70,
            cornerRadius=3,
            padAngle=0.012,
        )
        .encode(
            theta=alt.Theta(
                y,
                type="quantitative",
            ),
            color=alt.Color(
                x,
                type="nominal",
                title=x_title,
                scale=alt.Scale(
                    range=PALETTE
                ),
                legend=alt.Legend(
                    title=None,
                    orient="right",
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    x,
                    type="nominal",
                    title=x_title,
                ),
                alt.Tooltip(
                    y,
                    type="quantitative",
                    title=y_title,
                    format=fmt,
                ),
                alt.Tooltip(
                    "_share",
                    type="quantitative",
                    title="Share (%)",
                    format=".1f",
                ),
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
    )


# =====================================================================
# SCATTER
# =====================================================================


def _scatter_chart(
    df,
    x,
    y,
    color_col=None,
):

    cols = [
        x,
        y,
    ]

    if color_col:
        cols.append(
            color_col
        )

    plot_df = df[cols].copy()

    plot_df[x] = pd.to_numeric(
        plot_df[x],
        errors="coerce",
    )

    plot_df[y] = pd.to_numeric(
        plot_df[y],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            x,
            y,
        ]
    )

    if plot_df.empty:
        return None

    x_title = _pretty(x)
    y_title = _pretty(y)

    x_fmt = _axis_format(
        plot_df[x]
    )

    y_fmt = _axis_format(
        plot_df[y]
    )

    tooltip = [
        alt.Tooltip(
            x,
            type="quantitative",
            title=x_title,
            format=x_fmt,
        ),
        alt.Tooltip(
            y,
            type="quantitative",
            title=y_title,
            format=y_fmt,
        ),
    ]

    color_enc = alt.value(
        PRIMARY
    )

    if color_col:

        color_enc = alt.Color(
            color_col,
            type="nominal",
            title=_pretty(
                color_col
            ),
            scale=alt.Scale(
                range=PALETTE
            ),
        )

        tooltip.append(
            alt.Tooltip(
                color_col,
                type="nominal",
                title=_pretty(
                    color_col
                ),
            )
        )

    return (
        alt.Chart(plot_df)
        .mark_circle(
            size=90,
            opacity=0.75,
        )
        .encode(
            x=alt.X(
                x,
                type="quantitative",
                title=x_title,
                axis=alt.Axis(
                    format=x_fmt
                ),
            ),
            y=alt.Y(
                y,
                type="quantitative",
                title=y_title,
                axis=alt.Axis(
                    format=y_fmt
                ),
            ),
            color=color_enc,
            tooltip=tooltip,
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
        .interactive()
    )


# =====================================================================
# HISTOGRAM
# =====================================================================


def _histogram_chart(
    df,
    value,
):

    plot_df = _prepare_numeric_series(
        df,
        value,
    )

    if plot_df is None:
        return None

    if len(plot_df) < 2:
        return None

    return (
        alt.Chart(plot_df)
        .mark_bar(
            color=PRIMARY,
            opacity=0.85,
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        )
        .encode(
            x=alt.X(
                value,
                type="quantitative",
                bin=alt.Bin(
                    maxbins=30
                ),
                title=_pretty(
                    value
                ),
            ),
            y=alt.Y(
                "count():Q",
                title="Frequency",
            ),
            tooltip=[
                alt.Tooltip(
                    "count():Q",
                    title="Count",
                    format=",",
                )
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
    )


# =====================================================================
# BOXPLOT
# =====================================================================


def _boxplot_chart(
    df,
    category,
    value,
):

    cols = [
        category,
        value,
    ]

    plot_df = df[cols].copy()

    plot_df[value] = pd.to_numeric(
        plot_df[value],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            category,
            value,
        ]
    )

    if plot_df.empty:
        return None

    return (
        alt.Chart(plot_df)
        .mark_boxplot(
            size=35,
            color=PRIMARY,
        )
        .encode(
            x=alt.X(
                category,
                type="nominal",
                title=_pretty(
                    category
                ),
                axis=alt.Axis(
                    labelAngle=-30
                ),
            ),
            y=alt.Y(
                value,
                type="quantitative",
                title=_pretty(
                    value
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    category,
                    type="nominal",
                    title=_pretty(
                        category
                    ),
                ),
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
    )


# =====================================================================
# HEATMAP
# =====================================================================


def _heatmap_chart(
    df,
    x,
    category,
    value,
):

    plot_df = df[
        [
            x,
            category,
            value,
        ]
    ].copy()

    plot_df[value] = pd.to_numeric(
        plot_df[value],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            x,
            category,
            value,
        ]
    )

    if plot_df.empty:
        return None

    # Aggregate duplicate x/y combinations so Altair gets one cell
    # per matrix position.
    plot_df = (
        plot_df
        .groupby(
            [
                x,
                category,
            ],
            as_index=False,
        )[value]
        .sum()
    )

    if (
        plot_df[x].nunique()
        > MAX_HEATMAP_X_CATEGORIES
    ):
        return None

    if (
        plot_df[category].nunique()
        > MAX_HEATMAP_Y_CATEGORIES
    ):
        return None

    fmt = _axis_format(
        plot_df[value]
    )

    return (
        alt.Chart(plot_df)
        .mark_rect(
            cornerRadius=2
        )
        .encode(
            x=alt.X(
                x,
                type="ordinal",
                title=_pretty(
                    x
                ),
            ),
            y=alt.Y(
                category,
                type="ordinal",
                title=_pretty(
                    category
                ),
            ),
            color=alt.Color(
                value,
                type="quantitative",
                title=_pretty(
                    value
                ),
                scale=alt.Scale(
                    scheme="blues"
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    x,
                    type="ordinal",
                    title=_pretty(
                        x
                    ),
                ),
                alt.Tooltip(
                    category,
                    type="ordinal",
                    title=_pretty(
                        category
                    ),
                ),
                alt.Tooltip(
                    value,
                    type="quantitative",
                    title=_pretty(
                        value
                    ),
                    format=fmt,
                ),
            ],
        )
        .properties(
            height=DEFAULT_CHART_HEIGHT
        )
    )


# =====================================================================
# KPI CARDS
# =====================================================================


def _format_kpi_value(
    value,
    metric_name,
):
    """
    Format KPI values while remaining generic.

    Currency-style aliases receive $ formatting where appropriate.
    Percent-style aliases receive % formatting.
    """

    if pd.isna(value):
        return "—"

    metric_upper = str(
        metric_name
    ).upper()

    # Snowflake Decimals / numpy numbers / Python numbers.
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    is_percentage = any(
        token in metric_upper
        for token in [
            "PERCENT",
            "PERCENTAGE",
            "RATE",
            "MARGIN",
        ]
    )

    is_currency = any(
        token in metric_upper
        for token in [
            "REVENUE",
            "PROFIT",
            "SALES",
            "COST",
            "PRICE",
            "VALUE",
            "AOV",
            "AMOUNT",
        ]
    )

    if is_percentage:
        return f"{numeric:,.2f}%"

    if is_currency:

        absolute = abs(
            numeric
        )

        if absolute >= 1_000_000_000:
            return f"${numeric / 1_000_000_000:,.2f}B"

        if absolute >= 1_000_000:
            return f"${numeric / 1_000_000:,.2f}M"

        if absolute >= 1_000:
            return f"${numeric / 1_000:,.2f}K"

        return f"${numeric:,.2f}"

    # Integer-like values.
    if numeric.is_integer():
        return f"{int(numeric):,}"

    return f"{numeric:,.2f}"


def _render_kpis(
    df,
    metrics,
):
    """
    Render one or more aggregate values as Streamlit metric cards.
    """

    if df is None or df.empty:
        return False

    valid_metrics = [
        metric
        for metric in metrics
        if metric in df.columns
    ]

    if not valid_metrics:
        return False

    valid_metrics = valid_metrics[
        :MAX_KPI_CARDS
    ]

    # KPI questions should normally return one row. If multiple rows
    # exist, use the first row rather than failing the whole response.
    row = df.iloc[0]

    # Use up to 3 columns per row for readability.
    columns_per_row = min(
        3,
        len(valid_metrics),
    )

    for start in range(
        0,
        len(valid_metrics),
        columns_per_row,
    ):

        metric_group = valid_metrics[
            start:
            start + columns_per_row
        ]

        containers = st.columns(
            len(metric_group)
        )

        for container, metric in zip(
            containers,
            metric_group,
        ):

            value = row[metric]

            display_value = _format_kpi_value(
                value,
                metric,
            )

            with container:
                st.metric(
                    label=_pretty(
                        metric
                    ),
                    value=display_value,
                )

    return True


# =====================================================================
# BUILDER MAP
# =====================================================================


_BUILDERS = {
    "bar": _bar_chart,
    "line": _line_chart,
    "area": _area_chart,
    "pie": _donut_chart,
}


# =====================================================================
# PRESENTATION HELPERS
# =====================================================================


def _wrap_title(
    text,
    width=90,
    max_lines=2,
):

    lines = (
        textwrap.wrap(
            str(text).strip(),
            width=width,
        )
        or [
            str(text)
        ]
    )

    if len(lines) > max_lines:

        lines = lines[
            :max_lines
        ]

        lines[-1] = (
            lines[-1].rstrip()
            + "…"
        )

    return lines


def _finalize(
    chart,
    title,
):

    title_param = (
        alt.TitleParams(
            text=_wrap_title(
                title
            ),
            anchor="start",
            fontSize=15,
            fontWeight=600,
            color=TEXT_DARK,
            lineHeight=20,
        )
        if title
        else alt.Undefined
    )

    return (
        chart
        .properties(
            title=title_param
        )
        .configure_view(
            strokeWidth=0
        )
        .configure_axis(
            grid=True,
            gridColor=GRID_COLOR,
            gridOpacity=0.8,
            domainColor="#c7d0e0",
            tickColor="#c7d0e0",
            labelColor=TEXT_MUTED,
            titleColor=TEXT_DARK,
            labelFontSize=11,
            titleFontSize=12,
            labelFont="sans-serif",
            titleFont="sans-serif",
            titleFontWeight=600,
        )
        .configure_legend(
            labelColor=TEXT_MUTED,
            titleColor=TEXT_DARK,
            labelFontSize=11,
            titleFontSize=11,
            symbolSize=80,
        )
    )


# =====================================================================
# PUBLIC API
# =====================================================================


def has_chart(
    df,
    chart_type,
    x_axis=None,
    y_axis=None,
    category_column=None,
    value_column=None,
):
    """
    Cheap validation check used by app.py.

    Returns True when the recommendation can safely render against the
    actual DataFrame.
    """

    try:

        return (
            _validate_chart_spec(
                df=df,
                chart_type=chart_type,
                x_axis=x_axis,
                y_axis=y_axis,
                category_column=category_column,
                value_column=value_column,
            )
            is not None
        )

    except Exception:
        return False


def render_visualization(
    df,
    chart_type,
    x_axis=None,
    y_axis=None,
    category_column=None,
    value_column=None,
    question=None,
    key_prefix="viz",
):
    """
    Validate Gemini's chart recommendation against the actual result
    DataFrame and render it.

    Any visualization failure is deliberately swallowed so that a
    visualization problem never takes down the SQL result itself.
    """

    try:

        spec = _validate_chart_spec(
            df=df,
            chart_type=chart_type,
            x_axis=x_axis,
            y_axis=y_axis,
            category_column=category_column,
            value_column=value_column,
        )

        if spec is None:
            return

        resolved_type = spec[
            "chart_type"
        ]

        x = spec.get(
            "x_axis"
        )

        y_options = spec.get(
            "y_axis",
            [],
        )

        category = spec.get(
            "category_column"
        )

        value = spec.get(
            "value_column"
        )

        # ----------------------------------------------------------
        # Visualization heading
        # ----------------------------------------------------------

        st.markdown(
            '<div class="sql-label">📊 Visualization</div>',
            unsafe_allow_html=True,
        )

        chart_label_map = {
            "bar": "Bar",
            "grouped_bar": "Grouped Bar",
            "stacked_bar": "Stacked Bar",
            "line": "Line",
            "area": "Area",
            "pie": "Donut",
            "scatter": "Scatter",
            "histogram": "Histogram",
            "boxplot": "Box Plot",
            "heatmap": "Heatmap",
            "kpi": "KPI",
        }

        st.caption(
            "Recommended by Gemini · "
            f"{chart_label_map.get(resolved_type, resolved_type.title())}"
        )

        # ==========================================================
        # KPI
        # ==========================================================

        if resolved_type == "kpi":

            _render_kpis(
                df,
                y_options,
            )

            return

        # ==========================================================
        # HISTOGRAM
        # ==========================================================

        if resolved_type == "histogram":

            chart = _histogram_chart(
                df,
                value,
            )

        # ==========================================================
        # BOXPLOT
        # ==========================================================

        elif resolved_type == "boxplot":

            chart = _boxplot_chart(
                df,
                category,
                value,
            )

        # ==========================================================
        # HEATMAP
        # ==========================================================

        elif resolved_type == "heatmap":

            chart = _heatmap_chart(
                df,
                x,
                category,
                value,
            )

        # ==========================================================
        # SCATTER
        # ==========================================================

        elif resolved_type == "scatter":

            y = y_options[0]

            chart = _scatter_chart(
                df,
                x,
                y,
                spec.get(
                    "color_col"
                ),
            )

        # ==========================================================
        # GROUPED BAR
        # ==========================================================

        elif resolved_type == "grouped_bar":

            chart = _grouped_bar_chart(
                df,
                x,
                y_options,
            )

        # ==========================================================
        # STACKED BAR
        # ==========================================================

        elif resolved_type == "stacked_bar":

            chart = _stacked_bar_chart(
                df,
                x,
                y_options,
            )

        # ==========================================================
        # BAR / LINE / AREA / PIE
        # ==========================================================

        else:

            if not y_options:
                return

            y = y_options[0]

            # If multiple numeric metrics are returned for these
            # chart types, allow the user to select which measure is
            # displayed.
            if (
                len(y_options) > 1
                and resolved_type
                in (
                    "bar",
                    "line",
                    "area",
                )
            ):

                y = st.selectbox(
                    "Metric",
                    y_options,
                    key=f"{key_prefix}_metric",
                    format_func=_pretty,
                )

            plot_source = df
            truncated_from = None

            if (
                resolved_type == "bar"
                and len(df)
                > MAX_BAR_CATEGORIES
            ):

                truncated_from = len(
                    df
                )

                ranked = pd.to_numeric(
                    df[y],
                    errors="coerce",
                )

                top_indexes = (
                    ranked
                    .sort_values(
                        ascending=False
                    )
                    .index[
                        :MAX_BAR_CATEGORIES
                    ]
                )

                plot_source = df.loc[
                    top_indexes
                ]

            chart = _BUILDERS[
                resolved_type
            ](
                plot_source,
                x,
                y,
            )

            if chart is None:
                return

            chart_height = getattr(
                chart,
                "height",
                DEFAULT_CHART_HEIGHT,
            )

            if not isinstance(
                chart_height,
                (
                    int,
                    float,
                ),
            ):
                chart_height = (
                    DEFAULT_CHART_HEIGHT
                )

            title = (
                question.strip()
                if question
                else
                f"{_pretty(y)} by {_pretty(x)}"
            )

            final_chart = _finalize(
                chart,
                title,
            )

            if (
                chart_height
                > SCROLL_VIEWPORT_HEIGHT
            ):

                with st.container(
                    height=SCROLL_VIEWPORT_HEIGHT
                ):

                    st.altair_chart(
                        final_chart,
                        width="stretch",
                        theme=None,
                    )

                st.caption(
                    "Scroll within the chart "
                    f"to see all {len(plot_source):,} rows"
                )

            else:

                st.altair_chart(
                    final_chart,
                    width="stretch",
                    theme=None,
                )

            if truncated_from:

                st.caption(
                    f"Showing top {MAX_BAR_CATEGORIES} "
                    f"of {truncated_from:,} by {_pretty(y)} "
                    "— see the full result in the table above "
                    "or the CSV download below."
                )

            return

        # ----------------------------------------------------------
        # Final rendering for special chart types
        # ----------------------------------------------------------

        if chart is None:
            return

        title = (
            question.strip()
            if question
            else "Visualization"
        )

        final_chart = _finalize(
            chart,
            title,
        )

        st.altair_chart(
            final_chart,
            width="stretch",
            theme=None,
        )

    except Exception:
        # A visualization problem must never take down the SQL result
        # or the rest of the Streamlit application.
        return


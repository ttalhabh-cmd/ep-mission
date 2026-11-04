from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
DATA_FILE = APP_ROOT / "data" / "ep_thruster_models_v1.csv"
SOURCE_WORKBOOK = APP_ROOT / "data" / "EP_Thruster_Database_390_Models_v1.0.0.xlsx"

EVIDENCE_LABELS = {
    "verified": "Source verified",
    "manufacturer": "Manufacturer data",
    "compiled": "Compiled source",
    "unviewed": "Source not inspected",
}
EVIDENCE_ORDER = list(EVIDENCE_LABELS.values())
EVIDENCE_COLOURS = {
    "Source verified": "#35d3a3",
    "Manufacturer data": "#59a8ff",
    "Compiled source": "#f7b955",
    "Source not inspected": "#f27676",
}
MECHANISM_COLOURS = {
    "Electrostatic": "#5b9cff",
    "Electromagnetic": "#ff6b76",
    "Electrothermal": "#47cf95",
}
TYPE_LABELS = {
    "Hall": "Hall-effect",
    "GriddedIon": "Gridded ion",
    "MPD": "Magnetoplasmadynamic",
    "PPT": "Pulsed plasma",
    "FEEP": "FEEP",
    "Helicon": "Helicon",
    "Arcjet": "Arcjet",
    "Resistojet": "Resistojet",
    "Electrothermal": "Other electrothermal",
    "VacuumArc": "Vacuum arc",
    "Electrospray": "Electrospray",
    "ECR": "Electron cyclotron resonance",
}
COUNTRY_NAMES = {
    "AE": "United Arab Emirates",
    "AT": "Austria",
    "AU": "Australia",
    "BR": "Brazil",
    "CN": "China",
    "DE": "Germany",
    "ES": "Spain",
    "EU": "European collaboration",
    "FR": "France",
    "IL": "Israel",
    "IN": "India",
    "IT": "Italy",
    "JP": "Japan",
    "KR": "South Korea",
    "PK": "Pakistan",
    "RU": "Russia",
    "SE": "Sweden",
    "SG": "Singapore",
    "UK": "United Kingdom",
    "UA": "Ukraine",
    "US": "United States",
    "ZA": "South Africa",
}
POWER_CLASS_ORDER = [
    "micro (<50 W)",
    "sub-kilowatt (50 W - 1 kW)",
    "kilowatt (1 - 5 kW)",
    "high (5 - 30 kW)",
    "very-high (>30 kW)",
]
DISPLAY_COLUMNS = {
    "thruster_id": "Record ID",
    "thruster_name": "Thruster model",
    "manufacturer": "Developer / manufacturer",
    "country_name": "Country",
    "thruster_type": "Thruster type",
    "propellant": "Propellant",
    "acceleration_label": "Acceleration mechanism",
    "P_nom_W": "Nominal power (W)",
    "T_nom_mN": "Nominal thrust (mN)",
    "Isp_nom_s": "Nominal specific impulse (s)",
    "eta_percent": "Total efficiency (%)",
    "dry_mass_kg": "Dry mass (kg)",
    "lifetime_hr": "Lifetime (h)",
    "trl": "TRL",
    "maturity_label": "Maturity",
    "flight_status_label": "Development status",
    "parameter_set_count": "Documented parameter sets",
    "reconciliation_label": "Physics reconciliation",
    "evidence_label": "Evidence status",
    "source_grade": "Source grade",
    "trl_confidence_label": "TRL confidence",
    "reference": "Citation",
    "reference_url": "Source link",
}


st.set_page_config(
    page_title="EP Mission Design & Thruster Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["thruster_type"] = df["family"].map(TYPE_LABELS).fillna(df["family"])
    df["country_name"] = df["country_iso2"].map(COUNTRY_NAMES).fillna(df["country_iso2"])
    df["acceleration_label"] = df["acceleration_mechanism"].str.title()
    df["evidence_label"] = df["use_class"].map(EVIDENCE_LABELS).fillna("Unclassified")
    df["eta_percent"] = df["eta_nom"] * 100
    df["parameter_set_count"] = df["n_operating_points"]
    df["maturity_label"] = df["maturity_track"].str.replace("-", " ").str.title()
    df["flight_status_label"] = df["flight_status_raw"].str.replace("-", " ").str.title()
    df["reconciliation_label"] = df["reconciliation"].str.replace("-", " ").str.title()
    df["trl_confidence_label"] = df["trl_confidence"].str.title()
    df["reference_url"] = df["reference_url"].fillna("").astype(str)
    df.loc[~df["reference_url"].str.match(r"^https?://", na=False), "reference_url"] = ""
    return df


@st.cache_data(show_spinner=False)
def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def safe(value: object, fallback: str = "Not reported") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    value_text = str(value).strip()
    if not value_text or value_text.lower() == "nan":
        return fallback
    return html.escape(value_text)


def format_number(value: object, unit: str = "", digits: int = 3) -> str:
    if pd.isna(value):
        return "Not reported"
    number = float(value)
    magnitude = abs(number)
    if magnitude >= 100_000 or (0 < magnitude < 0.01):
        rendered = f"{number:.2e}"
    elif magnitude >= 1_000:
        rendered = f"{number:,.0f}"
    else:
        rendered = f"{number:,.{digits}g}"
    return f"{rendered} {unit}".strip()


def label_for_record(row: pd.Series) -> str:
    return f"{row['thruster_name']} — {row['manufacturer']} ({row['thruster_id']})"


def reset_filters() -> None:
    for key in (
        "search_filter",
        "evidence_filter",
        "type_filter",
        "mechanism_filter",
        "propellant_filter",
        "country_filter",
        "power_class_filter",
        "status_filter",
        "trl_filter",
        "minimum_power_filter",
        "maximum_power_filter",
    ):
        st.session_state.pop(key, None)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    search_text = st.session_state.get("search_filter", "").strip().lower()
    evidence = st.session_state.get("evidence_filter", ["Source verified"])
    thruster_types = st.session_state.get("type_filter", [])
    mechanisms = st.session_state.get("mechanism_filter", [])
    propellants = st.session_state.get("propellant_filter", [])
    countries = st.session_state.get("country_filter", [])
    power_classes = st.session_state.get("power_class_filter", [])
    development_status = st.session_state.get("status_filter", [])
    trl_min, trl_max = st.session_state.get("trl_filter", (1, 9))
    minimum_power = st.session_state.get("minimum_power_filter", float(df["P_nom_W"].min()))
    maximum_power = st.session_state.get("maximum_power_filter", float(df["P_nom_W"].max()))

    if search_text:
        fields = [
            "thruster_name",
            "manufacturer",
            "thruster_type",
            "country_name",
            "propellant",
            "subtype",
            "thruster_id",
        ]
        haystack = filtered[fields].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered[haystack.str.contains(search_text, regex=False)]
    if evidence:
        filtered = filtered[filtered["evidence_label"].isin(evidence)]
    if thruster_types:
        filtered = filtered[filtered["thruster_type"].isin(thruster_types)]
    if mechanisms:
        filtered = filtered[filtered["acceleration_label"].isin(mechanisms)]
    if propellants:
        filtered = filtered[filtered["propellant"].isin(propellants)]
    if countries:
        filtered = filtered[filtered["country_name"].isin(countries)]
    if power_classes:
        filtered = filtered[filtered["power_class"].isin(power_classes)]
    if development_status:
        filtered = filtered[filtered["flight_status_label"].isin(development_status)]
    filtered = filtered[filtered["trl"].between(trl_min, trl_max)]
    filtered = filtered[filtered["P_nom_W"].between(minimum_power, maximum_power)]
    return filtered.copy()


def inject_css() -> None:
    st.markdown("""
    <style>
    .block-container {max-width:1180px; padding-top:2.1rem; padding-bottom:3rem}
    [data-testid="stSidebar"] {border-right:1px solid #394245}
    h1,h2,h3 {letter-spacing:-.025em; font-weight:580}
    [data-testid="stMetric"] {padding:.65rem .8rem; border-top:2px solid #87ab9e; background:#192226}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums; font-size:1.55rem}
    .hero,.profile-head {border-top:2px solid #84ac9d; border-bottom:1px solid #445152; padding:1rem 0 .8rem; margin-bottom:1rem}
    .hero h1 {font-size:2.25rem; margin:.35rem 0}
    .eyebrow,.section-kicker,.footer {font-family:monospace;font-size:.77rem;color:#a7b9b3;letter-spacing:.07em}
    .validation-note,.method-card {padding:.8rem 0;color:#c4ceca}
    .footer {margin-top:2rem;border-top:1px solid #445152;padding-top:1rem}
    </style>""", unsafe_allow_html=True)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("## Explore the database")
        st.caption("Filters apply to Overview, Browser, Compare and Model profile.")
        st.text_input(
            "Search",
            placeholder="Model, developer, country…",
            key="search_filter",
        )
        st.multiselect(
            "Evidence status",
            options=EVIDENCE_ORDER,
            default=["Source verified"],
            help="The default view contains the 246 records checked against their cited sources. Clear the selection to include every evidence class.",
            key="evidence_filter",
        )
        with st.expander("Technology", expanded=True):
            st.multiselect(
                "Thruster type",
                sorted(df["thruster_type"].unique()),
                placeholder="All thruster types",
                key="type_filter",
            )
            st.multiselect(
                "Acceleration mechanism",
                sorted(df["acceleration_label"].unique()),
                placeholder="All mechanisms",
                key="mechanism_filter",
            )
            st.multiselect(
                "Propellant",
                sorted(df["propellant"].unique()),
                placeholder="All propellants",
                key="propellant_filter",
            )
        with st.expander("Readiness and origin"):
            st.slider("Technology readiness level", 1, 9, (1, 9), key="trl_filter")
            st.multiselect(
                "Development status",
                sorted(df["flight_status_label"].unique()),
                placeholder="All statuses",
                key="status_filter",
            )
            st.multiselect(
                "Country",
                sorted(df["country_name"].unique()),
                placeholder="All countries",
                key="country_filter",
            )
        with st.expander("Power"):
            st.multiselect(
                "Power class",
                POWER_CLASS_ORDER,
                placeholder="All power classes",
                key="power_class_filter",
            )
            st.number_input(
                "Minimum nominal power (W)",
                min_value=float(df["P_nom_W"].min()),
                max_value=float(df["P_nom_W"].max()),
                value=float(df["P_nom_W"].min()),
                key="minimum_power_filter",
            )
            st.number_input(
                "Maximum nominal power (W)",
                min_value=float(df["P_nom_W"].min()),
                max_value=float(df["P_nom_W"].max()),
                value=float(df["P_nom_W"].max()),
                key="maximum_power_filter",
            )
        st.button("Reset all filters", on_click=reset_filters, width="stretch")

    filtered = apply_filters(df)
    with st.sidebar:
        st.markdown(f"**{len(filtered):,} records shown**")
        st.caption(f"of {len(df):,} total records")
    return filtered


def render_header(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Open electric-propulsion research database</div>
          <h1>EP Technology Explorer</h1>
          <p>Search, compare and evaluate electric-propulsion thruster models across performance, readiness and evidence quality. Every point has the same visual size; colour identifies the acceleration mechanism.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="validation-note">
          <strong>Validation status:</strong> {int((df['use_class'] == 'verified').sum())} of {len(df)} records were checked against the cited source. This is literature verification, not independent laboratory validation. The remaining records remain visible when selected and are clearly labelled by evidence status.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(active: pd.DataFrame) -> None:
    columns = st.columns(5)
    verified_count = int((active["use_class"] == "verified").sum())
    values = [
        ("Records shown", f"{len(active):,}"),
        ("Source verified", f"{verified_count:,}"),
        ("Thruster types", f"{active['thruster_type'].nunique():,}"),
        ("Countries", f"{active['country_iso2'].nunique():,}"),
        ("Power span", f"{active['P_nom_W'].min():g}–{active['P_nom_W'].max():g} W"),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)


def base_plot_layout(fig: go.Figure, height: int = 470) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,14,30,.62)",
        font=dict(color="#c9d3e7"),
        height=height,
        margin=dict(l=18, r=18, t=50, b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(130,150,195,.13)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(130,150,195,.13)", zeroline=False)
    return fig


def render_overview(active: pd.DataFrame) -> None:
    st.markdown("### Performance overview")
    if active.empty:
        st.warning("No records match the current filters. Reset or broaden the filters to continue.")
        return
    render_metrics(active)
    st.markdown("#### Thrust–specific impulse trade space")
    control_a, control_b = st.columns([1, 4])
    with control_a:
        logarithmic = st.toggle("Logarithmic axes", value=True)
    with control_b:
        st.caption("Each marker represents one thruster model. Marker size is fixed and does not encode power.")

    fig = px.scatter(
        active,
        x="Isp_nom_s",
        y="T_nom_mN",
        color="acceleration_label",
        color_discrete_map=MECHANISM_COLOURS,
        labels={
            "Isp_nom_s": "Nominal specific impulse (s)",
            "T_nom_mN": "Nominal thrust (mN)",
            "acceleration_label": "Acceleration mechanism",
        },
        hover_name="thruster_name",
        hover_data={
            "manufacturer": True,
            "thruster_type": True,
            "P_nom_W": ":.4g",
            "trl": True,
            "evidence_label": True,
            "Isp_nom_s": ":.4g",
            "T_nom_mN": ":.4g",
            "acceleration_label": False,
        },
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.78, line=dict(width=0.7, color="rgba(240,245,255,.62)")),
        selector=dict(mode="markers"),
    )
    if logarithmic:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    base_plot_layout(fig, height=590)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    left, right = st.columns(2)
    with left:
        type_counts = (
            active.groupby(["thruster_type", "acceleration_label"], observed=True)
            .size()
            .reset_index(name="Models")
        )
        type_order = (
            type_counts.groupby("thruster_type")["Models"].sum().sort_values().index.tolist()
        )
        type_chart = px.bar(
            type_counts,
            x="Models",
            y="thruster_type",
            color="acceleration_label",
            orientation="h",
            category_orders={"thruster_type": type_order},
            color_discrete_map=MECHANISM_COLOURS,
            labels={"thruster_type": "Thruster type", "acceleration_label": "Mechanism"},
            title="Models by thruster type",
        )
        base_plot_layout(type_chart, height=470)
        type_chart.update_layout(barmode="stack", showlegend=False)
        st.plotly_chart(type_chart, width="stretch", config={"displaylogo": False})
    with right:
        maturity_counts = (
            active.groupby(["maturity_label", "evidence_label"], observed=True)
            .size()
            .reset_index(name="Models")
        )
        maturity_chart = px.bar(
            maturity_counts,
            x="maturity_label",
            y="Models",
            color="evidence_label",
            color_discrete_map=EVIDENCE_COLOURS,
            labels={"maturity_label": "Maturity", "evidence_label": "Evidence status"},
            title="Maturity and evidence coverage",
        )
        base_plot_layout(maturity_chart, height=470)
        maturity_chart.update_layout(barmode="stack", xaxis_tickangle=-18)
        st.plotly_chart(maturity_chart, width="stretch", config={"displaylogo": False})


def render_browser(active: pd.DataFrame) -> None:
    st.markdown("### Database browser")
    st.caption("The table contains one row per distinct thruster model. Select columns and download the filtered view for your own analysis.")
    if active.empty:
        st.warning("No records match the current filters.")
        return

    default_columns = [
        "Thruster model",
        "Developer / manufacturer",
        "Country",
        "Thruster type",
        "Propellant",
        "Nominal power (W)",
        "Nominal thrust (mN)",
        "Nominal specific impulse (s)",
        "TRL",
        "Evidence status",
        "Source link",
    ]
    chosen_labels = st.multiselect(
        "Visible columns",
        list(DISPLAY_COLUMNS.values()),
        default=default_columns,
        key="browser_columns",
    )
    if not chosen_labels:
        st.info("Choose at least one column to display the table.")
        return
    reverse_columns = {display: source for source, display in DISPLAY_COLUMNS.items()}
    chosen_columns = [reverse_columns[label] for label in chosen_labels]
    table = active[chosen_columns].rename(columns=DISPLAY_COLUMNS).copy()

    column_config: dict[str, object] = {
        "Source link": st.column_config.LinkColumn("Source link", display_text="Open source"),
        "Nominal power (W)": st.column_config.NumberColumn(format="%.3g"),
        "Nominal thrust (mN)": st.column_config.NumberColumn(format="%.3g"),
        "Nominal specific impulse (s)": st.column_config.NumberColumn(format="%d"),
        "Total efficiency (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Dry mass (kg)": st.column_config.NumberColumn(format="%.3g"),
        "Lifetime (h)": st.column_config.NumberColumn(format="%.3g"),
        "TRL": st.column_config.ProgressColumn(min_value=1, max_value=9, format="%d"),
    }
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=590,
        column_config=column_config,
    )

    download_left, download_right, note = st.columns([1, 1, 2.2])
    with download_left:
        st.download_button(
            "Download filtered CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="ep_thruster_filtered_view.csv",
            mime="text/csv",
            width="stretch",
        )
    with download_right:
        st.download_button(
            "Download full workbook",
            data=file_bytes(SOURCE_WORKBOOK),
            file_name=SOURCE_WORKBOOK.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with note:
        st.caption("The full workbook includes the data dictionary, TRL rubric, source-grade definitions and exclusion register.")


def percentile_position(series: pd.Series, value: object) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if pd.isna(value) or clean.empty:
        return None
    return float((clean <= float(value)).mean() * 100)


def render_compare(active: pd.DataFrame) -> None:
    st.markdown("### Compare thruster models")
    st.caption("Compare up to five models. The radial chart shows percentile position within the current filtered set; it is not a performance score.")
    if active.empty:
        st.warning("No records match the current filters.")
        return

    ordered = active.sort_values(["thruster_type", "thruster_name", "manufacturer"])
    record_lookup = ordered.set_index("thruster_id", drop=False)
    option_ids = ordered["thruster_id"].tolist()
    default_ids = ordered.drop_duplicates("thruster_type")["thruster_id"].head(3).tolist()
    selected_ids = st.multiselect(
        "Thruster models",
        option_ids,
        default=default_ids,
        max_selections=5,
        format_func=lambda record_id: label_for_record(record_lookup.loc[record_id]),
        key="comparison_models",
    )
    if not selected_ids:
        st.info("Select at least one thruster model to begin the comparison.")
        return
    selected = record_lookup.loc[selected_ids].copy()

    parameter_rows = [
        ("Thruster type", "thruster_type", lambda x: safe(x)),
        ("Developer / manufacturer", "manufacturer", lambda x: safe(x)),
        ("Country", "country_name", lambda x: safe(x)),
        ("Propellant", "propellant", lambda x: safe(x)),
        ("Nominal power", "P_nom_W", lambda x: format_number(x, "W")),
        ("Nominal thrust", "T_nom_mN", lambda x: format_number(x, "mN")),
        ("Nominal specific impulse", "Isp_nom_s", lambda x: format_number(x, "s")),
        ("Total efficiency", "eta_percent", lambda x: format_number(x, "%")),
        ("Thrust-to-power", "thrust_to_power_mN_per_kW", lambda x: format_number(x, "mN/kW")),
        ("Dry mass", "dry_mass_kg", lambda x: format_number(x, "kg")),
        ("Lifetime", "lifetime_hr", lambda x: format_number(x, "h")),
        ("Technology readiness", "trl", lambda x: f"TRL {int(x)}"),
        ("Evidence status", "evidence_label", lambda x: safe(x)),
    ]
    comparison: dict[str, list[str]] = {"Parameter": [row[0] for row in parameter_rows]}
    for _, record in selected.iterrows():
        column_name = f"{record['thruster_name']} ({record['thruster_id']})"
        comparison[column_name] = [formatter(record[field]) for _, field, formatter in parameter_rows]
    st.dataframe(pd.DataFrame(comparison).set_index("Parameter"), width="stretch")

    radial_metrics = [
        ("Power", "P_nom_W"),
        ("Thrust", "T_nom_mN"),
        ("Specific impulse", "Isp_nom_s"),
        ("Efficiency", "eta_percent"),
        ("Lifetime", "lifetime_hr"),
        ("TRL", "trl"),
    ]
    radar = go.Figure()
    palette = ["#62d7ff", "#ff7a89", "#55d6a6", "#f7bd5e", "#a789ff"]
    theta = [label for label, _ in radial_metrics]
    for colour, (_, record) in zip(palette, selected.iterrows()):
        scores = [percentile_position(active[field], record[field]) for _, field in radial_metrics]
        plotted = [score if score is not None else 0 for score in scores]
        radar.add_trace(
            go.Scatterpolar(
                r=plotted + [plotted[0]],
                theta=theta + [theta[0]],
                fill="toself",
                opacity=0.58,
                name=record["thruster_name"],
                line=dict(color=colour, width=2),
                hovertemplate="%{theta}: %{r:.0f}th percentile<extra>%{fullData.name}</extra>",
            )
        )
    radar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(8,14,30,.62)",
            radialaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(130,150,195,.18)"),
            angularaxis=dict(gridcolor="rgba(130,150,195,.18)"),
        ),
        height=540,
        margin=dict(l=35, r=35, t=35, b=35),
        legend=dict(orientation="h", y=-0.08),
    )
    st.plotly_chart(radar, width="stretch", config={"displaylogo": False})


def render_profile(active: pd.DataFrame) -> None:
    st.markdown("### Model profile")
    if active.empty:
        st.warning("No records match the current filters.")
        return
    ordered = active.sort_values(["thruster_name", "manufacturer"])
    lookup = ordered.set_index("thruster_id", drop=False)
    selected_id = st.selectbox(
        "Choose a thruster model",
        ordered["thruster_id"].tolist(),
        format_func=lambda record_id: label_for_record(lookup.loc[record_id]),
        key="profile_model",
    )
    row = lookup.loc[selected_id]
    st.markdown(
        f"""
        <div class="profile-head">
          <div class="section-kicker">{safe(row['thruster_type'])} · {safe(row['country_name'])}</div>
          <h2>{safe(row['thruster_name'])}</h2>
          <p>{safe(row['manufacturer'])} · Record {safe(row['thruster_id'])}</p>
          <span class="pill pill-{safe(row['use_class'])}">{safe(row['evidence_label'])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metrics = [
        ("Nominal power", format_number(row["P_nom_W"], "W")),
        ("Nominal thrust", format_number(row["T_nom_mN"], "mN")),
        ("Specific impulse", format_number(row["Isp_nom_s"], "s")),
        ("Total efficiency", format_number(row["eta_percent"], "%")),
        ("Readiness", f"TRL {int(row['trl'])}"),
    ]
    for column, (label, value) in zip(metric_columns, metrics):
        column.metric(label, value)

    performance_tab, evidence_tab, notes_tab = st.tabs(["Parameter record", "Evidence and readiness", "Notes"])
    with performance_tab:
        left, right = st.columns(2)
        parameter_table = pd.DataFrame(
            [
                ("Acceleration mechanism", safe(row["acceleration_label"])),
                ("Subtype", safe(row["subtype"])),
                ("Propellant", safe(row["propellant"])),
                ("Mode", safe(row["operation_mode"])),
                ("Power range", f"{format_number(row['P_min_W'], 'W')} – {format_number(row['P_max_W'], 'W')}"),
                ("Thrust range", f"{format_number(row['T_min_mN'], 'mN')} – {format_number(row['T_max_mN'], 'mN')}"),
                ("Specific-impulse range", f"{format_number(row['Isp_min_s'], 's')} – {format_number(row['Isp_max_s'], 's')}"),
                ("Documented parameter sets", safe(row["parameter_set_count"])),
            ],
            columns=["Parameter", "Value"],
        )
        engineering_table = pd.DataFrame(
            [
                ("Discharge voltage", format_number(row["Vd_V"], "V")),
                ("Dry mass", format_number(row["dry_mass_kg"], "kg")),
                ("Specific mass", format_number(row["specific_mass_kg_per_kW"], "kg/kW")),
                ("Lifetime", format_number(row["lifetime_hr"], "h")),
                ("Lifetime basis", safe(row["lifetime_basis"])),
                ("Thrust-to-power", format_number(row["thrust_to_power_mN_per_kW"], "mN/kW")),
                ("Throttleable", "Yes" if bool(row["throttleable"]) else "No"),
                ("Power class", safe(row["power_class"])),
            ],
            columns=["Parameter", "Value"],
        )
        left.dataframe(parameter_table, hide_index=True, width="stretch")
        right.dataframe(engineering_table, hide_index=True, width="stretch")
    with evidence_tab:
        evidence_left, evidence_right = st.columns([1.1, 1])
        with evidence_left:
            st.markdown("#### Source provenance")
            st.markdown(f"**Evidence status:** {safe(row['evidence_label'])}")
            st.markdown(f"**Source grade:** {safe(row['source_grade'])}")
            st.markdown(f"**Physics reconciliation:** {safe(row['reconciliation_label'])}")
            st.markdown(f"**Citation:** {safe(row['reference'])}")
            if row["reference_url"]:
                st.link_button("Open cited source", row["reference_url"])
            else:
                st.caption("No direct URL or DOI is stored for this record.")
        with evidence_right:
            st.markdown("#### Readiness assignment")
            st.markdown(f"**TRL:** {int(row['trl'])}")
            st.markdown(f"**TRL confidence:** {safe(row['trl_confidence_label'])}")
            st.markdown(f"**Maturity:** {safe(row['maturity_label'])}")
            st.markdown(f"**Basis:** {safe(row['trl_basis'])}")
            st.markdown(f"**Censoring:** {safe(row['trl_censoring'])}")
        if row["use_class"] == "verified":
            st.success("The cited document was inspected and the database value was checked or assembled from that source. This does not mean the model was independently tested by the database authors.")
        elif row["use_class"] == "manufacturer":
            st.info("This record is based on manufacturer or developer literature and has not been independently confirmed by a primary research source.")
        elif row["use_class"] == "compiled":
            st.warning("This record comes from a secondary compilation. Consult the original technical source before using it for detailed design.")
        else:
            st.error("The cited source was not inspected during the evidence audit. Treat this record as provisional.")
    with notes_tab:
        st.markdown(f"**Database notes:** {safe(row['notes'], 'No record-specific note.')}")
        st.markdown(f"**Reference notes:** {safe(row['reference_notes'], 'No reference-specific note.')}")
        st.markdown(f"**Original record origin:** {safe(row['record_origin'])}")


def evidence_definition_cards() -> None:
    cards = [
        (
            "Source verified",
            "The cited document was inspected and the stored value was checked directly or assembled transparently from that source. It is literature verification, not a new experiment.",
        ),
        (
            "Manufacturer data",
            "The value comes from developer or product literature. It may be authoritative for the product but is not independently peer reviewed in this audit.",
        ),
        (
            "Compiled source",
            "The value is inherited from a catalogue or review. The underlying primary document was not used to confirm the value.",
        ),
        (
            "Source not inspected",
            "A citation is recorded, but the document was not inspected. The record is retained for coverage and should be treated as provisional.",
        ),
    ]
    columns = st.columns(4)
    for column, (heading, body) in zip(columns, cards):
        column.markdown(
            f'<div class="method-card"><h4>{heading}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )


def render_evidence(df: pd.DataFrame) -> None:
    st.markdown("### Evidence, validation and methods")
    st.markdown(
        "The database has passed release-level integrity and physics-consistency checks, but that does not make every model independently validated. Evidence status and technology readiness answer different questions and must not be combined into one confidence claim."
    )
    evidence_definition_cards()

    st.markdown("#### Evidence coverage")
    left, right = st.columns([1, 1])
    with left:
        evidence_counts = (
            df["evidence_label"].value_counts().reindex(EVIDENCE_ORDER).rename_axis("Evidence status").reset_index(name="Records")
        )
        evidence_chart = px.pie(
            evidence_counts,
            values="Records",
            names="Evidence status",
            hole=0.63,
            color="Evidence status",
            color_discrete_map=EVIDENCE_COLOURS,
        )
        evidence_chart.update_traces(textposition="outside", textinfo="label+value")
        base_plot_layout(evidence_chart, height=440)
        evidence_chart.update_layout(showlegend=False, margin=dict(l=45, r=45, t=30, b=30))
        st.plotly_chart(evidence_chart, width="stretch", config={"displaylogo": False})
    with right:
        reconciliation_order = [
            "Agreed",
            "Recomputed",
            "Derived",
            "Not Applicable",
            "Chemically Augmented",
            "Irreconcilable",
        ]
        reconciliation_counts = (
            df["reconciliation_label"].value_counts().reindex(reconciliation_order).dropna().rename_axis("Outcome").reset_index(name="Records")
        )
        reconciliation_chart = px.bar(
            reconciliation_counts.sort_values("Records"),
            x="Records",
            y="Outcome",
            orientation="h",
            title="Physics reconciliation outcome",
            color="Records",
            color_continuous_scale=["#2a3e70", "#62d7ff"],
        )
        base_plot_layout(reconciliation_chart, height=440)
        reconciliation_chart.update_layout(coloraxis_showscale=False)
        st.plotly_chart(reconciliation_chart, width="stretch", config={"displaylogo": False})

    source_verified = int((df["use_class"] == "verified").sum())
    direct_links = int(df["reference_url"].ne("").sum())
    irreconcilable = int((df["reconciliation"] == "irreconcilable").sum())
    summary = pd.DataFrame(
        [
            ("Distinct model records", len(df), "One row per named thruster model"),
            ("Records with a citation", int(df["reference"].notna().sum()), "Citation token or resolved reference stored"),
            ("Records with a direct URL or DOI", direct_links, "Resolvable link stored in the release"),
            ("Source-verified records", source_verified, "Cited document inspected"),
            ("Irreconcilable parameter records", irreconcilable, "Reported values do not satisfy the applicable governing relation"),
        ],
        columns=["Measure", "Count", "Interpretation"],
    )
    st.dataframe(summary, hide_index=True, width="stretch")

    st.markdown("#### Correct use of the data")
    st.markdown(
        """
        - Use **source-verified** records for evidence-sensitive statistical analysis.
        - Use manufacturer and compiled records for broad technology screening, with the evidence label retained.
        - Inspect the cited source and the parameter range before detailed spacecraft sizing.
        - Treat TRL as documented technology maturity, not proof that every stored performance value is independently validated.
        - Exclude or investigate the four irreconcilable records when applying the thrust–power–efficiency relation.
        """
    )
    st.download_button(
        "Download full workbook with methods",
        data=file_bytes(SOURCE_WORKBOOK),
        file_name=SOURCE_WORKBOOK.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_footer() -> None:
    st.markdown(
        "<div class="
        '"footer">EP Technology Explorer · Database release 1.0.0 · Values are for research and preliminary screening; consult primary sources before engineering design.</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    if not DATA_FILE.exists():
        st.error(f"Database file not found: {DATA_FILE}")
        st.stop()
    df = load_data(DATA_FILE)
    workspace = st.sidebar.radio("Workspace", ["Transfer design", "Mission design", "Thruster explorer", "Reference animations"])
    if workspace == "Transfer design":
        from transfer_ui import transfer_page
        transfer_page(df)
        return
    if workspace == "Mission design":
        from mission_ui import design
        design(df)
        return
    if workspace == "Reference animations":
        from mission_ui import reference_library
        st.title("Reference animations")
        st.write("Browse the recovered project media and interactive mission viewers.")
        reference_library()
        return
    active = sidebar_filters(df)
    render_header(df)
    overview_tab, browser_tab, compare_tab, profile_tab, evidence_tab = st.tabs(
        ["Overview", "Database browser", "Compare", "Model profile", "Evidence & methods"]
    )
    with overview_tab:
        render_overview(active)
    with browser_tab:
        render_browser(active)
    with compare_tab:
        render_compare(active)
    with profile_tab:
        render_profile(active)
    with evidence_tab:
        render_evidence(df)
    render_footer()


if __name__ == "__main__":
    main()

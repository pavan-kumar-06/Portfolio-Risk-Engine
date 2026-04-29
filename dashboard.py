"""Streamlit dashboard for the portfolio risk engine."""
from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2

from data import MarketDataError, load_market_data, validate_tickers, validate_weights
from risk import (
    RiskComputationError,
    compute_full_risk_report,
    compute_portfolio_log_returns,
    rolling_historical_var_backtest,
)
from scenarios import (
    ScenarioComputationError,
    run_assignment_scenarios,
    scenario_correlation_spike,
    scenario_custom_shocks,
)


DEFAULT_TICKERS = ("AAPL", "GOOGL", "MSFT")
DEFAULT_WEIGHTS = (40.0, 35.0, 25.0)
DEFAULT_SHOCKS = (-8.0, -7.0, -6.0)
DEFAULT_PORTFOLIO_VALUE = 1_000_000.0
DEFAULT_PERIOD = "5y"
DEFAULT_SIMULATIONS = 100_000
DEFAULT_BACKTEST_WINDOW = 250
DEFAULT_CORRELATION = 0.85


def money(value: float, decimals: int = 0) -> str:
    return f"${value:,.{decimals}f}"


def pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def default_portfolio_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": list(DEFAULT_TICKERS),
            "Weight (%)": list(DEFAULT_WEIGHTS),
            "Custom Shock (%)": list(DEFAULT_SHOCKS),
        }
    )


def parse_portfolio_frame(
    portfolio_df: pd.DataFrame,
    normalize_weights: bool,
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, float]]:
    cleaned = portfolio_df.copy()
    cleaned["Ticker"] = cleaned["Ticker"].fillna("").astype(str).str.strip().str.upper()
    cleaned = cleaned[cleaned["Ticker"] != ""]
    if cleaned.empty:
        raise ValueError("Add at least one ticker to the portfolio table.")

    tickers = tuple(validate_tickers(cleaned["Ticker"].tolist(), min_count=1))
    weights_pct = pd.to_numeric(cleaned["Weight (%)"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    shocks_pct = pd.to_numeric(cleaned["Custom Shock (%)"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if np.any(weights_pct < 0):
        raise ValueError("Weights must be non-negative.")
    if float(weights_pct.sum()) <= 0:
        raise ValueError("At least one weight must be greater than zero.")

    if normalize_weights:
        weights = tuple((weights_pct / weights_pct.sum()).tolist())
    else:
        weights = tuple((weights_pct / 100.0).tolist())
        validate_weights(weights, expected_count=len(tickers), tol=1e-6)

    shocks = dict(zip(tickers, (shocks_pct / 100.0).tolist(), strict=True))
    return tickers, weights, shocks


@st.cache_data(ttl=900, show_spinner=False)
def run_analysis_cached(
    tickers: tuple[str, ...],
    weights: tuple[float, ...],
    portfolio_value: float,
    period: str,
    simulations: int,
    window: int,
    stressed_corr: float,
) -> dict[str, object]:
    symbols = tuple(validate_tickers(tickers, min_count=1))
    weights_array = validate_weights(weights, expected_count=len(symbols))
    weights_map = dict(zip(symbols, weights_array, strict=True))

    market = load_market_data(symbols, period=period, min_observations=max(252, window + 1))
    returns = market.log_returns
    covariance = returns.cov().values

    risk_report = compute_full_risk_report(
        returns=returns,
        weights=weights_array,
        portfolio_value=portfolio_value,
        simulations=simulations,
        seed=42,
    )
    backtest = rolling_historical_var_backtest(
        returns=returns,
        weights=weights_array,
        portfolio_value=portfolio_value,
        confidence=0.95,
        window=window,
    )
    stress = run_assignment_scenarios(weights_map, portfolio_value, covariance)
    corr = scenario_correlation_spike(
        weights_map,
        portfolio_value,
        covariance,
        stressed_correlation=stressed_corr,
    )
    stress["Correlation spike VaR(95)"] = corr.var_95
    stress["Correlation spike VaR(99)"] = corr.var_99

    portfolio_returns = compute_portfolio_log_returns(returns, weights_array)
    portfolio_pnl = portfolio_returns * portfolio_value
    historical_var_95_return = -risk_report.historical_95 / portfolio_value
    historical_var_99_return = -risk_report.historical_99 / portfolio_value
    tail_table = pd.DataFrame(
        {
            "Date": portfolio_returns.index,
            "Log Return": portfolio_returns.values,
            "P&L": portfolio_pnl.values,
            "Loss": np.maximum(-portfolio_pnl.values, 0.0),
            "Beyond 5% Tail": portfolio_returns.values <= historical_var_95_return,
            "Beyond 1% Tail": portfolio_returns.values <= historical_var_99_return,
        }
    )
    cumulative = portfolio_returns.cumsum().apply(np.exp) - 1.0

    return {
        "symbols": symbols,
        "weights": weights_map,
        "portfolio_value": portfolio_value,
        "period": period,
        "start_date": market.start_date,
        "end_date": market.end_date,
        "prices": market.prices,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "portfolio_pnl": portfolio_pnl,
        "tail_table": tail_table,
        "cumulative_returns": cumulative,
        "covariance": covariance,
        "correlation": returns.corr(),
        "risk_report": asdict(risk_report),
        "backtest": {
            "confidence": backtest.confidence,
            "window": backtest.window,
            "breaches": backtest.breaches,
            "observations": backtest.observations,
            "breach_rate": backtest.breach_rate,
            "expected_breach_rate": backtest.expected_breach_rate,
            "breach_series": backtest.breach_series,
            "var_series_dollar": backtest.var_series_dollar,
            "realized_loss_series_dollar": backtest.realized_loss_series_dollar,
        },
        "stress": stress,
    }


def make_var_bar(report: dict[str, float]) -> go.Figure:
    methods = ["Historical", "Parametric", "Monte Carlo"]
    var95 = [report["historical_95"], report["parametric_95"], report["monte_carlo_95"]]
    var99 = [report["historical_99"], report["parametric_99"], report["monte_carlo_99"]]

    fig = go.Figure()
    fig.add_bar(name="VaR 95", x=methods, y=var95, marker_color="#d89000", text=[money(v) for v in var95])
    fig.add_bar(name="VaR 99", x=methods, y=var99, marker_color="#a83a32", text=[money(v) for v in var99])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        barmode="group",
        height=390,
        margin=dict(l=24, r=24, t=18, b=24),
        yaxis_title="Dollar loss",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def make_distribution(portfolio_returns: pd.Series, report: dict[str, float], portfolio_value: float) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(
        x=portfolio_returns,
        nbinsx=80,
        marker_color="#24706f",
        opacity=0.78,
        name="Daily returns",
    )
    tail_x = portfolio_returns[portfolio_returns <= -report["historical_95"] / portfolio_value]
    if not tail_x.empty:
        fig.add_histogram(
            x=tail_x,
            nbinsx=30,
            marker_color="#a83a32",
            opacity=0.78,
            name="Worst 5% tail",
        )
    fig.add_vline(
        x=-report["historical_95"] / portfolio_value,
        line_width=2,
        line_dash="dash",
        line_color="#d89000",
        annotation_text=f"VaR 95 {money(report['historical_95'])}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=-report["historical_99"] / portfolio_value,
        line_width=2,
        line_dash="dash",
        line_color="#a83a32",
        annotation_text=f"VaR 99 {money(report['historical_99'])}",
        annotation_position="bottom left",
    )
    fig.update_layout(
        height=390,
        margin=dict(l=24, r=24, t=18, b=24),
        xaxis_title="Daily portfolio log return",
        yaxis_title="Frequency",
        showlegend=True,
    )
    return fig


def make_log_gain_loss_chart(portfolio_returns: pd.Series, portfolio_pnl: pd.Series) -> go.Figure:
    colors = np.where(portfolio_pnl.values >= 0, "#24706f", "#a83a32")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=portfolio_returns.index,
            y=portfolio_pnl.values,
            marker_color=colors,
            name="Daily P&L from log returns",
            hovertemplate="%{x|%Y-%m-%d}<br>P&L: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=24, r=24, t=18, b=24),
        xaxis_title="Date",
        yaxis_title="Daily dollar gain / loss",
        showlegend=False,
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def make_tail_outlier_chart(tail_table: pd.DataFrame, report: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tail_table["Date"],
            y=tail_table["Loss"],
            mode="markers",
            marker=dict(
                size=6,
                color=np.where(tail_table["Beyond 5% Tail"], "#a83a32", "#24706f"),
                opacity=0.72,
            ),
            name="Daily losses",
            hovertemplate="%{x|%Y-%m-%d}<br>Loss: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=report["historical_95"],
        line_color="#d89000",
        line_dash="dash",
        annotation_text=f"Historical VaR 95: {money(report['historical_95'])}",
    )
    fig.add_hline(
        y=report["historical_99"],
        line_color="#a83a32",
        line_dash="dash",
        annotation_text=f"Historical VaR 99: {money(report['historical_99'])}",
    )
    fig.update_layout(
        height=420,
        margin=dict(l=24, r=24, t=18, b=24),
        xaxis_title="Date",
        yaxis_title="Daily loss",
        showlegend=False,
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def make_backtest(backtest: dict[str, object]) -> go.Figure:
    var_series = backtest["var_series_dollar"]
    loss_series = backtest["realized_loss_series_dollar"]
    breaches = backtest["breach_series"]
    breach_points = loss_series[breaches]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=var_series.index,
            y=var_series.values,
            mode="lines",
            name="Rolling VaR 95",
            line=dict(color="#a83a32", width=2),
            fill="tozeroy",
            fillcolor="rgba(168,58,50,0.08)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=loss_series.index,
            y=loss_series.values,
            name="Actual daily loss",
            marker_color="#24706f",
            opacity=0.43,
        )
    )
    if not breach_points.empty:
        fig.add_trace(
            go.Scatter(
                x=breach_points.index,
                y=breach_points.values,
                mode="markers",
                name="Breach",
                marker=dict(color="#201914", size=8, symbol="x"),
            )
        )
    fig.update_layout(
        height=430,
        margin=dict(l=24, r=24, t=18, b=24),
        xaxis_title="Date",
        yaxis_title="Dollar loss",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def kupiec_pof_test(breaches: int, observations: int, expected_rate: float) -> tuple[float, float]:
    """Kupiec unconditional coverage test for VaR breach frequency."""
    if observations <= 0:
        return 0.0, 1.0
    observed_rate = breaches / observations
    eps = 1e-12
    observed_rate = float(np.clip(observed_rate, eps, 1.0 - eps))
    expected_rate = float(np.clip(expected_rate, eps, 1.0 - eps))
    likelihood_expected = (
        (observations - breaches) * np.log(1.0 - expected_rate)
        + breaches * np.log(expected_rate)
    )
    likelihood_observed = (
        (observations - breaches) * np.log(1.0 - observed_rate)
        + breaches * np.log(observed_rate)
    )
    lr_uc = max(0.0, -2.0 * (likelihood_expected - likelihood_observed))
    p_value = float(1.0 - chi2.cdf(lr_uc, df=1))
    return float(lr_uc), p_value


def make_breach_cumulative_chart(backtest: dict[str, object]) -> go.Figure:
    breaches = backtest["breach_series"].astype(int)
    expected_rate = float(backtest["expected_breach_rate"])
    cumulative_actual = breaches.cumsum()
    cumulative_expected = pd.Series(
        np.arange(1, len(breaches) + 1) * expected_rate,
        index=breaches.index,
        name="Expected breaches",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cumulative_actual.index,
            y=cumulative_actual.values,
            mode="lines",
            name="Actual cumulative breaches",
            line=dict(color="#a83a32", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_expected.index,
            y=cumulative_expected.values,
            mode="lines",
            name="Expected cumulative breaches",
            line=dict(color="#201914", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=24, r=24, t=18, b=24),
        xaxis_title="Date",
        yaxis_title="Cumulative breach count",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    return fig


def make_breach_monthly_chart(backtest: dict[str, object]) -> go.Figure:
    breaches = backtest["breach_series"].astype(int)
    expected_rate = float(backtest["expected_breach_rate"])
    monthly = breaches.resample("ME").agg(["sum", "count"])
    monthly["expected"] = monthly["count"] * expected_rate
    monthly.index = monthly.index.strftime("%Y-%m")

    fig = go.Figure()
    fig.add_bar(
        x=monthly.index,
        y=monthly["sum"],
        name="Actual breaches",
        marker_color="#a83a32",
    )
    fig.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["expected"],
            mode="lines+markers",
            name="Expected at 5%",
            line=dict(color="#201914", width=2),
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=24, r=24, t=18, b=80),
        xaxis_title="Month",
        yaxis_title="Breach count",
        legend=dict(orientation="h", y=1.08, x=0.0),
    )
    return fig


def make_breach_detail_table(backtest: dict[str, object]) -> pd.DataFrame:
    var_series = backtest["var_series_dollar"]
    loss_series = backtest["realized_loss_series_dollar"]
    breach_mask = backtest["breach_series"]
    detail = pd.DataFrame(
        {
            "Date": loss_series.index,
            "Realized Loss": loss_series.values,
            "Rolling VaR 95": var_series.values,
            "Excess Over VaR": loss_series.values - var_series.values,
            "Breach": breach_mask.values,
        }
    )
    detail = detail[detail["Breach"]].copy()
    detail["Date"] = detail["Date"].dt.strftime("%Y-%m-%d")
    return detail.sort_values("Excess Over VaR", ascending=False)


def make_cumulative_chart(cumulative_returns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode="lines",
            line=dict(color="#24706f", width=2.3),
            name="Portfolio",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=24, r=24, t=18, b=24),
        xaxis_title="Date",
        yaxis_title="Cumulative return",
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def make_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#a83a32"],
                [0.5, "#f5f1e8"],
                [1.0, "#24706f"],
            ],
            text=np.round(correlation.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=320, margin=dict(l=24, r=24, t=18, b=24))
    return fig


def make_stress_table(data: dict[str, object], custom_loss: float | None = None) -> pd.DataFrame:
    symbols = set(data["symbols"])
    if {"AAPL", "GOOGL", "MSFT"}.intersection(symbols):
        assignment_shock_label = "AAPL -8%, GOOGL -7%, MSFT -6%; other assets 0%"
    else:
        assignment_shock_label = "First three assets -8%, -7%, -6%; remaining assets 0%"

    rows = [
        {
            "Scenario": "2008-style equity shock",
            "Shock": assignment_shock_label,
            "Loss": data["stress"]["2008-style equity shock"],
        },
        {
            "Scenario": "Tech drawdown",
            "Shock": "All assets -10%",
            "Loss": data["stress"]["Tech drawdown (-10%)"],
        },
        {
            "Scenario": "Correlation spike VaR(95)",
            "Shock": "Pairwise correlation stressed",
            "Loss": data["stress"]["Correlation spike VaR(95)"],
        },
        {
            "Scenario": "Correlation spike VaR(99)",
            "Shock": "Pairwise correlation stressed",
            "Loss": data["stress"]["Correlation spike VaR(99)"],
        },
    ]
    if custom_loss is not None:
        rows.append({"Scenario": "Custom shock", "Shock": "Sidebar shock vector", "Loss": custom_loss})

    portfolio_value = float(data["portfolio_value"])
    df = pd.DataFrame(rows)
    df["Percent of Portfolio"] = df["Loss"] / portfolio_value
    df["Loss"] = df["Loss"].map(lambda value: money(float(value), 2))
    df["Percent of Portfolio"] = df["Percent of Portfolio"].map(lambda value: pct(float(value), 2))
    return df


def render_metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f8f4ea;
            --ink: #201914;
            --muted: #695f53;
            --line: #d9cdb7;
            --teal: #24706f;
            --amber: #d89000;
            --red: #a83a32;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(248,244,234,0.98), rgba(241,235,222,0.98)),
                repeating-linear-gradient(90deg, rgba(32,25,20,0.03) 0, rgba(32,25,20,0.03) 1px, transparent 1px, transparent 64px);
            color: var(--ink);
        }
        .block-container {
            max-width: 1420px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.54);
            border: 1px solid var(--line);
            padding: 0.75rem 0.9rem;
            border-radius: 8px;
        }
        .metric-card {
            min-height: 118px;
            background: rgba(255,255,255,0.62);
            border: 1px solid var(--line);
            border-left: 5px solid var(--teal);
            border-radius: 8px;
            padding: 0.9rem 1rem;
        }
        .metric-label {
            font-size: 0.76rem;
            color: var(--muted);
            text-transform: uppercase;
            font-weight: 700;
        }
        .metric-value {
            margin-top: 0.2rem;
            font-size: 1.75rem;
            line-height: 1.1;
            font-weight: 800;
            color: var(--ink);
        }
        .metric-caption {
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.84rem;
        }
        .section-note {
            color: var(--muted);
            border-left: 4px solid var(--amber);
            padding-left: 0.85rem;
            margin: 0.35rem 0 1.1rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        /* ── Tab styling ── */
        /* unselected tabs */
        button[data-testid="stTab"] {
            background: rgba(36, 112, 111, 0.08);
            color: var(--muted);
            font-weight: 700;
            font-size: 0.88rem;
            border-radius: 8px 8px 0px 0px;
            border: none;
            padding: 6px 18px;
        }
        button[data-testid="stTab"]:hover {
            background: rgba(36, 112, 111, 0.15);
            color: var(--ink);
        }
        /* selected tab */
        button[data-testid="stTab"][aria-selected="true"] {
            background: #24706f;
            color: #ffffff;
            font-weight: 800;
            font-size: 0.88rem;
            border: none;
            padding: 6px 18px;
        }
        /* active tab bar container */
        div[data-testid="stTabBar"] {
            border-bottom: 2px solid var(--teal);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    float,
    str,
    int,
    int,
    float,
    dict[str, float],
]:
    with st.sidebar:
        st.header("Portfolio")
        portfolio_df = st.data_editor(
            default_portfolio_frame(),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance ticker"),
                "Weight (%)": st.column_config.NumberColumn(
                    "Weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    format="%.2f",
                ),
                "Custom Shock (%)": st.column_config.NumberColumn(
                    "Custom Shock (%)",
                    min_value=-95.0,
                    max_value=100.0,
                    step=1.0,
                    format="%.2f",
                ),
            },
            key="portfolio_editor",
        )
        normalize_weights = st.checkbox("Normalize weights to 100%", value=True)
        tickers, weights, custom_shocks = parse_portfolio_frame(portfolio_df, normalize_weights)

        normalized_display = pd.DataFrame(
            {
                "Ticker": tickers,
                "Model Weight": [pct(weight) for weight in weights],
                "Custom Shock": [pct(custom_shocks[ticker]) for ticker in tickers],
            }
        )
        st.dataframe(normalized_display, hide_index=True, width="stretch")

        portfolio_value = st.number_input(
            "Portfolio value",
            min_value=10_000.0,
            value=DEFAULT_PORTFOLIO_VALUE,
            step=50_000.0,
            format="%.0f",
        )

        st.subheader("Model")
        period = st.selectbox("Lookback", ["2y", "3y", "5y", "10y", "max"], index=2)
        simulations = st.select_slider(
            "Monte Carlo scenarios",
            options=[10_000, 25_000, 50_000, 100_000, 250_000],
            value=DEFAULT_SIMULATIONS,
        )
        window = st.slider("Rolling VaR window", min_value=60, max_value=500, value=DEFAULT_BACKTEST_WINDOW, step=10)
        stressed_corr = st.slider("Stress correlation", min_value=0.10, max_value=0.99, value=DEFAULT_CORRELATION, step=0.01)

    return tickers, weights, portfolio_value, period, simulations, window, stressed_corr, custom_shocks


def render_dashboard(data: dict[str, object], custom_loss: float | None) -> None:
    symbols: Sequence[str] = data["symbols"]
    report: dict[str, float] = data["risk_report"]
    backtest: dict[str, object] = data["backtest"]
    portfolio_value = float(data["portfolio_value"])

    st.title("Portfolio Risk Engine")
    st.caption(
        f"{', '.join(symbols)} | {money(portfolio_value)} portfolio | "
        f"{data['start_date']} to {data['end_date']} | {len(data['returns'])} daily returns"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Historical VaR 95", money(report["historical_95"]), "Empirical 5th percentile loss")
    with c2:
        render_metric_card("Historical VaR 99", money(report["historical_99"]), "Empirical 1st percentile loss")
    with c3:
        render_metric_card("Expected Shortfall 99", money(report["cvar_99"]), "Normal CVaR beyond VaR")
    with c4:
        render_metric_card(
            "Backtest breach rate",
            pct(float(backtest["breach_rate"])),
            f"{backtest['breaches']} breaches over {backtest['observations']} days",
        )

    st.markdown(
        '<div class="section-note">VaR is reported as a positive dollar loss. At 95% confidence, losses should exceed VaR on roughly 5% of trading days.</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_var, tab_tail, tab_stress, tab_backtest, tab_data = st.tabs(
        ["Overview", "VaR Methods", "Tail / Outliers", "Stress Tests", "Backtest", "Data Quality"]
    )

    with tab_overview:
        left, right = st.columns([1.2, 0.8])
        with left:
            st.plotly_chart(make_var_bar(report), width="stretch")
        with right:
            st.plotly_chart(make_cumulative_chart(data["cumulative_returns"]), width="stretch")

        summary = pd.DataFrame(
            [
                ["Historical Simulation", report["historical_95"], report["historical_99"]],
                ["Parametric Normal", report["parametric_95"], report["parametric_99"]],
                ["Monte Carlo", report["monte_carlo_95"], report["monte_carlo_99"]],
            ],
            columns=["Method", "VaR 95", "VaR 99"],
        )
        st.dataframe(
            summary.style.format({"VaR 95": "${:,.2f}", "VaR 99": "${:,.2f}"}),
            width="stretch",
            hide_index=True,
        )

    with tab_var:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(
                make_distribution(data["portfolio_returns"], report, portfolio_value),
                width="stretch",
            )
        with col_b:
            st.plotly_chart(make_correlation_heatmap(data["correlation"]), width="stretch")

        st.dataframe(
            pd.DataFrame(
                [
                    ["CVaR 95", report["cvar_95"], report["cvar_95"] / portfolio_value],
                    ["CVaR 99", report["cvar_99"], report["cvar_99"] / portfolio_value],
                ],
                columns=["Metric", "Dollar Loss", "Percent of Portfolio"],
            ).style.format({"Dollar Loss": "${:,.2f}", "Percent of Portfolio": "{:.2%}"}),
            hide_index=True,
            width="stretch",
        )

    with tab_tail:
        tail_table: pd.DataFrame = data["tail_table"]
        tail_5 = tail_table[tail_table["Beyond 5% Tail"]].copy()
        tail_1 = tail_table[tail_table["Beyond 1% Tail"]].copy()
        expected_5 = len(tail_table) * 0.05

        t1, t2, t3 = st.columns(3)
        with t1:
            render_metric_card("Observed 5% tail days", f"{len(tail_5)}", f"Expected about {expected_5:.1f} days")
        with t2:
            render_metric_card("Observed 1% tail days", f"{len(tail_1)}", f"Worst empirical outliers")
        with t3:
            render_metric_card("Worst daily loss", money(float(tail_table["Loss"].max())), "From log-return P&L")

        st.plotly_chart(make_tail_outlier_chart(tail_table, report), width="stretch")
        st.plotly_chart(make_log_gain_loss_chart(data["portfolio_returns"], data["portfolio_pnl"]), width="stretch")

        worst_days = tail_table.sort_values("Loss", ascending=False).head(25).copy()
        worst_days["Date"] = worst_days["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            worst_days.style.format(
                {
                    "Log Return": "{:.3%}",
                    "P&L": "${:,.2f}",
                    "Loss": "${:,.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    with tab_stress:
        st.dataframe(make_stress_table(data, custom_loss), width="stretch", hide_index=True)

        fig = go.Figure()
        stress_df = make_stress_table(data, custom_loss)
        fig.add_bar(
            x=stress_df["Scenario"],
            y=[float(v.replace("$", "").replace(",", "")) for v in stress_df["Loss"]],
            marker_color=["#a83a32", "#d89000", "#24706f", "#24706f"] + (["#201914"] if custom_loss is not None else []),
        )
        fig.update_layout(height=380, margin=dict(l=24, r=24, t=18, b=80), yaxis_title="Dollar loss")
        fig.update_yaxes(tickprefix="$", separatethousands=True)
        st.plotly_chart(fig, width="stretch")

    with tab_backtest:
        expected = float(backtest["expected_breach_rate"])
        observed = float(backtest["breach_rate"])
        expected_breaches = int(round(float(backtest["observations"]) * expected))
        lr_uc, kupiec_p = kupiec_pof_test(
            int(backtest["breaches"]),
            int(backtest["observations"]),
            expected,
        )

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            render_metric_card("Actual breaches", f"{backtest['breaches']}", f"over {backtest['observations']} test days")
        with b2:
            render_metric_card("Expected breaches", f"{expected_breaches}", f"at {pct(expected)} expected rate")
        with b3:
            render_metric_card("Observed breach rate", pct(observed), "model calibration check")
        with b4:
            render_metric_card("Kupiec p-value", f"{kupiec_p:.3f}", f"LRuc {lr_uc:.2f}")

        st.plotly_chart(make_backtest(backtest), width="stretch")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_breach_cumulative_chart(backtest), width="stretch")
        with right:
            st.plotly_chart(make_breach_monthly_chart(backtest), width="stretch")

        breach_detail = make_breach_detail_table(backtest)
        st.dataframe(
            breach_detail.head(30).style.format(
                {
                    "Realized Loss": "${:,.2f}",
                    "Rolling VaR 95": "${:,.2f}",
                    "Excess Over VaR": "${:,.2f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )

        if observed > 0.075:
            st.error(
                f"Observed breach rate is {pct(observed)}, above the 7.5% monitoring threshold. "
                "The model is likely underestimating tail risk in this window."
            )
        else:
            st.success(
                f"Observed breach rate is {pct(observed)}, close to the expected {pct(expected)} rate for 95% VaR."
            )

    with tab_data:
        prices: pd.DataFrame = data["prices"]
        returns: pd.DataFrame = data["returns"]
        st.dataframe(prices.tail(10).style.format("${:,.2f}"), width="stretch")
        st.dataframe(
            pd.DataFrame(
                {
                    "Annualized Volatility": returns.std() * np.sqrt(252),
                    "Mean Daily Return": returns.mean(),
                    "Missing Values": prices.isna().sum(),
                }
            ).style.format({"Annualized Volatility": "{:.2%}", "Mean Daily Return": "{:.3%}"}),
            width="stretch",
        )

        st.markdown(
            """
            Assumptions: adjusted close prices from Yahoo Finance, daily log returns, fixed portfolio weights,
            one-day horizon, no transaction costs, and no intraday liquidity effects. Parametric and Monte Carlo
            methods assume multivariate normal returns; historical VaR is sample dependent.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Portfolio Risk Engine", layout="wide", initial_sidebar_state="expanded")
    page_style()

    try:
        tickers, weights, portfolio_value, period, simulations, window, stressed_corr, custom_shocks = sidebar_controls()
        data = run_analysis_cached(tickers, weights, portfolio_value, period, simulations, window, stressed_corr)
        custom_loss = scenario_custom_shocks(data["weights"], portfolio_value, custom_shocks)
        render_dashboard(data, custom_loss)
    except (MarketDataError, RiskComputationError, ScenarioComputationError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected dashboard error: {exc}")
        st.stop()


if __name__ == "__main__":
    main()

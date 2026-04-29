"""Command-line orchestrator for the portfolio VaR and stress-test engine."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import MarketDataError, load_market_data, validate_weights
from risk import (
    FullRiskReport,
    RiskComputationError,
    compute_full_risk_report,
    compute_portfolio_log_returns,
    rolling_historical_var_backtest,
)
from scenarios import ScenarioComputationError, run_assignment_scenarios


TICKERS = ["AAPL", "GOOGL", "MSFT"]
WEIGHTS = {"AAPL": 0.40, "GOOGL": 0.35, "MSFT": 0.25}
PORTFOLIO_VALUE = 1_000_000.0
PERIOD = "5y"
MONTE_CARLO_SIMULATIONS = 100_000
RANDOM_SEED = 42
BACKTEST_WINDOW = 250

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"


def money(value: float) -> str:
    """Format dollars consistently for terminal reports."""
    return f"${value:,.2f}"


def percent(value: float) -> str:
    """Format ratios as percentages."""
    return f"{value:.2%}"


def build_results_frame(report: FullRiskReport) -> pd.DataFrame:
    """Create a tidy VaR table for display and CSV export."""
    return pd.DataFrame(
        [
            ["Historical Simulation", report.historical_95, report.historical_99],
            ["Parametric Normal", report.parametric_95, report.parametric_99],
            ["Monte Carlo", report.monte_carlo_95, report.monte_carlo_99],
        ],
        columns=["Method", "VaR(95)", "VaR(99)"],
    )


def save_return_distribution_plot(
    portfolio_returns: pd.Series,
    report: FullRiskReport,
    output_path: Path,
) -> None:
    """Save histogram of portfolio returns with historical VaR markers."""
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(portfolio_returns, bins=80, color="#245a7a", alpha=0.78, ax=ax)
    ax.axvline(
        -report.historical_95 / PORTFOLIO_VALUE,
        color="#d08c00",
        linestyle="--",
        linewidth=2.2,
        label=f"Historical VaR(95): {money(report.historical_95)}",
    )
    ax.axvline(
        -report.historical_99 / PORTFOLIO_VALUE,
        color="#a51f2b",
        linestyle="--",
        linewidth=2.2,
        label=f"Historical VaR(99): {money(report.historical_99)}",
    )
    ax.set_title("Daily Portfolio Log Return Distribution")
    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_var_comparison_plot(report: FullRiskReport, output_path: Path) -> None:
    """Save side-by-side VaR comparison across methods."""
    df = build_results_frame(report)
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_95 = ax.bar(x - width / 2, df["VaR(95)"], width, label="VaR(95)", color="#d08c00")
    bars_99 = ax.bar(x + width / 2, df["VaR(99)"], width, label="VaR(99)", color="#a51f2b")

    for bars in (bars_95, bars_99):
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                money(height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("1-Day VaR Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Method"])
    ax.set_ylabel("Dollar loss")
    ax.yaxis.set_major_formatter(lambda y, _: f"${y:,.0f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_backtest_plot(backtest, output_path: Path) -> None:
    """Save rolling historical VaR backtest plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        backtest.var_series_dollar.index,
        backtest.var_series_dollar.values,
        color="#a51f2b",
        linewidth=1.8,
        label=f"Rolling VaR({int(backtest.confidence * 100)}), {backtest.window}d",
    )
    ax.bar(
        backtest.realized_loss_series_dollar.index,
        backtest.realized_loss_series_dollar.values,
        width=1.3,
        alpha=0.42,
        color="#245a7a",
        label="Actual daily loss",
    )
    breach_points = backtest.realized_loss_series_dollar[backtest.breach_series]
    if not breach_points.empty:
        ax.scatter(
            breach_points.index,
            breach_points.values,
            color="#2b2024",
            s=22,
            zorder=5,
            label=f"Breaches: {len(breach_points)}",
        )
    ax.set_title("Rolling Historical VaR Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Dollar loss")
    ax.yaxis.set_major_formatter(lambda y, _: f"${y:,.0f}")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_pipeline() -> dict[str, object]:
    """Run the complete assignment pipeline and persist report artifacts."""
    PLOTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    weights_array = validate_weights([WEIGHTS[t] for t in TICKERS])
    market = load_market_data(TICKERS, period=PERIOD)
    returns = market.log_returns
    covariance = returns.cov().values

    report = compute_full_risk_report(
        returns=returns,
        weights=weights_array,
        portfolio_value=PORTFOLIO_VALUE,
        simulations=MONTE_CARLO_SIMULATIONS,
        seed=RANDOM_SEED,
    )
    stress = run_assignment_scenarios(WEIGHTS, PORTFOLIO_VALUE, covariance)
    backtest = rolling_historical_var_backtest(
        returns=returns,
        weights=weights_array,
        portfolio_value=PORTFOLIO_VALUE,
        confidence=0.95,
        window=BACKTEST_WINDOW,
    )
    portfolio_returns = compute_portfolio_log_returns(returns, weights_array)

    save_return_distribution_plot(
        portfolio_returns,
        report,
        PLOTS_DIR / "portfolio_return_distribution.png",
    )
    save_var_comparison_plot(report, PLOTS_DIR / "var_comparison.png")
    save_backtest_plot(backtest, PLOTS_DIR / "rolling_var_backtest.png")

    var_table = build_results_frame(report)
    var_table.to_csv(REPORTS_DIR / "var_results.csv", index=False)

    stress_table = pd.DataFrame(
        [{"Scenario": name, "Dollar Loss": value, "Percent of Portfolio": value / PORTFOLIO_VALUE} for name, value in stress.items()]
    )
    stress_table.to_csv(REPORTS_DIR / "stress_results.csv", index=False)

    summary = {
        "portfolio_value": PORTFOLIO_VALUE,
        "tickers": TICKERS,
        "weights": WEIGHTS,
        "period": PERIOD,
        "start_date": market.start_date,
        "end_date": market.end_date,
        "observations": len(returns),
        "var": asdict(report),
        "stress": stress,
        "backtest": {
            "window": backtest.window,
            "confidence": backtest.confidence,
            "breaches": backtest.breaches,
            "observations": backtest.observations,
            "breach_rate": backtest.breach_rate,
            "expected_breach_rate": backtest.expected_breach_rate,
        },
        "artifacts": {
            "plots_dir": str(PLOTS_DIR),
            "reports_dir": str(REPORTS_DIR),
        },
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_report(summary: dict[str, object]) -> None:
    """Print an interview-friendly terminal report."""
    var = summary["var"]
    stress = summary["stress"]
    backtest = summary["backtest"]
    weights = summary["weights"]

    print("\nPortfolio Risk Engine")
    print("=" * 72)
    print(f"Portfolio value: {money(float(summary['portfolio_value']))}")
    print(
        "Portfolio: "
        + " | ".join(f"{ticker} {weight:.0%}" for ticker, weight in weights.items())
    )
    print(
        f"Data: {summary['start_date']} to {summary['end_date']} "
        f"({summary['observations']} daily returns)"
    )

    print("\n1-Day Value at Risk")
    print("-" * 72)
    print(f"{'Method':<26}{'VaR(95)':>18}{'VaR(99)':>18}")
    print(f"{'Historical Simulation':<26}{money(var['historical_95']):>18}{money(var['historical_99']):>18}")
    print(f"{'Parametric Normal':<26}{money(var['parametric_95']):>18}{money(var['parametric_99']):>18}")
    print(f"{'Monte Carlo':<26}{money(var['monte_carlo_95']):>18}{money(var['monte_carlo_99']):>18}")

    print("\nExpected Shortfall under normal assumption")
    print("-" * 72)
    print(f"CVaR(95): {money(var['cvar_95'])}")
    print(f"CVaR(99): {money(var['cvar_99'])}")

    print("\nStress Testing")
    print("-" * 72)
    for name, value in stress.items():
        print(f"{name:<34}{money(value):>18}")

    print("\nRolling VaR Backtest")
    print("-" * 72)
    print(
        f"Breaches: {backtest['breaches']} / {backtest['observations']} "
        f"({percent(backtest['breach_rate'])}); expected near {percent(backtest['expected_breach_rate'])}"
    )
    if backtest["breach_rate"] > 0.075:
        print("Interpretation: breach rate is elevated; historical VaR is likely underestimating tail risk.")
    else:
        print("Interpretation: breach rate is close to the expected rate for a 95% VaR model.")

    print("\nSaved artifacts")
    print("-" * 72)
    print(f"Plots:   {summary['artifacts']['plots_dir']}")
    print(f"Reports: {summary['artifacts']['reports_dir']}")


def main() -> int:
    try:
        summary = run_pipeline()
        print_report(summary)
        return 0
    except (MarketDataError, RiskComputationError, ScenarioComputationError, ValueError) as exc:
        print(f"Risk engine failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

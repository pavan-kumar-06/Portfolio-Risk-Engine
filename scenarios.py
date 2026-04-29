"""Stress testing scenarios for the portfolio risk engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from risk import RiskComputationError


class ScenarioComputationError(RuntimeError):
    """Raised when stress scenario computation fails."""


@dataclass(frozen=True)
class CorrelationSpikeResult:
    var_95: float
    var_99: float


def _ordered_weight_vector(weights: Mapping[str, float], ordered_tickers: Sequence[str]) -> np.ndarray:
    w = np.array([float(weights[t]) for t in ordered_tickers], dtype=float)
    if np.any(w < 0):
        raise ScenarioComputationError("Scenario weights must be non-negative.")
    total = float(w.sum())
    if abs(total - 1.0) > 1e-8:
        raise ScenarioComputationError(f"Scenario weights must sum to 1.0; got {total:.8f}.")
    return w


def scenario_2008_style_assignment(weights: Mapping[str, float], portfolio_value: float) -> float:
    """Assignment scenario: AAPL/GOOGL/MSFT shocks, or first three assets as fallback."""
    tickers = list(weights.keys())
    w = _ordered_weight_vector(weights, tickers)

    shock_map = {"AAPL": -0.08, "GOOGL": -0.07, "MSFT": -0.06}
    if any(ticker in shock_map for ticker in tickers):
        shock_vec = np.array([shock_map.get(ticker, 0.0) for ticker in tickers], dtype=float)
    else:
        shock_vec = np.zeros(len(tickers), dtype=float)
        shock_vec[: min(3, len(tickers))] = np.array([-0.08, -0.07, -0.06], dtype=float)[: len(tickers)]

    pnl_return = float(w @ shock_vec)
    return max(0.0, -pnl_return * portfolio_value)


def scenario_tech_drawdown_assignment(weights: Mapping[str, float], portfolio_value: float) -> float:
    """Assignment scenario: all three assets down 10% same day."""
    tickers = list(weights.keys())
    w = _ordered_weight_vector(weights, tickers)

    shock = -0.10
    pnl_return = float(w.sum() * shock)
    return max(0.0, -pnl_return * portfolio_value)


def scenario_correlation_spike(
    weights: Mapping[str, float],
    portfolio_value: float,
    covariance: np.ndarray,
    stressed_correlation: float = 0.85,
) -> CorrelationSpikeResult:
    """Keep vols fixed, set all pairwise correlations to stressed value, recompute parametric VaR."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ScenarioComputationError("Covariance matrix must be square.")
    if not (0.0 < stressed_correlation < 1.0):
        raise ScenarioComputationError("Stressed correlation must be in (0, 1).")

    tickers = list(weights.keys())
    w = _ordered_weight_vector(weights, tickers)

    if covariance.shape[0] != len(w):
        raise ScenarioComputationError(
            f"Covariance dimension ({covariance.shape[0]}) and number of weights ({len(w)}) mismatch."
        )

    vol = np.sqrt(np.diag(covariance))
    if np.any(vol <= 0) or np.any(~np.isfinite(vol)):
        raise ScenarioComputationError("Invalid volatility extracted from covariance matrix.")

    corr = np.full_like(covariance, stressed_correlation, dtype=float)
    np.fill_diagonal(corr, 1.0)

    d = np.diag(vol)
    stressed_cov = d @ corr @ d

    variance = float(w @ stressed_cov @ w)
    if variance < 0:
        raise ScenarioComputationError("Negative stressed variance computed.")
    sigma = float(np.sqrt(variance))

    var95 = float(norm.ppf(0.95) * sigma * portfolio_value)
    var99 = float(norm.ppf(0.99) * sigma * portfolio_value)
    return CorrelationSpikeResult(var_95=max(0.0, var95), var_99=max(0.0, var99))


def scenario_custom_shocks(
    weights: Mapping[str, float],
    portfolio_value: float,
    shocks: Mapping[str, float],
) -> float:
    """Generic scenario utility for dashboard ad-hoc stress testing."""
    tickers = list(weights.keys())
    if any(t not in shocks for t in tickers):
        missing = [t for t in tickers if t not in shocks]
        raise ScenarioComputationError(f"Missing shock values for: {missing}")

    w = _ordered_weight_vector(weights, tickers)
    s = np.array([float(shocks[t]) for t in tickers], dtype=float)

    if np.any(~np.isfinite(s)):
        raise ScenarioComputationError("Shock values must be finite numbers.")

    pnl_return = float(w @ s)
    return max(0.0, -pnl_return * portfolio_value)


def run_assignment_scenarios(
    weights: Mapping[str, float],
    portfolio_value: float,
    covariance: np.ndarray,
) -> dict[str, float]:
    """Run all required stress scenarios from the assignment brief."""
    try:
        s1 = scenario_2008_style_assignment(weights, portfolio_value)
        s2 = scenario_tech_drawdown_assignment(weights, portfolio_value)
        corr = scenario_correlation_spike(weights, portfolio_value, covariance)
    except RiskComputationError as exc:
        raise ScenarioComputationError(str(exc)) from exc

    return {
        "2008-style equity shock": s1,
        "Tech drawdown (-10%)": s2,
        "Correlation spike VaR(95)": corr.var_95,
        "Correlation spike VaR(99)": corr.var_99,
    }

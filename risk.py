"""VaR, Expected Shortfall, and backtesting methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


class RiskComputationError(RuntimeError):
    """Raised when VaR or related risk metrics cannot be computed safely."""


@dataclass(frozen=True)
class VarResult:
    confidence: float
    var_dollar: float


@dataclass(frozen=True)
class RollingBacktestResult:
    confidence: float
    window: int
    breaches: int
    observations: int
    breach_rate: float
    expected_breach_rate: float
    breach_series: pd.Series
    var_series_dollar: pd.Series
    realized_loss_series_dollar: pd.Series


@dataclass(frozen=True)
class FullRiskReport:
    historical_95: float
    historical_99: float
    parametric_95: float
    parametric_99: float
    cvar_95: float
    cvar_99: float
    monte_carlo_95: float
    monte_carlo_99: float


def _validate_returns_and_weights(
    returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise RiskComputationError("Returns input must be a non-empty DataFrame.")

    if np.any(~np.isfinite(returns.values)):
        raise RiskComputationError("Returns contain non-finite values.")

    w = np.asarray(weights, dtype=float)
    if returns.shape[1] != w.shape[0]:
        raise RiskComputationError(
            f"Weight vector length ({w.shape[0]}) does not match returns columns ({returns.shape[1]})."
        )
    if np.any(~np.isfinite(w)):
        raise RiskComputationError("Weights contain non-finite values.")

    total = float(w.sum())
    if abs(total - 1.0) > 1e-8:
        raise RiskComputationError(f"Weights must sum to 1.0; current sum is {total:.8f}.")

    if portfolio_value <= 0 or not np.isfinite(portfolio_value):
        raise RiskComputationError("Portfolio value must be a positive finite number.")

    return w


def compute_portfolio_log_returns(returns: pd.DataFrame, weights: Sequence[float]) -> pd.Series:
    """Compute aligned daily portfolio log returns."""
    w = np.asarray(weights, dtype=float)
    if returns.shape[1] != w.shape[0]:
        raise RiskComputationError("Weights and returns dimensions do not align.")

    portfolio_returns = returns.values @ w
    return pd.Series(portfolio_returns, index=returns.index, name="portfolio_log_return")


def historical_var(
    returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    confidence: float,
) -> VarResult:
    """Historical simulation VaR from empirical portfolio return quantile."""
    if not 0 < confidence < 1:
        raise RiskComputationError("Confidence must be between 0 and 1.")

    w = _validate_returns_and_weights(returns, weights, portfolio_value)
    portfolio_returns = compute_portfolio_log_returns(returns, w)

    left_tail_quantile = float(np.quantile(portfolio_returns.values, 1.0 - confidence))
    var_dollar = max(0.0, -left_tail_quantile * portfolio_value)
    return VarResult(confidence=confidence, var_dollar=var_dollar)


def _portfolio_sigma(returns: pd.DataFrame, weights: np.ndarray) -> float:
    cov = returns.cov().values
    variance = float(weights @ cov @ weights)
    if variance < 0:
        raise RiskComputationError("Computed negative portfolio variance; covariance matrix may be invalid.")
    return float(np.sqrt(max(variance, 0.0)))


def parametric_var(
    returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    confidence: float,
) -> VarResult:
    """Variance-covariance VaR under normal return assumption."""
    if not 0 < confidence < 1:
        raise RiskComputationError("Confidence must be between 0 and 1.")

    w = _validate_returns_and_weights(returns, weights, portfolio_value)
    sigma = _portfolio_sigma(returns, w)
    z = float(norm.ppf(confidence))
    if z <= 0:
        raise RiskComputationError("Invalid z-score computed for confidence level.")

    var_dollar = max(0.0, z * sigma * portfolio_value)
    return VarResult(confidence=confidence, var_dollar=var_dollar)


def parametric_expected_shortfall(
    returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    confidence: float,
) -> float:
    """Closed-form Expected Shortfall under normal assumption."""
    if not 0 < confidence < 1:
        raise RiskComputationError("Confidence must be between 0 and 1.")

    w = _validate_returns_and_weights(returns, weights, portfolio_value)
    sigma = _portfolio_sigma(returns, w)
    z = float(norm.ppf(confidence))
    es = sigma * norm.pdf(z) / (1.0 - confidence) * portfolio_value
    return max(0.0, float(es))


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    simulations: int = 100_000,
    seed: int = 42,
) -> dict[str, float]:
    """Monte Carlo VaR with multivariate normal simulations."""
    if simulations < 10_000:
        raise RiskComputationError("Use at least 10,000 simulations for stable Monte Carlo VaR.")

    w = _validate_returns_and_weights(returns, weights, portfolio_value)
    mean_vector = returns.mean().values
    covariance = returns.cov().values

    try:
        rng = np.random.default_rng(seed)
        simulated_returns = rng.multivariate_normal(mean=mean_vector, cov=covariance, size=simulations)
    except Exception as exc:  # noqa: BLE001
        raise RiskComputationError(f"Monte Carlo simulation failed: {exc}") from exc

    pnl = simulated_returns @ w * portfolio_value

    var_95 = max(0.0, float(-np.quantile(pnl, 0.05)))
    var_99 = max(0.0, float(-np.quantile(pnl, 0.01)))
    return {"VaR(95)": var_95, "VaR(99)": var_99}


def rolling_historical_var_backtest(
    returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    confidence: float = 0.95,
    window: int = 250,
) -> RollingBacktestResult:
    """Rolling historical VaR backtest and breach statistics."""
    if window < 30:
        raise RiskComputationError("Backtest window must be at least 30 trading days.")
    if not 0 < confidence < 1:
        raise RiskComputationError("Confidence must be between 0 and 1.")

    w = _validate_returns_and_weights(returns, weights, portfolio_value)
    portfolio_returns = compute_portfolio_log_returns(returns, w)

    if len(portfolio_returns) <= window:
        raise RiskComputationError(
            f"Need more observations than window size ({window}); got {len(portfolio_returns)}."
        )

    var_values: list[float] = []
    losses: list[float] = []
    breaches: list[bool] = []
    idx = portfolio_returns.index[window:]

    q = 1.0 - confidence
    values = portfolio_returns.values

    for end_i in range(window, len(values)):
        lookback = values[end_i - window : end_i]
        var_return = float(np.quantile(lookback, q))
        var_loss_dollar = max(0.0, -var_return * portfolio_value)

        realized_loss_dollar = max(0.0, -values[end_i] * portfolio_value)
        breach = realized_loss_dollar > var_loss_dollar

        var_values.append(var_loss_dollar)
        losses.append(realized_loss_dollar)
        breaches.append(breach)

    breach_series = pd.Series(breaches, index=idx, name="var_breach")
    observations = int(len(breach_series))
    breach_count = int(breach_series.sum())
    breach_rate = float(breach_count / observations)

    return RollingBacktestResult(
        confidence=confidence,
        window=window,
        breaches=breach_count,
        observations=observations,
        breach_rate=breach_rate,
        expected_breach_rate=1.0 - confidence,
        breach_series=breach_series,
        var_series_dollar=pd.Series(var_values, index=idx, name="rolling_var_dollar"),
        realized_loss_series_dollar=pd.Series(losses, index=idx, name="realized_loss_dollar"),
    )


def compute_full_risk_report(
    returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    simulations: int = 100_000,
    seed: int = 42,
) -> FullRiskReport:
    """Convenience function to compute all assignment VaR metrics."""
    hist_95 = historical_var(returns, weights, portfolio_value, 0.95).var_dollar
    hist_99 = historical_var(returns, weights, portfolio_value, 0.99).var_dollar

    param_95 = parametric_var(returns, weights, portfolio_value, 0.95).var_dollar
    param_99 = parametric_var(returns, weights, portfolio_value, 0.99).var_dollar

    cvar_95 = parametric_expected_shortfall(returns, weights, portfolio_value, 0.95)
    cvar_99 = parametric_expected_shortfall(returns, weights, portfolio_value, 0.99)

    mc = monte_carlo_var(
        returns=returns,
        weights=weights,
        portfolio_value=portfolio_value,
        simulations=simulations,
        seed=seed,
    )

    return FullRiskReport(
        historical_95=hist_95,
        historical_99=hist_99,
        parametric_95=param_95,
        parametric_99=param_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        monte_carlo_95=mc["VaR(95)"],
        monte_carlo_99=mc["VaR(99)"],
    )

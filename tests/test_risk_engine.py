"""Unit tests for core risk-engine math and validation."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data import validate_tickers, validate_weights
from risk import (
    compute_full_risk_report,
    historical_var,
    parametric_expected_shortfall,
    parametric_var,
    rolling_historical_var_backtest,
)
from scenarios import (
    scenario_2008_style_assignment,
    scenario_custom_shocks,
    scenario_tech_drawdown_assignment,
)


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.multivariate_normal(
            mean=[0.0004, 0.0003, 0.0002],
            cov=[
                [0.00035, 0.00012, 0.00010],
                [0.00012, 0.00030, 0.00011],
                [0.00010, 0.00011, 0.00025],
            ],
            size=600,
        )
        self.returns = pd.DataFrame(data, columns=["AAPL", "GOOGL", "MSFT"])
        self.weights = np.array([0.40, 0.35, 0.25])
        self.weights_map = {"AAPL": 0.40, "GOOGL": 0.35, "MSFT": 0.25}
        self.portfolio_value = 1_000_000.0

    def test_validation_rejects_bad_inputs(self) -> None:
        self.assertEqual(validate_tickers(["aapl", "googl", "msft"]), ["AAPL", "GOOGL", "MSFT"])
        self.assertEqual(validate_tickers(["aapl", "msft", "nvda", "meta"]), ["AAPL", "MSFT", "NVDA", "META"])
        with self.assertRaises(ValueError):
            validate_tickers(["AAPL", "AAPL", "MSFT"])
        with self.assertRaises(ValueError):
            validate_weights([0.5, 0.3, 0.3])

    def test_var_and_cvar_are_positive_and_ordered(self) -> None:
        hist95 = historical_var(self.returns, self.weights, self.portfolio_value, 0.95).var_dollar
        param95 = parametric_var(self.returns, self.weights, self.portfolio_value, 0.95).var_dollar
        cvar95 = parametric_expected_shortfall(self.returns, self.weights, self.portfolio_value, 0.95)

        self.assertGreater(hist95, 0)
        self.assertGreater(param95, 0)
        self.assertGreater(cvar95, param95)

    def test_full_report_contains_three_methods(self) -> None:
        report = compute_full_risk_report(
            self.returns,
            self.weights,
            self.portfolio_value,
            simulations=10_000,
            seed=11,
        )
        self.assertGreater(report.historical_99, report.historical_95)
        self.assertGreater(report.parametric_99, report.parametric_95)
        self.assertGreater(report.cvar_99, report.cvar_95)

    def test_assignment_scenarios_match_spec(self) -> None:
        self.assertAlmostEqual(
            scenario_2008_style_assignment(self.weights_map, self.portfolio_value),
            71_500.0,
            places=2,
        )
        self.assertAlmostEqual(
            scenario_tech_drawdown_assignment(self.weights_map, self.portfolio_value),
            100_000.0,
            places=2,
        )

    def test_assignment_scenarios_support_more_than_three_assets(self) -> None:
        weights = {"AAPL": 0.30, "GOOGL": 0.25, "MSFT": 0.20, "NVDA": 0.25}
        self.assertAlmostEqual(
            scenario_2008_style_assignment(weights, self.portfolio_value),
            53_500.0,
            places=2,
        )
        self.assertAlmostEqual(
            scenario_tech_drawdown_assignment(weights, self.portfolio_value),
            100_000.0,
            places=2,
        )

    def test_custom_shock_uses_weighted_loss(self) -> None:
        loss = scenario_custom_shocks(
            self.weights_map,
            self.portfolio_value,
            {"AAPL": -0.10, "GOOGL": 0.02, "MSFT": -0.05},
        )
        self.assertAlmostEqual(loss, 45_500.0, places=2)

    def test_backtest_produces_consistent_counts(self) -> None:
        result = rolling_historical_var_backtest(
            self.returns,
            self.weights,
            self.portfolio_value,
            confidence=0.95,
            window=250,
        )
        self.assertEqual(result.observations, len(self.returns) - 250)
        self.assertEqual(result.breaches, int(result.breach_series.sum()))
        self.assertGreaterEqual(result.breach_rate, 0.0)
        self.assertLessEqual(result.breach_rate, 1.0)


if __name__ == "__main__":
    unittest.main()

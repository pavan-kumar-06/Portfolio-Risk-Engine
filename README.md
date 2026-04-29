---
title: Portfolio Risk Engine
emoji: 📈
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Portfolio VaR and Stress Testing Engine

Production-style solution for the Quant Risk mini-project: a 3-stock equity portfolio risk engine with historical, parametric, and Monte Carlo VaR, Expected Shortfall, stress testing, rolling VaR backtesting, and a Streamlit dashboard.

## Portfolio

Default portfolio:

| Ticker | Weight |
| --- | ---: |
| AAPL | 40% |
| GOOGL | 35% |
| MSFT | 25% |

Portfolio value: `$1,000,000`

## What Is Implemented

| Area | Coverage |
| --- | --- |
| Data | Yahoo Finance adjusted close download, retries, validation, cleaning, log returns |
| Historical VaR | Empirical 5% and 1% left-tail portfolio return quantiles |
| Parametric VaR | Variance-covariance VaR under normal return assumption |
| Expected Shortfall | Closed-form normal CVaR at 95% and 99% |
| Monte Carlo VaR | 100,000 multivariate-normal one-day scenarios |
| Stress tests | 2008-style shock, tech drawdown, correlation spike |
| Bonus | Rolling 250-day historical VaR backtest with breach rate |
| Dashboard | Executive summary, VaR charts, stress table, correlation heatmap, rolling breach visualization |
| Quality | Defensive validation, typed dataclasses, unit tests, saved CSV/JSON artifacts |

## Run Locally

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

Outputs are written to:

| Path | Contents |
| --- | --- |
| `plots/portfolio_return_distribution.png` | Historical return distribution with VaR lines |
| `plots/var_comparison.png` | VaR comparison across all three methods |
| `plots/rolling_var_backtest.png` | Rolling VaR breach backtest |
| `reports/summary.json` | Machine-readable full result summary |
| `reports/var_results.csv` | VaR table |
| `reports/stress_results.csv` | Stress table |

## Run Dashboard

```bash
streamlit run dashboard.py
```

For Hugging Face Spaces, create a new Space with the Docker SDK and push this folder. The included `Dockerfile` runs Streamlit on port `7860`, and the YAML block above tells Spaces which port to expose. Store any deployment token only as a secret or local environment variable. Do not commit it to the repository.

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Interpretation Notes

VaR is reported as a positive dollar loss. A 1-day 95% VaR of `$25,000` means the model estimates a 5% chance that tomorrow's portfolio loss exceeds `$25,000`.

Historical VaR makes no distributional assumption, but it is fully driven by the selected lookback window. Parametric VaR and Monte Carlo VaR use a multivariate-normal assumption, which is fast and transparent but can understate tail risk when equity returns are fat-tailed or skewed.

The correlation spike scenario keeps individual asset volatilities unchanged and raises pairwise correlations to `0.85`, showing how diversification can break down during market stress.

The rolling backtest estimates VaR from the previous 250 trading days and checks whether the next day breaches that VaR. For a well-calibrated 95% VaR model, the breach rate should be close to 5%.

## Interview Talking Points

1. The risk engine separates data, risk methods, scenario logic, orchestration, dashboard, and tests.
2. Historical, parametric, and Monte Carlo VaR are intentionally compared because they answer the same risk question under different assumptions.
3. Expected Shortfall is included because VaR gives a threshold, while CVaR estimates the average loss after crossing that threshold.
4. Stress scenarios are deterministic and easy to audit, which complements probabilistic VaR.
5. The rolling backtest turns the model into something falsifiable by comparing forecasted VaR against realized losses.

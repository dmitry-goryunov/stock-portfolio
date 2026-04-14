# Stock Portfolio Optimization

Markowitz mean-variance portfolio optimization for UK-listed equities using Pyomo + IPOPT.

## Overview

This notebook finds the minimum-variance portfolio across a universe of FTSE 100 stocks and ETFs, subject to a minimum return constraint. It also computes the efficient frontier, visualizes correlations and allocations, exhaustively searches for the best 3-asset portfolio, and evaluates any custom allocation you define.

## Features

- **Mean-variance optimization** via Pyomo + IPOPT solver
- **Efficient frontier** computed across 40 return targets
- **Correlation heatmap** of asset returns
- **Optimal weights bar chart**
- **Best 3-asset portfolio** found by brute-forcing all C(n,3) combinations
- **Custom portfolio analyser** with support for a CASH (risk-free) position
- Dividend-adjusted or price-only return mode toggle

## Results Summary

| Metric | Value |
|---|---|
| Portfolio variance (monthly) | 0.000844 |
| Expected yearly return (with dividends) | 10.16% |
| Sharpe ratio (with dividends) | 0.56 |
| Weighted dividend yield | 3.33% |

**Top holdings:** ULVR.L (26.26%), NG.L (20.63%), AZN.L (12.56%), BA.L (12.16%), HSBA.L (10.58%)

**Best 3-asset portfolio:** BA.L (16.52%), HUKX.L (51.32%), NG.L (32.16%) — variance 0.001122, return 10.35% with dividends

## Requirements

```
yfinance
pandas
numpy
pyomo
idaes-pse        # provides APPSI-IPOPT; falls back to classic SolverFactory
seaborn
matplotlib
```

Install with:

```bash
pip install yfinance pandas numpy pyomo idaes-pse seaborn matplotlib
idaes get-extensions   # downloads IPOPT binary
```

## Configuration

All user-facing parameters are in **Section 1 (Configuration)**:

| Parameter | Default | Description |
|---|---|---|
| `selected_tickers` | 15 UK stocks/ETFs | Asset universe (Yahoo Finance symbols) |
| `start_date` | `'2019-10-01'` | Start of historical data window |
| `end_date` | `'2024-10-31'` | End of historical data window |
| `reward_floor` | `0.004` | Minimum monthly return (~5% annualised) |
| `risk_free_rate` | `0.045` | Annual risk-free rate for Sharpe ratio (UK gilt proxy) |
| `include_dividends` | `True` | `True` = total return; `False` = price-only |

**Default asset universe:**

| Sector | Tickers |
|---|---|
| Energy | SHEL.L, BP.L |
| Financials | HSBA.L, BARC.L |
| Healthcare | AZN.L, GSK.L |
| Consumer Staples | ULVR.L, DGE.L |
| Mining | RIO.L |
| Defence | BA.L |
| Utilities | NG.L |
| Telecoms | VOD.L |
| ETFs | ISF.L, VMID.L, HUKX.L |

## Usage

1. Open `portfolio_optimization.ipynb` in Jupyter.
2. Edit the configuration cell (Section 1) to adjust tickers, dates, and constraints.
3. Run all cells (`Kernel > Restart & Run All`).
4. Optionally, edit the **Custom Portfolio Analyser** cell (Section 10) to evaluate a specific allocation:

```python
custom_portfolio = {
    "ULVR.L": 0.25,
    "AZN.L":  0.25,
    "HSBA.L": 0.25,
    "CASH":   0.25,   # risk-free position at risk_free_rate
}
```

Weights are automatically normalised if they don't sum to 1.

## Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Configuration | Tickers, dates, return floor, risk-free rate |
| 2 | Download & Prepare Returns | Monthly returns via yfinance (60 months) |
| 3 | Build Pyomo Model | Covariance matrix, weights, variance objective |
| 4 | Solve | APPSI-IPOPT or classic SolverFactory |
| 5 | Results | Optimal weights, Sharpe ratio, dividend yield |
| 6 | Efficient Frontier | 40-point risk/return curve |
| 7 | Correlation Heatmap | Seaborn heatmap of pairwise correlations |
| 8 | Optimal Weights — Bar Chart | Weight distribution of optimal portfolio |
| 9 | Best 3-Asset Portfolio | Brute-force search over all C(n,3) combos |
| 10 | Custom Portfolio Analyser | Evaluate any user-defined allocation |

## Optimization Model

**Objective:** minimise portfolio variance

$$\min \sum_{i,j} w_i \, \Sigma_{ij} \, w_j$$

**Subject to:**

$$\sum_i w_i \mu_i \geq r_{\text{floor}}, \quad \sum_i w_i = 1, \quad w_i \geq 0$$

where $\Sigma$ is the sample covariance matrix of monthly returns and $\mu$ is the vector of mean monthly returns.

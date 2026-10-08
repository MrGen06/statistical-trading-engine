# Statistical Arbitrage & Pairs Trading Engine

An end-to-end, quantitative statistical arbitrage and pairs trading engine for equities (National Stock Exchange - NSE) built in Python.

The system rigorously identifies cointegrated asset pairs, models their long-run equilibrium spread, and executes a mean-reverting trading strategy incorporating execution lag, transaction costs, slippage, and out-of-sample validation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why Pairs Trading?](#why-pairs-trading)
- [System Architecture](#system-architecture)
- [Mathematical & Statistical Methodology](#mathematical--statistical-methodology)
  - [1. Return Correlation Screening](#1-return-correlation-screening)
  - [2. Engle-Granger Cointegration Test](#2-engle-granger-cointegration-test)
  - [3. Augmented Dickey-Fuller (ADF) Test](#3-augmented-dickey-fuller-adf-test)
  - [4. OLS Hedge Ratio & Spread Construction](#4-ols-hedge-ratio--spread-construction)
  - [5. Half-Life of Mean Reversion](#5-half-life-of-mean-reversion)
  - [6. Rolling Z-Score & State-Based Signals](#6-rolling-z-score--state-based-signals)
- [Backtesting & Execution Engine](#backtesting--execution-engine)
  - [Zero-Lookahead Execution](#zero-lookahead-execution)
  - [Realistic Friction Modeling](#realistic-friction-modeling)
- [Repository Structure](#repository-structure)
- [Configuration (.env)](#configuration-env)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Running the Main Pipeline](#running-the-main-pipeline)
  - [Running Chronological OOS Validation](#running-chronological-oos-validation)
  - [Outputs & Visualizations](#outputs--visualizations)
- [Performance Metrics](#performance-metrics)
- [Disclaimer](#disclaimer)

---

## Project Overview

Pairs trading is a market-neutral statistical arbitrage strategy designed to capitalize on temporary deviations from the equilibrium relationship between two historically co-moving assets.

This engine automates the full quantitative pipeline:
1. **Automated Data Ingestion:** Downloads historical daily adjusted close prices via Yahoo Finance (`yfinance`), applying data validation, forward filling, and handling MultiIndex outputs.
2. **Multi-Stage Statistical Filtering:** Prunes pairwise search space via daily return correlation ($\ge 0.7$), Engle-Granger cointegration ($p < 0.05$), and residual ADF stationarity ($p < 0.05$).
3. **Dynamic Modeling:** Solves Ordinary Least Squares (OLS) hedge ratio $\beta$, computes the cointegrating spread, and estimates the Ornstein-Uhlenbeck mean-reversion half-life.
4. **Zero-Lookahead Simulation:** Generates persistent state-machine signals ($+1, -1, 0$) based on rolling Z-score thresholds, shifting executed positions by 1 bar (`shift(1)`) to avoid lookahead bias.
5. **Cost-Aware Accounting:** Enforces linear transaction costs and bid-ask slippage on position transitions.
6. **Chronological OOS Validation:** Provides train/test splitting (70% in-sample / 30% out-of-sample) to evaluate walk-forward robustness and prevent overfitting.

---

## Why Pairs Trading?

Traditional directional trading relies on forecasting asset price trajectories. Pairs trading shifts the focus to relative pricing between two assets:

- **Correlation vs. Cointegration:**
  - *Correlation* quantifies short-term co-movement in returns. Two assets can be highly correlated while drifting apart in price levels over time.
  - *Cointegration* establishes that a linear combination of two non-stationary price series produces a stationary process with time-invariant mean and variance. This provides a mean-reverting spread suitable for statistical arbitrage.
- **Market Neutrality:** By holding long and short positions simultaneously weighted by the hedge ratio $\beta$, the portfolio aims to isolate the spread dynamics while hedging broad market directional risk.

---

## System Architecture

```text
       Historical Market Data (yfinance)
                      │
                      ▼
            Data Ingestion & Cleaning
          (Drop NaN, ffill short gaps)
                      │
                      ▼
         Pairwise Return Correlation
            (threshold >= 0.70)
                      │
                      ▼
        Engle-Granger Cointegration
               (p < 0.05)
                      │
                      ▼
         ADF Residual Stationarity
               (p < 0.05)
                      │
                      ▼
        Selected Pair (Lowest p-value)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
OLS Hedge Ratio (β)          Half-Life (t_1/2)
        │                           │
        └─────────────┬─────────────┘
                      ▼
            Spread = A - β * B
                      │
                      ▼
       Rolling Z-Score (Lookback N)
                      │
                      ▼
        Signal State Machine Generation
          (Entry: ±2.0 | Exit: ±0.5)
                      │
                      ▼
       Zero-Lookahead Backtester (t+1)
      (Transaction Costs + Slippage)
                      │
                      ▼
       Performance Analytics & Reports
      (Equity Curve, Sharpe, Max Drawdown)
                      │
                      ▼
       Chronological OOS Validation (70/30)
```

---

## Mathematical & Statistical Methodology

### 1. Return Correlation Screening
Calculates Pearson correlation of percentage daily returns across all candidate pairs:

$$\rho_{A, B} = \frac{\text{Cov}(R_A, R_B)}{\sigma_{R_A} \sigma_{R_B}}$$

Candidate pairs must satisfy $\rho \ge 0.70$ before undergoing computationally intensive tests.

### 2. Engle-Granger Cointegration Test
Tests whether two non-stationary $I(1)$ series share a common stochastic drift:

$$P_{A, t} = \alpha + \beta P_{B, t} + \epsilon_t$$

We test whether the residual series $\epsilon_t$ is stationary $I(0)$ at significance level $\alpha = 0.05$.

### 3. Augmented Dickey-Fuller (ADF) Test
Evaluates unit root hypothesis on the residual spread:

$$\Delta \epsilon_t = \gamma \epsilon_{t-1} + \sum_{i=1}^p \delta_i \Delta \epsilon_{t-i} + e_t$$

A rejection of $H_0$ ($p < 0.05$) confirms that the spread exhibits stationary mean-reverting properties.

### 4. OLS Hedge Ratio & Spread Construction
Estimates dynamic hedge ratio $\beta$ via Ordinary Least Squares:

$$\beta = \frac{\text{Cov}(P_A, P_B)}{\text{Var}(P_B)}$$

The stationary spread $S_t$ is computed as:

$$S_t = P_{A, t} - \beta P_{B, t}$$

### 5. Half-Life of Mean Reversion
Modeled as a continuous-time Ornstein-Uhlenbeck process, approximated using an AR(1) specification:

$$\Delta S_t = \lambda S_{t-1} + \mu + \epsilon_t$$

The expected duration for the spread to revert halfway back to its mean is:

$$t_{1/2} = -\frac{\ln(2)}{\lambda} \quad (\text{where } \lambda < 0)$$

### 6. Rolling Z-Score & State-Based Signals
Normalizes the spread using a rolling lookback window ($N$ periods):

$$Z_t = \frac{S_t - \mu_t(N)}{\sigma_t(N)}$$

**Trading Rules:**
| Condition | State Transition | Position |
| :--- | :--- | :--- |
| $Z_t \le -\text{Entry } Z$ | Go Long Spread | Long Asset A, Short Asset B ($\beta$) |
| $Z_t \ge +\text{Entry } Z$ | Go Short Spread | Short Asset A, Long Asset B ($\beta$) |
| $\|Z_t\| \le \text{Exit } Z$ | Reversion to Mean | Liquidate position to Flat ($0$) |
| $-\text{Entry } Z < Z_t < +\text{Entry } Z$ | Inside band | Maintain previous position |

---

## Backtesting & Execution Engine

### Zero-Lookahead Execution
To prevent forward-looking bias, the strategy does not execute trades at the close of bar $t$ when the signal is generated. Instead:

$$\text{Executed Position}_t = \text{Signal}_{t-1}$$

Trades are entered and marked on subsequent market returns, reflecting realistic end-of-day execution.

### Realistic Friction Modeling
- **Capital Allocation:** Weights are normalized by notional values:
  $$w_A = \frac{1}{1 + |\beta|}, \quad w_B = \frac{|\beta|}{1 + |\beta|}$$
- **Turnover:** Monitored as $|\Delta \text{Executed Position}_t|$.
- **Transaction Costs & Slippage:** Subtracted directly from daily returns whenever the target position changes:
  $$\text{Friction}_t = |\Delta \text{Position}_t| \times (\text{Cost}_{\text{bps}} + \text{Slippage}_{\text{bps}})$$

---

## Repository Structure

```text
statistical-trading-engine/
├── data/                      # Cached or local data files
├── results/                   # Generated backtest logs and charts
│   ├── backtest.csv           # Detailed trade-by-trade bar history
│   └── equity_curve.png       # Generated equity performance plot
├── src/
│   ├── __init__.py
│   ├── config.py              # Environment variable loader and defaults
│   ├── data.py                # Market data ingestion & cleaning
│   ├── screening.py           # Correlation, Cointegration & ADF screening
│   ├── strategy.py            # Spread, hedge ratio, half-life & signal logic
│   ├── backtest.py            # Execution simulator with cost modeling
│   ├── metrics.py             # Sharpe, Sortino, Drawdown, Total Return
│   └── validation.py          # Chronological train/test walk-forward validation
├── tests/
│   ├── __init__.py
│   └── test_strategy.py       # Unit tests
├── .env.example               # Example configuration parameters
├── .gitignore                 # Version control exclusions
├── main.py                    # Main pipeline orchestrator
├── README.md                  # Project documentation
└── requirements.txt           # Python package dependencies
```

---

## Configuration (.env)

Configuration parameters are managed through environment variables. Copy the template to `.env`:

```bash
cp .env.example .env
```

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `START_DATE` | string | `2018-01-01` | Start date for historical price download (`YYYY-MM-DD`) |
| `END_DATE` | string | `2026-01-01` | End date for historical price download (`YYYY-MM-DD`) |
| `INITIAL_CAPITAL`| float | `100000` | Initial portfolio cash ($ / ₹) |
| `ENTRY_Z` | float | `2.0` | Number of standard deviations to trigger entry |
| `EXIT_Z` | float | `0.5` | Number of standard deviations to close/exit position |
| `TRANSACTION_COST`| float | `0.0005` | Linear broker commission / exchange fee (5 bps) |
| `SLIPPAGE` | float | `0.0005` | Estimated bid-ask slippage penalty (5 bps) |
| `LOOKBACK` | int | `60` | Rolling lookback window for spread mean and standard deviation |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- `git`

### 2. Clone Repository
```bash
git clone https://github.com/MrGen06/statistical-trading-engine.git
cd statistical-trading-engine
```

### 3. Create & Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Main Pipeline

Execute the primary screening, trading, and backtesting script:

```bash
python main.py
```

**Workflow executed:**
1. Downloads adjusted close data for configured tickers (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, `ITC.NS`, `LT.NS`).
2. Screens all pair combinations for correlation ($\ge 0.70$), Engle-Granger cointegration ($p < 0.05$), and stationary ADF residuals.
3. Selects the most statistically significant pair (lowest cointegration $p$-value).
4. Solves OLS hedge ratio and computes mean-reversion half-life.
5. Generates long/short mean-reverting signals from rolling Z-score.
6. Simulates zero-lookahead backtest with fees and slippage.
7. Prints comprehensive summary table and saves outputs to `results/`.

### Running Chronological OOS Validation

To validate that your selected pair generalizes out-of-sample without lookahead or data snooping:

```python
from src.config import ENTRY_Z, EXIT_Z, LOOKBACK, START_DATE, END_DATE
from src.data import download_prices
from src.validation import chronological_validation

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS"]
prices = download_prices(TICKERS, start=START_DATE, end=END_DATE)

results = chronological_validation(
    prices=prices,
    lookback=LOOKBACK,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
)

print(results)
```

### Outputs & Visualizations
Upon completing a backtest, artifacts are stored in `results/`:
- `results/backtest.csv`: Per-bar time-series of dates, prices, signals, positions, daily returns, equity, and fees.
- `results/equity_curve.png`: High-resolution equity performance curve plot.

---

## Performance Metrics

The engine calculates industry-standard risk-adjusted return metrics:

- **Total Return:** Net percentage gain or loss over initial equity.
- **Sharpe Ratio (Annualized):** Mean excess return divided by standard deviation ($\times \sqrt{252}$).
- **Sortino Ratio (Annualized):** Downside-risk adjusted return assessing performance against negative volatility.
- **Maximum Drawdown (MDD):** Maximum peak-to-trough decline across the portfolio equity curve.
- **Number of Executed Trades:** Total distinct rebalances and position turns.

---

## Disclaimer

This software is for educational, research, and informational purposes only. It is not financial or investment advice. Algorithmic trading and statistical arbitrage involve substantial risk of financial loss. Past statistical relationships (cointegration and correlation) are non-stationary in real market environments and can break down at any time. Always test strategies thoroughly in simulated environments before committing real capital.
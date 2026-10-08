import pandas as pd

from src.backtest import backtest_pairs_strategy
from src.metrics import (
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_total_return,
)
from src.screening import screen_pairs
from src.strategy import (
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    generate_signals,
)


def chronological_validation(
    prices,
    lookback,
    entry_z,
    exit_z,
    initial_capital=100000,
    transaction_cost=0.0005,
    slippage=0.0005,
):
    split_index = int(len(prices) * 0.70)

    train_prices = prices.iloc[:split_index].copy()
    test_prices = prices.iloc[split_index:].copy()

    if train_prices.empty or test_prices.empty:
        raise ValueError("Insufficient data for chronological train/test split.")

    screened_pairs = screen_pairs(train_prices)

    if screened_pairs.empty:
        raise RuntimeError(
            "No statistically significant pairs found in training data."
        )

    selected = screened_pairs.iloc[0]

    ticker_a = selected["ticker_a"]
    ticker_b = selected["ticker_b"]

    train_series_a = train_prices[ticker_a]
    train_series_b = train_prices[ticker_b]

    _, hedge_ratio = calculate_hedge_ratio(
        train_series_a,
        train_series_b,
    )

    train_spread = calculate_spread(
        train_series_a,
        train_series_b,
        hedge_ratio,
    )

    test_series_a = test_prices[ticker_a]
    test_series_b = test_prices[ticker_b]

    test_spread = calculate_spread(
        test_series_a,
        test_series_b,
        hedge_ratio,
    )

    test_zscore = calculate_zscore(
        test_spread,
        lookback,
    )

    test_signals = generate_signals(
        test_zscore,
        entry_z=entry_z,
        exit_z=exit_z,
    )

    test_backtest = backtest_pairs_strategy(
        series_a=test_series_a,
        series_b=test_series_b,
        hedge_ratio=hedge_ratio,
        signals=test_signals,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )

    test_return = calculate_total_return(test_backtest["equity"])
    test_sharpe = calculate_sharpe(test_backtest["daily_return"])
    test_sortino = calculate_sortino(test_backtest["daily_return"])
    test_max_drawdown = calculate_max_drawdown(test_backtest["equity"])

    return {
        "selected_pair": f"{ticker_a} / {ticker_b}",
        "training_period": (
            train_prices.index[0],
            train_prices.index[-1],
        ),
        "test_period": (
            test_prices.index[0],
            test_prices.index[-1],
        ),
        "test_return": test_return,
        "test_sharpe": test_sharpe,
        "test_sortino": test_sortino,
        "test_max_drawdown": test_max_drawdown,
    }
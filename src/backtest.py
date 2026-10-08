import numpy as np
import pandas as pd


def backtest_pairs_strategy(
    series_a,
    series_b,
    hedge_ratio,
    signals,
    initial_capital=100000,
    transaction_cost=0.0005,
    slippage=0.0005,
):
    aligned = pd.concat([series_a, series_b, signals], axis=1).dropna()
    aligned.columns = ["price_a", "price_b", "signal"]

    aligned["executed_position"] = aligned["signal"].shift(1).fillna(0)

    returns_a = aligned["price_a"].pct_change().fillna(0)
    returns_b = aligned["price_b"].pct_change().fillna(0)

    notional_a = 1.0
    notional_b = abs(hedge_ratio)

    total_notional = notional_a + notional_b
    weight_a = notional_a / total_notional
    weight_b = notional_b / total_notional

    spread_returns = (
        weight_a * returns_a
        - np.sign(hedge_ratio) * weight_b * returns_b
    )

    aligned["daily_return"] = (
        aligned["executed_position"] * spread_returns
    )

    position_change = aligned["executed_position"].diff().abs().fillna(
        aligned["executed_position"].abs()
    )

    aligned["transaction_cost"] = position_change * transaction_cost
    aligned["daily_return"] -= aligned["transaction_cost"]

    slippage_cost = position_change * slippage
    aligned["daily_return"] -= slippage_cost

    aligned["equity"] = initial_capital * (
        1 + aligned["daily_return"]
    ).cumprod()

    aligned["turnover"] = position_change

    result = aligned.reset_index()
    result = result.rename(columns={result.columns[0]: "date"})

    return result[
        [
            "date",
            "price_a",
            "price_b",
            "signal",
            "executed_position",
            "daily_return",
            "equity",
            "transaction_cost",
        ]
    ]
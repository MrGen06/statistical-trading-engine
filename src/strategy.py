import numpy as np
import pandas as pd
import statsmodels.api as sm


def calculate_hedge_ratio(series_a, series_b):
    """Estimate A = alpha + beta * B using OLS."""

    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    aligned.columns = ["A", "B"]

    X = sm.add_constant(aligned["B"])
    model = sm.OLS(aligned["A"], X).fit()

    alpha = model.params["const"]
    beta = model.params["B"]

    return alpha, beta


def calculate_spread(series_a, series_b, beta):
    """Calculate the pair spread: A - beta * B."""

    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    aligned.columns = ["A", "B"]

    return aligned["A"] - beta * aligned["B"]


def calculate_half_life(spread):
    """Estimate mean-reversion half-life from an AR(1)-style regression."""

    spread = spread.dropna()

    lagged_spread = spread.shift(1)
    delta_spread = spread.diff()

    regression_data = pd.concat(
        [delta_spread, lagged_spread],
        axis=1,
    ).dropna()

    regression_data.columns = ["delta", "lagged"]

    if len(regression_data) < 2:
        return np.nan

    X = sm.add_constant(regression_data["lagged"])
    model = sm.OLS(regression_data["delta"], X).fit()

    lambda_value = model.params["lagged"]

    if lambda_value >= 0:
        return np.nan

    return -np.log(2) / lambda_value


def calculate_zscore(spread, lookback):
    """Calculate rolling Z-score of the spread."""

    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()

    return (spread - rolling_mean) / rolling_std


def generate_signals(zscore, entry_z=2.0, exit_z=0.5):
    """
    Generate persistent pair-trading signals.

    +1 = long spread
    -1 = short spread
     0 = flat
    """

    signals = pd.Series(0, index=zscore.index, dtype=int)

    position = 0

    for date, z in zscore.items():
        if pd.isna(z):
            signals.loc[date] = 0
            continue

        if position == 0:
            if z <= -entry_z:
                position = 1
            elif z >= entry_z:
                position = -1

        elif position == 1:
            if abs(z) <= exit_z:
                position = 0

        elif position == -1:
            if abs(z) <= exit_z:
                position = 0

        signals.loc[date] = position

    return signals
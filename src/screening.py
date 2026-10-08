import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def correlation_screen(prices, threshold=0.7):
    """Return pairs whose daily returns have correlation above threshold."""

    returns = prices.pct_change().dropna()
    correlation_matrix = returns.corr()

    candidate_pairs = []

    for ticker_a, ticker_b in itertools.combinations(correlation_matrix.columns, 2):
        correlation = correlation_matrix.loc[ticker_a, ticker_b]

        if correlation >= threshold:
            candidate_pairs.append(
                {
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "correlation": correlation,
                }
            )

    return candidate_pairs


def test_cointegration(series_a, series_b):
    """Run the Engle-Granger cointegration test."""

    test_statistic, p_value, _ = coint(series_a, series_b)

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "is_cointegrated": p_value < 0.05,
    }


def adf_test(series):
    """Run the Augmented Dickey-Fuller stationarity test."""

    test_statistic, p_value, _, _, _, _ = adfuller(series.dropna())

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "is_stationary": p_value < 0.05,
    }


def _calculate_residual_spread(series_a, series_b):
    """Construct the OLS residual spread used for the ADF test."""

    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    aligned.columns = ["A", "B"]

    X = sm.add_constant(aligned["B"])
    model = sm.OLS(aligned["A"], X).fit()

    return model.resid


def screen_pairs(prices):
    """Screen pairs using correlation, cointegration, and ADF tests."""

    candidate_pairs = correlation_screen(prices)

    results = []

    for pair in candidate_pairs:
        ticker_a = pair["ticker_a"]
        ticker_b = pair["ticker_b"]

        series_a = prices[ticker_a]
        series_b = prices[ticker_b]

        coint_result = test_cointegration(series_a, series_b)

        spread = _calculate_residual_spread(series_a, series_b)
        adf_result = adf_test(spread)

        results.append(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "correlation": pair["correlation"],
                "coint_pvalue": coint_result["p_value"],
                "adf_pvalue": adf_result["p_value"],
                "is_cointegrated": coint_result["is_cointegrated"],
                "is_stationary": adf_result["is_stationary"],
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "ticker_a",
                "ticker_b",
                "correlation",
                "coint_pvalue",
                "adf_pvalue",
                "is_cointegrated",
                "is_stationary",
            ]
        )

    result_df = pd.DataFrame(results)

    # Keep statistically significant pairs and rank by cointegration p-value.
    result_df = result_df[
        (result_df["coint_pvalue"] < 0.05)
        & (result_df["adf_pvalue"] < 0.05)
    ].copy()

    return result_df.sort_values("coint_pvalue").reset_index(drop=True)
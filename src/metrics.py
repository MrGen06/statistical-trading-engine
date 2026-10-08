import numpy as np
import pandas as pd


def calculate_sharpe(returns):
    returns = pd.Series(returns).dropna()

    if returns.empty or returns.std() == 0:
        return np.nan

    return returns.mean() / returns.std() * np.sqrt(252)


def calculate_sortino(returns):
    returns = pd.Series(returns).dropna()

    if returns.empty:
        return np.nan

    downside_returns = returns[returns < 0]

    if downside_returns.empty or downside_returns.std() == 0:
        return np.nan

    return returns.mean() / downside_returns.std() * np.sqrt(252)


def calculate_max_drawdown(equity):
    equity = pd.Series(equity).dropna()

    if equity.empty:
        return np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    return drawdown.min()


def calculate_total_return(equity):
    equity = pd.Series(equity).dropna()

    if len(equity) < 2:
        return np.nan

    return equity.iloc[-1] / equity.iloc[0] - 1
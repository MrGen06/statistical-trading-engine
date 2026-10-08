import matplotlib.pyplot as plt
from src.backtest import backtest_pairs_strategy
from src.metrics import (
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_total_return,
)

from src.config import END_DATE, ENTRY_Z, EXIT_Z, LOOKBACK, START_DATE
from src.data import download_prices
from src.screening import screen_pairs
from src.strategy import (
    calculate_half_life,
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    generate_signals,
)

from src.validation import chronological_validation


TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
]


def main():
    print("Downloading historical data...")

    prices = download_prices(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
    )

    print("\nPAIR SCREENING")
    print("-" * 64)

    screened_pairs = screen_pairs(prices)

    if screened_pairs.empty:
        print("No statistically significant pairs were found.")
        print("Pipeline stopped before strategy construction because the screening criteria were not satisfied.")
        return

    print(screened_pairs.to_string(index=False))

    # Select the valid pair with the lowest cointegration p-value.
    selected = screened_pairs.iloc[0]

    ticker_a = selected["ticker_a"]
    ticker_b = selected["ticker_b"]

    series_a = prices[ticker_a]
    series_b = prices[ticker_b]

    alpha, beta = calculate_hedge_ratio(series_a, series_b)

    spread = calculate_spread(
        series_a,
        series_b,
        beta,
    )

    half_life = calculate_half_life(spread)

    zscore = calculate_zscore(
        spread,
        LOOKBACK,
    )

    signals = generate_signals(
        zscore,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
    )

    current_zscore = zscore.iloc[-1]
    current_signal = signals.iloc[-1]

    signal_name = {
        1: "LONG",
        -1: "SHORT",
        0: "FLAT",
    }[current_signal]

    print("\nSELECTED PAIR")
    print("-" * 64)
    print(f"{ticker_a} / {ticker_b}")

    print(f"\nCorrelation:           {selected['correlation']:.4f}")
    print(f"Cointegration p:       {selected['coint_pvalue']:.6f}")
    print(f"ADF p-value:           {selected['adf_pvalue']:.6f}")
    print(f"Hedge Ratio:           {beta:.4f}")
    print(f"Half-Life:             {half_life:.2f} days")
    print(f"Current Z-score:       {current_zscore:.4f}")
    print(f"Signal:                {signal_name}")

    backtest = backtest_pairs_strategy(
        series_a=series_a,
        series_b=series_b,
        hedge_ratio=beta,
        signals=signals,
    )

    total_return = calculate_total_return(backtest["equity"])
    sharpe = calculate_sharpe(backtest["daily_return"])
    sortino = calculate_sortino(backtest["daily_return"])
    max_drawdown = calculate_max_drawdown(backtest["equity"])
    number_of_trades = int(
        (backtest["executed_position"].diff().abs() > 0).sum()
    )

    print("\nBACKTEST RESULTS")
    print("-" * 64)
    print(f"Selected Pair:         {ticker_a} / {ticker_b}")
    print(f"Initial Capital:       {backtest['equity'].iloc[0]:.2f}")
    print(f"Final Capital:         {backtest['equity'].iloc[-1]:.2f}")
    print(f"Total Return:          {total_return:.4%}")
    print(f"Sharpe Ratio:          {sharpe:.4f}")
    print(f"Sortino Ratio:         {sortino:.4f}")
    print(f"Maximum Drawdown:      {max_drawdown:.4%}")
    print(f"Number of Trades:      {number_of_trades}")

    backtest.to_csv("results/backtest.csv", index=False)   
    plt.figure(figsize=(10, 5))
    plt.plot(backtest["date"], backtest["equity"])
    plt.title("Pairs Trading Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig("results/equity_curve.png")
    plt.close()

    print("\nOUT-OF-SAMPLE VALIDATION")
    validation_result = chronological_validation(
        prices=prices,
        lookback=LOOKBACK,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
    )

    print(f"Selected Pair: {validation_result['selected_pair']}")
    print(f"Training Period: {validation_result['training_period']}")
    print(f"Test Period: {validation_result['test_period']}")
    print(f"Test Return: {validation_result['test_return']:.2%}")
    print(f"Test Sharpe: {validation_result['test_sharpe']:.4f}")
    print(f"Test Sortino: {validation_result['test_sortino']:.4f}")
    print(f"Test Max Drawdown: {validation_result['test_max_drawdown']:.2%}")

if __name__ == "__main__":
    main()
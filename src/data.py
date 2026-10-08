import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end):
    """Download and clean historical daily close prices."""

    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download market data: {exc}") from exc

    if data.empty:
        raise RuntimeError("No market data was downloaded.")

    # Handle yfinance MultiIndex output.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(1):
            prices = data.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("Could not find Close prices in yfinance output.")
    else:
        if "Close" not in data.columns:
            raise RuntimeError("Could not find Close prices in yfinance output.")
        prices = data[["Close"]].copy()

        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    # Remove columns with excessive missing values.
    min_valid = int(len(prices) * 0.9)
    prices = prices.dropna(axis=1, thresh=min_valid)

    # Fill only short gaps within the downloaded series.
    prices = prices.ffill()

    # Remove rows that still contain missing values.
    prices = prices.dropna()

    if prices.empty:
        raise RuntimeError("No clean price data remains after removing missing values.")

    return prices
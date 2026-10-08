from src.config import END_DATE, START_DATE
from src.data import download_prices


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

    print("\nDate range:")
    print(f"{prices.index.min().date()} → {prices.index.max().date()}")

    print(f"\nObservations: {len(prices)}")

    print("\nTickers:")
    for ticker in prices.columns:
        print(ticker)

    print("\nFirst 5 rows:")
    print(prices.head())


if __name__ == "__main__":
    main()
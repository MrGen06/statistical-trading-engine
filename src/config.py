import os

from dotenv import load_dotenv


load_dotenv()


START_DATE = os.getenv("START_DATE", "2016-01-01")
END_DATE = os.getenv("END_DATE", "2026-01-01")

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))

ENTRY_Z = float(os.getenv("ENTRY_Z", "2.0"))
EXIT_Z = float(os.getenv("EXIT_Z", "0.5"))

TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", "0.0005"))
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0005"))

LOOKBACK = int(os.getenv("LOOKBACK", "60"))
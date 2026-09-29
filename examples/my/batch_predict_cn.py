"""Update local A-share daily histories, then batch-run Kronos forecasts."""

import argparse
import re
import sys
import time
from datetime import datetime, time as datetime_time
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXAMPLES_DIR.parent
DATA_DIR = EXAMPLES_DIR / "data" / "CN"
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from prediction_cn_markets_day import PRED_LEN, create_predictor, predict_future

HISTORY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]
CHINA_TZ = ZoneInfo("Asia/Shanghai")
MARKET_CLOSE = datetime_time(15, 0)


def get_trade_calendar():
    dates = pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"])
    return pd.DatetimeIndex(dates).normalize().drop_duplicates().sort_values()


def get_as_of_trade_date(calendar, now=None):
    now = now or datetime.now(CHINA_TZ)
    today = pd.Timestamp(now.date())
    if now.time() >= MARKET_CLOSE:
        eligible = calendar[calendar <= today]
    else:
        eligible = calendar[calendar < today]
    if eligible.empty:
        raise ValueError(f"No completed A-share trading date found for {today.date()}")
    return eligible[-1]


def get_tencent_symbol(symbol):
    market_prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    return f"{market_prefix}{symbol}"


def normalize_history(df):
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }).copy()
    if "amount" not in df.columns and "turnover" in df.columns:
        df = df.rename(columns={"turnover": "amount"})

    missing_cols = [col for col in HISTORY_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Market data is missing columns: {missing_cols}")

    df = df[HISTORY_COLUMNS]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in HISTORY_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=HISTORY_COLUMNS).sort_values("date")
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_tencent_history(symbol, start_date, end_date):
    last_error = None
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=get_tencent_symbol(symbol),
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="",
            )
            if df is not None and not df.empty:
                return normalize_history(df)
            raise ValueError("Tencent returned no rows")
        except Exception as error:
            last_error = error
            print(f"⚠️ Tencent request {attempt}/3 failed for {symbol}: {error}")
            if attempt < 3:
                time.sleep(1.5)
    raise RuntimeError(f"Could not download history for {symbol}: {last_error}")


def update_local_history(symbol, as_of_date):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_DIR / f"{symbol}_daily.csv"

    if cache_file.is_file():
        cached = normalize_history(pd.read_csv(cache_file))
        last_cached_date = cached["date"].max()
        if last_cached_date >= as_of_date:
            print(f"✅ {symbol}: local history already reaches {last_cached_date.date()}; no download needed")
            return cache_file
        start_date = max(pd.Timestamp("1990-01-01"), last_cached_date - pd.Timedelta(days=7))
    else:
        cached = pd.DataFrame(columns=HISTORY_COLUMNS)
        start_date = pd.Timestamp("1990-01-01")

    new_data = fetch_tencent_history(symbol, start_date, as_of_date)
    if cached.empty:
        merged = new_data
    else:
        merged = normalize_history(pd.concat([cached, new_data], ignore_index=True))
    merged = merged[merged["date"] <= as_of_date]
    if merged.empty:
        raise ValueError(f"No historical rows for {symbol} through {as_of_date.date()}")

    merged.to_csv(cache_file, index=False)
    print(f"💾 {symbol}: saved {len(merged)} rows through {merged['date'].max().date()} to {cache_file}")
    return cache_file


def main(symbols):
    symbols = list(dict.fromkeys(symbols))
    for symbol in symbols:
        if not re.fullmatch(r"\d{6}", symbol):
            raise ValueError(f"Invalid A-share symbol {symbol!r}; expected six digits")

    calendar = get_trade_calendar()
    as_of_date = get_as_of_trade_date(calendar)
    future_dates = calendar[calendar > as_of_date][:PRED_LEN]
    if len(future_dates) < PRED_LEN:
        print(
            f"⚠️ Calendar only provides {len(future_dates)} future trading dates; "
            f"forecasting {len(future_dates)} instead of {PRED_LEN}"
        )
    if future_dates.empty:
        raise ValueError("The available trading calendar contains no future dates")

    print(f"📅 Completed through {as_of_date.date()}; forecasting {len(future_dates)} exchange trading dates")
    ready = []
    for symbol in symbols:
        try:
            cache_file = update_local_history(symbol, as_of_date)
            ready.append((symbol, cache_file))
        except Exception as error:
            print(f"❌ Skipping {symbol}: {error}")

    if not ready:
        raise RuntimeError("No symbols have usable local history")

    predictor = create_predictor()
    for symbol, cache_file in ready:
        try:
            predict_future(
                symbol=symbol,
                data_file=cache_file,
                as_of_date=as_of_date,
                y_timestamp=future_dates,
                predictor=predictor,
            )
        except Exception as error:
            print(f"❌ Prediction failed for {symbol}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True, help="Six-digit A-share codes")
    args = parser.parse_args()
    main(args.symbols)
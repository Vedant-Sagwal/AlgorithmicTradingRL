#using data downloaded from yfinance

from datetime import datetime
import pandas as pd
import os
import yfinance as yf
from scipy.stats import skew, kurtosis
import numpy as np

#constants
START_DATE = "2017-1-1"
END_DATE = "2020-12-31"
INTERVAL = "1d"
DATA_DIR = "./data"
TARGET = 'TSLA'
RATES_INDEX = "^FVX"        # 5 Year Treasury Note Yield
VOLATILITY_INDEX = "^VIX"   # CBOE Volatility Index
SMALLCAP_INDEX = "^RUT"     # Russell 2000 Index
GOLD_FUTURES = "GC=F"         # Gold futures
OIL_FUTURES = "CL=F"        # Crude Oil Futures
MARKET = "^SPX"             # S&P 500 Index
TICKER_SYMBOLS = [TARGET, RATES_INDEX, VOLATILITY_INDEX, SMALLCAP_INDEX, GOLD_FUTURES, MARKET, OIL_FUTURES]

def get_ticker_data(tickerList, start=START_DATE, end=END_DATE,interval =INTERVAL, data_dir = DATA_DIR):
    tickers = {}
    new_start = datetime.strptime(start,"%Y-%m-%d")
    new_end = datetime.strptime(end, "%Y-%m-%d")
    os.makedirs(data_dir,exist_ok=True)
    for ticker in tickerList:
        cached_file_path = f"{data_dir}/{ticker}--{start}--{end}--{interval}.csv"
        try:
            if os.path.exists(cached_file_path):
                df = pd.read_parquet(cached_file_path)
                df.index=pd.to_datetime(df.index)
                assert len(df) > 0
            else:
                df = yf.download(ticker, start=START_DATE, end = END_DATE, interval = INTERVAL)
                assert len(df) > 0
                df.to_parquet(cached_file_path, index=True,compression="snappy")
            min_date = df.index.min()
            max_date = df.index.max()
            nan_count = df["Close"].isnull().sum()
            skewness = np.round(skew(df["Close"].dropna()), 2)
            kurt = np.round(kurtosis(df["Close"].dropna()), 2)
            outliers_count = (df["Close"] > df["Close"].mean() + (3 * df["Close"].std())).sum()
            print(
                f"{ticker} => min_date: {min_date}, max_date: {max_date}, kurt:{kurt}, skewness:{skewness}, outliers_count:{outliers_count},  nan_count: {nan_count}"
            )
            tickers[ticker] = df
            if min_date > new_start:
                new_start = min_date
            if max_date < new_end:
                new_end = max_date
        except Exception as e:
            print(f"Error with ticker {ticker} : {e}")
    return tickers, new_start, new_end

tickers, latest_start, earliest_end = get_ticker_data(TICKER_SYMBOLS)
stock_df = tickers[TARGET].copy()
print(stock_df.tail(5))
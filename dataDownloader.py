#using data downloaded from yfinance

from datetime import datetime
import pandas as pd
import os
import yfinance as yf
from scipy.stats import skew, kurtosis
import numpy as np
from tradingEnv import TradingEnv
from gymnasium.utils.env_checker import check_env

from ta.trend import MACD,EMAIndicator
from ta.volatility import AverageTrueRange

from params import ACT_LONG, ACT_HOLD, ACT_SHORT



from params import START_DATE,SPLIT_DATE,END_DATE,INDEX,TARGET,RATES_INDEX,VOLATILITY_INDEX,SMALLCAP_INDEX,GOLD_FUTURES,OIL_FUTURES,MARKET,TICKER_SYMBOLS,INTERVAL,TRADING_DAYS_YEAR
from params import MODELS_PATH,LOGS_PATH,AGENT_HISTORY,DATA_DIR

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
# print(stock_df.tail(5))


macd = MACD(close=stock_df["Close"].iloc[:,0], window_slow=26,window_fast=13,window_sign=9,fillna=True)
stock_df["MACD"] = macd.macd()
stock_df["MACD_HIST"] = macd.macd_diff()
stock_df["MACD_SIG"] = macd.macd_signal()

atr = AverageTrueRange(stock_df["High"].iloc[:,0], stock_df["Low"].iloc[:,0], stock_df["Close"].iloc[:,0], window=14,fillna=True)
stock_df["ATR"] = atr.average_true_range()

ema = EMAIndicator(stock_df["Close"].iloc[:,0], window=12,fillna=True)
stock_df["EMA_SHORT"] = ema.ema_indicator()

ema = EMAIndicator(stock_df["Close"].iloc[:,0], window=26,fillna=True)
stock_df["EMA_MID"] = ema.ema_indicator()

ema = EMAIndicator(stock_df["Close"].iloc[:,0], window=200,fillna=True)
stock_df["EMA_LONG"] = ema.ema_indicator()

stock_df["VOLATILITY_INDEX"] = tickers[VOLATILITY_INDEX]["Close"].pct_change().fillna(0)
stock_df["SMALLCAP_INDEX"] = tickers[SMALLCAP_INDEX]["Close"].pct_change().fillna(0)
stock_df["RATES_INDEX"] = tickers[RATES_INDEX]["Close"].pct_change().fillna(0)
stock_df["GOLD_FUTURES"] = tickers[GOLD_FUTURES]["Close"].pct_change().fillna(0)
stock_df["OIL_FUTURES"] = tickers[OIL_FUTURES]["Close"].pct_change().fillna(0)
stock_df["MARKET"] = tickers[MARKET]["Close"].pct_change().fillna(0)

# print(stock_df.tail(5))
# print(stock_df[("ATR")])
# print(stock_df.columns)
train_data = stock_df[stock_df.index < pd.to_datetime(SPLIT_DATE)].copy()
test_data = stock_df[stock_df.index >= pd.to_datetime(SPLIT_DATE)].copy()

train_env = TradingEnv(train_data)
test_env = TradingEnv(test_data)


print(f"Action Space: {train_env.action_space}")
print(f"Observation Space: {train_env.observation_space}")
print(f"Observation Shape: {train_env.observation_space.shape}")
print(f"Action Count: {train_env.action_space.n}")



def execute_action_and_print_state(env, action):
    obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.int32))
    print(f'Action taken: {action} at step: {env.current_step}')
    print(f'Next state : ', obs.shape)
    print(f'New balance: {env.balance}')
    print(f'Total shares: {env.total_shares}')
    print(f'Reward: {reward}\n')

time_step = train_env.reset()

# Some dryruns to validate our env logic: Buy, Sell, we should have a positive balance with TSLA
print("------------------------------------------------------------------------------------------------")
execute_action_and_print_state(train_env, ACT_HOLD)
execute_action_and_print_state(train_env, ACT_LONG)
execute_action_and_print_state(train_env, ACT_SHORT)
execute_action_and_print_state(train_env, ACT_HOLD)
execute_action_and_print_state(train_env, ACT_LONG)
print("------------------------------------------------------------------------------------------------")
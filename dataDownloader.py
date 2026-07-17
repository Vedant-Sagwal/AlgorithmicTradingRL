#using data downloaded from yfinance

from datetime import datetime
import pandas as pd
import os

#constants
START_DATE = ""
END_DATE = ""
INTERVAL = "1d"
DATA_DIR = "./data"

def get_ticker_data(tickerList, start=START_DATE, end=END_DATE,interval =INTERVAL, data_dir = DATA_DIR):
    tickers = {}
    new_start = datetime.strptime(start,"%Y-%m-%d")
    new_end = datetime.strptime(end, "%Y-%m-%d")
    os.mkdirs(data_dir,exist_ok=True)

get_ticker_data([])
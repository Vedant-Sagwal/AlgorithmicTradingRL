import os


# Paths
MODELS_PATH = './models'
LOGS_PATH = './logs'
AGENT_HISTORY = f"{LOGS_PATH}/history"
DATA_DIR = "./data"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(AGENT_HISTORY, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Indices and tickers
START_DATE = "2016-01-01"   # Training dates
SPLIT_DATE = '2018-12-31'   # Testing cutoff
END_DATE = "2019-12-31"     # Testing dates
INDEX = "Date"
TARGET = 'TSLA'
RATES_INDEX = "^FVX"        # 5 Year Treasury Note Yield
VOLATILITY_INDEX = "^VIX"   # CBOE Volatility Index
SMALLCAP_INDEX = "^RUT"     # Russell 2000 Index
GOLD_FUTURES = "GC=F"       # Gold futures
OIL_FUTURES = "CL=F"        # Crude Oil Futures
MARKET = "^SPX"             # S&P 500 Index
TICKER_SYMBOLS = [TARGET, RATES_INDEX, VOLATILITY_INDEX, SMALLCAP_INDEX, GOLD_FUTURES, MARKET, OIL_FUTURES]
INTERVAL = "1d"
TRADING_DAYS_YEAR = 252     # Trading days in a year
RISK_FREE_RATE = 0.021      # Average riskfree 

# Trading Params
ACT_SHORT = 0
ACT_LONG = 1
ACT_HOLD = 2
ACTIONS = [ACT_SHORT, ACT_LONG]
CAPITAL = 100000
TRADE_COSTS_PERCENT = 10 / 100 / 100  # 10 basis points costs

# Hyperparameters
BATCH_SIZE = 64
LEARN_RATE = 1e-3           # The networks learning rate (min 1e-5)
TRAIN_EPISODES = 500        # Number of episodes to train the networks (max 1000)
COLLECT_SIZE = 1000         # Default memory buffer, should default to observation state size
STATE_LEN = 30              # How much historic timesteps to return from the environment
LOG_INTERVALS = 20          # Log every X iterations
TEST_INTERVALS = 100        # Every X iterations, validate
TARGET_UPDATE_ITERS = 20    # Update the target networks
VALIDATION_ITERS = 10       # Repeat the experiment N times for significant metrics
DISCOUNT = 0.4              # The gamma in the bellman, how much to discount past rewards
EPSILON_START = 1.0         # For epsilon greedy algos, how much to explore VS exploit
EPSILON_END = 0.01          # At the end there should be more exploitation
EPSILON_DECAY = 10000       # Used for anealing the epsilon
GRAD_CLIP = 1.0             # Stop exploding/disappearing gradients
REWARD_CLIP = 1.0           # Stop exploding/disappearing rewards
DROPOUT = 0.2               # Chance of tensor dropout
L2FACTOR = 1e-5             # Regularization wieghts
ADAM_WEIGHTS = 1e-5         # Weighted Optimizer
NEURONS = 512               # Network capacity
LAYERS = (NEURONS, NEURONS, NEURONS, NEURONS, NEURONS)



MACRO_FEATURES = [RATES_INDEX, VOLATILITY_INDEX, MARKET, GOLD_FUTURES, OIL_FUTURES]
TECH_INDICATORS = ['MACD', 'MACD_HIST', 'MACD_SIG', 'ATR', 'EMA_SHORT', 'EMA_MID', 'EMA_LONG']
OHLCV_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
FEATURES = ["Price Returns", "Price Delta", "Close Position", "Volume"]
TARGET_FEATURES = "Price Raw"

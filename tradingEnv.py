import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import math
from params import RISK_FREE_RATE,TRADING_DAYS_YEAR, ACT_HOLD, ACT_LONG, ACT_SHORT, DISCOUNT, FEATURES, CAPITAL, STATE_LEN, TARGET, TARGET_FEATURES, TRADE_COSTS_PERCENT, REWARD_CLIP
class TradingEnv(gym.Env):
    def __init__(self,data, features=FEATURES, money = CAPITAL, state_length = STATE_LEN, transaction_cost= 0 , market_costs = TRADE_COSTS_PERCENT,reward_discount=DISCOUNT):
        super().__init__()
        assert data is not None

        self.features = features
        self.data_dim = len(self.features)
        self.balance = money
        self.state_length = state_length
        self.transaction_cost = transaction_cost
        self.current_step = state_length
        self.reward_discount = reward_discount
        self.initial_balance = money
        self.epsilon = max(market_costs, np.finfo(float).eps)
        self.total_shares = 0
        self._episode_ended = False
        self._batch_size=1
        num_actions = ACT_LONG - ACT_SHORT + 1
        self.action_space = spaces.Discrete(num_actions, start = ACT_SHORT)
        self.observation_space = spaces.Box(low= -np.inf, high = np.inf, shape = (self.state_length * self.data_dim, ),dtype=np.float32)
        self.data = self.preprocess_data(data.copy())
        self.reset()

    def flatten_columns(self, data):
        if not isinstance(data.columns, pd.MultiIndex):
            return data
        flat = {}
        for col in data.columns:
            if isinstance(col, tuple):
                flat[col[0]] = data[col]
            else:
                flat[col] = data[col]
        return pd.DataFrame(flat, index=data.index)

    def cell(self, step, col):
        val = self.data.iloc[step][col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)

    def preprocess_data(self, data):
        data = self.flatten_columns(data)
        price_raw = data["Close"].copy()
        data["Price Returns"] = data["Close"].pct_change().fillna(0)
        data["Price Delta"] = data["High"] - data["Low"]
        data["Close Position"] = abs(data["Close"] - data["Low"]) / data["Price Delta"].replace(0, 0.5)
        for col in self.features[:-1]:
            data_min = data[col].min()
            data_max = data[col].max()
            if data_min != data_max:
                data[col] = (data[col] - data_min) / (data_max - data_min)
            else:
                data[col] = 0
        data_min = data["Volume"].min()
        data_max = data["Volume"].max()
        if data_min != data_max:
            data["Volume"] = (data["Volume"] - data_min) / (data_max - data_min)
        else:
            data["Volume"] = 0
        data[TARGET_FEATURES] = price_raw
        data["Sharpe"] = 0
        data["Position"] = 0
        data["Action"] = ACT_HOLD
        data["Holdings"] = 0.0
        data["Cash"] = float(self.balance)
        data["Money"] = data["Holdings"] + data["Cash"]
        data["Reward"] = 0.0
        return data
    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.state_length
        self._episode_ended = False
        self.total_shares = 0

        self.data["Sharpe"] = 0.0
        self.data["Reward"] = 0.0
        self.data["Action"] = ACT_HOLD
        self.data["Position"] = 0
        self.data["Cash"] = float(self.balance)
        self.data["Holdings"] = 0.0
        self.data["Money"] = self.data["Cash"] + self.data["Holdings"]
        self.data["Returns"] = 0.0
        obs = self._next_observation()
        info = {}
        return obs, info
    def _next_observation(self):
        start_idx = max(0, self.current_step - self.state_length + 1)
        end_idx = self.current_step + 1
        obs = self.data[self.features].iloc[start_idx:end_idx]
        obs_values = obs.values.flatten().astype(np.float32)
        return obs_values
    def step(self,action):
        if self._episode_ended:
            return self.reset()
        self.current_step += 1
        current_price = self.cell(self.current_step, TARGET_FEATURES)

        if action == ACT_HOLD:
            self.hold_position()
        elif action == ACT_SHORT:
            prev_price = self.cell(self.current_step - 1, TARGET_FEATURES)
            self.short_position(current_price, prev_price)
        elif action == ACT_LONG:
            self.long_position(current_price)
        else:
            raise Exception("Invalid action!!")
        self.financial_updates()
        terminated = True if self.current_step >= len(self.data) - 1 else False
        truncated = False
        reward = self.calc_sharpe_reward_signal()
        self.data.at[self.data.index[self.current_step], "Reward"] = reward
        obs = self._next_observation()
        if terminated or truncated:
            self._episode_ended = True
        info = {
            "step":self.current_step,
            "balance" : self.balance,
            "holdings" : self.total_shares
        }
        return obs, float(reward), terminated, truncated, info
    def get_lower_bound(self, cash, total_shares, price):
        delta = -cash - (total_shares * price * (1 + self.epsilon) * (1 + self.transaction_cost))
        if delta >=0:
            lowerBound= delta / (price * ((self.epsilon) * (1+ self.transaction_cost)))
        else:
            lowerBound = delta / ((price) * ((2 * self.transaction_cost) + (self.epsilon * (1 + self.transaction_cost))))
        if np.isinf(lowerBound):
            assert False
        return lowerBound
    def hold_position(self):
        index = self.data.index[self.current_step]
        self.data.at[index, "Position"] = 0
        self.data.at[index, "Action"] = ACT_HOLD
        self.data.at[index, "Cash"] = self.cell(self.current_step - 1, "Cash")
        self.data.at[index, "Holdings"] = self.cell(self.current_step - 1, "Holdings")
    def long_position(self, curr_price):
        index = self.data.index[self.current_step]
        prev_step = self.current_step - 1
        prev_position = self.cell(prev_step, "Position")
        prev_cash = self.cell(prev_step, "Cash")
        self.data.at[index, "Action"] = ACT_LONG
        self.data.at[index, "Position"] = 1
        if prev_position == 1:
            #more long
            self.data.at[index, "Cash"] = prev_cash
            self.data.at[index, "Holdings"] = self.total_shares * curr_price
            self.data.at[index, "Action"] = ACT_HOLD
        elif prev_position == 0:
            #new long
            self.total_shares = math.floor(prev_cash / (curr_price * (1 + self.transaction_cost)))
            self.data.at[index, "Cash"] = prev_cash - (self.total_shares * curr_price * (1 + self.transaction_cost))
            self.data.at[index, "Holdings"] = self.total_shares * curr_price
        else:
            #short to long
            cash_after_close = prev_cash - (self.total_shares * curr_price * (1 + self.transaction_cost))
            self.data.at[index, "Cash"] = cash_after_close
            self.total_shares = math.floor(cash_after_close / (curr_price * (1 + self.transaction_cost)))
            self.data.at[index, "Cash"] = cash_after_close - (self.total_shares * curr_price * (1 + self.transaction_cost))
            self.data.at[index, "Holdings"] = self.total_shares * curr_price
    def short_position(self, current_price, prev_price):
        step_idx = self.data.index[self.current_step]
        prev_step = self.current_step - 1
        prev_position = self.cell(prev_step, "Position")
        prev_cash = self.cell(prev_step, "Cash")
        self.data.at[step_idx, 'Position'] = -1
        self.data.at[step_idx, "Action"] = ACT_SHORT
        if prev_position == -1:
            # Short more
            low = self.get_lower_bound(prev_cash, -self.total_shares, prev_price)
            if low <= 0:
                self.data.at[step_idx, 'Cash'] = prev_cash
                self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
                self.data.at[step_idx, "Action"] = ACT_HOLD
            else:
                total_sharesToBuy = min(math.floor(low), self.total_shares)
                self.total_shares -= total_sharesToBuy
                self.data.at[step_idx, 'Cash'] = prev_cash - total_sharesToBuy * current_price * (1 + self.transaction_cost)
                self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
        elif prev_position == 0:
            # new short
            self.total_shares = math.floor(prev_cash / (current_price * (1 + self.transaction_cost)))
            self.data.at[step_idx, 'Cash'] = prev_cash + self.total_shares * current_price * (1 - self.transaction_cost)
            self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
        else:
            # long to short
            cash_after_close = prev_cash + self.total_shares * current_price * (1 - self.transaction_cost)
            self.data.at[step_idx, 'Cash'] = cash_after_close
            self.total_shares = math.floor(cash_after_close / (current_price * (1 + self.transaction_cost)))
            self.data.at[step_idx, 'Cash'] = cash_after_close + self.total_shares * current_price * (1 - self.transaction_cost)
            self.data.at[step_idx, 'Holdings'] = -self.total_shares * current_price
    def financial_updates(self):
        index = self.data.index[self.current_step]
        cash = self.cell(self.current_step, "Cash")
        holdings = self.cell(self.current_step, "Holdings")
        prev_money = self.cell(self.current_step - 1, "Money")
        money = holdings + cash
        self.balance = cash
        self.data.at[index, 'Money'] = money
        self.data.at[index, 'Returns'] = (money - prev_money) / prev_money if prev_money else 0.0
    def calculate_reward_signal(self, reward_clip=REWARD_CLIP):
        reward = self.cell(self.current_step, 'Returns')
        return np.clip(reward, -reward_clip, reward_clip)
    def calc_sharpe_reward_signal(self, risk_free_rate =RISK_FREE_RATE, trading_periods_per_year = TRADING_DAYS_YEAR,reward_clip =REWARD_CLIP):
        observed_returns = self.data['Returns'].iloc[:self.current_step + 1]
        period_risk_free_rate = risk_free_rate / trading_periods_per_year
        excess_returns = observed_returns - period_risk_free_rate

        rets = np.mean(excess_returns)
        std_rets = np.std(excess_returns)

        sr = rets / std_rets if std_rets > 0 else 0
        annual_sr = sr * np.sqrt(trading_periods_per_year)

        self.data.at[self.data.index[self.current_step], 'Sharpe'] = annual_sr

        return np.clip(annual_sr, -reward_clip, reward_clip)
    def get_trade_data(self):
        self.data['cReturns'] = np.cumprod(1 + self.data['Returns']) - 1
        return self.data.iloc[:self.current_step + 1]
    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.total_shares}')

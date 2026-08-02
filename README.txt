so we are building algo rl model. so first lets go and decide obs space, getting data 
through yfinance,

for obs space using 1. vix(cboe) 2. interest rate index(5 Year Treasury Note Yield) 
3. small cap index(russel 2000) 4. gold futures 5. crude oil futures 6. index(S&P index)

based on above obs space action will be calculated and reward will be calculated and 
model will learn on these params.

download data from yfinance and also include some macro factors(market factors) and 
stock indicators like MACD, ATR, EMA

so instead of only relying on ohlcv data for the obs space, we extend the obs space to many
such factors, which help in making better decisions.

after downloading data from yfinance, next step is defining the env

here we are using dqn(deep q network), see our goal is to maximize rewards, so we judge our
actions on basis of some value which is q value

their are 2 types of problem solving techniques in rl, 1. policy based, 2. value based, dqn is 
value based, policy based optimization cannot be applied easily to this problem, because
1. experience replay buffer 2. too expensive to apply again and again on env which is too expensive, 
what i mean to say is if again and again interacting with env is difficult then use value based
methods in rl. 3. in finance data so much noise is present, so in policy optimization due to noise
the method can suffer from high variance 4. our dataset is descrete so using dqn is preferred
over policy methods as they are much suitable for continous action space.

builing tradingENV
-> params init will take (data : containing the stock market data), (data_dim: which 
indicated dimension in which data to be used for each observation) ,(money :total cash from which stocks
will be bought), (transaction_cost:costs associated with trading actions), (state_length:
length upto which past states are to be considered)

shape of obs space is {macd, ema, atr, c,v, volatility index, small cap index,
spx index,gold index, oil index, price change, startegy}
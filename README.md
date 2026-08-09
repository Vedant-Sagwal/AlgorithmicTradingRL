# Algorithmic Trading Model Based on Deep Reinforcement Learning

### This project uses an innovative approach based on deep reinforcement learning (DRL) to solve the algorithmic trading problem of determining the optimal trading position at any point in time during a trading activity in the stock market. 

### Custom Trading Environment is created , which also takes into consideration various market factors(long term, short term index) and also various indicators are used to get better results from the model like MACD(Moving Average Convergence Divergence), EMA(Exponential Moving Average). This env is created using OpenAI Gymnasium.



### Double Deep Q-Network is the algorithm used in the project to train the RL environment. The policy is implemented from scratch and includes features like experience replay buffer, target and online q-networks etc.


### Objective Function used here is sharpe ratio which measures the excess return of an investment per unit of risk, calculated as portfolio return minus the risk-free rate, divided by standard deviation. 
### And it is used in industrial models also.

### In order to objectively assess the performance of trading strategies, the project also implements a novel, more rigorous performance assessment methodology.

### And The Model Performs better than buy-and hold and trend following strategies.

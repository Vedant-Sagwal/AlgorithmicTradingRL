from params import ADAM_WEIGHTS, LEARN_RATE
from tradingSim import TradingSimulator
from dataDownloader import train_env, test_env, train_data
from ddqn import DqnAgent, q_network
import torch.optim as optim

if __name__ == "__main__":
    print("Inside Main!!")
    q_net = q_network(train_env.observation_space.shape[0], train_env.action_space.n)
    t_q_net = q_network(train_env.observation_space.shape[0], train_env.action_space.n)
    optim = optim.AdamW(q_net.parameters(), lr=LEARN_RATE, weight_decay=ADAM_WEIGHTS)
    agent = DqnAgent(train_env.observation_space, train_env.action_space, q_net, t_q_net, optim)
    sim = TradingSimulator(train_env, agent, test_env, collect_steps=len(train_data))
    rewards, losses, q_values, t_q_values = sim.train()
    sim.plot_performance(rewards, losses, q_values, t_q_values)
from syslog import LOG_INFO
from ddqn import ReplayBuffer
from params import BATCH_SIZE, COLLECT_SIZE, LOG_INTERVALS, TEST_INTERVALS, TRAIN_EPISODES


class TradingSimulator():
    def __init__(self, env, agent, eval_env,episodes = TRAIN_EPISODES, batch_size=BATCH_SIZE, collect_steps = COLLECT_SIZE, log_interval = LOG_INTERVALS, eval_interval = TEST_INTERVALS, replay_capacity=None):
        self.env=env
        self.eval_env= eval_env
        self.agent = agent
        self.episodes = episodes
        self.batch_size = batch_size
        self.collect_steps = collect_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.replay_capacity = replay_capacity
        #initialize replay buffer
        self.replayBuffer = ReplayBuffer(self.replay_capacity)


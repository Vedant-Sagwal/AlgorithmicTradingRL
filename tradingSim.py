from syslog import LOG_INFO
from ddqn import ReplayBuffer
from params import ACT_LONG, ACT_SHORT, BATCH_SIZE, COLLECT_SIZE, EPSILON_DECAY, EPSILON_END, EPSILON_START, LOG_INTERVALS, LOGS_PATH, MODELS_PATH, TEST_INTERVALS, TRAIN_EPISODES
import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
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
        self.replay_capacity = replay_capacity or int(self.collect_steps *1.5)
        #initialize replay buffer
        self.replayBuffer = ReplayBuffer(self.replay_capacity)
        self.global_steps = 0
    def clear_model_directories(self, directories=[MODELS_PATH,LOGS_PATH]):
        try:
            for root, dirs, files in os.walk(directories, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    if not file_path.endswith('.zip'):
                        os.remove(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    shutil.rmtree(dir_path)
            print(f"Cleared all temporary files and directories in {directories}")
        except Exception as e:
            print(f"Error clearing directories: {e}")
    def get_q_values(self, state):
        self.agent.q_net.eval()
        self.agent.t_q_net.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.q_net(state_t).numpy()
            target_q_values = self.agent.t_q_net(state_t).numpy()
        return q_values, target_q_values
    def train(self, checkpoint_path=MODELS_PATH, initial_epsilon = EPSILON_START, end_epsilon = EPSILON_END, decay_rate = EPSILON_DECAY, only_purge_buffer = False):
        # 1. bringing checkpointed points
        checkpoint_dir = os.path.join(checkpoint_path, 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok = True)
        start_episode = 0
        if checkpoint_path and os.path.exists(checkpoint_dir):
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
            if ckpt_files:
                ckpt_path = os.path.join(checkpoint_dir, sorted(ckpt_files)[-1])
                checkpoint = torch.load(ckpt_path)
                self.agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
                self.agent.t_q_net.load_state_dict(checkpoint["t_q_net_state_dict"]) 
                start_episode = checkpoint["episode"] + 1
                self.agent.optimizer.load_state_dict(checkpoint["optim_state_dict"])
                self.global_steps = checkpoint.get("global_steps", 0)
                self.agent.epsilon = checkpoint.get("epsilon", initial_epsilon)
                print(f"Loaded checkpoint from {ckpt_path}, resuming at episode {start_episode}")
        # 2. start training 
        q_values_history = []
        t_q_values_history = [] 
        rewards_history = []
        losses_history = []
        episode = start_episode
        state, _ = self.env.reset()
        while episode < self.episodes:
            for _ in range(self.collect_steps):
                #get the action from current state to do
                action = self.agent.act(state)
                #take the action and get next state and reward
                next_state,reward, terminated, truncated,info = self.env.step(action)
                #push this to the replay buffer
                self.replayBuffer.push((state, action,reward,next_state,terminated))
                state = next_state if not terminated and not truncated else self.env.reset()[0]
                if len(self.replayBuffer) >= self.batch_size:
                    batch = self.replayBuffer.sample(self.batch_size)
                    loss = self.agent.learn(batch)
                    self.global_steps += 1
                    if self.log_interval and self.global_steps % self.log_interval == 0:
                        print(f'Step = {self.global_steps}: Loss = {loss:.6f}')
                        a,b = self.get_q_values(state)
                        q_values_history.append(a)
                        t_q_values_history.append(b)
                        losses_history.append(loss)
                        rewards_history.append(reward)
                    self.agent.decay_epsilon(end_epsilon,decay_rate,self.global_steps)
            #evaluation
            if (self.eval_interval and (self.global_steps % self.eval_interval == 0 or episode == self.episodes - 1)):
                total_returns, avg_returns = self.eval_metrics()
                rewards_history.append(np.mean(avg_returns))
                print(f'Step = {self.global_steps} | Avg Reward = {np.mean(avg_returns)} | Total = {total_returns[-1]}')
                # Save checkpoint
                if checkpoint_path:
                    torch.save({
                        'episode': episode,
                        'global_steps': self.global_steps,
                        'q_net_state_dict': self.agent.q_net.state_dict(),
                        't_q_net_state_dict': self.agent.t_q_net.state_dict(),
                        'optim_state_dict': self.agent.optim.state_dict(),
                        'epsilon': self.agent.epsilon,
                    }, os.path.join(checkpoint_dir, f'checkpoint_{self.global_steps}.pt'))
                    # Keep only latest 1 checkpoint
                    for f in sorted(os.listdir(checkpoint_dir))[:-1]:
                        os.remove(os.path.join(checkpoint_dir, f))

            episode += 1
            print("=========================================Episode : " , episode, "============================================")
        # Save final policy
        if checkpoint_path:
            policy_dir = os.path.join(checkpoint_path, 'policy')
            os.makedirs(policy_dir, exist_ok=True)
            torch.save(self.agent.q_net.state_dict(), os.path.join(policy_dir, 'q_net.pth'))
            print("Training complete and policy saved.")
            self.zip_directories(checkpoint_path)
            self.clear_model_directories(checkpoint_path)

        return rewards_history, losses_history, q_values_history, t_q_values_history

    def eval_metrics(self):
        total_returns = []
        avg_returns = []
        episode_returns = []
        state, _ = self.eval_env.reset()
        done = False
        while not done:
            action = self.agent.act(state)
            next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
            state = next_state
            episode_returns.append(reward)
            if terminated or truncated:
                total_return = np.sum(episode_returns)
                avg_return = np.mean(episode_returns)
                total_returns.append(total_return)
                avg_returns.append(avg_return)
                state, _ = self.eval_env.reset()
                episode_returns = []
                done = True
        return np.array(total_returns), np.array(avg_returns)
    
    def zip_directories(self, directories, output_filename=f'{MODELS_PATH}/model_files'):
        archive_path = shutil.make_archive(output_filename, 'zip', root_dir='.', base_dir=directories)
        print(f"Archived {directories} into {archive_path}")
    def load_and_eval_policy(self, policy_path):
        policy_dir = os.path.join(policy_path, 'policy')
        q_net_path = os.path.join(policy_dir, 'q_net.pth')
        if os.path.exists(q_net_path):
            self.agent.q_net.load_state_dict(torch.load(q_net_path))
            self.agent.target_q_net.load_state_dict(torch.load(q_net_path))
            print("Loaded policy from", q_net_path)
        else:
            # try checkpoint
            checkpoint_dir = os.path.join(policy_path, 'checkpoint')
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if ckpt_files:
                ckpt_path = os.path.join(checkpoint_dir, sorted(ckpt_files)[-1])
                checkpoint = torch.load(ckpt_path)
                self.agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
                self.agent.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
                print("Loaded checkpoint from", ckpt_path)
        total_rewards, avg_return = self.eval_metrics()
        print(f'Average Return = {np.mean(avg_return)}, Total Return = {np.mean(total_rewards)}')
        return self.agent, total_rewards, avg_return

    def plot_performance(self, average_rewards, losses, q_values, target_q_values):
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
        episodes_index = np.arange(len(average_rewards)) * self.eval_interval
        q_values_flat = np.array([np.mean(qv) if qv.ndim > 1 else qv for qv in q_values]).flatten()
        target_q_values_flat = np.array([np.mean(tqv) if tqv.ndim > 1 else tqv for tqv in target_q_values]).flatten()

        axs[0].plot(episodes_index, average_rewards, label='Average Rewards', color="yellow")
        axs[0].set_xlabel('Episodes')
        axs[0].set_ylabel('Rewards')
        axs[0].legend()
        axs[0].set_title('Average Rewards over Iterations')

        axs[1].plot(episodes_index, losses, label='Loss', color="red")
        axs[1].set_xlabel('Episodes')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].set_title('Loss over Iterations')

        min_q_len = min(len(q_values_flat), len(target_q_values_flat))
        q_episodes_index = np.arange(min_q_len) * self.log_interval
        axs[2].plot(q_episodes_index, q_values_flat[:min_q_len], label='Online Q-Values', color="green")
        axs[2].plot(q_episodes_index, target_q_values_flat[:min_q_len], label='Target Q-Values', color="blue")
        axs[2].set_xlabel('Episodes')
        axs[2].set_ylabel('Q-Values')
        axs[2].legend()
        axs[2].set_title('Networks Q-Values over Iterations')

        fig.suptitle('DDQN Performance', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    def plot_eval_trades(self, storage_dir=LOGS_PATH, file_name='backtest'):
        # Assumes eval_env has get_trade_data() method
        trades_df = self.eval_env.get_trade_data()
        assert len(trades_df) > 1, "No trades in evaluation environment. You need to call eval_metrics."

        print(f"Cumulative Return from the strategy: {trades_df['cReturns'].iloc[-1]*100.:.02f}%")
        buy_signals = trades_df[trades_df['Action'] == ACT_LONG]
        sell_signals = trades_df[trades_df['Action'] == ACT_SHORT]

        _, axes = plt.subplots(3, 1, figsize=(18, 11), gridspec_kw={'height_ratios': [4, 2, 2]})

        axes[0].plot(trades_df['Close'], label='Close', color='blue', alpha=0.6, linestyle='--')
        axes[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', label='Buy')
        axes[0].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', label='Sell')
        axes[0].set_title('Close Price and Signals')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(trades_df['cReturns'], label='Cumulative rets', color='purple')
        axes[1].set_title('Cumulative Returns')
        axes[1].set_ylabel('Cumulative rets')
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(trades_df['Reward'], label='Rewards', color='green')
        axes[2].set_title('Rewards or Penalties')
        axes[2].set_ylabel('Reward')
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            file_path = os.path.join(storage_dir, f'{file_name}.png')
            plt.savefig(file_path)
        plt.show()
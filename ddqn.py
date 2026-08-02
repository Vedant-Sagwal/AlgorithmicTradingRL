from collections import deque
import torch.nn as nn
import random
import numpy as np
import torch

from params import COLLECT_SIZE, DISCOUNT, DROPOUT, EPSILON_DECAY, EPSILON_START, GRAD_CLIP, L2FACTOR, LAYERS, TARGET_UPDATE_ITERS

class q_network(nn.Module):
    def __init__(self,numInputs, numOutputs, fc_layers_params = LAYERS, dropout=DROPOUT, lr_reg = L2FACTOR) -> None:
        super.__init__()
        layers = []
        in_features = numInputs
        for num_units in fc_layers_params:
            layers.append(nn.Linear(in_features, num_units))
            layers.append(nn.BatchNorm1d(num_units))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            in_features = num_units
        #output layer 
        layers.append(nn.Linear(in_features, numOutputs))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)



class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=COLLECT_SIZE)
    def push(self, x):
        self.buffer.append(x)
    def pop(self):
        return self.buffer.popleft()
    def sample(self,batch_size):
        #returns a mini batch of batch_size
        batch = random.sample(self.buffer, batch_size)
        state, action ,reward,next_obs,done = map(np.stack, (zip(*batch)))
        print("state shape : ", state.shape)
        print("reward shape : ", reward.shape)
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(done).unsqueeze(1)
        )
    def __len__(self):
        return len(self.buffer)

#dqn agent

class DqnAgent():
    def __init__(self, env,state_dim , action_dim,q_net, t_q_net, optimizer, epsilon=EPSILON_START,discount=DISCOUNT, target_update_iters = TARGET_UPDATE_ITERS, gradient_clipping = GRAD_CLIP):
        self.q_net = q_net
        self.t_q_net = t_q_net
        self.optim = optimizer
        self.epsilon = epsilon
        self.target_update_iters = target_update_iters
        self.gradient_clipping = gradient_clipping
        self.train_step = 0
        self.discount = discount
        self.update_taget_network()
        self.action_dim = action_dim
        self.state_dim = state_dim
    
    def update_target_network(self):
        self.t_q_net.load_state_dict(self.q_net.state_dict())
    def act(self, state, eval_mode=False):
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.TensorFloat(state).unqueeze(0)
        with torch.no_grad():
            action_logits = self.q_net(state_t)
        return action_logits.argmax().item()
    def learn(self,batch):
        states, actions,rewards,next_states,dones = batch
        current_q = self.q_net(states).gather(1,actions)
        with torch.no_grad():
            # q(st,at) = reward + (gamma * (tqnet(st+1, argmax(q_net(st+1)))))
            best_action = self.q_net(next_states).argmax(dim=1,keepdim = True)
            next_q = self.t_q_net(next_states).gather(1,best_action)
            target_q = rewards + ((1 - dones) * (self.discount * next_q))
        criterion = nn.MSELoss()
        loss = criterion(current_q, target_q)
        self.optim.zero_grad()
        loss.backward()
        if self.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clipping)
        self.optim.step()
        self.train_step += 1
        if self.train_step % self.target_update_period == 0:
            self.update_target_network(hard=True)

        return loss.item()
    def decay_epsilon(self, final_epsilon, decay_steps, current_step):
        self.epsilon = final_epsilon + (self.epsilon - final_epsilon) * np.exp(-current_step / decay_steps)

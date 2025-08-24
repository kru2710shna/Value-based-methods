import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------- Hyperparams ----------------
BUFFER_SIZE = int(1e5)   # replay buffer size
BATCH_SIZE = 64          # minibatch size
GAMMA      = 0.99        # discount factor
TAU        = 1e-3        # soft update factor
LR         = 5e-4        # learning rate
UPDATE_EVERY = 4         # how often to learn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------- Model ----------------
class QNetwork(nn.Module):
    """Simple MLP mapping state -> action values."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1  = nn.Linear(state_size, fc1_units)
        self.fc2  = nn.Linear(fc1_units, fc2_units)
        self.fc3  = nn.Linear(fc2_units, action_size)

        # (optional) small init
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # raw Q-values per action


# ---------------- Agent ----------------
class Agent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        self.state_size  = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-networks
        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY steps if enough samples
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        """Epsilon-greedy action selection."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(q_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value params from a batch of (s, a, r, s', done)."""
        states, actions, rewards, next_states, dones = experiences
        # --- Q targets ---
        # Max over next-state actions from target net (DQN)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Bellman target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # --- Q expected (from local net) ---
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # --- Loss & optimize ---
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Soft update target net ---
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for t_param, l_param in zip(target_model.parameters(), local_model.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


# ---------------- Replay Buffer ----------------
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states      = torch.from_numpy(np.vstack([e.state      for e in experiences])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones       = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

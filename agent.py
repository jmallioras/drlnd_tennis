import random
from model import Actor
from model import Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np
# Replay Buffer Size
BUFFER_SIZE = int(1e6)
# Minibatch Size
BATCH_SIZE = 512
# Discount Gamma
GAMMA = 0.995
# Soft Update Value
TAU = 1e-2
# Learning rates for each DNN
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
# Update the DNNs every 5 timesteps
UPDATE_STEP = 5
# Learn from random batches of experiences 4 times
TIMES_LEARN = 4



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent():
    """Interacts with and learns from the environment using the DDPG algorithm."""

    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Neural Network (Regular and target)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Neural Network (Regular and target)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Ensure that both networks have the same weights
        self.clone(self.actor_target, self.actor_local)
        self.clone(self.critic_target, self.critic_local)


    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        # Use LOCAL ACTOR in EVAL mode
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # Switch back to training
        self.actor_local.train()

        return np.clip(action,-1, 1)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # CRITIC TRAINING

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # clip gradients
        self.critic_optimizer.step()

        # ACTOR TRAINING

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update for target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        # Update the target network slowly to improve the stability
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def clone(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class MultiAgentDDPG():
    # Hosts and trains multiple DDPG Agents
    def __init__(self, state_size, action_size, n_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.timestep = 0
        self.n_agents = n_agents

        self.agents = []
        for i in range(n_agents):
            self.agents.append(DDPGAgent(state_size, action_size, random_seed))

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        # Save all experiences in replay buffer
        for i in range(self.n_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Train every UPDATE_STEP timesteps
        self.timestep = (self.timestep + 1) % UPDATE_STEP
        if len(self.memory) > BATCH_SIZE and self.timestep == 0:
            # Train TIMES_LEARN times
            for _ in range(TIMES_LEARN):
                for agent in self.agents:
                    # Train every agent
                    experiences = self.memory.sample()
                    agent.learn(experiences, GAMMA)

    def act(self, states, i_episode=0, add_noise=True):

        actions = []
        for agent, state in zip(self.agents, states):
            actions.append(np.squeeze(agent.act(np.expand_dims(state, axis=0)), axis=0))
        return np.stack(actions)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        action = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, action, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
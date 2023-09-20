# Import libraries
from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
from agent import *
from collections import deque
from agent import *
import torch
import time
import matplotlib.pyplot as plt

# Maximum length of an epsiode
MAX_TIMESTEPS = 1000

# Initialize environment and get default brain
env = UnityEnvironment(file_name="Tennis.app", no_graphics=False)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
# number of agents
n_agents = len(env_info.agents)
# size of each action
action_size = brain.vector_action_space_size
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
agents = MultiAgentDDPG(state_size, action_size, n_agents, random_seed=12)

for i, agent in enumerate(agents.agents):
            agent.actor_local.load_state_dict(torch.load(f'actor_local_{i}.pth', map_location=torch.device('cpu')))
            agent.critic_local.load_state_dict(torch.load(f'critic_local_{i}.pth', map_location=torch.device('cpu')))

scores =[] # Keep scores
average_scores = []

for e in range(100):
    score = np.zeros(n_agents)
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations

    for _ in range(MAX_TIMESTEPS):
        actions = agents.act(states)  # select an action (for each agent)

        env_info = env.step(actions)[brain_name]  # send all actions to tne environment

        # get next state (for each agent)
        next_states = env_info.vector_observations

        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished

        score += rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break

    scores.append(np.max(score))
    average_scores.append(np.mean(scores))
    print('\rEpisode {}\tAverage Score: {:.4f}'.format(e, np.mean(scores)), end="")

# Plot results
plt.plot(np.arange(1, len(scores)+1), scores, label = "Episode Score")
plt.plot(np.arange(1, len(scores)+1), average_scores, label = "Average Episode Score")
plt.ylabel('Score')
plt.xlabel('Episode Count')
plt.title("Score progression during testing")
plt.legend()
plt.savefig("test_progression.png")

env.close()
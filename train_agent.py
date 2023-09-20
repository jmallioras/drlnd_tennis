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


def train(agents, n_agents):
    n_episodes = 5000
    scores_window = deque(maxlen=100)                   # Array to bookkeep the best scores of the last 100 episodes
    scores = []
    average_scores = []
    # Measure the total time of training
    start_time = time.time()
    for e in range(1, n_episodes+1):
        episode_scores = np.zeros(n_agents)             # Keep current agents score

        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        states = env_info.vector_observations

        for _ in range(MAX_TIMESTEPS):
            actions = agents.act(states)                # select an action for each agent
            env_info = env.step(actions)[brain_name]    # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state for each agent

            rewards = env_info.rewards                  # get reward (for each agent)
            dones = env_info.local_done                 # see if episode finished
            agents.step(states, actions, rewards, next_states, dones)
            episode_scores += rewards
            states = next_states

            if np.any(dones):
                break

        scores_window.append(np.max(episode_scores))    # Bookkeeping
        scores.append(np.max(episode_scores))
        average_scores.append(np.mean(scores))

        print(f'\rEpisode {e}\t  Score: {np.max(episode_scores):.2f} | Avg. : {np.mean(scores):.2f}', end="")

        # Print every 100 episodes
        if e % 100==0:
            print(f'\rEpisode {e}\tAverage Score: {np.mean(scores_window):.2f}\n')
            for agent_idx, agent in enumerate(agents.agents):
                torch.save(agent.actor_local.state_dict(), f'actor_local_{agent_idx}.pth')
                torch.save(agent.critic_local.state_dict(), f'critic_local_{agent_idx}.pth')

        if np.mean(scores_window) >= 0.5:
            print(f"\rEnvironment solved in {e} episodes ({(time.time()-start_time)/60 :.2f} minutes) with a score of {np.max(episode_scores):.2f}.\nAverage last 100 episode score:{np.mean(scores_window)}")
            for agent_idx, agent in enumerate(agents.agents):
                torch.save(agent.actor_local.state_dict(), f'actor_local_{agent_idx}.pth')
                torch.save(agent.critic_local.state_dict(), f'critic_local_{agent_idx}.pth')
            break

    return scores, average_scores

# Initialize environment and get default brain
env = UnityEnvironment(file_name="Tennis.app", no_graphics=True)
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
# Train the agents
scores, average_scores = train(agents, n_agents)

# Save scores as csv
np.savetxt("scores.csv",scores)
np.savetxt("avgScores.csv",average_scores)

# Plot results
plt.plot(np.arange(1, len(scores)+1), scores, label = "Episode Score")
plt.plot(np.arange(1, len(scores)+1), average_scores, label = "Average Episode Score")
plt.ylabel('Score')
plt.xlabel('Episode Count')
plt.title("Score progression during training")
plt.legend()
plt.savefig("training_progression.png")

env.close()
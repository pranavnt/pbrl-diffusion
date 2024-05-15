import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from utils import ReplayBuffer
from sac import SACAgent
import random
import torch

drawer_open_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-open-v2-goal-observable"]
# drawer_close_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-goal-observable"]

env = drawer_open_cls()

env.max_path_length = 100000

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

action_range = (torch.tensor(env.action_space.low), torch.tensor(env.action_space.high))

device = "cuda" if torch.cuda.is_available() else "cpu"

policy = SACAgent(obs_dim, action_dim, action_range, device)


num_episodes = 100

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    step = 0
    replay_buffer = ReplayBuffer(capacity=10000000)

    prev_obs = obs

    while not done:
        action = policy.act(obs)
        obs, reward, done, truncated, additional_info = env.step(action)
        if truncated:
            done = True
        replay_buffer.push(prev_obs, action, obs, reward, done, additional_info=additional_info)
        if step % 1000 == 0:
            policy.update(replay_buffer, step)
        prev_obs = obs
        step += 1

    avg_reward = sum(entry[3] for entry in replay_buffer.buffer) / len(replay_buffer.buffer)
    print(f"Episode {episode}: Average reward: {avg_reward}, Length of trajectory: {len(replay_buffer)}")

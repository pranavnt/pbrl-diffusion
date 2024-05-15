import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from optimal_policies import OptimalDrawerClose, OptimalDrawerOpen
import numpy as np
from concurrent import futures
import pickle
from utils import ReplayBuffer
from sac import SACAgent
import random
import torch

TASK = "CLOSE"

if TASK == "OPEN":
    drawer_open_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-open-v2-goal-observable"]
    env = drawer_open_cls()
    policy = OptimalDrawerOpen()
elif TASK == "CLOSE":
    drawer_close_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-goal-observable"]
    env = drawer_close_cls()
    policy = OptimalDrawerClose()

env.max_path_length = 1000

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

action_range = (torch.tensor(env.action_space.low), torch.tensor(env.action_space.high))

# env.render_mode = "human"

rollouts = []

num_episodes = 1000

def _collect_rollout(episode_number):
    obs, _ = env.reset()

    for i in range(100):
        hand_pos = np.random.uniform(low=env.hand_low, high=env.hand_high)
        env.set_xyz_action(hand_pos)

    done = False
    step = 0
    replay_buffer = ReplayBuffer(capacity=10000000)

    prev_obs = obs

    while not done:
        action = policy.get_action(obs)
        obs, reward, done, truncated, additional_info = env.step(action)

        # env.render()
        if truncated:
            done = True

        replay_buffer.push(prev_obs, action, obs, reward, done, additional_info=additional_info)
        prev_obs = obs
        step += 1

    print(f"episode {episode_number} done")

    return replay_buffer

from multiprocessing import Pool

if __name__ == "__main__":
    with Pool() as pool:
        rollouts = pool.map(_collect_rollout, range(num_episodes))

    pickle.dump(rollouts, open("rollouts2.pkl", "wb"))

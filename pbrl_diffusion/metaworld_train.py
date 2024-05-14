import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from sac import SACAgent
import random
import torch


drawer_open_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-open-v2-goal-observable"]
# drawer_close_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["drawer-close-v2-goal-observable"]

env = drawer_open_cls()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high - env.action_space.low

device = "cuda" if torch.cuda.is_available() else "cpu"

policy = SACAgent(obs_dim, action_dim, action_range, device)

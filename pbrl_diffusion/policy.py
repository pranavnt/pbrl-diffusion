# diffusion
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from enum import Enum

class DiffusionPolicy(nn.Module):
  def __init__(self, model_config):
    super(DiffusionPolicy, self).__init__()
    ...

  def forward(self, x):
    ...

class GaussianPolicy(nn.Module):
  def __init__(self, model_config):
    super(GaussianPolicy, self).__init__()
    ...

  def forward(self, x):
    ...

class ActorType(Enum):
  DIFFUSION = DiffusionPolicy
  GAUSSIAN = GaussianPolicy

def instantiate_policy(policy_type, config=None):
    return policy_type.value(config)

if __name__ == "__main__":
  diffusion_policy = instantiate_policy(ActorType.DIFFUSION)
  gaussian_policy = instantiate_policy(ActorType.GAUSSIAN)
  print(diffusion_policy)
  print(gaussian_policy)

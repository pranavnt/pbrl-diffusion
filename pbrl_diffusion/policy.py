# diffusion
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

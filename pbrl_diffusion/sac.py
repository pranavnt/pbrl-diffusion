import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from utils import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.trunk = self._build_mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.apply(self._weight_init)

    def _build_mlp(self, input_dim, hidden_dim, output_dim, hidden_depth):
        layers = []
        for i in range(hidden_depth):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = torch.exp(log_std)
        return SquashedNormal(mu, std)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.Q1 = self._build_mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = self._build_mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.apply(self._weight_init)

    def _build_mlp(self, input_dim, hidden_dim, output_dim, hidden_depth):
        layers = []
        for i in range(hidden_depth):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2

class TanhTransform(torch.distributions.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(torch.distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class SACAgent:
    def __init__(self, obs_dim, action_dim, action_range, device,
                 hidden_dim=256, hidden_depth=2, log_std_bounds=[-20, 2],
                 discount=0.99, init_temperature=0.01, alpha_lr=1e-3,
                 alpha_betas=[0.9, 0.999], actor_lr=1e-3, actor_betas=[0.9, 0.999],
                 actor_update_frequency=1, critic_lr=1e-3, critic_betas=[0.9, 0.999],
                 critic_tau=0.005, critic_target_update_frequency=2,
                 batch_size=1024, learnable_temperature=True):

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = Critic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        action = action.cpu().data.numpy().flatten()
        return action

    def update_critic(self, obs, action, reward, next_obs, not_done):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)

        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

        target_Q = reward.view(-1, 1) + (not_done.view(-1, 1) * self.discount * target_V.view(-1, 1))
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, step):
        obs, action, next_obs, reward, not_done, additional_info = replay_buffer.sample(self.batch_size)

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            self._sync_critics()

    def _sync_critics(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.critic_tau * param.data + (1.0 - self.critic_tau) * target_param.data)
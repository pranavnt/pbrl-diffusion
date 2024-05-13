import torch
import torch.nn as nn
import torch.optim as optim

import random

class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(RewardModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.layers(state)

    def train_model(self, preference_labels, num_epochs, batch_size):
        optimizer = optim.Adam(self.parameters())
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            # Shuffle the preference labels
            random.shuffle(preference_labels)

            for i in range(0, len(preference_labels), batch_size):
                batch = preference_labels[i:i+batch_size]
                preferred_states = [label["preferred"] for label in batch]
                dispreferred_states = [label["dispreferred"] for label in batch]

                preferred_states = torch.FloatTensor(preferred_states)
                dispreferred_states = torch.FloatTensor(dispreferred_states)

                optimizer.zero_grad()
                preferred_rewards = self.forward(preferred_states)
                dispreferred_rewards = self.forward(dispreferred_states)

                # Bradley-Terry pairwise comparison
                preference_probs = torch.exp(preferred_rewards) / (torch.exp(preferred_rewards) + torch.exp(dispreferred_rewards))
                preference_labels = torch.ones_like(preference_probs)
                loss = criterion(preference_probs, preference_labels)

                loss.backward()
                optimizer.step()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, reward, done, additional_info=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward, done, additional_info)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones, additional_info = zip(*batch)
        return states, actions, next_states, rewards, dones, additional_info

    def __len__(self):
        return len(self.buffer)
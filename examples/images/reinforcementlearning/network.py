import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, dueling_DQN=False):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dueling_DQN = dueling_DQN

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        feature_size = self._get_feature_size()

        if self.dueling_DQN:
            self.advantage = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
            )
            self.value = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        else:
            self.action_value = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, self.num_actions),
            )

    def forward(self, x):
        x = x.float() / 255  # Normalize the input
        x = self.features(x)
        x = x.view(x.size(0), -1)

        if self.dueling_DQN:
            advantage = self.advantage(x)
            value = self.value(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.action_value(x)

        return q_values

    def _get_feature_size(self):
        self.features.eval()
        with torch.no_grad():
            return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

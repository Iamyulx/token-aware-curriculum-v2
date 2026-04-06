import torch
import torch.nn as nn
import torch.nn.functional as F

class RLSelector(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        return torch.sigmoid(self.net(features))


class RLTrainer:
    def __init__(self, selector, lr=1e-3):
        self.selector = selector
        self.optimizer = torch.optim.Adam(selector.parameters(), lr=lr)

    def update(self, features, rewards):
        probs = self.selector(features)
        loss = -torch.mean(torch.log(probs) * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

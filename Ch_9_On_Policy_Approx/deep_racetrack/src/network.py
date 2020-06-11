import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NN(nn.Module):
    """Neural network to approximate the value function."""

    def __init__(self):

        # if gpu is to be used
        super(NN, self).__init__()
        self.l1 = nn.Linear(4, 32)
        self.l2 = nn.Linear(32, 54)
        self.l3 = nn.Linear(54, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


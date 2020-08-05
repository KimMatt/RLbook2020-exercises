import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class NN(nn.Module):
    """Neural network to approximate the value function."""

    def __init__(self, input_size, output_size, softmax_out=False):
        super(NN, self).__init__()
        self.l1 = nn.Linear(input_size, output_size)
        self.softmax_out = softmax_out
        if self.softmax_out:
            self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.l1(x)
        if self.softmax_out:
            x = self.softmax(x)
        return x

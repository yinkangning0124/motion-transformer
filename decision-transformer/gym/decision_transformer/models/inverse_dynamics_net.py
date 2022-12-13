import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseDynamics(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.Linear1 = nn.Linear(2 * input_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, X):
        hidden_1 = self.Linear1(X)
        hidden_2 = self.Linear2(hidden_1)
        out = self.out(hidden_2)
        return out
        
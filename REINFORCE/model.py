import torch
from torch import nn
import numpy as np

class fc(nn.Module):
    def __init__(self, no_obv, no_actions, dropout=0.1, BN=True):
        super().__init__()
        self.layer1 = nn.Linear(no_obv, 128)
        self.out = nn.Linear(128, no_actions)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.act2 = nn.Softmax(dim=-1)
    
    def forward(self, obv):
        if not isinstance(obv, torch.FloatTensor):
            obv = torch.FloatTensor(obv)
        layer1out = self.act(self.layer1(torch.FloatTensor(obv)))
        layer1drop = self.dropout(layer1out)
        layer3out = self.out(layer1drop)
        return self.act2(layer3out)
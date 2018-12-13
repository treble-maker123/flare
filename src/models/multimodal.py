import torch
import torch.nn as nn
import torch.nn.functional as F

class TadpoleFeat(nn.Module):
    def __init__(self, num_input, num_output):
        super(TadpoleFeat, self).__init__()
        self.aff1 = nn.Linear(num_input, num_output)
        self.bn1 = nn.BatchNorm1d(num_output)
        self.dp1 = nn.Dropout(p=0.1)

        self.aff2 = nn.Linear(num_output, num_output)
        self.bn2 = nn.BatchNorm1d(num_output)       
        self.dp2 = nn.Dropout(p=0.5)

        self.aff3 = nn.Linear(num_output, num_output)
        self.bn3 = nn.BatchNorm1d(num_output)       
        self.dp3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.squeeze()
        x = self.dp1(self.bn1(F.relu(self.aff1(x))))
        x = self.dp2(self.bn2(F.relu(self.aff2(x))))
        x = self.dp3(self.bn3(F.relu(self.aff3(x))))
        return x


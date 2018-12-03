import torch
import torch.nn as nn
import torch.nn.functional as F

class Tadpole1:
    def __init__(self):
        super(Tadpole1, self).__init__()
        self.affine = nn.Linear(692, 500)
        
    def forward(self, x):
        x = x.squeeze()
        x = self.affine(x)
        x = nn.BatchNorm1d(x.shape[1])(x)
        x = F.relu(x)
        return x

class Tadpole2:
    def __init__(self):
        super(Tadpole2, self).__init__()
        self.affine1 = nn.Linear(692, 200)
        self.affine2 = nn.Linear(200, 10)
 
    def forward(self, x):
        x = x.squeeze()
        x = self.affine1(x)
        x = nn.BatchNorm1d(x.shape[1])(x)
        x = F.relu(x)

        x = self.affine2(x)
        x = nn.BatchNorm1d(x.shape[1])(x)
        x = F.relu(x)
        return x


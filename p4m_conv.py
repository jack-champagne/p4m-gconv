import torch
import torch.nn as nn
import torch.nn.functional as F
import gconvutils as gutils
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

class P4MNet(nn.Module):
    def __init__(self):
        super(P4MNet, self).__init__() 
        self.conv1 = P4MConvZ2(in_channels=1, out_channels=8, kernel_size=5, stride=1)
        self.conv2 = P4MConvP4M(in_channels=8, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = P4MConvP4M(in_channels=16, out_channels=32, kernel_size=5)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = gutils.plane_group_spatial_max_pooling(x, 1, 1) # Is this even doing anything?
        x = torch.max(x, dim=2)[0]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class P4MNetC(nn.Module):
    def __init__(self):
        super(P4MNetC, self).__init__() 
        self.conv1 = P4MConvZ2(in_channels=3, out_channels=12, kernel_size=5, stride=1)
        self.conv2 = P4MConvP4M(in_channels=12, out_channels=24, kernel_size=5, stride=1)
        self.conv3 = P4MConvP4M(in_channels=24, out_channels=48, kernel_size=5)

        self.fc1 = nn.Linear(48, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = torch.max(x, dim=2)[0]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
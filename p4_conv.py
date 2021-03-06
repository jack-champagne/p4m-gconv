import torch
import torch.nn as nn
import torch.nn.functional as F
import gconvutils as gutils
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

class P4Net(nn.Module):
    def __init__(self):
        super(P4Net, self).__init__() 
        self.conv1 = P4ConvZ2(in_channels=1, out_channels=8, kernel_size=5, stride=1)
        self.conv2 = P4ConvP4(in_channels=8, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = P4ConvP4(in_channels=16, out_channels=32, kernel_size=5)

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = gutils.plane_group_spatial_max_pooling(x, 1, 1)
        x = torch.max(x, dim=2)[0]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class P4NetC(nn.Module):
    def __init__(self):
        super(P4NetC, self).__init__() 
        self.conv1 = P4ConvZ2(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = P4ConvP4(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = P4ConvP4(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = gutils.plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = torch.max(x, dim=2)[0]
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
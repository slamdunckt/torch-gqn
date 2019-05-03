import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class Patcher(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64+7, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64+7, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        # self.pool  = nn.AvgPool2d(16)

    def forward(self, x, v):
        # Resisual connection
        patch_label =[];
        batch_size = np.shape(x)[0];
        context_size = np.shape(x)[3];
        for i in range(batch_size):
            ttt=[]
            for j in range(8):
                tt=[]
                for k in range(8):
                    t = [i//8+1, i%8+1]
                    tt.append(t)
                ttt.append(tt)
            patch_label.append(ttt)
        patch_all = np.zeros(np.shape(x))
        for i in range(context_size):
            patch_size = 8
            stride_size = 4
            patches = x[:,i].unfold(1,patch_size, stride_size).unfold(2,patch_size, stride_size)
            patches = torch.cat((patches, patch_label),3)
            patch_all[:,i] = patches

        skip_in  = F.relu(self.conv1(patch_all))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)

        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))

        # Pool
        # r = self.pool(r)

        return r

class PatchKey(nn.Module):
    def __init__(self):
        super(Tower, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=1, stride=1)

    def forward(self, x, v):
        # Resisual connection
        r = F.relu(self.conv1(x))
        r = F.relu(self.conv2(r))
        r = F.relu(self.conv3(r))
        r = F.relu(self.conv4(r))
        r = F.relu(self.conv5(r))
        r = F.relu(self.conv6(r))
        
        return r

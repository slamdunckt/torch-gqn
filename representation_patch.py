import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class Patcher(nn.Module):       #patcher of images and poses
    def __init__(self):
        super(Patcher, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64+7, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64+7, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=2, stride=1)

        self.unfold = nn.Unfold(kernel_size=(8,8),padding=2,stride=(4,4))
        self.patch_label = torch.Tensor(1,2,8,8).cuda()
        for i in range(8):
            for j in range(8):
                self.patch_label[0][0][i][j] = i
                self.patch_label[0][1][i][j] = j

    def forward(self, x, v):
        # Resisual connection
        batch_size = x.shape[0];
        context_size = x.shape[1];
        
        result = torch.Tensor().cuda()
        for i in range(batch_size):
            patches = self.unfold(x[i]).transpose(1,2).reshape(-1,3,8,8)
            patches = torch.cat((patches, self.patch_label.repeat(patches.shape[0],1,1,1)),1)

            skip_in  = F.relu(self.conv1(patches))
            skip_out = F.relu(self.conv2(skip_in))

            r = F.relu(self.conv3(skip_in))
            r = F.relu(self.conv4(r)) + skip_out

            # Broadcast
            v_repeat = v[i].reshape(context_size,v[i].shape[1], 1, 1).repeat(r.shape[0]//context_size, 1, 2, 2)

            # Resisual connection
            # Concatenate
            skip_in = torch.cat((r, v_repeat), dim=1)
            skip_out  = F.relu(self.conv5(skip_in))

            r = F.relu(self.conv6(skip_in))
            r = F.relu(self.conv7(r)) + skip_out
            r = F.relu(self.conv8(r)).unsqueeze(0)
            result = torch.cat((result,r),0)
        
        result = result.squeeze()
        result = result.reshape(batch_size,context_size,8,8,-1)
        return result

class PatchKey(nn.Module):  # patcher of pure images
    def __init__(self):
        super(PatchKey, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=1, stride=1)

    def forward(self, x):
        # Resisual connection
        batch_size = x.shape[0]
        result = torch.Tensor().cuda()
        for i in range(batch_size):
            r = F.relu(self.conv1(x[i]))
            r = F.relu(self.conv2(r))
            r = F.relu(self.conv3(r))
            r = F.relu(self.conv4(r))
            r = F.relu(self.conv5(r))
            r = F.relu(self.conv6(r)).unsqueeze(0)
            result = torch.cat((result,r),0)

        result = result.transpose(2,4)
        result = result.transpose(2,3)
        return result

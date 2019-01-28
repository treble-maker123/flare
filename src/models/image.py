import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet_utils import * 

class Tadpole1(nn.Module):
    def __init__(self, num_input, num_output):
        super(Tadpole1, self).__init__()
        self.aff1 = nn.Linear(num_input, num_output)
        self.bn1 = nn.BatchNorm1d(num_output)
        self.dp1 = nn.Dropout(p=0.1)

        self.aff2 = nn.Linear(num_output, num_output)
        self.bn2 = nn.BatchNorm1d(num_output) 
        self.dp2 = nn.Dropout(p=0.2)

        self.aff3 = nn.Linear(num_output, num_output)
        self.bn3 = nn.BatchNorm1d(num_output) 
        self.dp3 = nn.Dropout(p=0.2)

        self.aff4 = nn.Linear(num_output, num_output)
        self.bn4 = nn.BatchNorm1d(num_output)       
        self.dp4 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.squeeze()
        x = self.dp1(self.bn1(F.relu(self.aff1(x))))
        x = self.dp2(self.bn2(F.relu(self.aff2(x))))
        x = self.dp3(self.bn3(F.relu(self.aff3(x))))
        x = self.dp4(self.bn4(F.relu(self.aff3(x))))
        return x

class Tadpole2(nn.Module):
    def __init__(self, num_input, num_output):
        super(Tadpole2, self).__init__()
        self.aff1 = nn.Linear(num_input, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.dp1 = nn.Dropout(p=0.5)

        self.aff2 = nn.Linear(1000, 1000)
        self.bn2 = nn.BatchNorm1d(1000) 
        self.dp2 = nn.Dropout(p=0.2)

        self.aff3 = nn.Linear(1000, 1000)
        self.bn3 = nn.BatchNorm1d(1000)
        self.dp3 = nn.Dropout(p=0.2)

        self.aff4 = nn.Linear(1000, num_output)
        self.bn4 = nn.BatchNorm1d(num_output)
        self.dp4 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.squeeze()
        x = self.dp1(self.bn1(F.relu(self.aff1(x))))
        x = self.dp2(self.bn2(F.relu(self.aff2(x))))
        x = self.dp3(self.bn3(F.relu(self.aff3(x))))
        x = self.dp4(self.bn4(F.relu(self.aff4(x))))
        return x

class unet_3D(nn.Module):
    def __init__(self, num_classes, feature_scale=1, n_classes=3, is_deconv=True, in_channels=1, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        f = [64, 128, 256, 512, 1024]
        self.filters = [int(x / self.feature_scale) for x in f]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, self.filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(self.filters[0], self.filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(self.filters[1], self.filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(self.filters[2], self.filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(self.filters[3], self.filters[4], self.is_batchnorm)

        # final conv (without any concat)
        
        self.globalavgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.outputlayer = nn.Linear(self.filters[1],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
                #m = m.double()
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                #m = m.double()

    def forward(self, inputs):
        #print("before conv1")
        conv1 = self.conv1(inputs)
        #print("after conv1")
        
        maxpool1 = self.maxpool1(conv1)
        #print("shape maxpool1: ", maxpool1.size())

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        #print("shape maxpool2: ", maxpool2.size())
        gap = self.globalavgpool(maxpool2)
        gap = gap.view(-1,self.filters[1])
        print("shape gap: ", gap.size())

        out = self.outputlayer(gap)
        print("shape output: ", out.size())
        return out

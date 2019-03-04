import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class DeepMRI(nn.Module):
    '''
    Super deep network with many convolution layers. MUST RUN ON M40 GPU!
    '''
    def __init__(self, **kwargs):
        super().__init__()
        num_classes = kwargs.get("num_classes", 3)

        # input 256x256x256, output 254x254x254
        self.conv1 = ConvolutionBlock(1, 16, kernel_size=3,
                        conv_stride=1, max_pool=False, relu=True)
        # input 254x254x254, output 252x252x252
        self.conv2 = ConvolutionBlock(16, 16,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 252x252x252, output 250x250x250 -> 254x254x254
        self.conv3 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(16)
        nn.init.kaiming_normal_(self.conv3.weight)

        # input 254x254x254, output 252x252x252
        self.conv4 = ConvolutionBlock(16, 16,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 252x252x252, output 250x250x250
        self.conv5 = ConvolutionBlock(16, 16,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 250x250x250, output 248x248x248 -> 254x254x254
        self.conv6 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=0)
        self.bn6 = nn.BatchNorm3d(16)
        nn.init.kaiming_normal_(self.conv6.weight)
        # input 254x254x254, output 127x127x127
        self.mp6 = nn.MaxPool3d(2, stride=2)

        # input 127x127x127, output 125x125x125
        self.conv7 = ConvolutionBlock(16, 32,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 125x125x125, output 123x123x123
        self.conv8 = ConvolutionBlock(32, 32,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 123x123x123, output 121x121x121 -> 125x125x125
        self.conv9 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.bn9 = nn.BatchNorm3d(32)
        nn.init.kaiming_normal_(self.conv9.weight)

        # input 125x125x125, output 123x123x123
        self.conv10 = ConvolutionBlock(32, 32,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 123x123x123, output 121x121x121
        self.conv11 = ConvolutionBlock(32, 32,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 121x121x121, output 119x119x119 -> 125x125x125
        self.conv12 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.bn12 = nn.BatchNorm3d(32)
        nn.init.kaiming_normal_(self.conv12.weight)
        # input 125x125x125, 62x62x62
        self.mp12 = nn.MaxPool3d(2, stride=2)

        # input 62x62x62, output 60x60x60
        self.conv13 = ConvolutionBlock(32, 64,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 60x60x60, output 58x58x58
        self.conv14 = ConvolutionBlock(64, 64,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 58x58x58, output 56x56x56 -> 60x60x60
        self.conv15 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.bn15 = nn.BatchNorm3d(64)
        nn.init.kaiming_normal_(self.conv15.weight)

        # input 60x60x60, output 58x58x58
        self.conv16 = ConvolutionBlock(64, 64,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 58x58x58, output 56x56x56
        self.conv17 = ConvolutionBlock(64, 64,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 56x56x56, output 54x54x54 -> 60x60x60
        self.conv18 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.bn18 = nn.BatchNorm3d(64)
        nn.init.kaiming_normal_(self.conv18.weight)
        # input 60x60x60, output 30x30x30
        self.mp18 = nn.MaxPool3d(2, stride=2)

        # input 30x30x30, output 28x28x28
        self.conv19 = ConvolutionBlock(64, 128,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 28x28x28, output 26x26x26
        self.conv20 = ConvolutionBlock(128, 128,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 26x26x26, output 24x24x24 -> 28x28x28
        self.conv21 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=0)
        self.bn21 = nn.BatchNorm3d(128)
        nn.init.kaiming_normal_(self.conv21.weight)
        # input 28x28x28, output 14x14x14
        self.mp21 = nn.MaxPool3d(2, stride=2)

        # input 13x13x13, output 11x11x11
        self.conv22 = ConvolutionBlock(128, 128,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 11x11x11, output 9x9x9
        self.conv23 = ConvolutionBlock(128, 128,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 9x9x9, output 2x2x2
        self.conv24 = ConvolutionBlock(128, 128,
                        kernel_size=3, conv_stride=2, max_pool=True, relu=True)

        classification_layers = [
            nn.Linear(2*2*2*128, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, num_classes)
        ]

        self.classify = nn.Sequential(*classification_layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.bn3(self.conv3(conv2))
        conv3 = F.pad(conv3, (2, 2, 2, 2, 2, 2), value=0) + conv1
        F.relu(conv3, inplace=True)

        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.bn6(self.conv6(conv5))
        conv6 = F.pad(conv6, (3, 3, 3, 3, 3, 3), value=0) + conv1
        F.relu(conv6, inplace=True)
        conv6 = self.mp6(conv6)

        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        conv9 = self.bn9(self.conv9(conv8))
        conv9 = F.pad(conv9, (2, 2, 2, 2, 2, 2), value=0) + conv7
        F.relu(conv9, inplace=True)

        conv10 = self.conv10(conv9)
        conv11 = self.conv11(conv10)
        conv12 = self.bn12(self.conv12(conv11))
        conv12 = F.pad(conv12, (3, 3, 3, 3, 3, 3), value=0) + conv7
        F.relu(conv12, inplace=True)
        conv12 = self.mp12(conv12)

        conv13 = self.conv13(conv12)
        conv14 = self.conv14(conv13)
        conv15 = self.bn15(self.conv15(conv14))
        conv15 = F.pad(conv15, (2, 2, 2, 2, 2, 2), value=0) + conv13
        F.relu(conv15, inplace=True)

        conv16 = self.conv16(conv15)
        conv17 = self.conv17(conv16)
        conv18 = self.bn18(self.conv18(conv17))
        conv18 = F.pad(conv18, (3, 3, 3, 3, 3, 3), value=0) + conv13
        F.relu(conv18, inplace=True)
        conv18 = self.mp18(conv18)

        conv19  = self.conv19(conv18)
        conv20 = self.conv20(conv19)
        conv21 = self.bn21(self.conv21(conv20))
        conv21 = F.pad(conv21, (2, 2, 2, 2, 2, 2), value=0) + conv19
        F.relu(conv21, inplace=True)
        conv21 = self.mp21(conv21)

        conv22 = self.conv22(conv21)
        conv23 = self.conv23(conv22)
        conv24 = self.conv24(conv23)

        hidden = conv24.view(len(x), -1)
        scores = self.classify(hidden)

        return scores

    def reconstruct(self, x):
        raise NotImplementedError("Set pretrain num_epochs to 0.")

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def reconstruction_loss(self, x, y, hidden=None):
        loss = F.mse_loss(x, y)

        # sparsity constraint
        if hidden is not None:
            loss += torch.sum(torch.abs(hidden))

        return loss


class ConvolutionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        conv_params = {
            "kernel_size": kwargs.get("kernel_size", 3),
            "stride": kwargs.get("conv_stride", 1),
            "padding": kwargs.get("padding", 0)
        }
        batch_norm = kwargs.get("batch_norm", True)
        max_pool = kwargs.get("max_pool", True)
        max_pool_kernel = kwargs.get("pool_kernel", 2)
        max_pool_stride = kwargs.get("pool_stride", 2)
        relu = kwargs.get("relu", True)

        layers = []

        conv = nn.Conv3d(input_dim, output_dim, **conv_params)

        layers.append(conv)

        if max_pool:
            layers.append(nn.MaxPool3d(max_pool_kernel, stride=max_pool_stride))

        if relu:
            layers.append(nn.ReLU(True))
            nn.init.kaiming_normal_(conv.weight)
        else:
            nn.init.xavier_normal_(conv.weight)

        if batch_norm:
            layers.append(nn.BatchNorm3d(output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

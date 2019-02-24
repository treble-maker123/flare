import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class Hosseini(nn.Module):
    '''
    Model after https://arxiv.org/pdf/1607.00455.pdf
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_kernels = kwargs.get("num_kernels", [16, 32, 64, 128])
        num_classes = kwargs.get("num_classes", 3)

        self.encoder_layers = [
            # input 256x256x256, output 127x127x127
            ConvolutionBlock(1, num_kernels[0], kernel_size=3, conv_stride=1,
                             max_pool=True, pool_stride=2, relu=True),
            # input 127x127x127, output 62x62x62
            ConvolutionBlock(num_kernels[0], num_kernels[1], kernel_size=3,
                             conv_stride=1, max_pool=True, pool_stride=2,
                             relu=True),
            # input 62x62x62, output 30x30x30
            ConvolutionBlock(num_kernels[1], num_kernels[2], kernel_size=3,
                             conv_stride=1, max_pool=True, pool_stride=2,
                             relu=True),
            # input 30x30x30, output 14x14x14
            ConvolutionBlock(num_kernels[2], num_kernels[3], kernel_size=3,
                             conv_stride=1, max_pool=True, pool_stride=2,
                             relu=True),
        ]

        classification_layers = [
            nn.Linear(14*14*14*num_kernels[-1], 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, num_classes)
        ]

        self.encode = nn.Sequential(*self.encoder_layers)
        self.classify = nn.Sequential(*classification_layers)

    def forward(self, x):
        hidden = self.encode(x).view(len(x), -1)
        return self.classify(hidden)

    def reconstruct(self, x):
        hidden = self.encode(x)

        # Conv layers from the encoder
        conv1 = self.encoder_layers[0].block[0]
        conv2 = self.encoder_layers[1].block[0]
        conv3 = self.encoder_layers[2].block[0]
        conv4 = self.encoder_layers[3].block[0]

        # Tie the weights
        # input 14x14x14, output 30x30x30
        deconv1 = F.conv_transpose3d(hidden, conv4.weight,
                                     bias=conv3.bias, stride=2,
                                     output_padding=1)
        # input 30x30x30, output 62x62x62
        F.relu(deconv1, inplace=True)
        deconv2 = F.conv_transpose3d(deconv1, conv3.weight,
                                     bias=conv2.bias, stride=2,
                                     output_padding=1)
        F.relu(deconv2, inplace=True)
        # input 62x62x62, output 127x127x127
        deconv3 = F.conv_transpose3d(deconv2, conv2.weight,
                                     bias=conv1.bias, stride=2,
                                     output_padding=1)
        padded_3 = F.pad(deconv3, (1, 0, 1, 0, 1, 0),
                       mode="constant", value=0)
        F.relu(padded_3, inplace=True)
        # input 127x127x127, output 256x256x256
        deconv4 = F.conv_transpose3d(padded_3, conv1.weight, stride=2,
                                     output_padding=1)

        return torch.sigmoid(deconv4)

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def reconstruction_loss(self, x, y, hidden=None):
        loss = F.mse_loss(x, y)

        # sparsity constraint
        if hidden is not None:
            loss += torch.sum(torch.abs(hidden))

        return loss

class HosseiniThreeLayer(nn.Module):
    '''
    Model similart o Hosseini but has a three-layer autoencoder, another convolution layer for classification, as well as more kernels
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_kernels = kwargs.get("num_kernels", [32, 64, 128, 192])
        num_classes = kwargs.get("num_classes", 3)

        self.encoder_layers = [
            # input 256x256x256, output 127x127x127
            ConvolutionBlock(1, num_kernels[0], kernel_size=3, conv_stride=1,
                             max_pool=True, pool_stride=2, relu=True),
            # input 127x127x127, output 62x62x62
            ConvolutionBlock(num_kernels[0], num_kernels[1], kernel_size=3,
                             conv_stride=1, max_pool=True, pool_stride=2,
                             relu=True),
            # input 62x62x62, output 30x30x30
            ConvolutionBlock(num_kernels[1], num_kernels[2], kernel_size=3,
                             conv_stride=1, max_pool=True, pool_stride=2,
                             relu=True),
        ]

        # input 30x30x30, output 4x4x4
        self.conv = ConvolutionBlock(num_kernels[2], num_kernels[3],
                                     kernel_size=3, conv_stride=2,
                                     max_pool=True, pool_stride=3, relu=True)

        classification_layers = [
            nn.Linear(4*4*4*num_kernels[-1], 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, num_classes)
        ]

        self.encode = nn.Sequential(*self.encoder_layers)
        self.classify = nn.Sequential(*classification_layers)

    def forward(self, x):
        hidden = self.encode(x)
        hidden = self.conv(hidden).view(len(x), -1)
        return self.classify(hidden)

    def reconstruct(self, x):
        hidden = self.encode(x)

        # Conv layers from the encoder
        conv1 = self.encoder_layers[0].block[0]
        conv2 = self.encoder_layers[1].block[0]
        conv3 = self.encoder_layers[2].block[0]

        # Tie the weights
        # input 30x30x30, output 62x62x62
        deconv2 = F.conv_transpose3d(hidden, conv3.weight,
                                     bias=conv2.bias, stride=2,
                                     output_padding=1)
        F.relu(deconv2, inplace=True)
        # input 62x62x62, output 127x127x127
        deconv3 = F.conv_transpose3d(deconv2, conv2.weight,
                                     bias=conv1.bias, stride=2,
                                     output_padding=1)
        padded_3 = F.pad(deconv3, (1, 0, 1, 0, 1, 0),
                       mode="constant", value=0)
        F.relu(padded_3, inplace=True)
        # input 127x127x127, output 256x256x256
        deconv4 = F.conv_transpose3d(padded_3, conv1.weight, stride=2,
                                     output_padding=1)

        return torch.sigmoid(deconv4)

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
        max_pool_stride = kwargs.get("pool_stride", 2)
        relu = kwargs.get("relu", True)

        layers = []

        conv = nn.Conv3d(input_dim, output_dim, **conv_params)

        layers.append(conv)

        if max_pool:
            layers.append(nn.MaxPool3d(max_pool_stride))

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
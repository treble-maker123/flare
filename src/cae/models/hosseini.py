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

        # Tie the flipped weights
        # input 14x14x14, output 30x30x30
        deconv1 = F.conv_transpose3d(hidden, conv4.weight.flip(0,1,3,2,4),
                                     bias=conv3.bias, stride=2,
                                     output_padding=1)
        # input 30x30x30, output 62x62x62
        F.relu(deconv1, inplace=True)
        deconv2 = F.conv_transpose3d(deconv1, conv3.weight.flip(0,1,3,2,4),
                                     bias=conv2.bias, stride=2,
                                     output_padding=1)
        F.relu(deconv2, inplace=True)
        # input 62x62x62, output 127x127x127
        deconv3 = F.conv_transpose3d(deconv2, conv2.weight.flip(0,1,3,2,4),
                                     bias=conv1.bias, stride=2,
                                     output_padding=1)
        padded_3 = F.pad(deconv3, (1, 0, 1, 0, 1, 0),
                       mode="constant", value=0)
        F.relu(padded_3, inplace=True)
        # input 127x127x127, output 256x256x256
        deconv4 = F.conv_transpose3d(padded_3, conv1.weight.flip(0,1,3,2,4),
                                     stride=2, output_padding=1)

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

        # input 30x30x30, output 5x5x5
        self.conv = ConvolutionBlock(num_kernels[2], num_kernels[3],
                                     kernel_size=3, conv_stride=2,
                                     max_pool=True, pool_stride=3, relu=True)

        classification_layers = [
            nn.Linear(5*5*5*num_kernels[-1], 128),
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
        deconv2 = F.conv_transpose3d(hidden, conv3.weight.flip(0,1,3,2,4),
                                     bias=conv2.bias, stride=2,
                                     output_padding=1)
        F.relu(deconv2, inplace=True)
        # input 62x62x62, output 127x127x127
        deconv3 = F.conv_transpose3d(deconv2, conv2.weight.flip(0,1,3,2,4),
                                     bias=conv1.bias, stride=2,
                                     output_padding=1)
        padded_3 = F.pad(deconv3, (1, 0, 1, 0, 1, 0),
                       mode="constant", value=0)
        F.relu(padded_3, inplace=True)
        # input 127x127x127, output 256x256x256
        deconv4 = F.conv_transpose3d(padded_3, conv1.weight.flip(0,1,3,2,4),
                                     stride=2, output_padding=1)

        return torch.sigmoid(deconv4)

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def reconstruction_loss(self, x, y, hidden=None):
        loss = F.mse_loss(x, y)

        # sparsity constraint
        if hidden is not None:
            loss += torch.sum(torch.abs(hidden))

        return loss

class HosseiniSimple(nn.Module):
    '''
    Model with bigger kernel + stride and fewer layers.
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_kernels = kwargs.get("num_kernels", [64, 128])
        num_classes = kwargs.get("num_classes", 3)

        self.encoder_layers = [
            # input 256x256x256, output 42x42x42
            ConvolutionBlock(1, num_kernels[0], kernel_size=7, conv_stride=2,
                             max_pool=True, pool_kernel=3, pool_stride=3,
                             relu=True),
            # input 42x42x42, output 6x6x6
            ConvolutionBlock(num_kernels[0], num_kernels[1], kernel_size=5,
                             conv_stride=2, max_pool=True, pool_kernel=3,
                             pool_stride=3, relu=True),
        ]

        classification_layers = [
            nn.Linear(6*6*6*num_kernels[-1], 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
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

        # Tie the weights
        # input 6x6x6, output 42x42x42
        deconv1 = F.conv_transpose3d(hidden, conv2.weight.flip(0,1,3,2,4),
                                     bias=conv1.bias, stride=7,
                                     output_padding=4)
        F.relu(deconv1, inplace=True)
        # input 42x42x42, output 256x256x256
        deconv2 = F.conv_transpose3d(deconv1, conv1.weight.flip(0,1,3,2,4),
                                     stride=6, output_padding=5)

        return deconv2

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def reconstruction_loss(self, x, y, hidden=None):
        loss = F.mse_loss(x, y)

        # sparsity constraint
        if hidden is not None:
            loss += torch.sum(torch.abs(hidden))

        return loss

class HosseiniDeep(nn.Module):
    '''
    Deep network with many convolution layers. MUST RUN ON M40 GPU!
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_kernels = kwargs.get("num_kernels",
                                 [16, 16, 16, 32, 32, 32, 64, 64, 64])
        num_classes = kwargs.get("num_classes", 3)

        # input 256x256x256, output 254x254x254
        self.conv1 = ConvolutionBlock(1, num_kernels[0], kernel_size=3,
                        conv_stride=1, max_pool=False, relu=True)
        # input 254x254x254, output 252x252x252
        self.conv2 = ConvolutionBlock(num_kernels[0], num_kernels[1],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 252x252x252, output 250x250x250
        self.conv3 = ConvolutionBlock(num_kernels[0], num_kernels[1],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 250x250x250, output 62x62x62
        self.conv4 = ConvolutionBlock(num_kernels[1], num_kernels[2],
                        kernel_size=3, conv_stride=2, max_pool=True,
                        pool_stride=2, relu=True)

        # input 62x62x62, output 60x60x60
        self.conv5 = ConvolutionBlock(num_kernels[2], num_kernels[3],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 60x60x60, output 58x58x58
        self.conv6 = ConvolutionBlock(num_kernels[3], num_kernels[4],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 58x58x58, output 56x56x56
        self.conv7 = ConvolutionBlock(num_kernels[3], num_kernels[4],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 56x56x56, output 13x13x13
        self.conv8 = ConvolutionBlock(num_kernels[4], num_kernels[5],
                        kernel_size=3, conv_stride=2, max_pool=True,
                        pool_stride=2, relu=True)

        # input 13x13x13, output 11x11x11
        self.conv9 = ConvolutionBlock(num_kernels[5], num_kernels[6],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 11x11x11, output 9x9x9
        self.conv10 = ConvolutionBlock(num_kernels[6], num_kernels[7],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 9x9x9, output 7x7x7
        self.conv11 = ConvolutionBlock(num_kernels[6], num_kernels[7],
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        # input 7x7x7, output 3x3x3
        self.conv12 = ConvolutionBlock(num_kernels[7], num_kernels[8],
                        kernel_size=3, conv_stride=2, max_pool=False, relu=True)

        classification_layers = [
            nn.Linear(3*3*3*num_kernels[-1], 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, num_classes)
        ]

        self.classify = nn.Sequential(*classification_layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3 + conv1[:, :, 2:-2, 2:-2, 2:-2])

        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7 + conv5[:, :, 2:-2, 2:-2, 2:-2])

        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        conv11  = self.conv11(conv10)
        conv12 = self.conv12(conv11 + conv9[:, :, 2:-2, 2:-2, 2:-2])

        hidden = conv12.view(len(x), -1)
        return self.classify(hidden)

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

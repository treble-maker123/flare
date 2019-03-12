import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class DeepAutoencMRI(nn.Module):
    '''Super deep autoencoder network with pretraining routine. MUST RUN ON M40 GPU!
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)
        dropout = kwargs.get("dropout", 0.0)

        # input 145, output 143
        self.block1 = ResidualBlock(16, bottleneck=True, dropout=dropout)
        # input 143, output 71
        self.conv1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm3d(32)

        # input 71, output 69
        self.block2 = ResidualBlock(32, bottleneck=True, dropout=dropout)
        # input 69, output 34
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm3d(64)

        # input 34, output 32
        self.block3 = ResidualBlock(64, bottleneck=True, dropout=dropout)
        # input 32, output 15
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=0)
        self.bn3 = nn.BatchNorm3d(128)

        # input 15, output 13
        self.block4 = ResidualBlock(128, bottleneck=True, dropout=dropout)
        # input 13, output 6
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=0)
        self.bn4 = nn.BatchNorm3d(256)

        # input 6, output 4
        self.block5 = ResidualBlock(256, bottleneck=True, dropout=dropout)
        # input 4, output 2
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm3d(512)

        self.classification_dropout = nn.Dropout(dropout)

        classification_layers = [
            nn.Linear(2*2*2*512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        ]

        self.classify = nn.Sequential(*classification_layers)

    def forward(self, x):
        l1 = F.relu(self.bn1(self.conv1(self.block1(x))))
        l2 = F.relu(self.bn2(self.conv2(self.block2(x))))
        l3 = F.relu(self.bn3(self.conv3(self.block3(x))))
        l4 = F.relu(self.bn4(self.conv4(self.block4(x))))
        l5 = F.relu(self.bn5(self.conv5(self.block5(x))))

        dropped = self.classification_dropout(l5)

        return self.classify(dropped)

    def reconstruct(self, x):
        pass

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)

    def reconstruction_loss(self, pred, target, hidden_state=None):
        loss = F.mse_loss(pred, target)

        if hidden_state is not None:
            loss += torch.sum(torch.abs(hidden_state))

        return loss

    def freeze(self):
        '''Freeze the weights of the convolution layers
        '''
        self.block1.freeze()
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.bn1.weight.requires_grad = False
        self.bn1.bias.require_grad = False

        self.block2.freeze()
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.bn2.weight.requires_grad = False
        self.bn2.bias.require_grad = False

        self.block3.freeze()
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.bn3.weight.requires_grad = False
        self.bn3.bias.require_grad = False

        self.block4.freeze()
        self.conv4.weight.requires_grad = False
        self.conv4.bias.requires_grad = False
        self.bn4.weight.requires_grad = False
        self.bn4.bias.require_grad = False

        self.block5.freeze()
        self.conv5.weight.requires_grad = False
        self.conv5.bias.requires_grad = False
        self.bn5.weight.requires_grad = False
        self.bn5.bias.require_grad = False

    def unfreeze(self):
        '''Unfreeze the weights of the convolution layers
        '''
        self.block1.unfreeze()
        self.conv1.weight.requires_grad = True
        self.conv1.bias.requires_grad = True
        self.bn1.weight.requires_grad = True
        self.bn1.bias.require_grad = True

        self.block2.unfreeze()
        self.conv2.weight.requires_grad = True
        self.conv2.bias.requires_grad = True
        self.bn2.weight.requires_grad = True
        self.bn2.bias.require_grad = True

        self.block3.unfreeze()
        self.conv3.weight.requires_grad = True
        self.conv3.bias.requires_grad = True
        self.bn3.weight.requires_grad = True
        self.bn3.bias.require_grad = True

        self.block4.unfreeze()
        self.conv4.weight.requires_grad = True
        self.conv4.bias.requires_grad = True
        self.bn4.weight.requires_grad = True
        self.bn4.bias.require_grad = True

        self.block5.unfreeze()
        self.conv5.weight.requires_grad = True
        self.conv5.bias.requires_grad = True
        self.bn5.weight.requires_grad = True
        self.bn5.bias.require_grad = True

class ResidualBlock(nn.Module):
    '''Three-layer residual block.

    Follow "bottleneck" building block from https://arxiv.org/abs/1512.03385
    Follows "pre-activation" setup from https://arxiv.org/abs/1603.05027
    '''

    def __init__(self, num_chan, **kwargs):
        super().__init__()

        dropout = kwargs.get("dropout", 0.0)
        bottleneck = kwargs.get("bottleneck", True)

        hidden_chan = num_chan // 2 if bottleneck else num_chan

        self.dropout = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm3d(num_chan)
        self.conv1 = nn.Conv3d(num_chan, hidden_chan, kernel_size=1,
                                        stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.bn2 = nn.BatchNorm3d(hidden_chan)
        self.conv2 = nn.Conv3d(hidden_chan, hidden_chan, kernel_size=3,
                                        stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.bn3 = nn.BatchNorm3d(hidden_chan)
        self.conv3 = nn.Conv3d(hidden_chan, num_chan, kernel_size=1,
                                        stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.bn3_back = nn.BatchNorm3d(hidden_chan)
        self.bn2_back = nn.BatchNorm3d(hidden_chan)

    def forward(self, x):
        dropped = self.dropout(x)

        l1 = self.conv1(F.relu(self.bn1(x)))
        l2 = self.conv2(F.relu(self.bn2(l1)))
        l3 = self.conv3(F.relu(self.bn3(l2)))

        amount_to_pad = (l1.shape[-1] - l3.shape[-1]) // 2
        # times six because there are six sides to pad
        padding = (amount_to_pad, ) * 6

        return F.pad(l3, padding, value=0) + x

    def backward(self, hidden):
        weight_flip = (0, 1, 3, 2, 4)

        l3 = F.conv_transpose3d(hidden, self.conv3.weight.flip(**weight_flip),
                                bias=self.conv3.bias, output_padding=0)
        l3 = F.relu(self.bn3_back(l3))

        l2 = F.conv_transpose3d(l3, self.conv2.weight.flip(**weight_flip),
                                bias=self.conv2.bias, output_padding=0)
        l2 = F.relu(self.bn2_back(l2))

        l1 = F.conv_transpose3d(l2, self.conv1.weight.flip(**weight_flip),
                                bias=self.conv1.bias, output_padding=0)

        return torch.sigmoid(l1)

    def freeze(self):
        '''Freeze the weights of the residual block so that no gradient is calculated.
        '''
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        '''Unfreeze the weights of the residual block so that gradient is calculated.
        '''
        for param in self.parameters():
            param.requires_grad = True

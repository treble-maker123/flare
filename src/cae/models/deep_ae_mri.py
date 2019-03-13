import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class DeepAutoencMRI(nn.Module):
    '''Super deep autoencoder network with pretraining routine.

    MUST RUN ON M40 GPU!
        - Classification-only:
            |_ 4 images per GPU with num_blocks [1,1,1,1,1]
            |_ 3 images per GPU with num_blocks [2,2,2,2,2]
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)
        num_blocks = kwargs.get("num_blocks", [1, 1, 1, 1, 1])
        dropout = kwargs.get("dropout", 0.0)

        # input 145, output 143
        self.input_layer = nn.Sequential(
            nn.Conv3d(num_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        # input 143, output 143
        self.block1 = ResidualStack(32, num_blocks=num_blocks[0],
                                    bottleneck=True, dropout=dropout)
        # input 143, output 71
        self.ds1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        # input 71, output 71
        self.block2 = ResidualStack(64, num_blocks=num_blocks[1],
                                    bottleneck=True, dropout=dropout)
        # input 71, output 35
        self.ds2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        # input 35, output 35
        self.block3 = ResidualStack(128, num_blocks=num_blocks[2],
                                    bottleneck=True, dropout=dropout)
        # input 35, output 17
        self.ds3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        # input 17, output 17
        self.block4 = ResidualStack(256, num_blocks=num_blocks[3],
                                    bottleneck=True, dropout=dropout)
        # input 17, output 8
        self.ds4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )

        # input 8, output 8
        self.block5 = ResidualStack(512, num_blocks=num_blocks[4],
                                    bottleneck=True, dropout=dropout)

        self.ds5 = nn.Sequential(
            # input 8, output 6
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # input 6, output 4
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # input 4, output 2
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )

        self.classification_dropout = nn.Dropout(dropout)

        classification_layers = [
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            nn.Linear(2*2*2*512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        ]

        self.classify = nn.Sequential(*classification_layers)

    def forward(self, x):
        l1 = self.input_layer(x)

        l2 = self.ds1(self.block1(l1))
        l3 = self.ds2(self.block2(l2))
        l4 = self.ds3(self.block3(l3))
        l5 = self.ds4(self.block4(l4))
        final_conv = self.ds5(self.block5(l5))

        # pooled = self.global_pool(final_conv)

        flattened = final_conv.view(len(x), -1)
        dropped = self.classification_dropout(flattened)

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
        for params in self.input_layer.params():
            params.requires_grad = False

        self.block1.freeze()
        for params in self.ds1.params():
            params.requires_grad = False

        self.block2.freeze()
        for params in self.ds2.params():
            params.requires_grad = False

        self.block3.freeze()
        for params in self.ds3.params():
            params.requires_grad = False

        self.block4.freeze()
        for params in self.ds4.params():
            params.requires_grad = False

        self.block5.freeze()
        for params in self.ds5.params():
            params.requires_grad = False

class ResidualStack(nn.Module):
    '''A stack of residual blocks.
    '''
    def __init__(self, num_chan, num_blocks, **kwargs):
        super().__init__()

        layers = [ResidualBlock(num_chan, **kwargs) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        result = x

        for layer in self.blocks.children():
            result = layer.forward(result, x)

        return result

    def backward(self, x):
        result = x

        for layer in reversed(self.blocks.children()):
            result = layer.backward(result)

        return result

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

    def forward(self, x, prev_state=None):
        dropped = self.dropout(x)

        prev_state = prev_state if prev_state is not None else x

        l1 = self.conv1(F.relu(self.bn1(x)))
        l2 = self.conv2(F.relu(self.bn2(l1)))
        l3 = self.conv3(F.relu(self.bn3(l2)))

        amount_to_pad = (l1.shape[-1] - l3.shape[-1]) // 2
        # times six because there are six sides to pad
        padding = (amount_to_pad, ) * 6

        return F.pad(l3, padding, value=0) + prev_state

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

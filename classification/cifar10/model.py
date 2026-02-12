import torch
import torch.nn as nn
from equiv_blocks import conv2D, activation, get_norm_layer, ResidualConnection, InvConv2d_avg, InvConv2d_telescopic, AffineLinear_telescopic, AffineLinear_avg


class ResidualBlock(nn.Module):
    """
    A unified residual block that adapts its layers based on the `mode`.
    It uses factory functions from `equiv_blocks` to construct its layers.
    """
    def __init__(self, in_channels, out_channels, stride=1, mode='standard', dropout_rate=0.0, norm_type=None):

        super(ResidualBlock, self).__init__()
        self.mode = mode
        self.act = activation(mode)
        self.residual_connection = ResidualConnection(mode)

        # Main path
        self.conv1 = conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, mode=mode)
        self.bn1 = get_norm_layer(norm_type, out_channels, mode=mode)
        self.dropout1 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.conv2 = conv2D(out_channels, out_channels, kernel_size=3, padding=1, mode=mode)
        self.bn2 = get_norm_layer(norm_type, out_channels, mode=mode)
        self.dropout2 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Shortcut connection
        self.shortcut_seq = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            shortcut_conv = conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, mode=mode)
            shortcut_bn = get_norm_layer(norm_type, out_channels, mode=mode)
            self.shortcut_seq = nn.Sequential(shortcut_conv, shortcut_bn)

    def forward(self, x):
        shortcut_processed = self.shortcut_seq(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out = self.residual_connection(shortcut_processed, out)
        out = self.act(out)
        return out

class ResNet(nn.Module):
    """
    A unified ResNet model, adaptable for CIFAR-10 (ResNet-20 style).
    """
    def __init__(self, num_classes=10, mode='standard', in_channels=3, norm_type=None, dropout_rate=0.0, dataset='cifar10'):
        super(ResNet, self).__init__()
        self.mode = mode
        self.norm_type = norm_type
        self.dropout_rate = dropout_rate
        self.dataset = dataset

        if dataset == 'cifar10':
            # Configuration for CIFAR-10 (ResNet-20 style)
            block_config = [3, 3, 3]
            self.channels = [64, 128, 256]
            self.in_channels = 64

            # Determine the normalization type for the first layer, especially for 'mix' mode.
            initial_norm_type = self.norm_type
            if self.norm_type == 'mix':
                initial_norm_type = 'instance'

            # The first convolution is made invariant for SEqSI modes for a fair comparison.
            if self.mode == 'seqsi_avg':
                self.conv1 = InvConv2d_avg(in_channels, 64, kernel_size=3, stride=1, padding=1)
            elif self.mode == 'seqsi_telescopic':
                self.conv1 = InvConv2d_telescopic(in_channels, 64, kernel_size=3, stride=1, padding=1)
            else:
                self.conv1 = conv2D(in_channels, 64, kernel_size=3, stride=1, padding=1, mode=self.mode)
            self.bn1 = get_norm_layer(initial_norm_type, 64, mode=self.mode)
        
        else:
            raise ValueError(f"Dataset '{dataset}' not supported in ResNet model.")

        self.act = activation(self.mode)

        # Residual blocks
        self.layer1 = self._make_layer(self.channels[0], block_config[0], stride=1)
        self.layer2 = self._make_layer(self.channels[1], block_config[1], stride=2)
        self.layer3 = self._make_layer(self.channels[2], block_config[2], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        final_features = self.channels[-1]
        if mode == 'affeq_telescopic' : 
            self.fc = AffineLinear_telescopic(final_features, num_classes)
        elif mode == 'affeq_avg':
            self.fc = AffineLinear_avg(final_features, num_classes)
        else:
            # For standard models, a bias is used in the final layer.
            # For SE, SEqSI, the final layer must be bias-free to respect scale-equivariance.
            use_bias_fc = (mode == 'standard')
            self.fc = nn.Linear(final_features, num_classes, bias=use_bias_fc)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        # Logic for the 'mix' normalization mode
        current_norm_type = self.norm_type
        if self.norm_type == 'mix':
            # The first block (layer1, out_channels=64) uses InstanceNorm.
            # Subsequent blocks use BatchNorm.
            if out_channels == 64:
                current_norm_type = 'instance'
            else:
                current_norm_type = 'batch'

        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s, mode=self.mode, dropout_rate=self.dropout_rate, norm_type=current_norm_type))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
import torch
import torch.nn as nn
from torch.nn import functional as F

# This import is only required for 'seqsi' mode.
# The file should be present if using a non-standard mode.
try:
    from models.equiv_blocks import InvConv3d
    EQUIV_BLOCKS_AVAILABLE = True
except ImportError:
    EQUIV_BLOCKS_AVAILABLE = False


class ResidualBlock3D(nn.Module):
    """
    A unified 3D residual block that adapts based on the provided `mode`.
    For 'standard' mode, it's a conventional ResNet block with BatchNorm.
    For other modes (e.g., 'seqsi'), BatchNorm is replaced with an Identity layer,
    as normalization is handled differently in those architectures.
    """
    def __init__(self, in_channels, out_channels, stride=1, mode='standard'):
        super().__init__()
        self.mode = mode
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # Main path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # BatchNorm is only used in standard CNN mode.
        self.bn1 = nn.BatchNorm3d(out_channels) if mode == 'standard' else nn.Identity()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels) if mode == 'standard' else nn.Identity()

        # Shortcut connection to match dimensions if stride is not 1 or channels differ.
        self.shortcut_seq = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            shortcut_bn = nn.BatchNorm3d(out_channels) if mode == 'standard' else nn.Identity()
            self.shortcut_seq = nn.Sequential(shortcut_conv, shortcut_bn)

    def forward(self, x):
        shortcut_processed = self.shortcut_seq(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = shortcut_processed + out
        out = self.act(out)
        return out


class ResNet3D(nn.Module):
    """
    A 3D ResNet model, adaptable with different modes, inspired by a ResNet-18 architecture.
    The `mode` parameter allows switching between a standard CNN and other variants
    like SEqSI, which may use specialized equivariant layers.
    """
    def __init__(self, in_channels=1, num_classes=6, block_config=None, mode='standard'):
        super().__init__()
        if mode != 'standard' and not EQUIV_BLOCKS_AVAILABLE:
            raise ImportError(f"Mode '{mode}' requires 'models/equiv_blocks.py', which was not found.")
        self.mode = mode

        if block_config is None:
            block_config = [2, 2, 2, 2]
        
        self.channels = [64, 128, 256, 512]
        self.in_channels_current = 64

        # Input layer
        # For 'seqsi' mode, an shift invariant convolution is used as the first layer.
        if self.mode == 'seqsi':
            self.conv1 = InvConv3d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # BatchNorm is only applied in standard mode.
        self.bn1 = nn.BatchNorm3d(64) if mode == 'standard' else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(self.channels[0], block_config[0], stride=1)
        self.layer2 = self._make_layer(self.channels[1], block_config[1], stride=2)
        self.layer3 = self._make_layer(self.channels[2], block_config[2], stride=2)
        self.layer4 = self._make_layer(self.channels[3], block_config[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        final_features = self.channels[3]

        # Bias is typically disabled in equivariant networks' final layers.
        use_bias_fc = self.mode == 'standard'
        self.fc = nn.Linear(final_features, num_classes, bias=use_bias_fc)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock3D(self.in_channels_current, out_channels, stride=s, mode=self.mode))
            self.in_channels_current = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

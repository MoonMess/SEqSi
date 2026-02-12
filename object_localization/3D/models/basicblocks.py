import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_shuffle_3d(tensor, scale):

    batch_size, channels, in_depth, in_height, in_width = tensor.size()
    nOut = channels // scale ** 3

    out_depth = in_depth * scale
    out_height = in_height * scale
    out_width = in_width * scale

    input_view = tensor.contiguous().view(batch_size, nOut, scale, scale, scale, in_depth, in_height, in_width)

    output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

    return output.view(batch_size, nOut, out_depth, out_height, out_width)


class AffineConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode="reflect"):
        super().__init__(in_channels, out_channels, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, 
                                           groups=groups, padding_mode=padding_mode, bias=False)
        
    def affine(self, w):
        """ returns new kernels that encode affine combinations """
        return w.view(self.out_channels, -1).roll(1, 1).view(w.size()) - w + 1 / w[0, ...].numel()
    
    def forward(self, x):
        kernel = self.affine(self.weight)
        padding = tuple(elt for elt in reversed(self.padding) for _ in range(2)) # used to translate padding arg used by Conv module to the ones used by F.pad
        #print(padding)
        padding_mode = self.padding_mode if self.padding_mode != 'zeros' else 'constant' # used to translate padding_mode arg used by Conv module to the ones used by F.pad
        return F.conv3d(F.pad(x, padding, mode=padding_mode), kernel, stride=self.stride, dilation=self.dilation, groups=self.groups)


class InvConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode="reflect"):
        super().__init__(in_channels, out_channels, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation, 
                                           groups=groups, padding_mode=padding_mode, bias=False)
        
    def affine(self, w):
        """ returns new kernels that encode affine combinations """
        return w.view(self.out_channels, -1).roll(1, 1).view(w.size()) - w
    
    def forward(self, x):
        kernel = self.affine(self.weight)
        padding = tuple(elt for elt in reversed(self.padding) for _ in range(2)) # used to translate padding arg used by Conv module to the ones used by F.pad
        #print(padding)
        padding_mode = self.padding_mode if self.padding_mode != 'zeros' else 'constant' # used to translate padding_mode arg used by Conv module to the ones used by F.pad
        return F.conv3d(F.pad(x, padding, mode=padding_mode), kernel, stride=self.stride, dilation=self.dilation, groups=self.groups)


class AffineConvTranspose3d(nn.Module):
    """ Affine ConvTranspose2d with kernel=2 and stride=2, implemented using PixelShuffle """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = AffineConv3d(in_channels, 8*out_channels, 1)
        
    def forward(self, x):
        return pixel_shuffle_3d(self.conv1x1(x), 2)
    
class SortPool(nn.Module):
    """ Channel-wise sort pooling, C must be an even number """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # A trick with relu is used because the derivative for torch.aminmax is not yet implemented and torch.sort is slow.
        N, C, H, W, D = x.size()
        x1, x2 = torch.split(x.view(N, C//2, 2, H, W, D), 1, dim=2)
        diff = F.relu(x1 - x2, inplace=True)
        return torch.cat((x1-diff, x2+diff), dim=2).view(N, C, H, W, D)
    
class ResidualConnection(nn.Module):
    """ Residual connection """
    def __init__(self, mode='ordinary'):
        super().__init__()

        self.mode = mode
        if mode=='norm-equiv':
            self.alpha = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, x, y):
        if self.mode=='norm-equiv': 
            return self.alpha * x + (1 - self.alpha) * y
        return x + y

# the head function is not implemented in the classic implementation, only in the scale-eq+shift-inv    
def head(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', mode='ordinary'):
    if mode == "ordinary" or mode == "norm-equiv" or mode == "scale-equiv":
        return conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, mode=mode)
    elif mode == "seqsi":
        return InvConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode='reflect')
    else:
        raise NotImplementedError("Only ordinary, scale-equiv and norm-equiv modes are implemented")

def conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', mode='ordinary'):
    if mode=='ordinary' or mode=='scale-equiv' or mode == "seqsi":
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias if mode=='ordinary' else False, padding_mode=padding_mode)
    elif mode=='norm-equiv':
        return AffineConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode='reflect')
    else:
        raise NotImplementedError("Only ordinary, scale-equiv and norm-equiv modes are implemented")
    
def upscale3d(in_channels, out_channels, bias=True, mode='ordinary'):
    """ Upscaling using convtranspose with kernel 2x2 and stride 2"""
    if mode=='ordinary' or mode=='scale-equiv' or mode == "seqsi":
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=bias if mode=='ordinary' else False)
    elif mode=='norm-equiv':
        return AffineConvTranspose3d(in_channels, out_channels)
    else:
        raise NotImplementedError("Only ordinary, scale-equiv and norm-equiv modes are implemented")
    
def activation(mode='ordinary'):
    if mode=='ordinary' or mode=='scale-equiv' or mode=="seqsi":
        return nn.ReLU(inplace=True)
    elif mode=='norm-equiv':
        return SortPool()
    else:
        raise NotImplementedError("Only ordinary, scale-equiv and norm-equiv modes are implemented")
    
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False, mode="ordinary"):
        super().__init__()

        self.m_res = nn.Sequential(conv3d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias, mode=mode),
                                activation(mode),
                                conv3d(in_channels, out_channels, 3, stride=1, padding=1, bias=bias, mode=mode))

        self.sum = ResidualConnection(mode)
        self.final = activation(mode)
        
    def forward(self, x):
        return self.final(self.sum(x, self.m_res(x)))
    
class InvResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False, mode="ordinary"):
        super().__init__()

        self.m_res = nn.Sequential(conv3d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias, mode="ordinary"),
                                activation(mode),
                                conv3d(in_channels, out_channels, 3, stride=1, padding=1, bias=bias, mode="ordinary"))

        self.sum = ResidualConnection("ordinary")
        self.final = activation(mode)
        
    def forward(self, x):
        return self.final(self.sum(x, self.m_res(x)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basicblocks import *


class EquivUNet3D(nn.Module):

    def __init__(self, in_nc=1, out_nc=1, nc=[32, 64, 128, 256], nb=3, mode='norm-equiv'):
        super().__init__()

        bias = mode == 'ordinary'

        self.m_head = head(in_nc, nc[0], 3, stride=1, padding=1, bias=bias, mode=mode) # conv3d(in_nc, nc[0], 3, stride=1, padding=1, bias=bias, mode=mode) # 
        
        self.m_down = nn.ModuleList([nn.Sequential(
            *[ResBlock(nc[i], nc[i], bias=bias, mode=mode) for _ in range(nb)],
            conv3d(nc[i], nc[i+1], 2, stride=2, padding=0, bias=bias, mode=mode))
            for i in range(len(nc)-1)])

        self.m_body = nn.Sequential(*[ResBlock(nc[-1], nc[-1], bias=bias, mode=mode) for _ in range(nb)])

        self.m_up = nn.ModuleList([nn.Sequential(
            upscale3d(nc[i], nc[i-1], bias=bias, mode=mode),
            *[ResBlock(nc[i-1], nc[i-1], bias=bias, mode=mode) for _ in range(nb)])
            for i in range(len(nc)-1, 0, -1)])

        self.m_tail = conv3d(nc[0], out_nc, 3, stride=1, padding=1, bias=bias, mode=mode)

        self.res = nn.ModuleList([ResidualConnection(mode) for _ in range(len(nc))])

    def forward(self, x):
        # Size handling (h and w must divisible by v)
        _, _, h, w, d = x.size()
        scale = len(self.m_down)
        v = 2**scale
        r1, r2, r3 = h % v, w % v, d % v
        x = F.pad(x, pad=(0,v - r3 if r3 > 0  else 0,0, v-r2 if r2 > 0 else 0, 0, v-r1 if r1 > 0 else 0), mode='constant', value=float(x.mean()))
        

        layers = [self.m_head(x)]
        for i in range(scale):
            layers.append(self.m_down[i](layers[-1]))
            
        x = self.m_body(layers[-1])
        for i in range(scale):
            x = self.m_up[i](self.res[i](x, layers[-(1+i)]))
        x = self.m_tail(self.res[-1](x, layers[0]))
        
        return x[..., :h, :w, :d]
    


if __name__ == "__main__":

    unet = EquivUNet3D(mode = "norm-equiv").to(dtype=torch.float64)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Number of parameters : {total_params}")
    x = torch.rand((1,1,60,60,60), dtype=torch.float64)

    a, b = 3, 1

    with torch.no_grad():
        y = unet(x)
        z = unet(a*x+b)
    print(y.shape)

    test = (z - (a*y+b)).abs()

    print(test.max())
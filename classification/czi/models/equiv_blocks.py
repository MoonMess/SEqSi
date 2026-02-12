# /path/to/your/project/models/equiv_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

class InvConv3d(nn.Conv3d):
    """
    Shift-Invariant 3D Convolution Layer.

    This layer implements a convolution that is invariant to additive shifts in
    the input signal's intensity (e.g., f(x + c) = f(x)). This is achieved by
    enforcing a zero-sum constraint on the convolutional kernels.
    Bias is always set to False. This is a key component of SEqSI networks.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int], str] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 padding_mode: str = "reflect"):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=False,
                         padding_mode=padding_mode)

        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            self._reversed_padding_repeated_twice = []
            for p_val in reversed(self.padding):
                self._reversed_padding_repeated_twice.extend([p_val, p_val])

    def _zero_sum_kernel(self, w: torch.Tensor) -> torch.Tensor:
        """Modifies kernels to have a zero sum for each output channel."""
        w_flat = w.view(self.out_channels, -1)
        kernel_mean = torch.mean(w_flat, dim=1, keepdim=True)
        affine_w_flat = w_flat - kernel_mean
        return affine_w_flat.view_as(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero_kernel = self._zero_sum_kernel(self.weight)
        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            x_padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv3d(x_padded, zero_kernel, bias=None, stride=self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        else:
            return F.conv3d(x, zero_kernel, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)


class EquivConv3d(nn.Conv3d):
    """
    Affine-Equivariant 3D Convolution Layer.

    This layer implements a convolution that is equivariant to affine
    transformations of the input signal's intensity (e.g., f(a*x + b) = a*f(x) + b).
    This is achieved by constraining the convolutional kernels to sum to one.
    Bias is always set to False.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int], str] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 padding_mode: str = "reflect"):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=False,
                         padding_mode=padding_mode)

        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            self._reversed_padding_repeated_twice = []
            for p_val in reversed(self.padding):
                self._reversed_padding_repeated_twice.extend([p_val, p_val])

    def _one_sum_kernel(self, w: torch.Tensor) -> torch.Tensor:
        """Modifies kernels to have a sum of one for each output channel."""
        num_elements_per_kernel = w[0, ...].numel()
        w_flat = w.view(self.out_channels, -1)
        kernel_mean = torch.mean(w_flat, dim=1, keepdim=True)
        # This operation ensures the kernel weights sum to 1, making it affine-equivariant.
        affine_w_flat = w_flat - kernel_mean + (1.0 / num_elements_per_kernel)
        return affine_w_flat.view_as(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one_sum_kernel = self._one_sum_kernel(self.weight)
        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            x_padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv3d(x_padded, one_sum_kernel, bias=None, stride=self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        else:
            return F.conv3d(x, one_sum_kernel, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)

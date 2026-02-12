import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
import warnings

# --- Activation and Pooling ---

class SortPool(nn.Module):
    """
    Channel-wise sort pooling. The channel dimension C must be an even number.
    This is a key component for affine equivariance.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # A trick with relu is used for a fast, differentiable sort operation on pairs of channels.
        # This is faster than torch.sort and has a defined derivative, unlike torch.min/max.
        N, C, H, W = x.size()
        x1, x2 = torch.split(x.view(N, C//2, 2, H, W), 1, dim=2)
        diff = F.relu(x1 - x2, inplace=True)
        return torch.cat((x1-diff, x2+diff), dim=2).view(N, C, H, W)

def activation(mode: str = 'standard', nonlin: str = 'relu') -> nn.Module:
    """Returns the appropriate activation layer based on the model mode."""
    if mode in ['standard', 'normalized', 'seqsi_avg', 'se', 'seqsi_telescopic']:
        if nonlin == 'relu':
            return nn.ReLU(inplace=True)
        elif nonlin == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            raise NotImplementedError(f"Nonlinearity '{nonlin}' not implemented for mode '{mode}'.")
    elif mode in ['affeq_telescopic', 'affeq_avg']:
        return SortPool()
    else:
        raise NotImplementedError(f"Mode '{mode}' not implemented in activation factory.")

# --- Normalization Layer ---

def get_norm_layer(norm_type: Optional[str], num_features: int, mode: str = None) -> nn.Module:
    """
    Creates a 2D normalization layer.
    """
    if norm_type is None:
        return nn.Identity()
    elif norm_type == 'batch':
        # track_running_stats=True is the standard for BatchNorm to use accumulated stats during evaluation.
        return nn.BatchNorm2d(num_features)
    elif norm_type == 'instance':
        # affine=False: No learnable scale/shift. track_running_stats=False is standard for InstanceNorm.
        return nn.InstanceNorm2d(num_features)
    elif norm_type == 'layer':
        # GroupNorm(1, ...) is often used as a LayerNorm equivalent for Conv layers.
        return nn.GroupNorm(1, num_features)
    elif norm_type in ['group']:
        if num_features == 0: # Avoid division by zero
            print("Warning: num_features is 0, returning Identity layer for GroupNorm.")
            return nn.Identity()
        # Heuristic to find a suitable number of groups.
        num_groups_initial = 32 # Default desired number of groups
        num_groups = num_groups_initial

        if num_features % num_groups != 0:
            num_groups = 1 
            for i in range(5, 0, -1): # Powers of 2: 32, 16, 8, 4, 2
                pot_group = 2**i
                if num_features % pot_group == 0:
                    num_groups = pot_group
                    break
            if num_groups_initial != num_groups : 
                 warnings.warn(
                    f"num_features {num_features} not divisible by {num_groups_initial}. "
                    f"Using {num_groups} groups for GroupNorm instead.", UserWarning)
        
        if num_features % num_groups != 0: # Should ideally not happen if logic above is sound
            warnings.warn(
                f"num_features {num_features} is not divisible by the chosen num_groups {num_groups}. "
                f"Defaulting num_groups to 1.", UserWarning)
            num_groups = 1
        use_affine = (norm_type == 'groupAff')
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features, affine=use_affine)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

# --- Convolutions ---

class InvConv2d_avg(nn.Conv2d):
    """
    2D Invariant Convolution Layer (bias=False) using average subtraction.
    The kernel weights are modified to sum to zero per output channel by subtracting the mean.
    This corresponds to the 'seqsi_avg' mode.
    Padding is handled manually before convolution to support modes like 'reflect'.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int], str] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 padding_mode: str = "reflect"):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=False, padding_mode=padding_mode) 
        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            self._reversed_padding_repeated_twice = []
            for p_val in reversed(self.padding):
                self._reversed_padding_repeated_twice.extend([p_val, p_val])
            
    def _zero_kernel_avg(self, w: torch.Tensor) -> torch.Tensor:
        """Computes the zero-sum kernel by subtracting the per-channel mean."""
        num_elements_per_kernel = w[0, ...].numel()
        if num_elements_per_kernel == 0: return w

        # View as (out_channels, num_elements_in_kernel)
        w_flat = w.view(self.out_channels, -1)
        kernel_mean = torch.mean(w_flat, dim=1, keepdim=True)
        zero_sum_w_flat = w_flat - kernel_mean
        return zero_sum_w_flat.view_as(w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the zero-sum kernel on the fly
        zero_kernel = self._zero_kernel_avg(self.weight)
  
        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            x_padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(x_padded, zero_kernel, bias=None, stride=self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        else:
            return F.conv2d(x, zero_kernel, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)

class InvConv2d_telescopic(nn.Conv2d):
    """
    2D Invariant Convolution Layer (bias=False) using telescopic subtraction.
    The kernel weights are modified to sum to zero per output channel using `w_rolled - w_flat`.
    This corresponds to the 'seqsi_telescopic' mode.
    Padding is handled manually.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int], str] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 padding_mode: str = "reflect"):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=False, padding_mode=padding_mode)
        
        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            self._reversed_padding_repeated_twice = []
            for p_val in reversed(self.padding):
                self._reversed_padding_repeated_twice.extend([p_val, p_val])

    def _zero_kernel_telescopic(self, w: torch.Tensor) -> torch.Tensor:
        """
        Calculates the zero-sum kernel using telescopic subtraction.

        NUMERICAL STABILITY NOTE:
        This method, while mathematically correct for achieving a zero sum, is
        numerically unstable. The `roll(w) - w` operation can amplify the
        magnitude (norm) of the weights, especially if they become large or
        alternate in sign during training. This can lead to exploding values.
        The mean subtraction method in `InvConv2d_avg` is numerically stable.
        """
        w_flat = w.view(self.out_channels, -1)
        w_rolled = torch.roll(w_flat, shifts=1, dims=1)
        zero_sum_w_flat = w_rolled - w_flat
        return zero_sum_w_flat.view_as(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero_kernel = self._zero_kernel_telescopic(self.weight)
  
        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            x_padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(x_padded, zero_kernel, bias=None, stride=self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        else:
            return F.conv2d(x, zero_kernel, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)


class AffineConv2d_telescopic(nn.Conv2d):
    """
    Affine 2D Convolution where kernel weights are constrained to sum to 1.
    This version uses telescopic subtraction, which is numerically unstable.
    Uses bias=False and padding_mode='reflect' by default.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int], str] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1, 
                 padding_mode: str = "reflect"): 
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=False, padding_mode=padding_mode) 

        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            self._reversed_padding_repeated_twice = []
            for p_val in reversed(self.padding):
                self._reversed_padding_repeated_twice.extend([p_val, p_val])

    def _affine_kernel(self, w: torch.Tensor) -> torch.Tensor:
        """
        Computes the affine kernel (weights sum to 1) via telescopic subtraction.
        See `InvConv2d_telescopic` for notes on numerical instability.
        """
        num_elements_per_kernel = w[0, ...].numel()
        if num_elements_per_kernel == 0: return w

        w_flat = w.view(self.out_channels, -1)
        w_rolled = torch.roll(w_flat, shifts=1, dims=1)
        affine_w_flat = w_rolled - w_flat + (1.0 / num_elements_per_kernel)
        return affine_w_flat.view_as(w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        affine_kernel = self._affine_kernel(self.weight)

        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            x_padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(x_padded, affine_kernel, bias=None, stride=self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        else:
            return F.conv2d(x, affine_kernel, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)


class AffineConv2d_avg(nn.Conv2d):
    """
    Affine 2D Convolution where kernel weights are constrained to sum to 1.
    This version uses mean subtraction, which is numerically stable.
    Uses bias=False and padding_mode='reflect' by default.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int], str] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1, 
                 padding_mode: str = "reflect"): 
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=False, padding_mode=padding_mode) 

        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            self._reversed_padding_repeated_twice = []
            for p_val in reversed(self.padding):
                self._reversed_padding_repeated_twice.extend([p_val, p_val])

    def _affine_kernel(self, w: torch.Tensor) -> torch.Tensor:
        """Computes the affine kernel (weights sum to 1) via mean subtraction."""
        num_elements_per_kernel = w[0, ...].numel()
        if num_elements_per_kernel == 0: return w

        w_flat = w.view(self.out_channels, -1)
        kernel_mean = torch.mean(w_flat, dim=1, keepdim=True)
        affine_w_flat = w_flat - kernel_mean + (1.0 / num_elements_per_kernel)
        return affine_w_flat.view_as(w)
    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        affine_kernel = self._affine_kernel(self.weight)

        if self.padding_mode != 'zeros' and any(p > 0 for p in self.padding):
            x_padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(x_padded, affine_kernel, bias=None, stride=self.stride,
                            padding=0, dilation=self.dilation, groups=self.groups)
        else:
            return F.conv2d(x, affine_kernel, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups)



def conv2D(in_channels: int, out_channels: int,
           kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[int, Tuple[int, int], str] = 0,
           dilation: Union[int, Tuple[int, int]] = 1,
           groups: int = 1, bias: bool = True,
           padding_mode: str = 'zeros', mode: str = 'standard') -> nn.Module:
    """Factory function for creating a 2D convolution layer based on the mode."""

    if mode in ['standard', 'se', 'seqsi_avg', 'seqsi_telescopic']:
        # Standard PyTorch convolution. Bias is only used for the 'standard' mode.
        use_bias = bias and (mode == 'standard')
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=use_bias, padding_mode=padding_mode)
    elif "affeq_telescopic" == mode :
        return AffineConv2d_telescopic(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups,
                            padding_mode='reflect')
    elif "affeq_avg" == mode :
        return AffineConv2d_avg(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups,
                            padding_mode='reflect')
    else:
        raise NotImplementedError(f"Mode '{mode}' not implemented for conv2D factory.")

# --- Linear Layers ---

class AffineLinear_avg(nn.Linear):
    """
    A linear layer where weights are constrained to sum to 1 for each output neuron.
    This is the fully-connected counterpart to `AffineConv2d_avg`.
    The constraint is applied dynamically on each forward pass. Bias is disabled.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        # Make weights sum to zero by subtracting the mean (numerically stable).
        w_mean = torch.mean(w, dim=1, keepdim=True)
        w_zero_sum = w - w_mean

        # Add a constant to make the weights sum to 1.
        num_in_features = w.size(1)
        w_sum_one = w_zero_sum + (1.0 / num_in_features)
        
        return F.linear(x, w_sum_one, bias=None)
    
class AffineLinear_telescopic(nn.Linear):
    """
    A linear layer where weights are constrained to sum to 1.
    This version uses telescopic subtraction, which is numerically unstable.
    Bias is disabled.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        # Make weights sum to zero using telescopic subtraction.
        w_rolled = torch.roll(w, shifts=1, dims=1)
        w_zero_sum = w_rolled - w
        
        # Add a constant to make the weights sum to 1.
        num_in_features = w.size(1)
        w_sum_one = w_zero_sum + (1.0 / num_in_features)
        
        return F.linear(x, w_sum_one, bias=None)

# --- Residual Connection ---

class ResidualConnection(nn.Module):
    """
    Implements the residual connection for a Residual Block.
    For affine equivariant modes ('affeq_...'), it uses a learnable weighted sum.
    For other modes, it uses a simple addition.
    """
    def __init__(self, mode: str = 'standard'):
        super().__init__()
        self.mode = mode
        self.use_weighted_sum = mode in ['affeq_telescopic', 'affeq_avg']
        if self.use_weighted_sum:
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.use_weighted_sum:
            # x is the shortcut connection, residual is the main path output.
            alpha_val = torch.sigmoid(self.alpha) # Constrain alpha to be between 0 and 1.
            return (1.0 - alpha_val) * residual + alpha_val * x
        else:
            # Simple addition for 'standard', 'normalized', 'se' modes.
            return residual + x

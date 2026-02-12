import torch
import math

class _BaseLinearTransform(object):
    """
    Base class to mark transformations that are purely linear and should not be clamped.
    This is used to identify transformations for which equivariance is expected.
    """
    pass

# --- Refactored Transformation Blocks ---

# Base class for transformations where params can be drawn once (init) or per-call.
class _PerCallOrInit(object):
    def __init__(self, *args, **kwargs):
        # Subclasses should call super().__init__ and then self._draw_params() if they are 'init' type.
        pass
    
    def _draw_params(self):
        raise NotImplementedError

    def _apply_transform(self, img_tensor):
        raise NotImplementedError

    def __call__(self, img_tensor):
        # 'Per-image' subclasses will call _draw_params here.
        return self._apply_transform(img_tensor)

# --- Spatially-Varying Linear Transformation ---
class _RandomScaleShiftLinearBase(_BaseLinearTransform, _PerCallOrInit):
    """Base class for spatially-varying linear scale and shift."""
    def __init__(self, scale_range=(0.5, 1.5), shift_range=(-0.5, 0.5)):
        super().__init__()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.scale0, self.scale1, self.shift0, self.shift1, self.angle = [None] * 5

    def _draw_params(self):
        self.scale0, self.scale1 = torch.empty(2).uniform_(*self.scale_range)
        self.shift0, self.shift1 = torch.empty(2).uniform_(*self.shift_range)
        self.angle = torch.empty(1).uniform_(0, 2 * math.pi).item()

    def _apply_transform(self, img_tensor):
        C, H, W = img_tensor.shape
        dx, dy = math.cos(self.angle), math.sin(self.angle)
        x_coords = torch.linspace(-1, 1, W, device=img_tensor.device)
        y_coords = torch.linspace(-1, 1, H, device=img_tensor.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        projection = dx * grid_x + dy * grid_y
        min_proj, max_proj = projection.min(), projection.max()
        if max_proj - min_proj > 1e-6:
            normalized_projection = (projection - min_proj) / (max_proj - min_proj) * 2 - 1
        else:
            normalized_projection = torch.zeros_like(projection)
        interp_factor = (normalized_projection + 1) / 2
        
        scale0, scale1 = self.scale0.to(img_tensor.device), self.scale1.to(img_tensor.device)
        shift0, shift1 = self.shift0.to(img_tensor.device), self.shift1.to(img_tensor.device)

        L_scale = scale0 + (scale1 - scale0) * interp_factor
        L_shift = shift0 + (shift1 - shift0) * interp_factor
        
        L_scale = L_scale.unsqueeze(0).expand_as(img_tensor)
        L_shift = L_shift.unsqueeze(0).expand_as(img_tensor)
        return img_tensor * L_scale + L_shift

class RandomScaleShiftLinear(_RandomScaleShiftLinearBase):
    """Applies a PURELY linear, spatially-varying scale and shift. Params are fixed at init."""
    def __init__(self, scale_range=(0.5, 1.5), shift_range=(-0.5, 0.5)):
        super().__init__(scale_range, shift_range)
        self._draw_params()

class RandomScaleShiftLinearPerImage(_RandomScaleShiftLinearBase):
    """Applies a PURELY linear, spatially-varying scale and shift. Params are drawn per-image."""
    def __call__(self, img_tensor):
        self._draw_params()
        return self._apply_transform(img_tensor)

# --- Global Saturated Scale and Shift ---
class _RandomScaleShiftSaturatedBase(_PerCallOrInit):
    """Base class for constant scale and shift with saturation."""
    def __init__(self, scale_range=(0.05, 1.0), shift_range=(-0.3, 0.3)):
        super().__init__()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.saturation_shift = 0.5
        self.scale, self.shift = None, None

    def _draw_params(self):
        self.scale = torch.empty(1).uniform_(*self.scale_range).item()
        self.shift = torch.empty(1).uniform_(*self.shift_range).item()

    def _apply_transform(self, img_tensor):
        return torch.clamp(self.scale * img_tensor + self.shift + self.saturation_shift, 0.0, 1.0)

class RandomScaleShiftSaturated(_RandomScaleShiftSaturatedBase):
    """Applies a constant scale/shift with saturation. Params are fixed at init."""
    def __init__(self, scale_range=(0.05, 1.0), shift_range=(-0.3, 0.3)):
        super().__init__(scale_range, shift_range)
        self._draw_params()

class RandomScaleShiftSaturatedPerImage(_RandomScaleShiftSaturatedBase):
    """Applies a constant scale/shift with saturation. Params are drawn per-image."""
    def __init__(self, scale_range=(0.7, 1.3), shift_range=(-0.3, 0.3)):
        super().__init__(scale_range, shift_range)

    def __call__(self, img_tensor):
        self._draw_params()
        return self._apply_transform(img_tensor)

# --- Global Saturated Scale ---
class _RandomScaleSaturatedBase(_PerCallOrInit):
    """Base class for random scale with clamping."""
    def __init__(self, scale_range=(0.05, 1.0)):
        super().__init__()
        self.scale_range = scale_range
        self.scale = None

    def _draw_params(self):
        self.scale = torch.empty(1).uniform_(*self.scale_range).item()

    def _apply_transform(self, img_tensor):
        return torch.clamp(self.scale * img_tensor, 0.0, 1.0)

class RandomScaleSaturated(_RandomScaleSaturatedBase):
    """Applies a random scale with clamping. Params are fixed at init."""
    def __init__(self, scale_range=(0.05, 1.0)):
        super().__init__(scale_range)
        self._draw_params()

class RandomScaleSaturatedPerImage(_RandomScaleSaturatedBase):
    """Applies a random scale with clamping. Params are drawn per-image."""
    def __call__(self, img_tensor):
        self._draw_params()
        return self._apply_transform(img_tensor)

# --- Global Saturated Shift ---
class _RandomShiftSaturatedBase(_PerCallOrInit):
    """Base class for random shift with clamping."""
    def __init__(self, shift_range=(-0.5, 0.5)):
        super().__init__()
        self.shift_range = shift_range
        self.shift = None

    def _draw_params(self):
        self.shift = torch.empty(1).uniform_(*self.shift_range).item()

    def _apply_transform(self, img_tensor):
        return torch.clamp(img_tensor + self.shift, 0.0, 1.0)

class RandomShiftSaturated(_RandomShiftSaturatedBase):
    """Applies a random shift with clamping. Params are fixed at init."""
    def __init__(self, shift_range=(-0.5, 0.5)):
        super().__init__(shift_range)
        self._draw_params()

class RandomShiftSaturatedPerImage(_RandomShiftSaturatedBase):
    """Applies a random shift with clamping. Params are drawn per-image."""
    def __init__(self, shift_range=(-0.8, 0.8)): # Note: original had different range
        super().__init__(shift_range)
    def __call__(self, img_tensor):
        self._draw_params()
        return self._apply_transform(img_tensor)

# --- Global Linear Scale ---
class _RandomScaleBase(_BaseLinearTransform, _PerCallOrInit):
    """Base class for pure linear scale."""
    def __init__(self, scale_range=(0.1, 10.0)):
        super().__init__()
        self.scale_range = scale_range
        self.scale = None

    def _draw_params(self):
        self.scale = torch.empty(1).uniform_(*self.scale_range).item()

    def _apply_transform(self, img_tensor):
        return self.scale * img_tensor

class RandomScale(_RandomScaleBase):
    """Applies a pure linear scale. Params are fixed at init."""
    def __init__(self, scale_range=(0.1, 10.0)):
        super().__init__(scale_range)
        self._draw_params()

class RandomScalePerImage(_RandomScaleBase):
    """Applies a pure linear scale. Params are drawn per-image."""
    def __call__(self, img_tensor):
        self._draw_params()
        return self._apply_transform(img_tensor)

# --- Global Linear Shift ---
class _RandomShiftBase(_BaseLinearTransform, _PerCallOrInit):
    """Base class for pure linear shift."""
    def __init__(self, shift_range=(-0.5, 0.5)):
        super().__init__()
        self.shift_range = shift_range
        self.shift = None

    def _draw_params(self):
        self.shift = torch.empty(1).uniform_(*self.shift_range).item()

    def _apply_transform(self, img_tensor):
        return img_tensor + self.shift

class RandomShift(_RandomShiftBase):
    """Applies a pure linear shift. Params are fixed at init."""
    def __init__(self, shift_range=(-0.5, 0.5)):
        super().__init__(shift_range)
        self._draw_params()

class RandomShiftPerImage(_RandomShiftBase):
    """Applies a pure linear shift. Params are drawn per-image."""
    def __call__(self, img_tensor):
        self._draw_params()
        return self._apply_transform(img_tensor)

# --- Global Linear Scale and Shift ---
class _RandomScaleShiftBase(_BaseLinearTransform, _PerCallOrInit):
    """Base class for global, pure linear scale and shift."""
    def __init__(self, scale_range=(0.5, 2.0), shift_range=(-0.3, 0.3)):
        super().__init__()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.scale, self.shift = None, None

    def _draw_params(self):
        self.scale = torch.empty(1).uniform_(*self.scale_range).item()
        self.shift = torch.empty(1).uniform_(*self.shift_range).item()

    def _apply_transform(self, img_tensor):
        return self.scale * img_tensor + self.shift

class RandomScaleShift(_RandomScaleShiftBase):
    """Applies a global, pure linear scale and shift. Params are fixed at init."""
    def __init__(self, scale_range=(0.5, 2.0), shift_range=(-0.3, 0.3)):
        super().__init__(scale_range, shift_range)
        self._draw_params()

class RandomScaleShiftPerImage(_RandomScaleShiftBase):
    """Applies a global, pure linear scale and shift. Params are drawn per-image."""
    def __call__(self, img_tensor):
        self._draw_params()
        return self._apply_transform(img_tensor)

# --- Other Photometric Transformations ---

class AdditiveGaussianNoise(object):
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, img_tensor):
        noise = torch.randn_like(img_tensor) * self.std
        return torch.clamp(img_tensor + noise, 0.0, 1.0)
    
class RandomAdditiveGaussianNoisePerImage(object):
    """
    Applies additive Gaussian noise with a random standard deviation (std) for each image.
    The std value is chosen from a range on each call.
    """
    def __init__(self, range=(0.0, 0.25)):
        self.std_range = range

    def __call__(self, img_tensor):
        std = torch.empty(1).uniform_(*self.std_range).item()
        noise = torch.randn_like(img_tensor) * std
        return torch.clamp(img_tensor + noise, 0.0, 1.0)

class GammaCorrection(object):
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, img_tensor):
        return torch.clamp(torch.pow(img_tensor, self.gamma), 0.0, 1.0)

class RandomGammaCorrectionPerImage(object):
    """
    Applies a random gamma correction to each image. The gamma value is
    chosen from a range on each call.
    """
    def __init__(self, range=(0.2, 5.0)):
        self.gamma_range = range

    def __call__(self, img_tensor):
        gamma = torch.empty(1).uniform_(*self.gamma_range).item()
        return torch.clamp(torch.pow(img_tensor, gamma), 0.0, 1.0)

# --- Deterministic Transformations (for visualization/specific tests) ---

class ScaleShift(object):
    def __init__(self, scale=1.0, shift=0.0):
        self.scale = scale
        self.shift = shift

    def __call__(self, img_tensor):
        perturbed_img = self.scale * img_tensor + self.shift
        return torch.clamp(perturbed_img, 0.0, 1.0)

class HighContrast(ScaleShift):
    def __init__(self):
        super().__init__(scale=2.0, shift=-0.5)

class LowContrast(ScaleShift):
    def __init__(self):
        super().__init__(scale=0.5, shift=0.25)

class BrightSaturate(ScaleShift):
    def __init__(self):
        super().__init__(scale=1.5, shift=0.5)

# --- Central Transform Registry ---
# 'saturated' means the output is clamped to [0, 1] after transformation.
TRANSFORM_REGISTRY = {
    'standard': {
        "RandomScaleShiftLinear": RandomScaleShiftLinear,
        "RandomScaleShiftSaturated": RandomScaleShiftSaturated,
        "RandomScaleShift": RandomScaleShift,
        "RandomScaleSaturated": RandomScaleSaturated,
        "RandomShiftSaturated": RandomShiftSaturated,
        "RandomScale": RandomScale,
        "RandomShift": RandomShift,
        "HighContrast": HighContrast,
        "LowContrast": LowContrast,
        "BrightSaturate": BrightSaturate,
        "AdditiveGaussianNoise": AdditiveGaussianNoise,
        "GammaCorrection": GammaCorrection,
    },
    'per_image': {
        "RandomScaleShiftLinear": RandomScaleShiftLinearPerImage,
        "RandomScaleShiftSaturated": RandomScaleShiftSaturatedPerImage,
        "RandomScaleSaturated": RandomScaleSaturatedPerImage,
        "RandomShiftSaturated": RandomShiftSaturatedPerImage,
        "RandomScaleShift": RandomScaleShiftPerImage,
        "RandomScale": RandomScalePerImage,
        "RandomShift": RandomShiftPerImage,
        "RandomGammaCorrection": RandomGammaCorrectionPerImage,
        "RandomAdditiveGaussianNoise": RandomAdditiveGaussianNoisePerImage,
        # Deterministic transforms are the same for both modes
        "HighContrast": HighContrast,
        "LowContrast": LowContrast,    }
}
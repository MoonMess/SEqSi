# -*- coding: utf-8 -*-

"""
Central configuration file for photometric transformations.

This file replaces the old `config.json` to allow for comments and
better maintainability.

The structure is a dictionary named `CONFIG` containing a "transformations" key.
"""

CONFIG = {
    "transformations": {
        # Group 1: Basic affine perturbations.
        # These transformations demonstrate the effects of additive (shift) and
        # multiplicative (scale) changes.
        "affine": {
            "Shift": {
                "type": "RandomShift", "params": { "shift_range": [-2.0, 2.0] }, "per_image_randomness": True
            }, 
  
            "Scale (<1)": {
                "type": "RandomScale", "params": {  "scale_range": [0.0, 1.0], }, "per_image_randomness": True
            },
            "Scale (>1)": {
                "type": "RandomScale", "params": {  "scale_range": [1.0, 4.0], }, "per_image_randomness": True
            },
            "Affine": {
                "type": "RandomScaleShift", "params": { "scale_range": [0.0, 4.0], "shift_range": [-2.0, 2.0] }, "per_image_randomness": True
            }
        },

        "non_affine": {
            "Shift Saturated": {
                "type": "RandomShiftSaturated", "params": { "shift_range": [-0.7, 0.7] }, "per_image_randomness": True
            },
            "Scale (>1) Saturated": {
                "type": "RandomScaleSaturated", "params": {  "scale_range": [1.0, 3.0], }, "per_image_randomness": True
            },
            "Affine Saturated": {
                "type": "RandomScaleShiftSaturated", "params": { "scale_range": [0.7, 1.3], "shift_range": [-0.3, 0.3] }, "per_image_randomness": True
            },
            "Linear": { "type": "RandomScaleShiftLinear", "params": { "scale_range": [0.1, 0.5], "shift_range": [-1, 1] }, "per_image_randomness": True },
            "Linear (Contrast Invert)": { "type": "RandomScaleShiftLinear", "params": { "scale_range": [-1.0, -0.2], "shift_range": [0.0, 0.0] }, "per_image_randomness": True },
            "Noise (low)": { "type": "RandomAdditiveGaussianNoise", "params": { "range": [0.0,0.03] }, "per_image_randomness": True },
            "Noise (high)": { "type": "RandomAdditiveGaussianNoise", "params": { "range": [0.15,0.25] }, "per_image_randomness": True },
            "Gamma (lighten)": { "type": "RandomGammaCorrection", "params": { "range":[0.2,1.0] }, "per_image_randomness": True },
            "Gamma (darken)": { "type": "RandomGammaCorrection", "params": { "range": [1.00,5.0] }, "per_image_randomness": True },
    
        },
    }
}
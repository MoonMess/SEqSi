import numpy as np
from random import random

def alter_image(image, alteration, verbose = False):

    if verbose: print(f"Alteration : {alteration['name']}")

    if alteration["name"] == "none":
        return image

    if alteration["name"] == "noise":
        img_shape = image.shape
        noise = np.random.normal(0, alteration["noise_lvl"], img_shape)
        new_img = image + noise
        m, M = new_img.min(), new_img.max()
        return (new_img - m)/(M-m)
    
    if alteration["name"] == "gamma":
        return image**alteration["gamma"]
    
    if alteration["name"] == "affine":
        tmp = image*alteration["scale"] + alteration["shift"]
        if alteration["saturated"]:
            return tmp*(tmp>0)*(tmp<=1) + 1.*(tmp>1)
        else:
            return tmp
    
    else:
        print("Unknown alteration")
        assert False


MAX_NOISE = 0.2
MAX_SHIFT = 2
MIN_SHIFT = - MAX_SHIFT
MAX_SCALE = 4
MAX_GAMMA = 5
INV_GAMMA = 1/MAX_GAMMA
FIXED_NOISE = 0.05



def generate_alteration_parameters(alteration_name):

    if alteration_name == "none":
        alteration = {
            "name": "none"
        }

    if alteration_name == "noise":
        alteration = {
            "name": alteration_name,
            "noise_lvl": MAX_NOISE*random()
            }
    
    if alteration_name == "noise_fixed":
        alteration = {
            "name": "noise",
            "noise_lvl": FIXED_NOISE
        }
        
    if alteration_name == "gamma":
        bright = random()>0.5
        x = INV_GAMMA+(1-INV_GAMMA)*random()
        alteration = {
            "name": alteration_name,
            "gamma": x if bright else 1/x
        }

    
    if alteration_name == "affine":
        alteration = {
            "name": "affine",
            "scale":  (random() if random()>0.5 else 1+random()*(MAX_SCALE-1)),
            "shift":  MIN_SHIFT + (MAX_SHIFT - MIN_SHIFT)*random(),
            "saturated": False
        }


    if alteration_name == "affine_saturated":
        alteration = {
            "name": "affine",
            "scale":  0.5 + random(),
            "shift": (1-2*(random()>0.5))*(0.4+0.4*random()),
            "saturated": True
        }

    if alteration_name == "noise_high":
        alteration = {
            "name": "noise",
            "noise_lvl": 0.15 + 0.1*random()
        }

    if alteration_name == "noise_low":
        alteration = {
            "name": "noise",
            "noise_lvl": 0.01 + 0.02*random()
        }

    if alteration_name == "shift":
        alteration = {
            "name": "affine",
            "scale": 1,
            "shift": MIN_SHIFT + (MAX_SHIFT-MIN_SHIFT)*random(),
            "saturated": False
        }

    if alteration_name == "scale":
        alteration = {
            "name": "affine",
            "scale": (random() if random()>0.5 else 1+random()*(MAX_SCALE-1)),
            "shift": 0,
            "saturated": False
        }
    
    if alteration_name == "scale_low":
        alteration = {
            "name": "affine",
            "scale": random(),
            "shift": 0,
            "saturated": False
        }

    if alteration_name == "scale_high":
        alteration = {
            "name": "affine",
            "scale": 1 + (MAX_SCALE-1)*random(),
            "shift": 0,
            "saturated": False
        }

    if alteration_name == "gamma_dark":
        alteration = {
            "name": "gamma",
            "gamma": 1/(INV_GAMMA+(1-INV_GAMMA)*random())
        }

    if alteration_name == "gamma_light":
        alteration = {
            "name": "gamma",
            "gamma": (INV_GAMMA+(1-INV_GAMMA)*random())
        }

    if alteration_name == "shift_saturated":
        alteration = {
            "name": "affine",
            "scale": 1,
            "shift": (1-2*(random()>0.5))*(0.4+0.4*random()),
            "saturated": True
        }

    if alteration_name == "scale_saturated":
        alteration = {
            "name": "affine",
            "scale": 1+3*random(),
            "shift": 0,
            "saturated": True
        }
    
    
    return alteration

    


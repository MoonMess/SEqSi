import numpy as np
from random import random

def norm_a_b(img : np.array, a : int, b):
    float_img = img.astype(float)
    m, M = float_img.min(), float_img.max()
    return a + (b-a)*(float_img-m)/(M-m)


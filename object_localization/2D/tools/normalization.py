import numpy as np
from random import random

def norm_a_b(img : np.array, a, b):
    float_img = img.astype(float)
    m, M = float_img.min(), float_img.max()
    return a + (b-a)*(float_img-m)/(M-m)

def norm_lamb_mu(img: np.array, lamb, mu):
    return lamb*img + mu


def random_norm(img):

    a = 2*random()
    b = a + 3*random()

    return norm_a_b(img, a, b)
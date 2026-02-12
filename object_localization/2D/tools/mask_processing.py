import numpy as np


def centers_from_mask(mask):
    lbls = np.unique(mask)[1:]

    centers = list()

    for lbl in lbls:
            
        crt_mask = (mask == lbl)

        N = np.sum(crt_mask)

        nx, ny = crt_mask.shape
        vx, vy = np.arange((nx))[:,None], np.arange((ny))[None, :]

        bx, by = np.sum(crt_mask*vx)/N, np.sum(crt_mask*vy)/N
        centers.append([bx, by])
    
    return np.array(centers)
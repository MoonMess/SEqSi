from argparse import ArgumentParser
from tifffile import imread, imwrite
import numpy as np
from glob import glob
from os.path import join, basename, splitext


def bbox_and_init(contour):
    x, y = np.nonzero(contour)
    x_min, y_min = x.min(), y.min()
    return x_min, x.max(), y_min, y.max(), (x[0]-x_min,y[0]-y_min)


def compute_proba(mask):
    nx, ny = mask.shape
    vx, vy = np.arange(nx)[:,None], np.arange(ny)[None,:]
    x,y = (mask*vx).sum()/mask.sum(), (mask*vy).sum()/mask.sum()

    dist = np.sqrt((vx-x)**2 + (vy-y)**2)*mask

    proba = (dist.max() - dist)*mask

    norm_proba = proba/proba.max()

    return norm_proba , (x,y)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    
    args = parser.parse_args()

    out_dir = args.output

    imgs = sorted(glob(join(args.input, "*.tif")))

    for im_path in imgs:

        mask = imread(im_path)

        proba = np.zeros_like(mask, dtype=np.float32)

        labels = np.unique(mask)[1:]

        for l in labels:

            crt_mask = (mask == l)*1.

            x_min, x_max, y_min, y_max, first = bbox_and_init(crt_mask)

            small_mask = crt_mask[x_min:x_max+1, y_min:y_max+1]

            small_proba, bary = compute_proba(small_mask)
            proba[x_min:x_max+1, y_min:y_max+1] = np.maximum(small_proba, proba[x_min:x_max+1, y_min:y_max+1])


        bname = basename(im_path)
        imwrite(join(out_dir, bname), proba)
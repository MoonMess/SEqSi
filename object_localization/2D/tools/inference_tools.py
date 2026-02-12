import torch
from skimage.filters import gaussian
import numpy as np

k_size = 5
nb_chan = k_size*k_size-1
mid_k = nb_chan/2-1
conv_kernel = torch.zeros((nb_chan, 1, k_size, k_size))
pad = (k_size-1)//2
for k in range(nb_chan):
    p = k + 1*(k>mid_k)
    i, j = p%k_size, p//k_size
    conv_kernel[k, 0, i,j] = -1
    conv_kernel[k, 0, (k_size-1)//2,(k_size-1)//2] = 1


def find_centers(spots_map, threshold):
    blurred = torch.tensor(gaussian(spots_map.cpu(), sigma = 1)).unsqueeze(0).to(device = spots_map.device, dtype=torch.float32)
    comparison_map = torch.nn.functional.conv2d(blurred, conv_kernel.to(spots_map.device), padding=pad)
    min_map = comparison_map.min(dim=0)[0]
    return ((min_map>=0)*(spots_map>threshold)).nonzero()



def pairing(C, maximize = False):
    if maximize:
        C = -C
    nx,ny = C.shape
    
    n = min(nx,ny)

    M = C.max()

    ind_row = []
    ind_col = []
    values = []

    for _ in range(n):
        tmp = C.argmin()
        i, j = tmp//ny, tmp%ny
        ind_row.append(i)
        ind_col.append(j)
        values.append(C[i,j])
        C[i] = M
        C[:,j] = M
        

    return np.array(ind_row), np.array(ind_col), np.array(values)

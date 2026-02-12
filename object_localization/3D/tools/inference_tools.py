import torch
from skimage.filters import gaussian
import torch.nn.functional as F
import numpy as np

k_size = 5
nb_chan = k_size**3-1
mid_k = nb_chan/2-1
conv_kernel = torch.zeros((nb_chan, 1, k_size, k_size, k_size))
pad = (k_size-1)//2
for n in range(nb_chan):
    p = n + 1*(n>mid_k)
    i, tmp = p%k_size, p//k_size
    j, k = tmp%k_size, tmp//k_size
    conv_kernel[n, 0, i,j,k] = -1
    conv_kernel[n, 0, (k_size-1)//2,(k_size-1)//2, (k_size-1)//2] = 1


def _find_centers(spots_map, threshold):
    padded = F.pad(spots_map, )
    comparison_map = F.conv3d(spots_map.float().unsqueeze(0), conv_kernel.to(spots_map.device), padding=pad)
    min_map = comparison_map.min(dim=0)[0]
    #max_map = comparison_map.max(dim=0)[0]
    return ((min_map>=0)*(spots_map>threshold)).nonzero()


def find_centers(spots_map, threshold, nb_tiles = (1,1,1), overlap = 0.1):
    blurred = torch.tensor(gaussian(spots_map.cpu(), sigma = 1)).unsqueeze(0).to(device = spots_map.device, dtype=torch.float32)
    nx, ny, nz = spots_map.shape
    
    min_map = torch.zeros_like(spots_map)
    #nb_votes = torch.zeros((nx,ny,nz), dtype=torch.uint8)

    nx, ny, nz = spots_map.shape
    nb_x, nb_y, nb_z = nb_tiles

    Dx, Dy, Dz = nx//nb_x + 1,  ny//nb_y + 1, nz//nb_z + 1
    dx, dy, dz = int(Dx*overlap)+1, int(Dy*overlap)+1, int(Dz*overlap)+1

    for i in range(nb_x):
        lx, rx = max(0, i*Dx-dx), min((i+1)*Dx+dx, nx)
        ax, bx = i*Dx-lx, rx-(i+1)*Dx
        for j in range(nb_y):
            ly, ry = max(0, j*Dy-dy), min((j+1)*Dy+dy, ny)
            ay, by = j*Dy-ly, ry-(j+1)*Dy
            for k in range(nb_z):
                lz, rz = max(0, k*Dz-dz), min((k+1)*Dz+dz, nz)
                az, bz = k*Dz-lz, rz-(k+1)*Dz
                #if verbose: print(f"Processing tile X:{(lx,rx)}, Y:{(ly,ry)}, Z:{(lz,rz)}")

                crt_blurred = blurred[...,lx:rx,ly:ry,lz:rz]
                crt_compar = torch.nn.functional.conv3d(crt_blurred, conv_kernel.to(spots_map.device), padding=pad)
                crt_min_map = crt_compar.min(dim=0)[0]

                min_map[i*Dx:(i+1)*Dx,j*Dy:(j+1)*Dy,k*Dz:(k+1)*Dz] = crt_min_map[ax:ax+Dx,ay:ay+Dy,az:az+Dz]
    
    
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
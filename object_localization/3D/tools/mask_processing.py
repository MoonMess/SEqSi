import numpy as np


def centers_from_mask(mask):
    lbls = np.unique(mask)[1:]

    centers = list()

    for lbl in lbls:

        #print(lbl)
            
        crt_mask = (mask == lbl)

        N = np.sum(crt_mask)

        X,Y,Z = np.nonzero(crt_mask)
        xmin, xmax = int(np.floor(X.min())), int(np.ceil(X.max()))
        ymin, ymax = int(np.floor(Y.min())), int(np.ceil(Y.max()))
        zmin, zmax = int(np.floor(Z.min())), int(np.ceil(Z.max()))

        small_mask = crt_mask[xmin:xmax,ymin:ymax,zmin:zmax]

        nx, ny, nz = xmax-xmin, ymax-ymin, zmax-zmin
        vx, vy, vz = np.arange((nx))[:,None , None], np.arange((ny))[None, :, None],  np.arange((nz))[None,None,:]

        bx, by, bz = np.sum(small_mask*vx)/N, np.sum(small_mask*vy)/N, np.sum(small_mask*vz)/N
        centers.append([bx+xmin, by+ymin, bz+zmin])
    
    return np.array(centers)


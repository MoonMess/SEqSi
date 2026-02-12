import numpy as np
import matplotlib.pyplot as plt

def plot_depths_map(img, ax, cmap='jet_r'):
    """Affiche la depth map sur l'axe 0 (f est une image 3D format ZYX."""

    nz, ny, nx = img.shape

    colormap = plt.get_cmap(cmap)
    map = colormap(np.linspace(0, 1, nz))
    R = np.zeros((ny, nx, nz))
    G = np.zeros((ny, nx, nz))
    B = np.zeros((ny, nx, nz))
    for i in range(nz):
        R[:, :, i] = img[i, :, :] * map[i, 0]
        G[:, :, i] = img[i, :, :] * map[i, 1]
        B[:, :, i] = img[i, :, :] * map[i, 2]
    proj = np.stack((np.sum(R, axis=2), np.sum(G, axis=2), np.sum(B, axis=2)), axis=2) / nz
    proj = proj / np.max(proj)

    ax.imshow(proj)
    ax.axis('off')

    return ax
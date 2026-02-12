from os.path import join, basename
from glob import glob
from tifffile import imread
from random import randint
from csbdeep.utils import normalize
from torch.utils.data import Dataset
import torch

class EquiSet(Dataset):

  def __init__(self, path, patch_size, device, norm_img = True):
    super().__init__()
    self.path = path
    self.img_dir = join(path, "images")
    self.proba_dir = join(path, "samplings")
    self.patch_size = patch_size

    self.device = device

    self.norm_img = norm_img

    self.img_files = sorted(glob(join(self.img_dir, "*.tif")))

  def __len__(self) -> int:
    return len(self.img_files)

  def crop(self, img):
    nx, ny = img.shape
    x, y = randint(0,nx-self.patch_size-1), randint(0,ny-self.patch_size-1)

    return img[x:x+self.patch_size, y:y+self.patch_size], x, y

  def __getitem__(self, index):
    img_path = self.img_files[index]
    name = basename(img_path)
    img = imread(img_path)
    proba = imread(join(self.proba_dir, name))

    cropped_proba, x, y = self.crop(proba)
    while cropped_proba.max()<0.5:
      cropped_proba, x, y = self.crop(proba)
    cropped_img = img[x:x+self.patch_size, y:y+self.patch_size]
    if self.norm_img:
      #cropped_img = normalize(cropped_img, pmin=1, pmax=99.8, axis=(0,1), clip=True)
      m, M = cropped_img.min(), cropped_img.max()
      cropped_img = (cropped_img - m)/(M-m)

    return torch.tensor(cropped_img[None,...], device=self.device, dtype = torch.float32), torch.tensor(cropped_proba, device=self.device, dtype = torch.float32)
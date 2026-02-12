from os.path import join, basename
from glob import glob
from tifffile import imread
from random import randint
from torch.utils.data import Dataset
import torch

from .alteration import generate_alteration_parameters, alter_image

class EquiSet(Dataset):

  def __init__(self, path, patch_size, device, data_aug = []):
    super().__init__()
    self.path = path
    self.img_dir = join(path, "images")
    self.proba_dir = join(path, "samplings")
    self.patch_size = patch_size

    self.device = device
    self.img_files = sorted(glob(join(self.img_dir, "*.tif")))
    self.data_aug = data_aug

  def __len__(self) -> int:
    return len(self.img_files)

  def crop(self, img):
    nx, ny, nz = img.shape
    x, y, z = randint(0,nx-self.patch_size-1), randint(0,ny-self.patch_size-1), randint(0,nz-self.patch_size-1)

    return img[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size], x, y, z

  def __getitem__(self, index):
    img_path = self.img_files[index]
    name = basename(img_path)
    img = imread(img_path)
    proba = imread(join(self.proba_dir, name))

    cropped_proba, x, y, z = self.crop(proba)
    while cropped_proba.max()<0.5:
      cropped_proba, x, y, z = self.crop(proba)
    cropped_img = img[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]

    m, M = cropped_img.min(), cropped_img.max()
    cropped_img = (cropped_img - m)/(M-m)

    if len(self.data_aug) > 0:
      k = randint(0, len(self.data_aug)-1)
      alteration = generate_alteration_parameters(self.data_aug[k])
      cropped_img = alter_image(cropped_img, alteration=alteration)

    return torch.tensor(cropped_img[None,...], device=self.device, dtype = torch.float32), torch.tensor(cropped_proba, device=self.device, dtype = torch.float32)




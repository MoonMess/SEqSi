from os.path import join, basename
from glob import glob
from tifffile import imread
from torch.utils.data import Dataset
import torch
from random import randint

from .alteration import alter_image, generate_alteration_parameters

class OptimSet(Dataset):

  def __init__(self, path, device, data_aug = []):
    super().__init__()
    self.path = path
    self.img_dir = join(path, "images")
    self.mask_dir = join(path, "masks")

    self.device = device

    self.data_aug = data_aug

    self.img_files = sorted(glob(join(self.img_dir, "*.tif")))

  def __len__(self) -> int:
    return len(self.img_files)

  def __getitem__(self, index):
    img_path = self.img_files[index]
    name = basename(img_path)
    img = imread(img_path)
    mask = imread(join(self.mask_dir, name))

    m, M = img.min(), img.max()
    img = (img - m)/(M-m)

    if len(self.data_aug) > 0:
      k = randint(0, len(self.data_aug)-1)
      alteration = generate_alteration_parameters(self.data_aug[k])
      img = alter_image(img, alteration=alteration)

    return torch.tensor(img[None,...], device=self.device, dtype = torch.float32), mask

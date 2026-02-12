from os.path import join, basename
from glob import glob
from tifffile import imread
from csbdeep.utils import normalize
from torch.utils.data import Dataset
import torch

class OptimSet(Dataset):

  def __init__(self, path, device, norm_img = True):
    super().__init__()
    self.path = path
    self.img_dir = join(path, "images")
    self.mask_dir = join(path, "masks")

    self.device = device

    self.norm_img = norm_img

    self.img_files = sorted(glob(join(self.img_dir, "*.tif")))

  def __len__(self) -> int:
    return len(self.img_files)

  def __getitem__(self, index):
    img_path = self.img_files[index]
    name = basename(img_path)
    img = imread(img_path)
    mask = imread(join(self.mask_dir, name))

    if self.norm_img:
      # img = normalize(img, pmin=1, pmax=99.8, axis=(0,1), clip=True)
      m, M = img.min(), img.max()
      img = (img - m)/(M-m)

    return torch.tensor(img[None,...], device=self.device, dtype = torch.float32), mask
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np
import random

class CryoETClassificationDataset(Dataset):
    """
    Dataset for Cryo-ET patch classification.
    Loads patches and metadata from a pre-separated folder structure (train/val/test).
    """
    def __init__(self, data_root, target_patch_size, mode='train', use_augmentation=True):
        """
        Args:
            data_root (str or Path): Root directory for this split (e.g., '.../classification/train').
                                     Must contain 'metadata.csv' and a 'patches' folder.
            target_patch_size (tuple[int, int, int]): Target size of the patches (D, H, W).
                                                      Patches will be cropped or padded to this size.
            mode (str): 'train', 'val', or 'test'.
            use_augmentation (bool): If True, applies data augmentation (for 'train' mode).
        """
        self.data_root = Path(data_root)
        self.target_patch_size = target_patch_size
        self.mode = mode
        self.use_augmentation = use_augmentation and self.mode == 'train'

        metadata_path = self.data_root / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv file not found in {self.data_root}")
        
        self.metadata_df = pd.read_csv(metadata_path)
        
        print(f"Dataset in '{self.mode}' mode initialized from '{self.data_root}' with {len(self.metadata_df)} samples.")
        if self.use_augmentation:
            print("  - Data augmentation enabled (rotations, flips).")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata_df.iloc[idx]
        patch_relative_path = row['patch_path']
        patch_path = self.data_root / patch_relative_path
        
        label = int(row['class_id'])

        try:
            # Load the .npy patch
            patch = np.load(patch_path).astype(np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Patch not found at location: {patch_path}")
        except Exception as e:
            raise IOError(f"Error loading patch {patch_path}: {e}")

        # Ensure the patch has the correct size (crop/pad)
        patch = self._crop_or_pad_to_target(patch)

        # Data augmentation
        if self.use_augmentation:
            patch = self._apply_augmentations(patch)

        # Add a channel dimension (C, D, H, W)
        patch = np.expand_dims(patch, axis=0)
        
        return {
            "patch": torch.from_numpy(patch.copy()), # .copy() to avoid numpy/torch memory issues
            "label": torch.tensor(label, dtype=torch.long)
        }

    def _crop_or_pad_to_target(self, patch):
        """
        Center-crops or pads the patch to reach the target size.
        Padding is done with the mean value of the patch to minimize the introduction
        of strong edge artifacts.
        """
        current_shape = patch.shape
        target_shape = self.target_patch_size
        
        # Padding if the patch is smaller
        pad_needed = [(max(0, t - c) // 2, max(0, t - c) - max(0, t - c) // 2) for c, t in zip(current_shape, target_shape)]
        if any(p[0] > 0 or p[1] > 0 for p in pad_needed):
            patch = np.pad(patch, pad_needed, mode='constant', constant_values=np.mean(patch))

        # Cropping if the patch is larger
        current_shape = patch.shape # Update shape after padding
        if any(c > t for c, t in zip(current_shape, target_shape)):
            start = [(c - t) // 2 for c, t in zip(current_shape, target_shape)]
            end = [s + t for s, t in zip(start, target_shape)]
            patch = patch[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            
        return patch

    def _apply_augmentations(self, patch):
        """
        Applies simple 3D augmentations: random rotations and flips.
        These augmentations help the model generalize by exposing it to varied orientations.
        """
        # 90-degree rotation on one of the axes
        if random.random() > 0.5:
            # We only rotate in the XZ and YZ planes (around Y and X axes respectively).
            # This preserves the notion of "up" and "down" in the Z-axis, which can be relevant in tomography.
            axis_to_rotate = random.choice([(0, 2), (0, 1)])
            k = random.randint(1, 3) # 90, 180, or 270 degrees
            patch = np.rot90(patch, k=k, axes=axis_to_rotate)

        # Flip on a random axis
        if random.random() > 0.5:
            axis_to_flip = random.choice([0, 1, 2])
            patch = np.flip(patch, axis=axis_to_flip)
            
        return patch

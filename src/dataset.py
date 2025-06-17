# ==============================================================================
# src/dataset.py
#
# This file defines the custom PyTorch Dataset for loading paired
# H&E and Reticulin image patches.
# ==============================================================================

import os
import glob
import random
import logging
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

def load_tiff_to_tensor_raw(path, target_channels):
    """
    Loads a TIFF image file, handles different channel configurations,
    and normalizes it to a [-1, 1] tensor.
    """
    try:
        img_np = tifffile.imread(path)
        if img_np is None: 
            raise IOError("tifffile.imread returned None")
        
        # Ensure 3 channels (for RGB)
        if img_np.ndim == 2:  
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[-1] == 4:  
            img_np = img_np[..., :3] # Drop alpha channel
        elif img_np.shape[-1] == 1:  
            img_np = np.concatenate([img_np] * 3, axis=-1)
        
        # Normalize based on bit depth
        if img_np.dtype == np.uint8:
            tensor = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1) / 255.0
        else: # Assume uint16 or other, normalize to [0, 1]
            tensor = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1) / 65535.0
            
        # Normalize to [-1, 1] range for the Tanh activation in the generator
        return (tensor * 2.0) - 1.0
    except Exception as e:
        logging.error(f"LOAD_TENSOR_ERROR for {path}: {e}")
        return None

class PairedImageDataset(Dataset):
    """
    A PyTorch Dataset for loading unpaired images from two domains (H&E and Reticulin).
    """
    def __init__(self, root_H_folder, root_R_folder, domain_name, steps_per_epoch=None, batch_size=1):
        self.paths_H = sorted(glob.glob(os.path.join(root_H_folder, "*.tif")))
        self.paths_R = sorted(glob.glob(os.path.join(root_R_folder, "*.tif")))
        self.len_H = len(self.paths_H)
        self.len_R = len(self.paths_R)
        self.steps_per_epoch = steps_per_epoch

        logging.info(f"--- {domain_name} Dataset Initialized ---")
        logging.info(f"Found {self.len_H} H&E images and {self.len_R} Reticulin images.")

        if self.len_H == 0 or self.len_R == 0:
            self.length = 0
            raise ValueError(f"CRITICAL: {domain_name} dataset is empty. Check paths.")
        elif self.steps_per_epoch is not None:
            # For training, create a fixed-length epoch for consistent timing
            self.length = self.steps_per_epoch * batch_size
            logging.info(f"Using fixed steps per epoch. Epoch will contain {self.length} items.")
        else:
            # For testing/inference, use the full length of the larger dataset
            self.length = max(self.len_H, self.len_R)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # This retry loop makes the data loading robust to corrupt files
        while True:
            # Get random images from each domain
            img_H_path = self.paths_H[random.randint(0, self.len_H - 1)]
            img_R_path = self.paths_R[random.randint(0, self.len_R - 1)]
            
            img_H_tensor = load_tiff_to_tensor_raw(img_H_path, 3)
            img_R_tensor = load_tiff_to_tensor_raw(img_R_path, 3)
            
            # If both images load successfully, return them
            if img_H_tensor is not None and img_R_tensor is not None:
                return img_H_tensor, img_R_tensor
            
            # If a file fails, the loop will automatically try another random pair
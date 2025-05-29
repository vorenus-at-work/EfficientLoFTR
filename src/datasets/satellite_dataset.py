import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
import kornia as K
import kornia.augmentation as KA

from src.utils.dataset import read_megadepth_gray


class SatelliteDataset(Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 augment_fn=None,
                 fp16=False,
                 **kwargs):
        """
        Dataset for satellite images that generates warped pairs for training.
        
        Args:
            root_dir (str): Directory containing satellite images
            mode (str): options are ['train', 'val', 'test']
            img_resize (int, optional): the longer edge of resized images
            df (int, optional): image size division factor
            img_padding (bool): If set to 'True', zero-pad the image to squared size
            augment_fn (callable, optional): augments images with pre-defined visual effects
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        
        # Get all image files
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(('.jpg', '.png', '.tif', '.tiff')):
                    self.image_paths.append(osp.join(root, f))
        
        # Parameters for image resizing and padding
        if mode == 'train':
            assert img_resize is not None and img_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        
        # For training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        self.fp16 = fp16
        
        # Define warp parameters
        self.warp_params = {
            'degrees': (-20, 20),
            'translate': (0.1, 0.1),
            'scale': (0.8, 1.2),
            'shear': (-20, 20),
            'perspective': 0.2,
        }
        
    def __len__(self):
        return len(self.image_paths)
    
    def generate_warp(self, image_size):
        """Generate random homography for warping"""
        H, W = image_size
        # Random affine + perspective transform
        perspective_transformer = KA.RandomPerspective(
            self.warp_params['perspective'], p=1.0, return_transform=True)
        affine_transformer = KA.RandomAffine(
            degrees=self.warp_params['degrees'],
            translate=self.warp_params['translate'],
            scale=self.warp_params['scale'],
            shear=self.warp_params['shear'],
            p=1.0, return_transform=True
        )
        
        dummy_img = torch.zeros(1, 1, H, W)
        _, affine_mat = affine_transformer(dummy_img)
        _, persp_mat = perspective_transformer(dummy_img)
        
        # Combine transforms
        H_mat = persp_mat @ affine_mat
        return H_mat[0]  # Remove batch dimension

    def __getitem__(self, idx):
        # Read original image
        img_name = self.image_paths[idx]
        image0, mask0, scale0 = read_megadepth_gray(
            img_name, self.img_resize, self.df, self.img_padding, self.augment_fn)
        
        # Generate homography and create warped image
        H_0to1 = self.generate_warp(image0.shape[-2:])
        image1 = K.geometry.warp_perspective(image0, H_0to1[None], image0.shape[-2:])
        mask1 = K.geometry.warp_perspective(mask0[None].float(), H_0to1[None], mask0.shape[-2:]).bool()[0]
        
        # Convert to half precision if needed
        if self.fp16:
            image0, image1 = image0.half(), image1.half()
            H_0to1 = H_0to1.half()
        
        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'H_0to1': H_0to1,  # (3, 3)
            'H_1to0': torch.inverse(H_0to1),  # (3, 3)
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale0,  # Same scale since image1 is warped from image0
            'dataset_name': 'Satellite',
            'pair_id': idx,
            'pair_names': (img_name, f"{img_name}_warped"),
        }
        
        # Add masks for LoFTR training
        if mask0 is not None and self.coarse_scale:
            ts_mask = torch.stack([mask0, mask1], dim=0)[None].float()
            [ts_mask_0, ts_mask_1] = F.interpolate(
                ts_mask, scale_factor=self.coarse_scale,
                mode='nearest', recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
        
        return data

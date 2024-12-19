import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


class ImageDatasetWithTransforms(Dataset):
    def __init__(self, folder_path, norm_transform=None, quality_transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'JPG'))]
        self.norm_transform = norm_transform
        self.quality_transform = quality_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Load as RGB

        original_image = self.norm_transform(image) if self.norm_transform else image

        # Apply transformations if defined
        transformed_image = self.quality_transform(image) if self.quality_transform else image

        return original_image, transformed_image


class CustomLoss(nn.Module):
    def __init__(self, lambda_smooth=1.0):
        super(CustomLoss, self).__init__()
        self.lambda_smooth = lambda_smooth  # Smoothness weight

    def image_comparison_loss(self, original, transformed):
        # L1 loss (Mean Absolute Error) between original and transformed images
        return torch.mean(torch.abs(original - transformed)) * 10

    def total_variation_loss(self, image):
        # Total Variation Loss (Smoothness penalty)
        diff_i = image[:, :, :, 1:] - image[:, :, :, :-1]  # Horizontal difference
        diff_j = image[:, :, 1:, :] - image[:, :, :-1, :]  # Vertical difference
        b, c, h, w = image.shape
        tv_loss = (torch.sum(torch.abs(diff_i)) + torch.sum(torch.abs(diff_j)))/(b*c*h*w)
        return tv_loss

    def forward(self, original, transformed):
        # Compute the L1 loss and the smoothness (TV loss)
        comparison_loss = self.image_comparison_loss(original, transformed)
        smoothness_loss = self.total_variation_loss(transformed)

        # Total loss = comparison loss + smoothness regularization
        total_loss = comparison_loss + self.lambda_smooth * smoothness_loss
        return total_loss
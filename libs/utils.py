import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from libs.image_transformers import reverse_transform


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
    def __init__(self, lambda_smooth=0.5):
        super(CustomLoss, self).__init__()
        self.lambda_smooth = lambda_smooth  # Smoothness weight
        # Define Sobel edge-detection kernels
        sobel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        mean_kernel = torch.tensor([[1/25, 1/25, 1/25, 1/25, 1/25],
                                         [1/25, 1/10, 1/10, 1/10, 1/25],
                                         [1/25, 1/10, 1/5, 1/10, 1/25],
                                         [1/25, 1/10, 1/10, 1/10, 1/25],
                                         [1/25, 1/25, 1/25, 1/25, 1/25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('mean_kernel', mean_kernel)

        self.edge_mask = None

    def edge_comparison_loss(self, original, transformed):
        grayscale = original.mean(dim=1, keepdim=True)

        # Apply Sobel filters
        edges_x = F.conv2d(grayscale, self.sobel_x, padding=1)  # Horizontal edges
        edges_y = F.conv2d(grayscale, self.sobel_y, padding=1)  # Vertical edges
        # edges_r_u = F.conv2d(grayscale, self.sobel_r_u, padding=1)
        # edges_r_b = F.conv2d(grayscale, self.sobel_r_b, padding=1)

        # Combine edge maps
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

        edges_mask = F.conv2d(edges, self.mean_kernel, padding=2)
        edges_mask = F.conv2d(edges_mask, self.mean_kernel, padding=2)

        normalized_edges_mask = torch.clamp(edges_mask, 0.2, 0.95)

        self.edge_mask = normalized_edges_mask

        """fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

        # Plot the first image
        axes[0].imshow(reverse_transform(gray_scale[0] * normalized_edges_mask[0]), cmap="gray")
        axes[0].axis("off")  # Turn off axes

        axes[1].imshow(reverse_transform(original[0]))
        axes[1].axis("off")  # Turn off axes

        plt.tight_layout()
        plt.show()"""

        diff = torch.abs(original - transformed)

        diff = diff * normalized_edges_mask

        loss = torch.sum(diff) / torch.sum(normalized_edges_mask)

        # L1 loss (Mean Absolute Error) between original and transformed images

        return loss

    def total_variation_loss(self, image):
        # Total Variation Loss (Smoothness penalty)
        diff_i_r = image[:, :, :, 1:] - image[:, :, :, :-1]  # Horizontal difference
        diff_i_l = image[:, :, :, :-1] - image[:, :, :, 1:]  # Horizontal difference
        zeros = torch.zeros((diff_i_r.size(0), diff_i_r.size(1), diff_i_r.size(2), 1)).to(image.device)
        diff_i_r = torch.cat((zeros, diff_i_r), dim=3)
        diff_i_l = torch.cat((diff_i_l, zeros), dim=3)
        diff_i = (diff_i_r + diff_i_l) / 2

        diff_j_u = image[:, :, :-1, :] - image[:, :, 1:, :]  # Vertical difference
        diff_j_b = image[:, :, 1:, :] - image[:, :, :-1, :]  # Vertical difference
        zeros = torch.zeros((diff_i_r.size(0), diff_i_r.size(1), 1, diff_j_u.size(3))).to(image.device)
        diff_j_u = torch.cat((diff_j_u, zeros), dim=2)
        diff_j_b = torch.cat((zeros, diff_j_b), dim=2)
        diff_j = (diff_j_u + diff_j_b) / 2

        inversed_edge_mask = -1 * self.edge_mask + 1
        diff = ((diff_i + diff_j) / 2) * inversed_edge_mask
        tv_loss = torch.sum(torch.abs(diff)) / torch.sum(inversed_edge_mask)

        return tv_loss

    def forward(self, original, transformed):
        # Compute the L1 loss and the smoothness (TV loss)
        edge_dif_loss = self.edge_comparison_loss(original, transformed)
        smoothness_loss = self.total_variation_loss(transformed)

        # Total loss = comparison loss + smoothness regularization
        total_loss = edge_dif_loss + self.lambda_smooth * smoothness_loss
        return total_loss

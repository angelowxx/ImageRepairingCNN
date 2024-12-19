import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np

import torch.nn
from sklearn.model_selection import KFold
from torch.utils.data import random_split, DataLoader, Subset
from torchsummary import summary
from tqdm import tqdm

from libs.utils import ImageDatasetWithTransforms, CustomLoss
from libs.image_transformers import *
from libs.model import ImageRepairingCNN
from libs.variables import *


def train_model(image_folder_path=kaggle_data_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model_save_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    save_model_str = os.path.join(model_save_dir, 'image_repairing_model')

    dataset = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_data, test_data = random_split(dataset, [train_size, test_size])

    channels, img_height, img_width = train_data[0][0].shape
    input_shape = (channels, img_height, img_width)

    model = ImageRepairingCNN(input_shape=input_shape).to(device)

    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    criterion = CustomLoss(lambda_smooth=0.1).to(device)

    train(model=model, train_data=train_data, test_data=test_data, criterion=criterion, device=device)

    torch.save(model.state_dict(), save_model_str)

    return model


def train(model=None, train_data=None, test_data=None, criterion=None, device=None, batch_size=16, folds=5,
          num_epochs=10):
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

    lr = 0.005

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):

        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=num_epochs,
            T_mult=2
        )
        lr *= 0.5

        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Fold [{}/{}]'.format(fold + 1, folds))
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            train_loss = train_fn(model, optimizer, criterion, train_loader, val_loader, device)

            # eval_loss = eval_fn(model=model, criterion=criterion, test_loader=val_loader, device=device, display=False)

            scheduler.step()

        model_save_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, 'image_repairing_model')
        torch.save(model.state_dict(), save_model_str)
        break



def train_fn(model, optimizer, criterion, train_loader, val_loader, device, display=False):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    model.train()
    time_train = 0

    t = tqdm(train_loader)
    sum_loss = 0
    n = 0
    for original_images, transformed_images in t:
        original_images = original_images.to(device)
        transformed_images = transformed_images.to(device)

        optimizer.zero_grad()
        repaired_images = model(transformed_images)
        loss = criterion(original_images, repaired_images)
        loss *= 100
        loss.backward()
        optimizer.step()

        sum_loss += loss
        length = repaired_images.size(0)
        n += length

        t.set_description('(=> Training) Loss: {:.4f}'.format(sum_loss/n))

        if display:
            # Create a figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

            # Plot the first image
            axes[0].imshow(reverse_transform(original_images[length-1]))
            axes[0].set_title("original_image")
            axes[0].axis("off")  # Turn off axes

            # Plot the second image
            axes[1].imshow(reverse_transform(repaired_images[length-1]))
            axes[1].set_title("repaired_image")
            axes[1].axis("off")  # Turn off axes


            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()



    time_train += time.time() - time_begin
    print('(=> Training) Loss: {:.4f}'.format(sum_loss/n))
    print('training time: ' + str(time_train))
    return sum_loss/n


def eval_fn(model, criterion, test_loader, device, display=False):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    model.eval()
    time_train = 0

    t = tqdm(test_loader)
    sum_loss = 0
    n = 0
    for original_images, transformed_images in t:
        original_images = original_images.to(device)
        transformed_images = transformed_images.to(device)

        repaired_images = model(transformed_images)
        loss = criterion(original_images, repaired_images)
        loss *= 100

        sum_loss += loss
        length = repaired_images.size(0)
        n += length

        t.set_description('(=> Training) Loss: {:.4f}'.format(sum_loss/n))

        if display:
            # Create a figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

            # Plot the first image
            axes[0].imshow(reverse_transform(original_images[length-1]))
            axes[0].set_title("original_image")
            axes[0].axis("off")  # Turn off axes

            # Plot the second image
            axes[1].imshow(reverse_transform(repaired_images[length-1]))
            axes[1].set_title("repaired_image")
            axes[1].axis("off")  # Turn off axes


            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()


    time_train += time.time() - time_begin
    print('(=> Training) Loss: {:.4f}'.format(sum_loss/n))
    print('training time: ' + str(time_train))
    return sum_loss/n

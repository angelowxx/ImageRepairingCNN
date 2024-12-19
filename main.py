import logging
import os

import torch
from torch.utils.data import DataLoader

from libs.image_transformers import normalize_img_size, downward_img_quality
from libs.training import train_model, eval_fn
from libs.utils import ImageDatasetWithTransforms


def evaluate_model(model, image_folder_path='D:\\python\\Animal Classification\\data\\raw-img\\cane'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model_save_dir = os.path.join(os.getcwd(), 'models', 'image_repairing_model')
    criterion = torch.nn.MSELoss().to(device)
    model = model.to(device)
    dataset = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    test_loss = eval_fn(model=model, criterion=criterion, test_loader=test_loader, device=device, display=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = train_model()
    print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

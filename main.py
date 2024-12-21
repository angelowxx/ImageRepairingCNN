import logging
import os

import torch
from torch.utils.data import DataLoader

from libs.image_transformers import normalize_img_size, downward_img_quality
from libs.training import train_model, eval_fn
from libs.utils import ImageDatasetWithTransforms, CustomLoss
from libs.model import ImageRepairingCNN
from libs.variables import *


def evaluate_model():
    image_folder_path = os.path.join(os.getcwd(), 'data', 'split_images')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model_save_dir = os.path.join(os.getcwd(), 'models', 'image_repairing_model')
    criterion = CustomLoss().to(device)
    dataset = ImageDatasetWithTransforms(image_folder_path, normalize_img_size, downward_img_quality)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    channels, img_height, img_width = dataset[0][0].shape
    input_shape = (channels, img_height, img_width)

    model = ImageRepairingCNN(input_shape=input_shape).to(device)

    model.load_state_dict(torch.load(model_save_dir, map_location=torch.device('cpu')))

    test_loss = eval_fn(model=model, criterion=criterion, test_loader=test_loader, device=device, display=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    log_lvl = logging.INFO
    logging.basicConfig(level=log_lvl)
    model = train_model()
    evaluate_model()
    print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

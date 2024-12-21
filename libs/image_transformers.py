import torch
from torchvision.transforms import transforms
from libs.variables import *

downward_img_quality = transforms.Compose([
    transforms.Resize((clip_height//4, clip_width//4)),
    transforms.Resize((clip_height, clip_width)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda img: torch.clamp(img + torch.randn_like(img) * 0.05, 0, 1))
])

normalize_img_size = transforms.Compose([
    transforms.Resize((clip_height, clip_width)),
    transforms.ToTensor()
])

reverse_transform = transforms.Compose([
    transforms.ToPILImage()  # Convert the tensor back to a PIL image
])

import torch
from torchvision.transforms import transforms

downward_img_quality = transforms.Compose([
    transforms.Resize((200, 300)),
    transforms.Resize((400, 600)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda img: torch.clamp(img + torch.randn_like(img) * 0.05, 0, 1))
])

normalize_img_size = transforms.Compose([
    transforms.Resize((400, 600)),
    transforms.ToTensor()
])

reverse_transform = transforms.Compose([
    transforms.ToPILImage()  # Convert the tensor back to a PIL image
])

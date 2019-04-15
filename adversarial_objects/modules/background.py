import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Background(nn.Module):
    def __init__(self, filepath, image_size):
        super(Background, self).__init__()
        self.image = PIL.Image.open(filepath)
        self.image_size = image_size

    def render_image(self, center_crop=False, batch_size=None):
        transform = transforms.Compose([
            transforms.CenterCrop((self.image_size, self.image_size)) if center_crop else transforms.RandomCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        data = np.transpose(transform(self.image), [1, 2, 0]).detach().numpy()
        data = torch.tensor((data - data.min()) / (data.max() - data.min()), device='cuda')
        if batch_size is not None:
            data = data.view(1, *data.shape).expand(batch_size, *data.shape)
        return data

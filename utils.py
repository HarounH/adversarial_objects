from collections import defaultdict
import numpy as np
import csv
from skimage.io import imread, imsave
import imageio
import torch


def create_rotation_y(angle):
    rotation_y = torch.eye(4)
    rotation_y[0, 0] = rotation_y[2, 2] = torch.cos(torch.tensor(angle*np.pi/180))
    rotation_y[0, 2] = -torch.sin(torch.tensor(angle*np.pi/180))
    rotation_y[2, 0] = -rotation_y[0, 2]
    rotation_y = rotation_y.unsqueeze(0)
    return rotation_y


def save_torch_image(path, tensor):
    ''' One tensor in (RGB, H, W) format.
    '''
    assert (len(tensor.shape) == 3)
    # import pdb; pdb.set_trace()
    data = tensor.permute(1, 2, 0).detach().cpu().numpy()
    data[data > 1] = 1.0
    data[data < -1] = -1.0
    # data -= data.min()
    # data /= data.max()
    # data *= 2
    # data -= 1
    imsave(path, data)


def save_torch_gif(path, tensors):
    ''' List of images in (RGB, H, W) format/shape
    '''
    assert(len(tensors[0].shape) == 3)
    N = len(tensors) if isinstance(tensors, list) else tensors.shape[0]
    writer = imageio.get_writer(path, mode='I')
    for i in range(N):
        writer.append_data((
            255 * tensors[i].detach().cpu().permute(1, 2, 0).numpy()
        ).astype(np.uint8))
    writer.close()


class LossHandler:
    def __init__(self, print_every=1):
        # name -> epoch -> list (per batch)
        self.logs = defaultdict(lambda: defaultdict(list))
        self.print_every = print_every

    def __getitem__(self, *args, **kwargs):
        return self.logs.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.logs.__setitem__(*args, **kwargs)

    def log_epoch(self, writer, epoch):
        for k, v in self.logs.items():
            if epoch in v:
                if (epoch < 0) or (epoch % self.print_every == 0):
                    print("{}[{}]: {}".format(k, epoch, np.mean(v[epoch])))
                writer.add_scalar(k, np.mean(v[epoch]), epoch)


class SignReader:
    def __init__(self, filename='victim_0/signnames.csv', reader_mode='rb'):
        self.signs = {}
        with open(filename, reader_mode) as f:
            reader = csv.reader(f)
            header = next(reader)
            for i,signname in reader:
                self.signs[int(i)] = signname
    def __getitem__(self, *args, **kwargs):
        return self.signs.__getitem__(*args, **kwargs)


class ImagenetReader:
    def __init__(self, filename='imagenet/imagenet_labels.csv', reader_mode='rb'):
        self.signs = {}
        with open(filename, reader_mode) as f:
            reader = csv.reader(f)
            for i,signname in reader:
                self.signs[int(i)] = signname
    def __getitem__(self, *args, **kwargs):
        return self.signs.__getitem__(*args, **kwargs)

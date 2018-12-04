""" File that loads model, scene, creates adversary
Unlike attack.py, it doesn't bother using a model
"""
from __future__ import print_function, absolute_import
import os
import argparse
import torch
from torch import nn
import numpy as np
import tqdm
import imageio
import pdb
import neural_renderer as nr
from skimage.io import imread, imsave
import PIL
from torchvision import transforms


class Background(nn.Module):
    def __init__(self, filepath, args):
        super(Background, self).__init__()
        self.image = PIL.Image.open(filepath)
        self.image_size = args.image_size

    def render_image(self):
        transform = transforms.Compose([
            transforms.CenterCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        data = np.transpose(transform(self.image), [1, 2, 0]).detach().numpy()
        return torch.tensor((data - data.min()) / (data.max() - data.min()), device='cuda')


class Object(nn.Module):
    def __init__(self, obj_filename, texture_wrapping='REPEAT', use_bilinear=True, adversarial_affine=False, adversarial_textures=False):
        super(Object, self).__init__()
        assert torch.cuda.is_available()
        vertices, faces, textures = nr.load_obj(
            obj_filename,
            load_texture=True,
            texture_size=4,
            texture_wrapping=texture_wrapping,
            use_bilinear=use_bilinear,
            normalization=False,
        )
        self.vertices = vertices[None, :, :].cuda()
        self.faces = faces[None, :, :].cuda()

        if adversarial_affine:
            self.adversarial_affine_transform = nn.Parameter(0.1 * torch.eye(4).float() + 0.005 * torch.randn((4, 4)).float())
            self.adversarial_affine_transform[0, 3] = 0.5
            self.adversarial_affine_transform[1, 3] = 0.5
        if adversarial_textures:
            self.textures = nn.Parameter(textures.unsqueeze(0)).cuda()
        else:
            self.textures = textures.unsqueeze(0).cuda()
        self.cuda()

    def render_parameters(self, affine_transform=None):
        vertices, faces, textures = self.vertices, self.faces, self.textures
        if affine_transform is not None:
            # vertices are bs, nv, 3
            bs = vertices.shape[0]
            affine_transform = affine_transform.unsqueeze(0).expand([bs] + list(affine_transform.shape)).cuda()
            ones = torch.ones((list(vertices.shape[:-1]) + [1])).float().cuda()
            vertices = torch.cat((vertices, ones), dim=2)
            vertices = torch.bmm(vertices, affine_transform)[:, :, :3]
        elif hasattr(self, 'adversarial_affine_transform'):
            bs = vertices.shape[0]
            affine_transform = self.adversarial_affine_transform.unsqueeze(0).expand([bs, 4, 4]).cuda()
            ones = torch.ones((list(vertices.shape[:-1]) + [1])).float().cuda()
            vertices = torch.cat((vertices, ones), dim=2)
            vertices = torch.bmm(vertices, affine_transform)[:, :, :3]
        else:
            vertices = self.vertices
        faces = self.faces
        textures = self.textures
        return [vertices.cuda(), faces.cuda(), textures.cuda()]

def combine_images_in_order(image_list, args):
    result = torch.zeros(image_list[0].shape, dtype=torch.float, device='cuda')
    for image in image_list:
        selector = (torch.abs(image).sum(dim=2, keepdim=True) == 0).float()
        result = result * selector + image
    result = (result - result.min()) / (result.max() - result.min())
    return result


if __name__ == '__main__':
    # directory set up
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    output_dir = os.path.join(current_dir, 'output')
    evil_cube_filename = 'evil_cube.obj'

    try:
        os.makedirs(output_dir)
    except:
        pass

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-bg", "--background", dest="background", type=str, default="highway.jpg", help="Path to background file (image)")
    parser.add_argument("-obj", "--object", dest="object", type=str, default="custom_stop_sign.obj", help="Name of .obj file containing stop sign")
    parser.add_argument("-o", "--output", dest="output_filename", type=str, default="custom_stop_sign.png", help="Filename for output image")
    parser.add_argument("-angle", "--azimuth", dest="azimuth", type=float, default=90.0, help="Azimuth angle to use for rendering")
    parser.add_argument("--camera_distance", dest="camera_distance", type=float, default=2.732, help="Camera distance to use for rendering")
    parser.add_argument("--image_size", type=int, default=256, help="Size of square image")
    args = parser.parse_args()


    raise NotImplementedError("combine_images_in_order takes tensors, but given numpy matrices.")
    # TODO: Load background
    background = Background(os.path.join(data_dir, args.background), args)
    bg_img = background.render_image()
    # Load stop-sign
    wrapping = 'REPEAT'
    bilinear = True
    stop_sign = Object(
        os.path.join(data_dir, args.object),
    )
    # Render into image
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    camera_distance = args.camera_distance
    elevation = 30.0
    azimuth = args.azimuth
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    obj_image = renderer(*(stop_sign.render_parameters()))
    obj_image = obj_image.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    # Create adversary
    evil_cube = Object(
        os.path.join(data_dir, evil_cube_filename),
        adversarial_affine=True,
    )
    evil_image = renderer(*evil_cube.render_parameters())
    evil_image = evil_image.detach().cpu().numpy()[0].transpose((1, 2, 0))
    # Output
    image = combine_images_in_order([bg_img, obj_image, evil_image], args)
    imsave(os.path.join(output_dir, args.output_filename), image)

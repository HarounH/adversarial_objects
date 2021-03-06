
import os
import json
import argparse
import numpy as np
import scipy
from random import shuffle
from time import time
import sys
import pdb
import pickle as pk
from collections import defaultdict
import itertools
# pytorch imports
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data.sampler as samplers
import tqdm
import imageio
from skimage.io import imread, imsave
import PIL
import pdb
import neural_renderer as nr
from torchvision import transforms


class Background(nn.Module):
    def __init__(self, filepath, args):
        super(Background, self).__init__()
        self.image = PIL.Image.open(filepath)
        self.image_size = args.image_size

    def render_image(self, rotation_fn=lambda x: x, pytorch_mode=False):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        data = np.transpose(transform(rotation_fn(self.image)), [1, 2, 0]).detach().numpy()
        retval = torch.tensor((data - data.min()) / (data.max() - data.min()), device='cuda')
        if pytorch_mode:
            return retval.permute(2, 0, 1)
        else:
            return retval


class Object(nn.Module):
    def __init__(self, obj_filename, texture_size=2, adv_ver=False, adv_tex=False, rng_tex=False):
        super(Object, self).__init__()
        assert torch.cuda.is_available()
        self.adv_ver = adv_ver
        self.adv_tex = adv_tex
        self.rng_tex = rng_tex
        vertices, faces, textures = nr.load_obj(
            obj_filename,
            load_texture=True,
            texture_size=texture_size,
            texture_wrapping='REPEAT',
            use_bilinear=True,
            normalization=False,
        )
        # pdb.set_trace()
        if rng_tex:
            textures = 0.1 + 0.8 * torch.rand_like(textures, device='cuda')
        # Case 1: Textures are parameters
        if adv_tex:
            self.textures = nn.Parameter(textures.unsqueeze(0)).cuda()
        else:
            self.textures = textures.unsqueeze(0).cuda()

        # Case 2: Vertices are parameters
        if adv_ver:
            self.vertices = vertices+torch.randn(vertices.shape, device='cuda')/1000
            xvals, self.const_inds = self.vertices[:, 0].topk(3, largest=False)  # nv, 3
            _, self.var_inds = self.vertices[:, 0].topk(self.vertices.shape[0] - 3, largest=True)  # nv, 3
            var_inds = self.var_inds
            const_inds = self.const_inds
            # pdb.set_trace()
            self.vertices = self.vertices - torch.tensor([xvals[0], 0.0, 0.0],device = 'cuda')
            self.vertices[const_inds[1]] -= torch.tensor([xvals[1] - xvals[0], 0.0, 0.0],device = 'cuda')
            self.vertices[const_inds[2]] -= torch.tensor([xvals[2] - xvals[0], 0.0, 0.0],device = 'cuda')
            self.vertices += torch.tensor([0.02, 0.0, 0.0],device = 'cuda')
            # self.vertices += torch.tensor([2.0, 0.0, 0.0],device = 'cuda')
            self.vertices_constants = self.vertices[const_inds, :]
            self.vertices_vars = nn.Parameter(self.vertices[var_inds, :])
            self.vertices = torch.zeros(self.vertices.shape, dtype=torch.float, device='cuda')
            self.vertices[const_inds] = self.vertices_constants
            self.vertices[var_inds] = self.vertices_vars
            # pdb.set_trace()

        self.vertices = vertices[None, :, :].cuda()
        self.faces = faces[None, :, :].cuda()
        self.cuda()

    def render_parameters(self, affine_transform=None):
        vertices, faces, textures = self.vertices, self.faces, self.textures
        if self.adv_ver:
            vertices = torch.zeros(self.vertices.shape, dtype=torch.float, device='cuda')
            vertices[:, self.const_inds] = self.vertices_constants
            vertices[:, self.var_inds] = self.vertices_vars
        if affine_transform is not None:
            # vertices are bs, nv, 3
            bs = vertices.shape[0]
            affine_transform = affine_transform.unsqueeze(0).expand([bs] + list(affine_transform.shape)).cuda()
            ones = torch.ones((list(vertices.shape[:-1]) + [1])).float().cuda()
            vertices = torch.cat((vertices, ones), dim=2)
            vertices = torch.bmm(vertices, affine_transform)[:, :, :3]

        faces = self.faces
        textures = self.textures
        # pdb.set_trace()
        return [vertices.cuda(), faces.cuda(), textures.cuda()]


def combine_objects(vs, fs, ts):
    n = len(vs)
    v = vs[0]
    f = fs[0]
    t = ts[0]
    # pdb.set_trace()
    for i in range(1, n):
        face_offset = v.shape[1]
        v = torch.cat([v, vs[i]], dim=1)
        f = torch.cat([f, face_offset + fs[i]], dim=1)
        t = torch.cat([t, ts[i]], dim=1)

    return [v, f, t]

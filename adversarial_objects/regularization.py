

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
import pdb


def fna(vertices, faces):
    """
    Face adjacency is fixed asper evil_cube_1.obj
[(0, 1), (0, 5), (0, 7), (1, 0), (1, 9), (1, 10), (2, 3), (2, 8), (2, 11), (3, 2), (3, 4), (3, 6), (4, 3), (4, 5), (4, 10), (5, 0), (5, 4), (5, 6), (6, 3), (6, 5), (6, 7), (7, 0), (7, 6), (7, 8), (8, 2), (8, 7), (8, 9), (9, 1), (9, 8), (9, 11), (10, 1), (10, 4), (10, 11), (11, 2), (11, 9), (11, 10)]
    """
    val = 0.0
    for bidx in range(faces.shape[0]):
        temp_f = faces[bidx, :, :].long()
        temp_v = vertices[bidx, :, :]
        nf = temp_f.shape[1]
        for i, j in zip(range(nf), range(nf)):
            if len(set(temp_f.detach()[i, :].tolist()).intersection(set(temp_f.detach()[j, :].tolist()))) == 2:
                vi = []
                vj = []
                for k in range(faces.shape[2]):
                    vi.append(F.embedding(temp_f[i, k], temp_v))
                    vj.append(F.embedding(temp_f[j, k], temp_v))
                fi_n = torch.cross(vi[1] - vi[0], vi[2] - vi[0], 1)
                fi_n /= fi_n.norm(2)
                fj_n = torch.cross(vj[1] - vj[0], vj[2] - vi[0], 1)
                fj_n /= fj_n.norm(2)
                val += (fi_n * fi_j).sum()
    return val / faces.shape[0]

def fna_ad(vertices, faces, vertices_base):
    """
    Face adjacency is fixed asper evil_cube_1.obj
[(0, 1), (0, 5), (0, 7), (1, 0), (1, 9), (1, 10), (2, 3), (2, 8), (2, 11), (3, 2), (3, 4), (3, 6), (4, 3), (4, 5), (4, 10), (5, 0), (5, 4), (5, 6), (6, 3), (6, 5), (6, 7), (7, 0), (7, 6), (7, 8), (8, 2), (8, 7), (8, 9), (9, 1), (9, 8), (9, 11), (10, 1), (10, 4), (10, 11), (11, 2), (11, 9), (11, 10)]
    """
    val = 0.0
    
    for bidx in range(faces.shape[0]):
        temp_f = faces[bidx, :, :].long()
        temp_v = vertices[bidx, :, :]
        temp_v_base = vertices_base[bidx, :, :]
        v = []
        v_base = []
        for i in range(faces.shape[2]):
            v.append(F.embedding(temp_f[:, i], temp_v))
            v_base.append(F.embedding(temp_f[:, i], temp_v_base))
        n = torch.cross(v[1] - v[0], v[2] - v[0], 1)
        n_base = torch.cross(v_base[1] - v_base[0], v_base[2] - v_base[0], 1)
        # pdb.set_trace()
        val+=(n-n_base).norm(2)
    return val / faces.shape[0]

def edge_length(vertices, faces):
    """
    args,
        vertices: bs, nv, 3
    return
        val: scalar
    """
    length = 0.0
    for bidx in range(faces.shape[0]):
        temp_f = faces[bidx, :, :].long()
        temp_v = vertices[bidx, :, :]
        v = []
        for i in range(faces.shape[2]):
            v.append(F.embedding(temp_f[:, i], temp_v))
        length += (v[1] - v[0]).norm(2)
        length += (v[2] - v[0]).norm(2)
        length += (v[1] - v[2]).norm(2)
    return length / (3*faces.shape[0])

def surface_area(vertices, faces):
    """
    args,
        vertices: bs, nv, 3
    return
        val: scalar
    """
    area = 0.0
    for bidx in range(faces.shape[0]):
        temp_f = faces[bidx, :, :].long()
        temp_v = vertices[bidx, :, :]
        v = []
        for i in range(faces.shape[2]):
            v.append(F.embedding(temp_f[:, i], temp_v))
        area += torch.cross(v[1] - v[0], v[2] - v[0], 1).norm(2)
    return area / faces.shape[0]


def aabb_volume(vertices, faces):
    """
    args,
        vertices: bs, nv, 3
    return
        val: scalar
    """
    return torch.log(vertices.max(dim=1)[0] - vertices.min(dim=1)[0]).sum(1).exp().mean()

def radius_volume(vertices, faces):
    """
    args,
        vertices: bs, nv, 3
    return
        val: scalar
    """
    #vertices: bs, nv, 3
    centroids = vertices.mean(1, keepdim=True)
    return (1.33 * 3.14 * (vertices - centroids).norm(2, dim=2).max(1)[0]**3).mean()  #bs, nv, 3


function_lookup = {
    'fna': fna,
    'surface_area': surface_area,
    'aabb_volume': aabb_volume,
    'radius_volume': radius_volume,
    'edge_length': edge_length,
}


# Contains all black and others
DEFAULT_PRINTABLE_PIXELS = [torch.zeros((1, 3, 1, 1), dtype=torch.float)] + [torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(2).unsqueeze(3) for x in itertools.combinations_with_replacement([0.1, 0.5, 0.9], 3)]


def nps(image, printable_pixels=DEFAULT_PRINTABLE_PIXELS):
    # image: [bs, 3, h, w] tensor
    val = None
    for phat_ in printable_pixels:
        if not(str(image.device) == 'cpu'):
            phat = phat_.cuda()  # Optimize and do it only once
        else:
            phat = phat_

        if val is None:
            val = (image - phat).abs()
        else:
            val = val * (image - phat).abs()
    return val.mean()

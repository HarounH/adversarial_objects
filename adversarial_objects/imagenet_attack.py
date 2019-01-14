""" File that loads model, scene, creates adversary
Then trains the adversary too.
"""

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
import torch.utils.data.sampler as samplers
import tqdm
import imageio
from skimage.io import imread, imsave
import pdb
import neural_renderer as nr
from draw import (
    Background,
)
from object import Object, combine_objects


def combine_images_in_order(image_list, args):
    result = torch.zeros(image_list[0].shape, dtype=torch.float, device='cuda')
    for image in image_list:
        selector = (torch.abs(image).sum(dim=2, keepdim=True) == 0).float()
        result = result * selector + image
    # result = (result - result.min()) / (result.max() - result.min())
    return result

from tensorboardX import SummaryWriter
from utils import LossHandler
from utils import SignReader
import regularization
from victim_0.network import get_victim

def get_args():
    # parameters
    parser = argparse.ArgumentParser()
    # Input output specifications
    parser.add_argument("--image_size", default=128, type=int, help="Square Image size that neural renderer should create for attacking.")
    parser.add_argument("--victim_path", default="victim_0/working_model_91.chk", help="Path relative current_dir to attack model.")
    parser.add_argument("--signnames_path", default="victim_0/signnames.csv", help="Path where the signnames.csv is located.")

    parser.add_argument("-bg", "--background", dest="background", type=str, default="table.jpg", help="Path to background file (image)")
    parser.add_argument("-bo", "--base_object", dest="base_object", type=str, default="coffeemug.obj", help="Name of .obj file containing base object to attack")
    parser.add_argument("-ap", "--attacker_path", dest="evil_cube_path", default="evil_cube_1.obj", help="Path to basic cube shape")

    parser.add_argument("--data_dir", type=str, default='data', help="Location where data is present")
    parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
    parser.add_argument("--output_dir", type=str, default='output', help="Location where data is present")

    parser.add_argument("-o", "--output", dest="output_filename", type=str, default="coffeemug.png", help="Filename for output image")
    # Optimization
    parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
    parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
    parser.add_argument("--bs", default=4, type=int, help="Batch size")
    parser.add_argument("--nobj", default=1, type=int, help="Batch size")
    # Attack specification
    parser.add_argument("--nps", dest="nps", default=False, action="store_true")  # noqa
    parser.add_argument("--fna_ad", dest="fna_ad", default=False, action="store_true")  # noqa
    parser.add_argument("--reg", nargs='+', dest="reg", default="", type=str, choices=[""] + list(regularization.function_lookup.keys()), help="Which function to use for shape regularization")
    parser.add_argument("--reg_w", default=0.05, type=float, help="Weight on shape regularization")
    parser.add_argument("--scale0", default=0.15, type=float, help="Weight on shape regularization")
    parser.add_argument("--translation_clamp", default=5.0, type=float, help="L1 constraint on translation. Clamp applied if it is greater than 0.")
    parser.add_argument("--rotation_clamp", default=0, type=float, help="L1 constraint on rotation. Clamp applied if it is greater than 0.")
    parser.add_argument("--scaling_clamp", default=0, type=float, help="L1 constraint on allowed scaling. Clamp applied if it is greater than 0.")
    parser.add_argument("--adv_tex", action="store_true", default=False, help="Attack using texture too?")
    parser.add_argument("--rng_tex", action="store_true", default=False, help="Attack using random init texture too?")
    parser.add_argument("--adv_ver", action="store_true", default=False, help="Attack using vertices too?")
    parser.add_argument("--ts", dest="ts",  type=int, default=2, help="Textre suze")
    parser.add_argument("--target_class", default=-1, type=int, help="Class of the target that you want the object to be classified as. Negative if not using a targeted attack")
    # Hardware
    parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa

    parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")
    parser.add_argument("--validation_range", default=30, type=int, help="Range over which to validate the image")
    parser.add_argument("--training_range", default=30, type=int, help="Range over which to train the image")

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    current_dir = os.path.dirname(os.path.realpath(__file__))  # WTF AM I DOING
    args.data_dir = os.path.join(current_dir, args.data_dir)
    args.output_dir = os.path.join(current_dir, args.output_dir)
    args.tensorboard_dir = os.path.join(current_dir, args.output_dir, args.tensorboard_dir)

    try:
        os.makedirs(args.output_dir)
    except:
        pass
    try:
        os.makedirs(args.tensorboard_dir)
    except:
        pass

    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args


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

def create_affine_transform(scaling, translation, rotation):
    scaling_matrix = torch.eye(4)
    for i in range(3):
        scaling_matrix[i, i] = scaling[i]
    translation_matrix = torch.eye(4)
    for i in range(1 if args.adv_ver else 0, 3):
        translation_matrix[3, i] = translation[i]
    rotation_x = torch.eye(4)
    rotation_x[1, 1] = rotation_x[2, 2] = torch.cos(rotation[0])
    rotation_x[1, 2] = torch.sin(rotation[0])
    rotation_x[2, 1] = -rotation_x[1, 2]
    rotation_y = torch.eye(4)
    rotation_y[0, 0] = rotation_y[2, 2] = torch.cos(rotation[1])
    rotation_y[0, 2] = -torch.sin(rotation[1])
    rotation_y[2, 0] = -rotation_y[0, 2]
    rotation_z = torch.eye(4)
    rotation_z[0, 0] = rotation_z[1, 1] = torch.cos(rotation[2])
    rotation_z[0, 1] = torch.sin(rotation[2])
    rotation_z[1, 0] = -rotation_z[0, 1]
    return scaling_matrix.mm(rotation_y.mm(rotation_z.mm(rotation_x.mm(translation_matrix))))


if __name__ == '__main__':
    args = get_args()
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    camera_distance = 2.72-0.75  # Constant
    elevation = 0.0
    azimuth = 90.0
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    background = Background(os.path.join(args.data_dir, args.background), args.image_size)
    base_object = Object(
        os.path.join(args.data_dir, args.base_object),
        texture_size=args.ts,
        adv_ver=False,
        adv_tex=False,
    )
    base_object.vertices -= base_object.vertices.mean(1)
    base_object.vertices /= 6.0
    base_vft = (base_object.render_parameters())
    pdb.set_trace()
    obj_image = renderer(*base_vft) # [1, RGB, is, is]
    obj_image = obj_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]

    bg_img = background.render_image().cuda()
    image = combine_images_in_order([bg_img, obj_image], args) # [is, is, RGB]
    imsave(os.path.join(args.output_dir, "safe_" + args.output_filename), image.detach().cpu().numpy())
    image = image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]

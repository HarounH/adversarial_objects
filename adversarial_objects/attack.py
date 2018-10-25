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
from torch.autograd import Variable
import torch.utils.data.sampler as samplers
import tqdm
import imageio
from skimage.io import imread, imsave
import pdb
import neural_renderer as nr
from draw import (
    Background,
    Object,
    combine_images_in_order,
)
from victim_0.network import get_victim
from tensorboardX import SummaryWriter
from utils import LossHandler


ALLOWED_PROJECTIONS = [
    "azimuth",
    "camera_distance",
    "elevation",
]
# parameters
parser = argparse.ArgumentParser()
# Input output specifications
parser.add_argument("--image_size", default=32, type=int, help="Square Image size that neural renderer should create for attacking.")
parser.add_argument("--victim_path", default="victim_0/working_model_91.chk", help="Path relative current_dir to attack model.")

parser.add_argument("-bg", "--background", dest="background", type=str, default="highway.jpg", help="Path to background file (image)")
parser.add_argument("-bo", "--base_object", dest="base_object", type=str, default="custom_stop_sign.obj", help="Name of .obj file containing stop sign")
parser.add_argument("-cp", "--cube_path", dest="evil_cube_path", default="evil_cube.obj", help="Path to basic cube shape")

parser.add_argument("--data_dir", type=str, default='data', help="Location where data is present")
parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
parser.add_argument("--output_dir", type=str, default='output', help="Location where data is present")

parser.add_argument("-o", "--output", dest="output_filename", type=str, default="custom_stop_sign.png", help="Filename for output image")
# Optimization
parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa

# Attack specification
parser.add_argument("--translation_clamp", default=1.0, type=float, help="L1 constraint on translation. Clamp applied if it is greater than 0.")
parser.add_argument("--rotation_clamp", default=2.0 * np.pi, type=float, help="L1 constraint on rotation. Clamp applied if it is greater than 0.")
parser.add_argument("--scaling_clamp", default=1.0, type=float, help="L1 constraint on allowed scaling. Clamp applied if it is greater than 0.")
# Projection specifications
parser.add_argument("--projection_modes", nargs='+', choices=ALLOWED_PROJECTIONS, type=str, help="What kind of projections to use for attack.")
# Hardware
parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, args.data_dir)
output_dir = os.path.join(current_dir, args.output_dir)
tensorboard_dir = os.path.join(current_dir, args.tensorboard_dir)

if __name__ == '__main__':
    # Load background
    background = Background(os.path.join(data_dir, args.background), args)
    bg_img = background.render_image().cuda()
    # Load stop-sign
    stop_sign = Object(
        os.path.join(data_dir, args.base_object),
    )
    stop_sign_translation = torch.tensor([0.75, -1.5, 0.0]).cuda()
    stop_sign.vertices += stop_sign_translation
    # Create adversary
    base_cube = Object(
        os.path.join(data_dir, args.evil_cube_path)
    )
    parameters = {}
    if args.translation_clamp > 0:
        translation_param = torch.randn((3,), requires_grad=True)
        parameters['translation'] = translation_param
    if args.rotation_clamp > 0:
        rotation_param = torch.randn((3,), requires_grad=True)
        parameters['rotation'] = rotation_param
    if args.scaling_clamp > 0:
        scaling_param = torch.randn((3,), requires_grad=True)
        parameters['scaling'] = scaling_param

    optimizer = optim.Adam(
        [v for _, v in parameters.items()],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Render into image
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    camera_distance = 2.72  # Constant
    elevation = 0.0
    azimuth = 90.0
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    obj_image = renderer(*(stop_sign.render_parameters())) # [1, RGB, is, is]
    obj_image = obj_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]

    image = combine_images_in_order([bg_img, obj_image], args) # [is, is, RGB]
    imsave(os.path.join(output_dir, "safe." + args.output_filename), image.detach().cpu().numpy())
    image = image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
    # Load model
    victim = get_victim(args.victim_path).cuda()  # nn.Module
    # Ensure that adversary is adversarial
    y = victim(image)
    ypred = torch.argmax(y)
    print("y: {}".format(y))
    print("ypred: {}".format(ypred))

    raise NotImplementedError("Stuff after this isn't implemented")
    # Optimize loss function
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    loss_handler = LossHandler()

    if args.cuda:
        raise NotImplementedError("Move stuff to cuda")
    else:
        raise NotImplementedError("Figure it out")
    for i in range(args.max_iterations):
        # TODO: Consider batching by parallelizing over renderers.
        # Sample a projection
        # Create image using projection, parameters
        # Run victim on created image.
        # Construct Loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print out the loss
        loss_handler['loss'][i].append(loss.item())
        loss_handler.log_epoch(writer, i)
    # Output
    raise NotImplementedError("Output final adversarial image")

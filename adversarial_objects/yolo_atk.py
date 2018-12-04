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
)
from object import Object, combine_objects

from tensorboardX import SummaryWriter
from utils import LossHandler
from utils import SignReader
import regularization


from yolo_v3.models import *
from yolo_v3.utils.utils import *

def combine_images_in_order(image_list):
    result = torch.zeros(image_list[0].shape, dtype=torch.float, device='cuda')
    for image in image_list:
        selector = (torch.abs(image).sum(dim=2, keepdim=True) == 0).float()
        result = result * selector + image
    result = (result - result.min()) / (result.max() - result.min())
    return result


###########################
### parser
###########################
# parameters
parser = argparse.ArgumentParser()
# Input output specifications
parser.add_argument("--image_size", default=416, type=int, help="Square Image size that neural renderer should create for attacking (input to YOLO)")
# Model is fixed - pretrained YOLO

# Required for input generation
parser.add_argument("--data_dir", type=str, default='data', help="Location where data is present")
parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
parser.add_argument("--output_dir", type=str, default='output/yolo', help="Location where data is present")

parser.add_argument("-bg", "--background", dest="background", type=str, default="highway.jpg", help="Path to background file (image)")
parser.add_argument("-bo", "--base_object", dest="base_object", type=str, default="custom_stop_sign.obj", help="Name of .obj file containing stop sign")
parser.add_argument("-ap", "--attacker_path", dest="attacker_path", default="evil_cube_1.obj", help="Path to basic cube shape")

# Optimization
parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
parser.add_argument("--bs", default=4, type=int, help="Batch size")
# Attack specification
parser.add_argument("--nps", dest="nps", default=False, action="store_true")  # noqa
parser.add_argument("--reg", nargs='+', dest="reg", default="", type=str, choices=[""] + list(regularization.function_lookup.keys()), help="Which function to use for shape regularization")
parser.add_argument("--reg_w", default=0.05, type=float, help="Weight on shape regularization")
parser.add_argument("--translation_clamp", default=5.0, type=float, help="L1 constraint on translation. Clamp applied if it is greater than 0.")
parser.add_argument("--rotation_clamp", default=0, type=float, help="L1 constraint on rotation. Clamp applied if it is greater than 0.")
parser.add_argument("--scaling_clamp", default=0, type=float, help="L1 constraint on allowed scaling. Clamp applied if it is greater than 0.")
parser.add_argument("--adv_tex", action="store_true", default=False, help="Attack using texture too?")
parser.add_argument("--adv_ver", action="store_true", default=False, help="Attack using vertices too?")
parser.add_argument("--ts", dest="ts", default=2, help="Textre suze")
parser.add_argument("--target_class", default=-1, type=int, help="Class of the target that you want the object to be classified as. Negative if not using a targeted attack")

parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, args.data_dir)
output_dir = os.path.join(current_dir, args.output_dir)
tensorboard_dir = os.path.join(current_dir, args.tensorboard_dir)

for dir in [output_dir, tensorboard_dir]:
    try:
        os.makedirs(dir)
    except:
        pass

np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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
    for i in range(1, 3):
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
    return rotation_x.mm(rotation_y.mm(rotation_z.mm(translation_matrix.mm(scaling_matrix))))



if __name__ == '__main__':
    # Load Background
    background = Background(os.path.join(data_dir, args.background), args)
    bg_img = background.render_image()  # 416, 416, 3

    # First we need to create an object that is correctly detected
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    camera_distance = 2.72-0.75  # Constant
    elevation = 0.0
    azimuth = 90.0
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    stop_sign = Object(
        os.path.join(data_dir, args.base_object),
        texture_size=args.ts,
        adv_ver=False,
        adv_tex=False,
    )
    stop_sign_translation = torch.tensor([0.0, -1.5, 0.0]).cuda()
    stop_sign.vertices += stop_sign_translation
    obj_image = renderer(*(stop_sign.render_parameters())) # [1, RGB, is, is]
    obj_image = obj_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]

    image = combine_images_in_order([bg_img, obj_image]) # [is, is, RGB]

    image = image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]



    # Set up model Check if it detects well.
    model = Darknet("yolo_v3/config/yolov3.cfg", img_size=args.image_size)
    model.load_weights("yolo_v3/weights/yolov3.weights")
    model = model.cuda()  # Always on CUDA
    classes = load_classes('yolo_v3/data/coco.names') # Extracts class labels from file
    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]


    with torch.no_grad():
        detections = model(image)
        detections = non_max_suppression(detections, 80, 0.8, 0.4)

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(os.path.join(output_dir, "noatk.png"), bbox_inches='tight', pad_inches=0.0)
    plt.close()
    pdb.set_trace()

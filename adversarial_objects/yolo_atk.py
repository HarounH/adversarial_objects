""" File that loads model, scene, creates adversary
Then trains the adversary too.
"""

import os
import json
import argparse
import numpy as np
import scipy
import random
from random import shuffle
import time
import sys
import pdb
import pickle as pk
from collections import defaultdict
import itertools
# pytorch imports
import matplotlib
matplotlib.use('agg')
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data.sampler as samplers
import torchvision
import tqdm
import imageio
from skimage.io import imread, imsave
import pdb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import neural_renderer as nr
from object import (
    Background,
    Object,
    combine_objects,
)

from tensorboardX import SummaryWriter
from utils import LossHandler
from utils import SignReader
import regularization


from yolo_v3.models import *
from yolo_v3.utils.utils import *
YOLO_NUM_CLASSES = 80
YOLO_MAX_OBJECTS = 50

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
parser.add_argument("-o", "--output", dest="output_filename", type=str, default="ywdo", help="Filename for output GIF")

parser.add_argument("-bg", "--background", dest="background", type=str, default="highway1.jpg", help="Path to background file (image)")
parser.add_argument("-bo", "--base_object", dest="base_object", type=str, default="custom_stop_sign.obj", help="Name of .obj file containing stop sign")
parser.add_argument("-ap", "--attacker_path", dest="attacker_path", default="evil_cube_1.obj", help="Path to basic cube shape")

# Optimization
parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
parser.add_argument("--bs", default=4, type=int, help="Batch size")
# Attack specification
parser.add_argument("--nobj", default=1, type=int, help="Number of objects to attack with")
parser.add_argument("--nps", dest="nps", default=False, action="store_true")  # noqa
parser.add_argument("--reg", nargs='+', dest="reg", default="", type=str, choices=[""] + list(regularization.function_lookup.keys()), help="Which function to use for shape regularization")
parser.add_argument("--reg_w", default=0.05, type=float, help="Weight on shape regularization")
parser.add_argument("--scale0", default=0.2, type=float, help="Weight on shape regularization")
parser.add_argument("--translation_clamp", default=5.0, type=float, help="L1 constraint on translation. Clamp applied if it is greater than 0.")
parser.add_argument("--rotation_clamp", default=0, type=float, help="L1 constraint on rotation. Clamp applied if it is greater than 0.")
parser.add_argument("--scaling_clamp", default=0, type=float, help="L1 constraint on allowed scaling. Clamp applied if it is greater than 0.")
parser.add_argument("--adv_tex", action="store_true", default=False, help="Attack using texture too?")
parser.add_argument("--adv_ver", action="store_true", default=False, help="Attack using vertices too?")
parser.add_argument("--ts", dest="ts", default=2, help="Textre suze")
parser.add_argument("--correct_class", default=11, type=int, help="Which class we want to avoid")
parser.add_argument("-tc", "--target_class", default=-1, type=int, help="Class of the target that you want the object to be classified as. Negative if not using a targeted attack")

parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa
parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")
parser.add_argument("--validation_range", default=30, type=int, help="One sided angle range over which to validate the image")

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


def get_mAP(all_correct_detections, all_attacked_detections, attacked_class=11):
    average_precisions = {}
    for label in range(YOLO_NUM_CLASSES):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in range(len(all_correct_detections)):
            detections = all_attacked_detections[i][label]
            annotations = all_correct_detections[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for detect in detections:
                score = detect[-1]
                bbox = detect[:-1]
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                # pdb.set_trace()
                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= 0.5 and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    # We dont care about this.
    # print("Average Precisions:")
    # for c, ap in average_precisions.items():
    #     print(f"+ Class '{c}' - AP: {ap}")

    print("AP[attacked class {}]={}".format(attacked_class, average_precisions[attacked_class]))
    mAP = np.mean(list(average_precisions.values()))
    print("mAP: {}".format(mAP))


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


def combine_images_in_order(image_list, output_shape, color_dim):
    result = torch.zeros(output_shape, dtype=torch.float, device='cuda')
    for image in image_list:
        selector = (torch.abs(image).sum(dim=color_dim, keepdim=True) == 0).float()
        result = result * selector + image
    result = (result - result.min()) / (result.max() - result.min())
    return result


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


def save_image_with_detections(img, detections, filename):
    # img: h, w, 3 shaped array.
    plt.figure()
    fig, ax = plt.subplots(1)
    # pdb.set_trace()
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (args.image_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (args.image_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = args.image_size - pad_y
    unpad_w = args.image_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[0][:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0].tolist():

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf))

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
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.0)
    plt.close()



def prepare_adversary(args):
    parameters = {}
    adv_objs = {}
    for k in range(args.nobj):
        adv_obj = Object(
            os.path.join(data_dir, args.attacker_path),
            texture_size=args.ts,
            adv_tex=args.adv_tex,
            adv_ver=args.adv_ver,
        )
        adv_objs[k] = adv_obj
        if args.adv_ver:
            parameters['vertices{}'.format(k)] = adv_obj.vertices_vars
        if args.translation_clamp > 0:
            translation_param = torch.tensor([0, 0.02, 0.02], device="cuda") * torch.randn((3,), device='cuda') + torch.tensor([0.1,  -1.5 + np.cos(2 * np.pi * k / args.nobj), 1.5 + np.sin(2 * np.pi * k / args.nobj)], dtype=torch.float, device='cuda')
            translation_param.requires_grad_(True)
            parameters['translation{}'.format(k)] = translation_param
        if args.rotation_clamp > 0:
            rotation_param = torch.randn((3,), requires_grad=True, device='cuda')
            parameters['rotation{}'.format(k)] = rotation_param

        else:
            parameters['rotation{}'.format(k)] = torch.zeros((3,),requires_grad=False,device='cuda')

        if args.scaling_clamp > 0:
            scaling_param = args.scale0 * (torch.ones((3,),requires_grad=False,device='cuda') + torch.rand((3,), requires_grad=False, device='cuda'))
            scaling_param.requires_grad_(True)
            parameters['scaling{}'.format(k)] = scaling_param
        else:
            parameters['scaling{}'.format(k)] = torch.ones((3,),requires_grad=False,device='cuda') * args.scale0
        if args.adv_tex:
            parameters['texture{}'.format(k)] = adv_obj.textures
    return adv_objs, parameters


if __name__ == '__main__':
    # Load Background
    background = Background(os.path.join(data_dir, args.background), args)
    bg_img = background.render_image()  # 416, 416, 3

    # First we need to create an object that is correctly detected
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    camera_distance = 10.0  # Constant
    elevation = 0.0
    azimuth = 90.0
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    stop_sign = Object(
        os.path.join(data_dir, args.base_object),
        texture_size=args.ts,
        adv_ver=False,
        adv_tex=False,
    )
    stop_sign_translation = torch.tensor([0.0, -1.5, 1.5]).cuda()
    stop_sign.vertices += stop_sign_translation
    obj_image = renderer(*(stop_sign.render_parameters())) # [1, RGB, is, is]
    obj_image = obj_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]

    image = combine_images_in_order([bg_img, obj_image], obj_image.shape, 2) # [is, is, RGB]

    image = image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]



    # Set up model Check if it detects well.
    model = Darknet("yolo_v3/config/yolov3.cfg", img_size=args.image_size)
    model.load_weights("yolo_v3/weights/yolov3.weights")
    model.train()
    model = model.cuda()  # Always on CUDA
    classes = load_classes('yolo_v3/data/coco.names') # Extracts class labels from file
    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]


    with torch.no_grad():
        detections = model(image)
        detections = non_max_suppression(detections, 80, 0.8, 0.4)

    # Create plot
    save_image_with_detections(image.detach()[0].cpu().numpy().transpose(1, 2, 0), detections, "noatk.png")

    # Load adversary
    adv_objs, parameters = prepare_adversary(args)


    optimizer = optim.Adam(
        list(filter(lambda p : p.requires_grad, [v for _, v in parameters.items()])),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    for itr in range(args.max_iterations):
        start = time.time()
        # Construct components
        obj_vft = stop_sign.render_parameters()

        adv_vfts = [adv_obj.render_parameters(
            affine_transform=create_affine_transform(
                parameters['scaling{}'.format(k)],
                parameters['translation{}'.format(k)],
                parameters['rotation{}'.format(k)],
            )) for k, adv_obj in adv_objs.items()]

        rot_matrices = []
        bg_imgs = []
        for idx in range(args.bs):
            angle = np.random.uniform(75, 105)
            rotation_fn = lambda img: img  # torchvision.transforms.functional.rotate(img, angle)
            bg_imgs.append(background.render_image(rotation_fn, pytorch_mode=True))  # 3, 416, 416
            rotation_y = torch.eye(4)
            rotation_y[0, 0] = rotation_y[2, 2] = torch.cos(torch.tensor(angle))
            rotation_y[0, 2] = -torch.sin(torch.tensor(angle))
            rotation_y[2, 0] = -rotation_y[0, 2]
            rotation_y = rotation_y.unsqueeze(0)
            rot_matrices.append(rotation_y)
        rot_matrices = torch.cat(rot_matrices).cuda()
        bg_img = torch.stack(bg_imgs)

        # Use to make a detection without adversary
        rotated_obj_vft = [torch.tensor(lol.detach(), device='cuda') for lol in obj_vft]
        rotated_obj_vft[0] = torch.bmm(
            torch.cat(
                (
                    rotated_obj_vft[0].expand(args.bs, *(rotated_obj_vft[0].shape[1:])),
                    torch.ones(([args.bs] + list(rotated_obj_vft[0].shape[1:-1]) + [1])).float().cuda(),
                ),
                dim=2),
            rot_matrices,
        )[:, :, :3]
        rotated_obj_vft[1] = rotated_obj_vft[1].expand(args.bs, *(rotated_obj_vft[1].shape[1:]))
        rotated_obj_vft[2] = rotated_obj_vft[2].expand(args.bs, *(rotated_obj_vft[2].shape[1:]))
        # IMAGE


        vft = combine_objects(
            [obj_vft[0]] + [adv_vft[0] for adv_vft in adv_vfts],
            [obj_vft[1]] + [adv_vft[1] for adv_vft in adv_vfts],
            [obj_vft[2]] + [adv_vft[2] for adv_vft in adv_vfts],
        )
        vft[0] = torch.bmm(
            torch.cat(
                (
                    vft[0].expand(args.bs, *(vft[0].shape[1:])),
                    torch.ones(([args.bs] + list(vft[0].shape[1:-1]) + [1])).float().cuda(),
                ),
                dim=2),
            rot_matrices,
        )[:, :, :3]
        vft[1] = vft[1].expand(args.bs, *(vft[1].shape[1:]))
        vft[2] = vft[2].expand(args.bs, *(vft[2].shape[1:]))

        with torch.no_grad():
            # Construct image without adversary to get target
            rotated_obj_img = renderer(*rotated_obj_vft)
            normal_image = combine_images_in_order([bg_img, rotated_obj_img], rotated_obj_img.shape, color_dim=1)
            correct_detections = model(normal_image).detach()
            correct_target = non_max_suppression(correct_detections, 80, 0.8, 0.4)

        # Construct the image with target
        adv_img = renderer(*vft)
        attacked_image = combine_images_in_order([bg_img, adv_img], adv_img.shape, color_dim=1)

        # CONSTRUCT A LOSS!
        if args.target_class == -1:
            attacked_detections = model(attacked_image)
            # Don't detect the correct class at all!
            loss = (attacked_detections[:, :, 4] * attacked_detections[:, :, (5 + args.correct_class)]).sum()
        else:
            # Construct a target and obtain a loss to minimize!
            formatted_attacked_target = []
            attack_count = 0
            for bidx, btarget in enumerate(correct_target):
                detached_btarget = btarget.detach()
                # detached_btarget is a ? * 7 tensor
                formatted_attacked_target.append(torch.zeros((YOLO_MAX_OBJECTS, 5), device='cuda'))
                for obj_idx in range(detached_btarget.shape[0]):
                    if abs(detached_btarget[obj_idx, -1] - args.correct_class) < 1e-4:
                        attack_count += 1
                        formatted_attacked_target[-1][obj_idx, 0] = args.target_class
                    else:
                        formatted_attacked_target[-1][obj_idx, 0] = detached_btarget[obj_idx, -1].item()
                    formatted_attacked_target[-1][obj_idx, 1] = (detached_btarget[obj_idx, 0] + detached_btarget[obj_idx, 2]).item() / (2 * args.image_size)
                    formatted_attacked_target[-1][obj_idx, 2] = (detached_btarget[obj_idx, 1] + detached_btarget[obj_idx, 3]).item() / (2 * args.image_size)
                    formatted_attacked_target[-1][obj_idx, 3] = (detached_btarget[obj_idx, 0] - detached_btarget[obj_idx, 2]).abs().item() / args.image_size
                    formatted_attacked_target[-1][obj_idx, 4] = (detached_btarget[obj_idx, 1] - detached_btarget[obj_idx, 3]).abs().item() / args.image_size
            formatted_attacked_target = torch.stack(formatted_attacked_target)
            # Need to transform to target representation though.
            loss = model(attacked_image, formatted_attacked_target)

        # regularization
        if args.reg != '':
            for k, adv_vft in enumerate(adv_vfts):
                loss += sum(args.reg_w * regularization.function_lookup[reg](adv_vft[0], adv_vft[1]) for reg in args.reg)
        if args.nps:
            loss += regularization.nps(attacked_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.scaling_clamp>0.0:
            [parameters['scaling{}'.format(k)].data.clamp_(0.01, args.scaling_clamp) for k in range(args.nobj)]
        if args.translation_clamp>0.0:
            [parameters['translation{}'.format(k)].data.clamp_(- args.translation_clamp, args.translation_clamp) for k in range(args.nobj)]
        if args.rotation_clamp>0.0:
            [parameters['rotation{}'.format(k)].data.clamp_(-args.rotation_clamp, args.rotation_clamp) for k in range(args.nobj)]
        if args.adv_tex:
            [parameters['texture{}'.format(k)].data.clamp_(-0.9, 0.9) for k in range(args.nobj)]

        print("[{}/{}:{:.2f}s] loss={} {}".format(itr, args.max_iterations, time.time() - start, loss.item(), "attack_count={}/batch_size={}".format(attack_count, args.bs) if args.target_class > 0 else ""))

    print("DONE TRAINING!")
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    # DONE TRAINING
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    # TESTING
    model.eval()
    renderer_attacked_gif = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    renderer_attacker_gif = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    writer_attacked = imageio.get_writer(os.path.join(output_dir, "final_attacked_" + args.output_filename + '.gif'), mode='I')
    writer_attacker = imageio.get_writer(os.path.join(output_dir, "final_attacker_" + args.output_filename + '.gif'), mode='I')

    all_correct_detections = []
    all_attacked_detections = []  # These are used to compute some score.

    loop = range(90 - args.validation_range, 90 + args.validation_range, 1)
    actual_bs = args.bs
    args.bs = 1
    for num, azimuth in enumerate(loop):
        with torch.no_grad():
            obj_vft = stop_sign.render_parameters()
            adv_vfts = [adv_obj.render_parameters(
                affine_transform=create_affine_transform(
                    parameters['scaling{}'.format(k)],
                    parameters['translation{}'.format(k)],
                    parameters['rotation{}'.format(k)],
                )) for k, adv_obj in adv_objs.items()]
            rot_matrices = torch.stack([torch.eye(4)]).cuda()
            bg_img = background.render_image(lambda x: x, pytorch_mode=True)
            # Use to make a detection without adversary
            rotated_obj_vft = [torch.tensor(lol.detach(), device='cuda') for lol in obj_vft]
            rotated_obj_vft[0] = torch.bmm(
                torch.cat(
                    (
                        rotated_obj_vft[0].expand(args.bs, *(rotated_obj_vft[0].shape[1:])),
                        torch.ones(([args.bs] + list(rotated_obj_vft[0].shape[1:-1]) + [1])).float().cuda(),
                    ),
                    dim=2),
                rot_matrices,
            )[:, :, :3]
            rotated_obj_vft[1] = rotated_obj_vft[1].expand(args.bs, *(rotated_obj_vft[1].shape[1:]))
            rotated_obj_vft[2] = rotated_obj_vft[2].expand(args.bs, *(rotated_obj_vft[2].shape[1:]))
            # IMAGE
            vft = combine_objects(
                [obj_vft[0]] + [adv_vft[0] for adv_vft in adv_vfts],
                [obj_vft[1]] + [adv_vft[1] for adv_vft in adv_vfts],
                [obj_vft[2]] + [adv_vft[2] for adv_vft in adv_vfts],
            )
            vft[0] = torch.bmm(
                torch.cat(
                    (
                        vft[0].expand(args.bs, *(vft[0].shape[1:])),
                        torch.ones(([args.bs] + list(vft[0].shape[1:-1]) + [1])).float().cuda(),
                    ),
                    dim=2),
                rot_matrices,
            )[:, :, :3]
            vft[1] = vft[1].expand(args.bs, *(vft[1].shape[1:]))
            vft[2] = vft[2].expand(args.bs, *(vft[2].shape[1:]))

            attacker_vft = combine_objects(
                [adv_vft[0] for adv_vft in adv_vfts],
                [adv_vft[1] for adv_vft in adv_vfts],
                [adv_vft[2] for adv_vft in adv_vfts],
            )
            attacker_vft[0] = torch.bmm(
                torch.cat(
                    (
                        attacker_vft[0].expand(args.bs, *(attacker_vft[0].shape[1:])),
                        torch.ones(([args.bs] + list(attacker_vft[0].shape[1:-1]) + [1])).float().cuda(),
                    ),
                    dim=2),
                rot_matrices,
            )[:, :, :3]
            attacker_vft[1] = attacker_vft[1].expand(args.bs, *(attacker_vft[1].shape[1:]))
            attacker_vft[2] = attacker_vft[2].expand(args.bs, *(attacker_vft[2].shape[1:]))

            # Construct image without adversary to get target
            rotated_obj_img = renderer(*rotated_obj_vft)
            normal_image = combine_images_in_order([bg_img, rotated_obj_img], rotated_obj_img.shape, color_dim=1)
            correct_detections = model(normal_image)
            correct_target = non_max_suppression(correct_detections, 80, 0.8, 0.4)[0]

            # Construct the image with target
            adv_img = renderer(*vft)
            attacked_image = combine_images_in_order([bg_img, adv_img], adv_img.shape, color_dim=1)
            attacked_detections = model(attacked_image)
            attacked_target = non_max_suppression(attacked_detections, 80, 0.8, 0.4)[0]

            if azimuth == 90:
                save_image_with_detections(attacked_image.detach()[0].cpu().numpy().transpose(1, 2, 0), [attacked_target], "atk.png")


            # Construct the image with ONLY the target
            attacker_image = renderer(*attacker_vft)
            # Write images!
            writer_attacked.append_data((255 * (attacked_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())).astype(np.uint8))
            writer_attacker.append_data((255 * (attacker_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())).astype(np.uint8))

            # Save detections to all_correct_detections, all_attacked_detections
            for is_not_annotation, (output, all_detections) in enumerate(zip([correct_target, attacked_target], [all_correct_detections, all_attacked_detections])):
                all_detections.append([np.array([]) for _ in range(YOLO_NUM_CLASSES)])
                if output is not None:
                    # Get predicted boxes, confidence scores and labels
                    pred_boxes = output[:, :5].cpu().numpy()
                    scores = output[:, 4].cpu().numpy()
                    pred_labels = output[:, -1].cpu().numpy()

                    # Order by confidence
                    sort_i = np.argsort(scores)
                    pred_labels = pred_labels[sort_i]
                    pred_boxes = pred_boxes[sort_i]

                    if is_not_annotation == 0:
                        for label in range(YOLO_NUM_CLASSES):
                            all_detections[-1][label] = pred_boxes[pred_labels == label][..., :4]  # Don't need scores here.
                    else:
                        for label in range(YOLO_NUM_CLASSES):
                            all_detections[-1][label] = pred_boxes[pred_labels == label]
    args.bs = actual_bs
    get_mAP(all_correct_detections, all_attacked_detections)

    writer_attacked.close()
    writer_attacker.close()

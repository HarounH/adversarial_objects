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

def combine_images_in_order(image_list, args):
    result = torch.zeros(image_list[0].shape, dtype=torch.float, device='cuda')
    for image in image_list:
        selector = (torch.abs(image).sum(dim=-1, keepdim=True) < 1e-4).float()
        result = result * selector + image
    # result = (result - result.min()) / (result.max() - result.min())
    return result

from tensorboardX import SummaryWriter
from utils import LossHandler
from utils import ImagenetReader
import regularization
import pretrainedmodels
from save_obj import save_obj

# parameters
parser = argparse.ArgumentParser()
# Input output specifications
parser.add_argument("--image_size", default=299, type=int, help="Square Image size that neural renderer should create for attacking.")
parser.add_argument("--victim_name", default="inceptionv3", help="Path relative current_dir to attack model.")
parser.add_argument("--imagenet_path", default="imagenet/imagenet_labels.csv", help="Path where the imagenet_labels.csv is located.")

parser.add_argument("-bg", "--background", dest="background", type=str, default="table.jpg", help="Path to background file (image)")
parser.add_argument("-bo", "--base_object", dest="base_object", type=str, default="coffeemug.obj", help="Name of .obj file containing stop sign")
parser.add_argument("-ap", "--attacker_path", dest="evil_cube_path", default="evil_cube_1.obj", help="Path to basic cube shape")

parser.add_argument("--data_dir", type=str, default='data', help="Location where data is present")
parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
parser.add_argument("--output_dir", type=str, default='output', help="Location where data is present")

parser.add_argument("-o", "--output", dest="output_filename", type=str, default="coffee_mug.png", help="Filename for output image")
# Optimization
parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa
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
parser.add_argument("--validation_range", default=15, type=int, help="Range over which to validate the image")
parser.add_argument("--training_range", default=30, type=int, help="Range over which to train the image")

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, args.data_dir)
output_dir = os.path.join(current_dir, args.output_dir)
tensorboard_dir = os.path.join(current_dir, args.output_dir, args.tensorboard_dir)
try:
    os.makedirs(output_dir)
    os.makedirs(tensorboard_dir)
except:
    print(output_dir)
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
    # Load imagenet_labels
    imagenet_labels = ImagenetReader(args.imagenet_path)
    # Load background
    background = Background(os.path.join(data_dir, args.background), args.image_size)
    background_big = Background(os.path.join(data_dir, args.background), 1*args.image_size)
    # Load stop-sign
    base_object = Object(
        os.path.join(data_dir, args.base_object),
        texture_size=args.ts,
        adv_ver=False,
        adv_tex=False,
    )
    base_object.vertices -= base_object.vertices.mean(1)
    base_object.vertices /= 6.0 #0.5 #2.0 #
    base_object.vertices += torch.tensor([-0.5,0.0,-0.5], device="cuda") #torch.tensor([-0.1,-0.0,+0.1], device = "cuda") #torch.tensor([-0.3,-0.2,+0.1], device = "cuda") #
    # Create adversary
    parameters = {}
    adv_objs = {}
    adv_objs_base = {}
    for k in range(args.nobj):
        adv_obj = Object(
            os.path.join(data_dir, args.evil_cube_path),
            texture_size=args.ts,
            adv_tex=args.adv_tex,
            adv_ver=args.adv_ver,
            rng_tex=args.rng_tex,
        )
        adv_objs[k] = adv_obj
        adv_obj_base = Object(
            os.path.join(data_dir, args.evil_cube_path),
        )
        adv_objs_base[k] = adv_obj_base
        if args.adv_ver:
            parameters['vertices{}'.format(k)] = adv_obj.vertices_vars

        if args.translation_clamp > 0:
            translation_param = torch.tensor([
                0.4,
                3 * args.scale0 * np.cos(2 * np.pi * k / args.nobj) - 0.6,
                3 * args.scale0*np.sin(2 * np.pi * k / args.nobj)
                ], dtype=torch.float, device='cuda')
            translation_param.requires_grad_(True if args.adv_ver else False)
            parameters['translation{}'.format(k)] = translation_param

        if args.rotation_clamp > 0:
            rotation_param = torch.randn((3,), requires_grad=True, device='cuda')
            parameters['rotation{}'.format(k)] = rotation_param
        else:
            parameters['rotation{}'.format(k)] = torch.zeros((3,),requires_grad=False,device='cuda')

        if args.scaling_clamp > 0:
            scaling_param = args.scale0 * (torch.ones((3.,),requires_grad=False,device='cuda'))
            scaling_param.requires_grad_(True)
            parameters['scaling{}'.format(k)] = scaling_param
        else:
            parameters['scaling{}'.format(k)] = torch.ones((3.,),requires_grad=False,device='cuda') * args.scale0

        if args.adv_tex:
            parameters['texture{}'.format(k)] = adv_obj.textures

    optimizer = optim.Adam(
        list(filter(lambda p : p.requires_grad, [v for _, v in parameters.items()])),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Render into image
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    renderer2 = nr.Renderer(camera_mode='look_at', image_size=1*args.image_size)
    renderer_gif = nr.Renderer(camera_mode='look_at', image_size=1*args.image_size)
    camera_distance = 2.72-0.75  # Constant
    elevation = 5.0
    azimuth = 90.0
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    renderer2.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    obj_image = renderer(*(base_object.render_parameters())) # [1, RGB, is, is]
    obj_image = obj_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]

    bg_img = background.render_image(center_crop = True).cuda()
    image = combine_images_in_order([bg_img, obj_image], args) # [is, is, RGB]
    imsave(os.path.join(output_dir, args.output_filename), image.detach().cpu().numpy())
    image = image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
    # Load model
    victim = pretrainedmodels.__dict__[args.victim_name](num_classes=1000, pretrained='imagenet').cuda()  # nn.Module
    victim.eval()
    # Ensure that adversary is adversarial
    ytrue = (F.softmax(victim(image),dim=1))
    ##TODO(kk20): Try using multiple labels
    ytrue_label = int(torch.argmax(ytrue).detach().cpu().numpy())
    ytopk = torch.topk(ytrue,5)[1].detach().cpu().numpy()
    #  504,  700,  999,  899,  968,  725,  505,  686,  647,  901, 438,  653,  849,  470,  720,  631,  680,  804,  844,  773,
    #  632,  756,  969,  898,  550,  838,  828,  489,  412,  463
    # pdb.set_trace()
    print("Raw image classified by the classifier as: {} with p: {}".format(imagenet_labels[ytrue_label], ytrue[0][ytrue_label]))
    if args.target_class>-1:
        print("Using a targeted attack to classify the image as: {}".format(imagenet_labels[args.target_class]))

    # Optimize loss function
    writer_tf = SummaryWriter(log_dir=tensorboard_dir)
    loss_handler = LossHandler()

    for i in range(args.max_iterations):
        # TODO: Consider batching by parallelizing over renderers.
        # Sample a projection
        # Create image
        obj_vft = base_object.render_parameters()

        adv_vfts = [adv_obj.render_parameters(
            affine_transform=create_affine_transform(
                parameters['scaling{}'.format(k)],
                parameters['translation{}'.format(k)],
                parameters['rotation{}'.format(k)],
            )) for k, adv_obj in adv_objs.items()]
        adv_vfts_base = [adv_obj_base.render_parameters(
            affine_transform=create_affine_transform(
                parameters['scaling{}'.format(k)],
                parameters['translation{}'.format(k)],
                parameters['rotation{}'.format(k)],
            )) for k, adv_obj_base in adv_objs_base.items()]
        # pdb.set_trace()

        vft = combine_objects(
            [obj_vft[0]] + [adv_vft[0] for adv_vft in adv_vfts],
            [obj_vft[1]] + [adv_vft[1] for adv_vft in adv_vfts],
            [obj_vft[2]] + [adv_vft[2] for adv_vft in adv_vfts],
        )

        rot_matrices = []
        for idx in range(args.bs):
            angle = np.random.uniform(-args.training_range, args.training_range)
            # pdb.set_trace()
            rotation_y = torch.eye(4)
            rotation_y[0, 0] = rotation_y[2, 2] = torch.cos(torch.tensor(angle*np.pi/180))
            rotation_y[0, 2] = -torch.sin(torch.tensor(angle*np.pi/180))
            rotation_y[2, 0] = -rotation_y[0, 2]
            rotation_y = rotation_y.unsqueeze(0)
            rot_matrices.append(rotation_y)
        rot_matrices = torch.cat(rot_matrices).cuda()
        # projection, parameters
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

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

        # Shape regularization
        if args.reg is '':
            loss = 0.0
        else:
            loss = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                loss += sum(args.reg_w * regularization.function_lookup[reg](adv_vft[0], adv_vft[1]) for reg in args.reg)

        image = renderer(*vft)  # [bs, 3, is, is]
        image = image.squeeze().permute(0, 2, 3, 1)  # [image_size, image_size, RGB]
        img2 = image
        bg_img = background.render_image(center_crop = True).cuda()
        image = combine_images_in_order([bg_img, image], args)
        image = image.permute(0, 3, 1, 2) # [1, RGB, is, is]
        # pdb.set_trace()
        if args.nps:
            loss += 5*regularization.nps(image)
        if args.fna_ad:
            for k, adv_vft in enumerate(adv_vfts):
                loss += 2*regularization.fna_ad(adv_vft[0], adv_vft[1],adv_vfts_base[k][0])

        if i % 10 ==0:
            for bi in range(1):
                imsave(
                    os.path.join(output_dir, "batch{}.iter{}.".format(bi, i) + args.output_filename),
                    np.transpose(image.detach().cpu().numpy()[bi], (1, 2, 0)),
                )

        # Run victim on created image.

        y = (F.softmax(victim(image),dim=1))
        # Construct Loss
        loss += y[:,ytrue_label].mean()

        if args.target_class > -1:
            # pdb.set_trace()
            loss += y[:, ytopk].mean(0).sum()+(-10*(y[:, args.target_class]).mean(0).sum())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scaling_clamp>0.0:
            [parameters['scaling{}'.format(k)].data.clamp_(0.01, args.scaling_clamp) for k in range(args.nobj)]
        # if args.translation_clamp>0.0:
        #     [parameters['translation{}'.format(k)].data.clamp_(- args.translation_clamp, args.translation_clamp) for k in range(args.nobj)]
        if args.rotation_clamp>0.0:
            [parameters['rotation{}'.format(k)].data.clamp_(-args.rotation_clamp, args.rotation_clamp) for k in range(args.nobj)]
        if args.adv_tex:
            [parameters['texture{}'.format(k)].data.clamp_(-0.9, 0.9) for k in range(args.nobj)]

        # Print out the loss
        if i%1==0:
            loss_handler['loss'][i].append(loss.item())
            loss_handler['target_probability'][i].append(y[:, args.target_class].mean(0).sum().detach().cpu().numpy())
            loss_handler['initial_class_probability'][i].append(y[:, ytrue_label].mean(0).sum().detach().cpu().numpy())

            fna_ad_val = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                fna_ad_val += regularization.fna_ad(adv_vft[0], adv_vft[1],adv_vfts_base[k][0]).item()
            loss_handler['fna_ad'][i].append(fna_ad_val)

            loss_handler['nps'][i].append(regularization.nps(image).item())
            surface_area_reg = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                surface_area_reg += (regularization.function_lookup['surface_area'](adv_vft[0], adv_vft[1]))
            edge_length_reg = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                edge_length_reg += (regularization.function_lookup['edge_length'](adv_vft[0], adv_vft[1]))
            edge_variance = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                edge_variance += (regularization.function_lookup['edge_variance'](adv_vft[0], adv_vft[1]))
            loss_handler['surface_area'][i].append(surface_area_reg.item())
            loss_handler['edge_variance'][i].append(edge_variance.item())
            loss_handler['edge_length'][i].append(edge_length_reg.item())
            loss_handler.log_epoch(writer_tf, i)
    # pdb.set_trace()
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    # DONE TRAINING
    ###############################################
    ###############################################
    ###############################################
    ###############################################
    # Output
    for top_counts in [1,2,3,4,5]:
    print(torch.argmax(y.detach()))
    # Count how many raw images are classified as the true_label
    correct_raw = 0
    # Count how many adversarial images are classified as the true_label
    correct_adv = 0
    # Count how many adversarial images are classified as the target_label
    correct_target = 0
    # The labels of the adversarial image from different azimuths when the detection is succesful
    adv_labels = []
    loop = range(90 - 2*args.validation_range, 90 + 2*args.validation_range, 1)
    # loop = tqdm.tqdm(range(0, 360, 4))
    writer = imageio.get_writer(os.path.join(output_dir, "final" + args.output_filename + '.gif'), mode='I')
    writer2 = imageio.get_writer(os.path.join(output_dir, "final_cube_" + args.output_filename + '.gif'), mode='I')
    writer3 = imageio.get_writer(os.path.join(output_dir, "hq_final_cube_" + args.output_filename + '.gif'), mode='I')

    tf_gif_cube = torch.zeros([1,len(loop),3,1*args.image_size,1*args.image_size], device='cuda')
    tf_gif = torch.zeros([1,len(loop),3,1*args.image_size,1*args.image_size], device='cuda')
    bg_img = background.render_image(center_crop = True).cuda()
    bg_img_big = background_big.render_image(center_crop = True).cuda()
    for num, azimuth in enumerate(loop):
        # projection, parameters
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        renderer_gif.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        renderer2.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth*180.0/args.validation_range)
        # Create image

        obj_vft = ((base_object.render_parameters())) # [1, RGB, is, is]
        adv_vfts = [adv_obj.render_parameters(
            affine_transform=create_affine_transform(
                parameters['scaling{}'.format(k)],
                parameters['translation{}'.format(k)],
                parameters['rotation{}'.format(k)],
            )) for k, adv_obj in adv_objs.items()]

        vft = combine_objects(
            [obj_vft[0]] + [adv_vft[0] for adv_vft in adv_vfts],
            [obj_vft[1]] + [adv_vft[1] for adv_vft in adv_vfts],
            [obj_vft[2]] + [adv_vft[2] for adv_vft in adv_vfts],
        )

        adv_vft = combine_objects(
            [adv_vft[0] for adv_vft in adv_vfts],
            [adv_vft[1] for adv_vft in adv_vfts],
            [adv_vft[2] for adv_vft in adv_vfts],
        )

        if num == 0:
            save_obj('{}/printable_coffeemug.obj'.format(output_dir),adv_vft[0][0],adv_vft[1][0],adv_vft[2][0])


        cube_image = renderer2(*adv_vft)
        # pdb.set_trace()
        raw_image = renderer(*obj_vft)
        adv_image = renderer(*vft)
        adv_image_big = renderer_gif(*vft)



        adv_image = adv_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]
        adv_image_big = adv_image_big.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]
        raw_image = raw_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]
        cube_image = cube_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]


        adv_image = combine_images_in_order([bg_img, adv_image], args)
        raw_image = combine_images_in_order([bg_img, raw_image], args)
        adv_image_big = combine_images_in_order([bg_img_big, adv_image_big], args)

        adv_image_ = adv_image.detach().cpu().numpy()
        adv_image_big_ = adv_image_big.detach().cpu().numpy()
        cube_image_ = cube_image.detach().cpu().numpy()


        # image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255*adv_image_).astype(np.uint8))
        writer2.append_data((255*cube_image_).astype(np.uint8))
        writer3.append_data((255*adv_image_big_).astype(np.uint8))

        # Validation
        adv_image = adv_image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
        cube_image = cube_image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
        raw_image = raw_image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
        adv_image_big = adv_image_big.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]

        adv_img_gif = torch.stack([adv_image_big[0,2,:,:],adv_image_big[0,1,:,:],adv_image_big[0,0,:,:]], dim=0)
        cube_image_gif = torch.stack([cube_image[0,2,:,:],cube_image[0,1,:,:],cube_image[0,0,:,:]], dim=0)
        # pdb.set_trace()
        tf_gif[0,num,:,:,:] = (255*adv_img_gif.detach())
        tf_gif_cube[0,num,:,:,:] = (255*cube_image_gif.detach())
        # Run victim on created image.
        y_adv = victim(adv_image)
        y_raw = victim(raw_image)
        y_adv_label = torch.topk(y_adv,5)[1].detach().cpu().numpy()
        y_raw_label = torch.topk(y_raw,5)[1].detach().cpu().numpy()
        # pdb.set_trace()

        if ytrue_label in y_raw_label :
            correct_raw += 1
            adv_labels.append(y_adv_label[0])
        if ytrue_label in y_raw_label and ytrue_label in y_adv_label:
            correct_adv += 1
        if args.target_class > -1 and ytrue_label in y_raw_label and args.target_class in y_adv_label:
            correct_target += 1
    writer.close()
    writer2.close()
    writer3.close()
    print("Raw accuracy: {}/{} Attack accuracy: {}/{}".format(correct_raw,len(loop),correct_raw-correct_adv,correct_raw))
    # most_frequent_attack_label = int(max(set(adv_labels), key=adv_labels.count).detach().cpu().numpy())
    # print("Most frequently predicted as {}: {}, {} out of {} times ".format(
    #     most_frequent_attack_label,
    #     imagenet_labels[most_frequent_attack_label],
    #     adv_labels.count(most_frequent_attack_label),
    #     len(adv_labels)))

    if args.target_class > -1:
        print("Target attack on {}: {}, {} out of {} times ".format(
            args.target_class,
            imagenet_labels[args.target_class],
            correct_target,
            len(adv_labels)))
        loss_handler['targeted_attack'][-1].append((correct_target/(0.0+len(adv_labels))))
    loss_handler['raw_accuracy'][-1].append((correct_raw/(0.0+len(loop))))
    loss_handler['attack_accuracy'][-1].append(((correct_raw-correct_adv)/(0.0+correct_raw)))
    loss_handler.log_epoch(writer_tf, -1)
    # tf_gif = tf_gif.permute(0,2,1,3,4)
    # tf_gif_cube = tf_gif_cube.permute(0,2,1,3,4)
    # writer_tf.add_video("Scene_4_gif",tf_gif,fps=2)
    # writer_tf.add_video("Object_4_gif",tf_gif_cube,fps=2)
    # pdb.set_trace()

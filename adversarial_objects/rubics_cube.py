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

from victim_0.network import get_victim
from tensorboardX import SummaryWriter
from utils import LossHandler
from utils import SignReader


class Object(nn.Module):
    def __init__(self, obj_filename, texture_size=4, texture_wrapping='REPEAT', use_bilinear=True, adversarial_affine=False, adversarial_textures=False, adversarial_object = False):
        super(Object, self).__init__()
        assert torch.cuda.is_available()
        vertices, faces, textures = nr.load_obj(
            obj_filename,
            load_texture=True,
            texture_size=texture_size,
            texture_wrapping=texture_wrapping,
            use_bilinear=use_bilinear,
            normalization=False,
        )
        self.adversarial_object = adversarial_object
        self.vertices = vertices[None, :, :].cuda()
        self.faces = faces[None, :, :].cuda()
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
        return [vertices.cuda(), faces.cuda(), textures.cuda()]


parser = argparse.ArgumentParser()
# Input output specifications
parser.add_argument("--image_size", default=32, type=int, help="Square Image size that neural renderer should create for attacking.")
parser.add_argument("--victim_path", default="victim_0/working_model_91.chk", help="Path relative current_dir to attack model.")
parser.add_argument("--signnames_path", default="victim_0/signnames.csv", help="Path where the signnames.csv is located.")

parser.add_argument("-bg", "--background", dest="background", type=str, default="highway.jpg", help="Path to background file (image)")
parser.add_argument("-bo", "--base_object", dest="base_object", type=str, default="custom_stop_sign.obj", help="Name of .obj file containing stop sign")
parser.add_argument("-cp", "--cube_path", dest="evil_cube_path", default="evil_cube_1.obj", help="Path to basic cube shape")

parser.add_argument("--data_dir", type=str, default='data', help="Location where data is present")
parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
parser.add_argument("--output_dir", type=str, default='output', help="Location where data is present")

parser.add_argument("-o", "--output", dest="output_filename", type=str, default="custom_stop_sign.png", help="Filename for output image")
# Optimization
parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
parser.add_argument("--bs", default=4, type=int, help="Batch size")
# Attack specification
parser.add_argument("--translation_clamp", default=5.0, type=float, help="L1 constraint on translation. Clamp applied if it is greater than 0.")
parser.add_argument("--rotation_clamp", default=0, type=float, help="L1 constraint on rotation. Clamp applied if it is greater than 0.")
parser.add_argument("--scaling_clamp", default=0, type=float, help="L1 constraint on allowed scaling. Clamp applied if it is greater than 0.")
parser.add_argument("--adv_tex", action="store_true", default=False, help="Attack using texture too?")
parser.add_argument("--ts", dest='tex_size', default=4, type=int, help="Texture size")
# Hardware
parser.add_argument("--cuda", dest="cuda", default=False, action="store_true")  # noqa

parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")
parser.add_argument("--validation_range", default=30, type=int, help="Range over which to validate the image")

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, args.data_dir)
output_dir = os.path.join(current_dir, args.output_dir)
tensorboard_dir = os.path.join(current_dir, args.tensorboard_dir)


np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def combine_objects(vs,fs,ts):
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



def combine_images_in_order(image_list, args):
    result = torch.zeros(image_list[0].shape, dtype=torch.float, device='cuda')
    for image in image_list:
        selector = (torch.abs(image).sum(dim=2, keepdim=True) == 0).float()
        result = result * selector + image
    result = (result - result.min()) / (result.max() - result.min())
    return result


def create_affine_transform(scaling, translation, rotation):
    scaling_matrix = torch.eye(4)
    for i in range(3):
        scaling_matrix[i, i] = scaling[i]
    translation_matrix = torch.eye(4)
    for i in range(1,3):
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
    # Load signnames
    signnames = SignReader(args.signnames_path)
    # Load background
    background = Background(os.path.join(data_dir, args.background), args)
    bg_img = background.render_image().cuda()
    # Load stop-sign
    stop_sign = Object(
        os.path.join(data_dir, args.base_object),
        texture_size=args.tex_size,
    )
    stop_sign_translation = torch.tensor([0.0, -1.5, 0.0]).cuda()
    stop_sign.vertices += stop_sign_translation
    # Create adversary


    parameters = {}
    if args.translation_clamp > 0:
        translation_param = torch.tensor([0, 0.2, 0.2],device="cuda")*torch.randn((3,), device='cuda') + torch.tensor([3.0,0,0],device='cuda')
        translation_param.requires_grad_(True)
        parameters['translation'] = translation_param
    if args.rotation_clamp > 0:
        rotation_param = torch.randn((3,), requires_grad=True, device='cuda')
        parameters['rotation'] = rotation_param
        print("HI:)")
    else:
        parameters['rotation'] = torch.zeros((3,),requires_grad=False,device='cuda')
    if args.scaling_clamp > 0:
        scaling_param = torch.rand((3,), requires_grad=True, device='cuda')
        parameters['scaling'] = scaling_param
        print("HI:)")
    else:
        parameters['scaling'] = torch.ones((3,),requires_grad=False,device='cuda') * 0.1

    # Rubics cube.
    cubes = [
        Object(
            os.path.join(data_dir, args.evil_cube_path),
            texture_size=args.tex_size,
            adversarial_textures=args.adv_tex,
            adversarial_object=True,
        ) for _ in range(27)
    ]

    cube_centers = list(itertools.product([-2.1, 0.0, 2.1], repeat=3))
    for i in range(27):
        cubes[i].vertices = cubes[i].vertices + torch.tensor(cube_centers[i], device='cuda')
        if args.adv_tex:
            parameters['textures[{}]'.format(i)] = cubes[i].textures


    optimizer = optim.Adam(
        list(filter(lambda p : p.requires_grad, [v for _, v in parameters.items()])),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Render into image
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    renderer2 = nr.Renderer(camera_mode='look_at', image_size=3*args.image_size)
    camera_distance = 2.72-0.75  # Constant
    elevation = 0.0
    azimuth = 90.0
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    renderer2.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    obj_image = renderer(*(stop_sign.render_parameters())) # [1, RGB, is, is]
    obj_image = obj_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]

    image = combine_images_in_order([bg_img, obj_image], args) # [is, is, RGB]
    imsave(os.path.join(output_dir, "safe." + args.output_filename), image.detach().cpu().numpy())
    image = image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
    # Load model
    victim = get_victim(args.victim_path).cuda()  # nn.Module
    victim.eval()
    # Ensure that adversary is adversarial
    ytrue = victim(image)
    ytrue_label = int(torch.argmax(ytrue).detach().cpu().numpy())
    print("Raw image classified by the classifier as: {}".format(signnames[ytrue_label]))
    # print("y: {}".format(y))
    # print("ypred: {}".format(ypred))

    # Optimize loss function
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    loss_handler = LossHandler()

    for i in range(args.max_iterations):

        # TODO: Consider batching by parallelizing over renderers.
        # Sample a projection
        # Create image
        cube_vfts = [(cube.render_parameters(
            affine_transform=create_affine_transform(
                parameters['scaling'],
                parameters['translation'],
                parameters['rotation'],
            ))) for cube in cubes]
        # cube_image = cube_image.squeeze().permute(1, 2, 0)

        obj_vft = ((stop_sign.render_parameters())) # [1, RGB, is, is]
        vft = combine_objects(
            [obj_vft[0]] + [cube_vft[0] for cube_vft in cube_vfts],
            [obj_vft[1]] + [cube_vft[1] for cube_vft in cube_vfts],
            [obj_vft[2]] + [cube_vft[2] for cube_vft in cube_vfts],
        )

        rot_matrices = []
        for idx in range(args.bs):
            angle = np.random.uniform(75, 105)
            rotation_y = torch.eye(4)
            rotation_y[0, 0] = rotation_y[2, 2] = torch.cos(torch.tensor(angle))
            rotation_y[0, 2] = -torch.sin(torch.tensor(angle))
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

        image = renderer(*vft)  # [bs, 3, is, is]

        if i%10 ==0:
            for bi in range(1):
                try:
                    img = np.transpose(image.detach().cpu().numpy()[bi], (1, 2, 0))
                    img = (img - img.min()) / (img.max() - img.min())
                    imsave(
                        os.path.join(output_dir, "batch{}.iter{}.".format(bi, i) + args.output_filename),
                        img,
                    )
                except:
                    print(np.transpose(image.detach().cpu().numpy()[bi], (1, 2, 0)).min())
                    print(np.transpose(image.detach().cpu().numpy()[bi], (1, 2, 0)).max())
                    pdb.set_trace()

        # Run victim on created image.

        y = victim(image)

        # Construct Loss
        loss = y[:,ytrue_label].mean()

        # Stepping.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scaling_clamp>0.0:
            parameters['scaling'].data.clamp_(0, args.scaling_clamp)
        if args.translation_clamp>0.0:
            parameters['translation'].data.clamp_(-args.translation_clamp, args.translation_clamp)
        if args.rotation_clamp>0.0:
            parameters['rotation'].data.clamp_(-args.rotation_clamp, args.rotation_clamp)
        if args.adv_tex:
            for cube_idx in range(27):
                parameters['textures[{}]'.format(cube_idx)].data.clamp_(-0.9, 0.9)

        # Print out the loss
        loss_handler['loss'][i].append(loss.item())
        loss_handler.log_epoch(writer, i)


    # Output
    # pdb.set_trace()
    print(torch.argmax(y.detach()))
    # Count how many raw images are classified as the true_label
    correct_raw = 0
    # Count how many adversarial images are classified as the true_label
    correct_adv = 0
    # The labels of the adversarial image from different azimuths when the detection is succesful
    adv_labels = []
    loop = range(90-args.validation_range, 90+args.validation_range, 1)
    # loop = tqdm.tqdm(range(0, 360, 4))
    writer = imageio.get_writer(os.path.join(output_dir, "final" + args.output_filename + '.gif'), mode='I')
    writer2 = imageio.get_writer(os.path.join(output_dir, "final_cube_" + args.output_filename + '.gif'), mode='I')
    for num, azimuth in enumerate(loop):
        # pdb.set_trace()
        # loop.set_description('Drawing')
        # projection, parameters
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        renderer2.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth*180.0/args.validation_range)
        # Create image
        cube_vfts = [(cube.render_parameters(
            affine_transform=create_affine_transform(
                parameters['scaling'],
                parameters['translation'],
                parameters['rotation'],
            ))) for cube in cubes]
        # cube_image = cube_image.squeeze().permute(1, 2, 0)

        obj_vft = ((stop_sign.render_parameters())) # [1, RGB, is, is]
        vft = combine_objects(
            [obj_vft[0]] + [cube_vft[0] for cube_vft in cube_vfts],
            [obj_vft[1]] + [cube_vft[1] for cube_vft in cube_vfts],
            [obj_vft[2]] + [cube_vft[2] for cube_vft in cube_vfts],
        )

        cube_image = renderer2(*combine_objects(
            [cube_vft[0] for cube_vft in cube_vfts],
            [cube_vft[1] for cube_vft in cube_vfts],
            [cube_vft[2] for cube_vft in cube_vfts],
        ))
        raw_image = renderer(*obj_vft)
        adv_image = renderer(*vft)
        adv_image = adv_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]
        raw_image = raw_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]
        cube_image = cube_image.squeeze().permute(1, 2, 0)  # [image_size, image_size, RGB]


        adv_image_ = adv_image.detach().cpu().numpy()
        cube_image_ = cube_image.detach().cpu().numpy()
        # image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255*adv_image_).astype(np.uint8))
        writer2.append_data((255*cube_image_).astype(np.uint8))

        # Validation
        adv_image = adv_image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
        raw_image = raw_image.unsqueeze(0).permute(0, 3, 1, 2) # [1, RGB, is, is]
        # Run victim on created image.
        y_adv = victim(adv_image)
        y_raw = victim(raw_image)
        y_adv_label = torch.argmax(y_adv)
        y_raw_label = torch.argmax(y_raw)

        if y_raw_label == ytrue_label:
            correct_raw += 1
            adv_labels.append(y_adv_label)
        if y_raw_label == ytrue_label and y_adv_label==ytrue_label:
            correct_adv += 1
    writer.close()
    writer2.close()
    print("Raw accuracy: {}/{} Attack accuracy: {}/{}".format(correct_raw,len(loop),correct_raw-correct_adv,correct_raw))
    most_frequent_attack_label = int(max(set(adv_labels), key=adv_labels.count).detach().cpu().numpy())
    print("Most frequently predicted as {}: {}, {} out of {} times ".format(
        most_frequent_attack_label,
        signnames[most_frequent_attack_label],
        adv_labels.count(most_frequent_attack_label),
        len(adv_labels)))

''' Combined file for attacking classifiers (ImageNet, GTSRB)
'''

# General imports
import os
import sys
import json
import argparse
import tqdm
import imageio
from skimage.io import imread, imsave
from time import time
# ML imports
from random import shuffle
import numpy as np
import scipy
from collections import defaultdict
import itertools
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as samplers
from tensorboardX import SummaryWriter

# NR
import neural_renderer as nr
# Custom imports
from modules import (
    background,
    wavefront,
    combiner,
    regularization,
    renderers,
    loss_fns,
)
import utils
from victim_0.network import get_victim


HIGH_RES = 400


def get_imagenet_constructor(name):
    def get_imagenet(*args, **kwargs):
        pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')
        return model
    return get_imagenet


classifiers = {
    'inceptionv3': [
        get_imagenet_constructor('inceptionv3'),
        utils.ImagenetReader,
        '',
        'imagenet/imagenet_labels.csv',
        299,
    ],
    'gtsrb': [
        get_victim,
        utils.SignReader,
        'victim_0/gtsrb_us_stop_signs_latest.chk',
        'victim_0/signnames.csv',
        32
    ],
}


def get_classifier(classifier_name, path_='', labels_path_=''):
    fn, label_reader, path, labels_path, img_size = *(classifiers[classifier_name])
    if path_ != '':
        path = path_
    if labels_path_ != '':
        labels_path = labels_path_
    return fn(path).eval().cuda(), label_reader(labels_path)


def get_args():
    parser = argparse.ArgumentParser()
    # Victim specification
    parser.add_argument('classifier', choices=list(classifiers.keys()), help='Which classifier to attack')
    parser.add_argument('-m', '--classifier_path', default='', help='Specify to override defaults')
    parser.add_argument('-l', '--labels_path', default='', help='Specify to override defaults')
    parser.add_argument("-bg", "--background", dest="background", type=str, default="highway2.jpg", help="Path to background file (image)")

    # Attacker specification
    parser.add_argument('-s', '--scene_name', default='stopsign', choices=list(wavefront.objects_dict.keys()), help='name of object to use to attack')
    parser.add_argument('-a', '--attacker_name', default='cube', choices=list(wavefront.objects_dict.keys()), help='name of object to use to attack')
    parser.add_argument('-k', '--nobj', default=1, type=int, help='Number of attacking objects')
    parser.add_argument("--target_class", default=-1, type=int, help="Class of the target that you want the object to be classified as. Negative if not using a targeted attack")

    # Parameterization specification
    parser.add_argument("--ts", dest="ts",  type=int, default=2, help="Textre suze")
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

    # Training specification
    parser.add_argument("--validation_range", default=60, type=int, help="Range over which to validate the image")
    parser.add_argument("--training_range", default=30, type=int, help="Range over which to train the image")

    parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
    parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=1e-5, help='Weight decay')  # noqa
    parser.add_argument("--bs", default=4, type=int, help="Batch size")

    # Output
    parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")
    parser.add_argument("--data_dir", type=str, default='data', help="Location where data is present")
    parser.add_argument("--base_output", dest="base_output", default="./new_output/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be a folder in the output directory')  # noqa
    parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
    parser.add_argument("--output_dir", type=str, default='output', help="Location where data is present")
    parser.add_argument('-o', '--output_dir', default='', help='Specify to override run code mechanism')

    args = parser.parse_args()

    args.cuda = True
    args.device = 'cuda'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.debug:
        args.run_code = "debug"
    os.makedirs(args.base_output, exist_ok=True)
    if len(args.run_code) == 0:
        # Generate a run code by counting number of directories in oututs
        run_count = len(os.listdir(args.base_output))
        args.run_code = 'run{}'.format(run_count)
    if args.output_dir == '':
        args.output_dir = os.path.join(args.base_output, args.run_code)
    os.makedirs(args.output_dir, exist_ok=True)
    print("Using run_code: {}".format(args.run_code))
    print('Output dir: {}'.format(args.output_dir))
    return args


def train(
        args,
        bg=None,
        base_object=None,
        model=None,
        label_names=None,
        adv_objs=None,
        parameters=None,
        bg_big=None,
        adv_objs_base=None,
        ):
    optimizer = optim.Adam(
        list(filter(lambda p : p.requires_grad, [v for _, v in parameters.items()])),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
    renderer_high_res = nr.Renderer(camera_mode='look_at', image_size=HIGH_RES)
    camera_distance = 2.72-0.75  # Constant
    elevation = 5.0
    azimuth = 90.0

def main():
    args = get_args()
    # Get classifiers
    model, label_names, img_size = get_classifier(args.classifier, args.classifier_path, args.labels_path)
    # Instantiate objects
    bg = background.Background(args.background, img_size)
    bg_big = background.Background(args.background, HIGH_RES)
    base_object = wavefront.load_obj(args.scene_name, texture_size=args.ts)

    # Initialize objects
    parameters = {}
    adv_objs = {}
    adv_objs_base = {}
    for k in range(args.nobj):
        adv_obj = wavefront.load_obj(
            args.attacker_name,
            texture_size=args.ts,
            adv_tex=args.adv_tex,
            adv_ver=args.adv_ver,
            rng_tex=args.rng_tex,
        )
        adv_obj_base = wavefront.load_obj(
            os.path.join(data_dir, args.evil_cube_path),
        )
        adv_objs[k] = adv_obj
        adv_objs_base[k] = adv_obj_base

        for param_type, v in adv_obj.init_parameters(args).items():
            parameters['{}_{}'.format(param_type, k)] = v

    # Train
    train(
        args,
        bg=bg,
        base_object=base_object,
        model=model,
        label_names=label_names,
        adv_objs=adv_objs,
        parameters=parameters,
        bg_big=bg_big,
        adv_objs_base=adv_objs_base,
    )
    # Evaluate
    evaluate(
        args,
        bg=bg,
        base_object=base_object,
        model=model,
        label_names=label_names,
        adv_objs=adv_objs,
        parameters=parameters,
        bg_big=bg_big,
        adv_objs_base=adv_objs_base,
    )


if __name__ == '__main__':
    main()

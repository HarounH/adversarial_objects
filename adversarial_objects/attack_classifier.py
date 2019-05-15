''' Combined file for attacking classifiers (ImageNet, GTSRB)
'''

# General imports
import os
import sys
import json
import argparse
import tqdm
import imageio
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
from save_obj import save_obj
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
import pretrainedmodels
import json
PHOTO_EVERY = 100
EVAL_EVERY = 1
HIGH_RES = 400
TOP_COUNTS = [1, 2, 3, 4, 5]
center_crops = {
    'coffeemug': False,
    'stopsign': True,
}

with open('prepared_shapenet_info.json','r') as json_file:
    data = json.load(json_file)
    # TODO: What do we do here?
    print("TODO: Centercrop for shapenet??")
    for k,v in data.items():
        center_crops[k] = True


def get_imagenet_constructor(name):
    def get_imagenet(*args, **kwargs):
        return pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')
    return get_imagenet


classifiers = {
    'inceptionv3': [
        get_imagenet_constructor('inceptionv3'),
        lambda x: utils.ImagenetReader(x, reader_mode='r'),
        '',
        'imagenet/imagenet_labels.csv',
        299,
    ],
    'gtsrb': [
        get_victim,
        lambda x: utils.SignReader(x, reader_mode='r'),
        'victim_0/gtsrb_us_stop_signs_latest.chk',
        'victim_0/signnames.csv',
        32
    ],
}


def get_classifier(classifier_name, path_='', labels_path_=''):
    fn, label_reader, path, labels_path, img_size = tuple(classifiers[classifier_name])
    if path_ != '':
        path = path_
    if labels_path_ != '':
        labels_path = labels_path_
    return fn(path).eval().cuda(), label_reader(labels_path), img_size


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
    parser.add_argument("-el_min", "--min_training_elevation_delta", default=0, type=int, help="")
    parser.add_argument("-el_max", "--max_training_elevation_delta", default=5, type=int, help="")
    parser.add_argument("-el_val_min", "--min_validation_elevation_delta", default=0, type=int, help="")
    parser.add_argument("-el_val_max", "--max_validation_elevation_delta", default=50, type=int, help="")
    parser.add_argument('-la_min', '--light_intensity_ambient_min', default=0.4)
    parser.add_argument('-la_max', '--light_intensity_ambient_max', default=0.6)
    parser.add_argument('-ld_min', '--light_intensity_directional_min', default=0.4)
    parser.add_argument('-ld_max', '--light_intensity_directional_max', default=0.6)

    parser.add_argument('-naz', '--num_azimuth', default=-1, type=int, help='If >0, it is the number of different azimuth values to use for training')

    parser.add_argument("-iter", "--max_iterations", type=int, default=100, help="Number of iterations to attack for.")
    parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Rate at which to do steps.")
    parser.add_argument("--weight_decay", dest="weight_decay", type=float, metavar='<float>', default=0, help='Weight decay')  # noqa
    parser.add_argument("-b", "--bs", dest="batch_size", default=4, type=int, help="Batch size")

    # Output
    parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")
    parser.add_argument("--data_dir", type=str, default='adversarial_objects/data', help="Location where data is present")
    parser.add_argument("--base_output", dest="base_output", default="adversarial_objects/new_output/", help="Directory which will have folders per run")  # noqa
    parser.add_argument("-r", "--run", dest='run_code', type=str, default='', help='Name this run. It will be a folder in the output directory')  # noqa
    parser.add_argument("--tensorboard_dir", dest="tensorboard_dir", type=str, default="tensorboard", help="Subdirectory to save logs using tensorboard")  # noqa
    parser.add_argument('-o', '--output_dir', default='', help='Specify to override run code mechanism')

    args = parser.parse_args()

    args.cuda = True
    args.device = 'cuda'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not os.path.exists(args.base_output):
        os.makedirs(args.base_output)
    if len(args.run_code) == 0:
        # Generate a run code by counting number of directories in oututs
        run_count = len(os.listdir(args.base_output))
        args.run_code = 'run{}'.format(run_count)
    if args.output_dir == '':
        args.output_dir = os.path.join(args.base_output, args.run_code)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.tensorboard_dir = os.path.join(args.output_dir, args.tensorboard_dir)
    print("Using run_code: {}".format(args.run_code))
    print('Output dir: {}'.format(args.output_dir))
    return args


def get_metrics(image, pred_prob_y, true_y, target_y=-1, loss=None, adv_vfts=None, adv_vfts_base=None):
    metrics = {}
    with torch.no_grad():
        if loss is not None:
            metrics['loss'] = loss.item()
        metrics['initial_class_probability'] = pred_prob_y[:, true_y].mean(0).sum().detach().cpu().numpy()
        if target_y > 0:
            metrics['target_probability'] = pred_prob_y[:, target_y].mean(0).sum().detach().cpu().numpy()

        if adv_vfts is not None and adv_vfts_base is not None:
            # FNA
            fna_ad_val = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                fna_ad_val += regularization.fna_ad(adv_vft[0], adv_vft[1],adv_vfts_base[k][0]).item()
            metrics['fna_ad'] = fna_ad_val
            # NPS
            metrics['nps'] = regularization.nps(image).item()
            # Others
            surface_area_reg = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                surface_area_reg += (regularization.function_lookup['surface_area'](adv_vft[0], adv_vft[1]))
            edge_length_reg = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                edge_length_reg += (regularization.function_lookup['edge_length'](adv_vft[0], adv_vft[1]))
            edge_variance = 0.0
            for k, adv_vft in enumerate(adv_vfts):
                edge_variance += (regularization.function_lookup['edge_variance'](adv_vft[0], adv_vft[1]))
            metrics['surface_area'] = surface_area_reg.item()
            metrics['edge_variance'] = edge_variance.item()
            metrics['edge_length'] = edge_length_reg.item()
    return metrics


def test(
        args,
        bg=None,
        base_object=None,
        model=None,
        label_names=None,
        adv_objs=None,
        parameters=None,
        bg_big=None,
        adv_objs_base=None,
        writer_tf=None,
        loss_handler=None
        ):
    renderer, camera_distance, elevation, azimuth = renderers.get_renderer(args.image_size, base_object=args.scene_name)
    renderer_high_res, _, _, _ = renderers.get_renderer(HIGH_RES, base_object=args.scene_name)

    obj_vft = base_object.render_parameters()
    with torch.no_grad():
        base_image = renderer(*(obj_vft))  # 1, 3, is, is
        if bg:
            bg_img = bg.render_image(center_crop=center_crops[args.scene_name], batch_size=1)
            base_image = combiner.combine_images_in_order([bg_img, base_image], bg_img.shape)

        ytrue = (F.softmax(model(base_image),dim=1))
        ytrue_label = int(torch.argmax(ytrue).detach().cpu().numpy())
        ytopk = torch.topk(ytrue, 5)[1].detach().cpu().numpy()

    loop = range(90 - 2 * args.validation_range,
                 90 + 2 * args.validation_range,
                 2)

    NUM_TEST_ELEVATIONS = 10  # (args.max_validation_elevation_delta - args.min_validation_elevation_delta) // 2
    elevation_loop = np.linspace(
        (elevation + args.min_validation_elevation_delta),
        elevation + args.max_validation_elevation_delta,
        num=NUM_TEST_ELEVATIONS)

    correct_raw = {i: 0 for i in TOP_COUNTS}
    correct_adv = {i: 0 for i in TOP_COUNTS}
    correct_target = {i: 0 for i in TOP_COUNTS}
    adv_labels = []  # {i: [] for i in TOP_COUNTS}

    for num_elevation, elevation in enumerate(elevation_loop):
        gif_tensors = []
        gif_tensors_big = []
        gif_tensors_adversary = []
        print('Starting elevation iter {}/{}'.format(num_elevation, len(elevation_loop)))
        for num, azimuth in enumerate(loop):
            adv_vfts = [adv_obj.render_parameters(
                affine_transform=wavefront.create_affine_transform(
                    parameters['scaling{}'.format(k)],
                    parameters['translation{}'.format(k)],
                    parameters['rotation{}'.format(k)],
                    args.adv_ver,
                )) for k, adv_obj in adv_objs.items()]

            vft = combiner.combine_objects(
                *([[obj_vft[i]] + [adv_vft[i] for adv_vft in adv_vfts] for i in range(3)])
            )

            adv_vft = combiner.combine_objects(
                *([[adv_vft[i] for adv_vft in adv_vfts] for i in range(3)])
            )

            if num == 0 and num_elevation == 0:
                save_obj(
                    os.path.join(args.output_dir, 'adversary.obj'),
                    adv_vft[0][0],
                    adv_vft[1][0],
                    adv_vft[2][0]
                )

            if (args.light_intensity_ambient_min != args.light_intensity_ambient_max) or (args.light_intensity_directional_min != args.light_intensity_directional_max):
                renderer = nr.Renderer(
                    camera_mode=renderers.DEFAULT_CAMERA_MODE,
                    image_size=args.image_size,
                    light_intensity_ambient=np.random.uniform(args.light_intensity_ambient_min, args.light_intensity_ambient_max),
                    light_intensity_directional=np.random.uniform(args.light_intensity_directional_min, args.light_intensity_directional_max),
                )
                renderer_high_res = nr.Renderer(
                    camera_mode=renderers.DEFAULT_CAMERA_MODE,
                    image_size=HIGH_RES,
                    light_intensity_ambient=np.random.uniform(args.light_intensity_ambient_min, args.light_intensity_ambient_max),
                    light_intensity_directional=np.random.uniform(args.light_intensity_directional_min, args.light_intensity_directional_max),
                )
            renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
            renderer_high_res.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

            raw_image = renderer(*obj_vft)
            adv_image = renderer(*vft)
            adv_image_big = renderer_high_res(*vft)

            if bg:
                bg_img = bg.render_image(center_crop=center_crops[args.scene_name], batch_size=1)
                raw_image = combiner.combine_images_in_order([bg_img, renderer(*obj_vft)], bg_img.shape)
                adv_image = combiner.combine_images_in_order([bg_img, renderer(*vft)], bg_img.shape)
            if bg_big:
                bg_img_big = bg_big.render_image(center_crop=center_crops[args.scene_name], batch_size=1)
                adv_image_big = combiner.combine_images_in_order([bg_img_big, renderer_high_res(*vft)], bg_img_big.shape)

            renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth * 180.0 / args.validation_range)
            cube_image = renderer(*adv_vft)

            y_adv = model(adv_image)
            y_raw = model(raw_image)

            # These 3 will be passed to a writer
            gif_tensors.append(adv_image)
            gif_tensors_big.append(adv_image_big)
            gif_tensors_adversary.append(cube_image)

            for k in TOP_COUNTS:
                y_adv_label = torch.topk(y_adv, k)[1].detach().cpu().numpy()
                y_raw_label = torch.topk(y_raw, k)[1].detach().cpu().numpy()
                if ytrue_label in y_raw_label:
                    correct_raw[k] += 1
                    adv_labels.append(y_adv_label[0])
                if ytrue_label in y_raw_label and ytrue_label in y_adv_label:
                    correct_adv[k] += 1
                if args.target_class > -1 and ytrue_label in y_raw_label and args.target_class in y_adv_label:
                    correct_target[k] += 1

        # Draw gifs
        utils.save_torch_gif(
            os.path.join(
                args.output_dir,
                '{}_and_{}{}_ele{}.gif'.format(args.scene_name, args.nobj, args.attacker_name, elevation)),
            torch.cat(gif_tensors, dim=0))
        utils.save_torch_gif(
            os.path.join(
                args.output_dir,
                '{}_and_{}{}_ele{}_hq.gif'.format(args.scene_name, args.nobj, args.attacker_name, elevation)),
            torch.cat(gif_tensors_big, dim=0))
        utils.save_torch_gif(
            os.path.join(
                args.output_dir,
                '{}{}_ele{}.gif'.format(args.nobj, args.attacker_name, elevation)),
            torch.cat(gif_tensors_adversary, dim=0))

    # Report metrics
    for k in TOP_COUNTS:
        print("########")
        print("TOP-{}".format(k))
        print("Raw accuracy:{}".format(correct_raw[k] / (len(elevation_loop) * len(loop))))
        print("Attack accuracy:{}".format((correct_raw[k] - correct_adv[k]) / (0.0 + correct_raw[k])))

        if args.target_class > -1:
            print("Target attack on {}: {}, {} out of {} times ".format(
                args.target_class,
                label_names[args.target_class],
                correct_target[k],
                correct_raw[k]))
            loss_handler['targeted_attack'][top_counts].append(correct_target[k] / (0.0 + correct_raw[k]))

        loss_handler['raw_accuracy'][k].append((correct_raw[k] / (len(elevation_loop) * len(loop))))
        loss_handler['attack_accuracy'][k].append(((correct_raw[k] - correct_adv[k])/(0.0 + correct_raw[k])))
        loss_handler['correct_raw'][k].append(((correct_raw[k])))
        loss_handler['correct_adv'][k].append(((correct_adv[k])))
        loss_handler['correct_target'][k].append(correct_target[k])
        loss_handler.log_epoch(writer_tf, k)
        print("########")
        print("########")


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
        writer_tf=None,
        loss_handler=None,
        ):
    renderer, camera_distance, elevation, azimuth = renderers.get_renderer(args.image_size, base_object=args.scene_name)
    renderer_high_res, _, _, _ = renderers.get_renderer(HIGH_RES, base_object=args.scene_name)

    obj_vft = base_object.render_parameters()
    with torch.no_grad():
        base_image = renderer(*(obj_vft))  # 1, 3, is, is
        if bg:
            bg_img = bg.render_image(center_crop=center_crops[args.scene_name], batch_size=1)
            base_image = combiner.combine_images_in_order([bg_img, base_image], bg_img.shape)
        utils.save_torch_image(os.path.join(args.output_dir, 'original_image.png'), base_image[0])

        ytrue = (F.softmax(model(base_image),dim=1))
        ytrue_label = int(torch.argmax(ytrue).detach().cpu().numpy())
        ytopk = torch.topk(ytrue, 5)[1].detach().cpu().numpy()

    print("Raw image classified by the classifier as: {} with p: {}".format(label_names[ytrue_label], ytrue[0][ytrue_label]))
    if args.target_class>-1:
        print("Using a targeted attack to classify the image as: {}".format(label_names[args.target_class]))

    # Define an optimizer
    optimizer = optim.Adam(
        list(filter(lambda p: p.requires_grad, [v for _, v in parameters.items()])),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    batch_size = args.batch_size
    for i in range(args.max_iterations):
        # Forward pass
        # obj_vft is alreay defined
        adv_vfts = [adv_obj.render_parameters(
            affine_transform=wavefront.create_affine_transform(
                parameters['scaling{}'.format(k)],
                parameters['translation{}'.format(k)],
                parameters['rotation{}'.format(k)],
                args.adv_ver,
            )) for k, adv_obj in adv_objs.items()]

        vft = combiner.combine_objects(
            *([[obj_vft[i]] + [adv_vft[i] for adv_vft in adv_vfts] for i in range(3)])
        )

        # Add ability to sample from set of azimuths.
        if batch_size > 1:
            rot_matrices = []
            for idx in range(batch_size):  # batch_size=1 works really.
                if args.num_azimuth > 0:
                    angles = np.linspace(-args.training_range, args.training_range, args.num_azimuth)
                    angle_idx = np.random.randint(0, args.num_azimuth - 1)
                    angle = angles[angle_idx]
                else:
                    angle = np.random.uniform(-args.training_range, args.training_range)
                rot_matrices.append(utils.create_rotation_y(angle))
            rot_matrices = torch.cat(rot_matrices).cuda()
            vft = wavefront.prepare_y_rotated_batch(vft, batch_size, rot_matrices)
            delta_azimuth = 0.0
        else:
            # projection, parameters
            if args.num_azimuth > 0:
                angles = np.linspace(-args.max_training_azimuth_deviation, args.max_training_azimuth_deviation, args.num_azimuth)
                angle_idx = np.random.randint(0, args.num_azimuth - 1)
                angle = angles[angle_idx]
            else:
                delta_azimuth = np.random.uniform(-args.max_training_azimuth_deviation, args.max_training_azimuth_deviation)

        if args.num_azimuth > 0:
            delta_elevation = 0
        else:
            delta_elevation = np.random.uniform(args.min_training_elevation_delta, args.max_training_elevation_delta)

        # Training light intensity.
        if (args.num_azimuth == -1) and ((args.light_intensity_ambient_min != args.light_intensity_ambient_max) or (args.light_intensity_directional_min != args.light_intensity_directional_max)):
            renderer = nr.Renderer(
                camera_mode=renderers.DEFAULT_CAMERA_MODE,
                image_size=args.image_size,
                light_intensity_ambient=np.random.uniform(args.light_intensity_ambient_min, args.light_intensity_ambient_max),
                light_intensity_directional=np.random.uniform(args.light_intensity_directional_min, args.light_intensity_directional_max),
            )

        renderer.eye = nr.get_points_from_angles(
            camera_distance,
            elevation + delta_elevation,
            azimuth + delta_azimuth
        )

        image = renderer(*vft)  # [bs, 3, is, is]
        if bg:
            bg_img = bg.render_image(center_crop=center_crops[args.scene_name], batch_size=batch_size)
            image = combiner.combine_images_in_order([bg_img, image], image.shape)

        pred_prob_y = F.softmax(model(image), dim=1)

        loss = loss_fns.untargeted_loss_fn(pred_prob_y, ytrue_label)
        if args.target_class > -1:
            loss += loss_fns.targeted_loss_fn(pred_prob_y, ytopk, target_class)

        # Regularization terms
        adv_vfts_base = [adv_obj_base.render_parameters(
            affine_transform=wavefront.create_affine_transform(
                parameters['scaling{}'.format(k)],
                parameters['translation{}'.format(k)],
                parameters['rotation{}'.format(k)],
                args.adv_ver,
            )) for k, adv_obj_base in adv_objs_base.items()]

        if args.nps:
            loss += 5 * regularization.nps(image)
        if args.fna_ad:
            for k, adv_vft in enumerate(adv_vfts):
                loss += 2*regularization.fna_ad(adv_vft[0], adv_vft[1],adv_vfts_base[k][0])

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

        # Evaluate attack and log numbers.
        if i % EVAL_EVERY == 0:
            iteration_metrics = get_metrics(
                image,
                pred_prob_y,
                ytrue_label,
                target_y=args.target_class,
                adv_vfts=adv_vfts,
                adv_vfts_base=adv_vfts_base
            )
            for k, v in iteration_metrics.items():
                loss_handler[k][i].append(v)
            loss_handler.log_epoch(writer_tf, i)
        if i % PHOTO_EVERY == 0:
            utils.save_torch_image(
                os.path.join(args.output_dir, 'training_iter{}.png'.format(i)), image[0])


            if (args.num_azimuth == -1) and ((args.light_intensity_ambient_min != args.light_intensity_ambient_max) or (args.light_intensity_directional_min != args.light_intensity_directional_max)):
                renderer_high_res = nr.Renderer(
                    camera_mode=renderers.DEFAULT_CAMERA_MODE,
                    image_size=HIGH_RES,
                    light_intensity_ambient=np.random.uniform(args.light_intensity_ambient_min, args.light_intensity_ambient_max),
                    light_intensity_directional=np.random.uniform(args.light_intensity_directional_min, args.light_intensity_directional_max),
                )

            image_hq = renderer_high_res(*vft)  # [bs, 3, is, is]
            if bg_big:
                bg_img_hq = bg_big.render_image(center_crop=center_crops[args.scene_name], batch_size=batch_size)
            image_hq = combiner.combine_images_in_order([bg_img_hq, image_hq], image_hq.shape)
            utils.save_torch_image(
                os.path.join(args.output_dir, 'training_iter{}_hq.png'.format(i)), image_hq[0])


def main():
    args = get_args()
    # Get classifiers
    model, label_names, img_size = get_classifier(args.classifier, args.classifier_path, args.labels_path)
    args.image_size = img_size
    # Instantiate objects
    if args.background != '':
        bg = background.Background(os.path.join(args.data_dir, args.background), img_size)
        bg_big = background.Background(os.path.join(args.data_dir, args.background), HIGH_RES)
    else:
        bg = bg_big = None
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
            args.attacker_name,
        )
        adv_objs[k] = adv_obj
        adv_objs_base[k] = adv_obj_base

        # INITIALIZATION OF PARAMETERS IS IMPORTANT
        for param_name, v in adv_obj.init_parameters(args, k, base_object).items():
            parameters['{}{}'.format(param_name, k)] = v

    writer_tf = SummaryWriter(log_dir=args.tensorboard_dir)
    loss_handler = utils.LossHandler()

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
        writer_tf=writer_tf,
        loss_handler=loss_handler,
    )
    # Evaluate
    with torch.no_grad():
        test(
            args,
            bg=bg,
            base_object=base_object,
            model=model,
            label_names=label_names,
            adv_objs=adv_objs,
            parameters=parameters,
            bg_big=bg_big,
            adv_objs_base=adv_objs_base,
            writer_tf=writer_tf,
            loss_handler=loss_handler,
        )


if __name__ == '__main__':
    main()

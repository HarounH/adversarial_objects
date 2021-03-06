import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import neural_renderer as nr
import json
import pdb
def parse_shapenet_code(name):
    # Returns T/F, ImagenetClass/F, model_idx/F
    if name.startswith('shapenet.'):
        return True, name.split('.')[1], name.split('.')[2]
    else:
        return False, None, None


class Object(nn.Module):
    def __init__(self, obj_filename, texture_size=2, adv_ver=False, adv_tex=False, rng_tex=False, normalization=False):
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
            normalization=normalization,
        )
        if rng_tex:
            textures = 0.1 + 0.8 * torch.rand_like(textures, device='cuda')
        # Case 1: Textures are parameters
        if adv_tex:
            self.textures = nn.Parameter(textures.unsqueeze(0)).cuda()
        else:
            self.textures = textures.unsqueeze(0).cuda()

        # Case 2: Vertices are parameters
        if adv_ver:
            self.vertices = vertices
            xvals, self.const_inds = self.vertices[:, 0].topk(3, largest=False)  # nv, 3
            _, self.var_inds = self.vertices[:, 0].topk(self.vertices.shape[0] - 3, largest=True)  # nv, 3
            var_inds = self.var_inds
            const_inds = self.const_inds

            self.vertices = self.vertices - torch.tensor([xvals[0], 0.0, 0.0],device = 'cuda')
            self.vertices[const_inds[1]] -= torch.tensor([xvals[1] - xvals[0], 0.0, 0.0],device = 'cuda')
            self.vertices[const_inds[2]] -= torch.tensor([xvals[2] - xvals[0], 0.0, 0.0],device = 'cuda')
            self.vertices += torch.tensor([0.02, 0.0, 0.0],device = 'cuda')
            self.vertices_constants = self.vertices[const_inds, :]
            self.vertices_vars = nn.Parameter(self.vertices[var_inds, :])
            self.vertices = torch.zeros(self.vertices.shape, dtype=torch.float, device='cuda')
            self.vertices[const_inds] = self.vertices_constants
            self.vertices[var_inds] = self.vertices_vars

        self.vertices = vertices[None, :, :].cuda()
        self.faces = faces[None, :, :].cuda()
        self.info = {}
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
        return [vertices.cuda(), faces.cuda(), textures.cuda()]

    def init_parameters(self, args, k, base_object):
        assert (self.adv_ver or self.adv_tex), "init_parameters was called on non-adversarial object"
        parameters = {}
        if args.adv_ver:
            parameters['vertices'] = self.vertices_vars
        if args.adv_tex:
            parameters['texture'] = self.textures

        if args.translation_clamp > 0:
            if args.scene_name == 'coffeemug':
                translation_param = torch.tensor([
                        0.4,
                        3 * args.scale0 * np.cos(2 * np.pi * k / args.nobj) - 0.6,
                        3 * args.scale0 * np.sin(2 * np.pi * k / args.nobj)
                    ], dtype=torch.float, device='cuda')
            elif args.scene_name == 'stopsign':
                translation_param = (
                    torch.tensor([0,0.02,0.02], device='cuda') * torch.randn((3, ), device='cuda')
                    + torch.tensor([
                        0.02,
                        5 * args.scale0*np.cos(2 * np.pi * k / args.nobj),
                        5 * args.scale0*np.sin(2 * np.pi * k / args.nobj)
                    ], dtype=torch.float, device='cuda')
                )
            elif args.scene_name.startswith('shapenet'):
                translation_param_info = base_object.info['translation_param_init']
                translation_param = (
                    torch.tensor(translation_param_info['group'], device='cuda')
                    + (
                        torch.tensor(translation_param_info['random_multiplier'], device='cuda')
                        * torch.randn((3, ), device='cuda'))
                    + torch.tensor([
                        0.0,
                        translation_param_info['circle_radius'][1] * np.cos(2 * np.pi * k / args.nobj),
                        translation_param_info['circle_radius'][2] * np.sin(2 * np.pi * k / args.nobj)
                    ], dtype=torch.float, device='cuda')
                )
            else:
                raise NotImplementedError(
                    'Implement initialization of parameters in \
                        modules.wavefront.init_parameters for \
                        scene {}'.format(args.scene_name))

            translation_param.requires_grad_(True if args.adv_ver else False)
            parameters['translation'] = translation_param

        if args.rotation_clamp > 0:
            if args.scene_name.startswith('shapenet'):
                rotation_param_info = base_object.info['rotation_param_init']
                rotation_param_info = torch.tensor(rotation_param_info['absolute'], device='cuda', requires_grad=True)
            else:
                rotation_param = 0
            rotation_param += torch.randn((3,), requires_grad=True, device='cuda')
            parameters['rotation'] = rotation_param
        else:
            parameters['rotation'] = torch.zeros((3,), requires_grad=False, device='cuda')

        if args.scaling_clamp > 0:
            scaling_param = args.scale0 * (torch.ones((3,), requires_grad=False, device='cuda'))
            if args.scene_name.startswith('shapenet'):
                scaling_param *= torch.tensor(base_object.info['scaling_param_init']['multiplier'])
            scaling_param.requires_grad_(True)
            parameters['scaling'] = scaling_param
        else:
            parameters['scaling'] = torch.ones((3,), requires_grad=False, device='cuda') * args.scale0
        return parameters


def prep_stop_sign(stop_sign):
    stop_sign_translation = torch.tensor([0.0, -1.5, 0.0]).cuda()
    stop_sign.vertices += stop_sign_translation
    return stop_sign


def prep_mug(base_object):
    base_object.vertices -= base_object.vertices.mean(1)
    base_object.vertices /= 6.0 #0.5 #2.0 #
    base_object.vertices += torch.tensor([-0.5, 0.0, -0.5], device='cuda')
    return base_object


def prep_shapenet_maker(scale=1, translation=[0, 0, 0], info={}):
    def prep_shapenet(base_object):
        print('prep shapenet with scale={}'.format(scale))
        base_object.vertices -= torch.tensor(translation, device='cuda')
        base_object.vertices /= scale
        base_object.info = info
        return base_object
    return prep_shapenet


objects_dict = {
    'stopsign': ['adversarial_objects/data/custom_stop_sign.obj', prep_stop_sign],
    'coffeemug': ['adversarial_objects/data/coffeemug.obj', prep_mug],
    'cube': ['adversarial_objects/data/evil_cube_1.obj', None],
    'slab': ['adversarial_objects/data/evil_slab_1.obj', None],
    'small_icosphere': ['adversarial_objects/data/obj2.obj', None],
    'big_icosphere': ['adversarial_objects/data/obj3.obj', None],
    'cylinder': ['adversarial_objects/data/obj4.obj', None],
    'divcube': ['adversarial_objects/data/subdivided_cube.obj', None],
}

with open('prepared_shapenet_info.json','r') as json_file:
    data = json.load(json_file)
    for k, v in data.items():
        objects_dict[k] = [
            v['wavefront_file'],
            prep_shapenet_maker(
                scale=v['base_object_init']['scale'],
                translation=v['base_object_init']['translation'],
                info={
                    'rotation_param_init': v['rotation_param_init'],
                    'scaling_param_init': v['scaling_param_init'],
                    'translation_param_init': v['translation_param_init']
                }
            )
        ]
        # lambda x: prep_shapenet(x,
        #     scale=v['base_object_init']['scale'],
        #     translation=v['base_object_init']['translation'],
        #     info={'rotation_param_init':v['rotation_param_init'],
        #         'scaling_param_init':v['scaling_param_init'],
        #         'translation_param_init':v['translation_param_init']
        #         }
        #     )


def load_obj(obj_name, prep_fn_=None, *args, **kwargs):
    # Shapenet objects
    is_shapenet, class_name, model_idx = parse_shapenet_code(obj_name)
    path, prep_fn = tuple(objects_dict[obj_name])

    if prep_fn_ is not None:
        prep_fn = prep_fn_

    if prep_fn is None:
        prep_fn = lambda x: x

    with torch.no_grad():
        obj = Object(path, normalization=is_shapenet, *args, **kwargs)
        obj.name = obj_name
    return prep_fn(obj)


def create_affine_transform(scaling, translation, rotation, adv_ver):
    scaling_matrix = torch.eye(4)
    for i in range(3):
        scaling_matrix[i, i] = scaling[i]
    translation_matrix = torch.eye(4)
    for i in range(1 if adv_ver else 0, 3):
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


def prepare_y_rotated_batch(vft, batch_size, rot_matrices):
    new_v = torch.bmm(
        torch.cat(
            (
                vft[0].expand(batch_size, *(vft[0].shape[1:])),
                torch.ones(([batch_size] + list(vft[0].shape[1:-1]) + [1])).float().cuda(),
            ),
            dim=2),
        rot_matrices,
    )[:, :, :3]

    new_f = vft[1].expand(batch_size, *(vft[1].shape[1:]))
    new_t = vft[2].expand(batch_size, *(vft[2].shape[1:]))
    return [new_v, new_f, new_t]

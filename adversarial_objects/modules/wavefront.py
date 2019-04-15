import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import neural_renderer as nr


class Object(nn.Module):
    def __init__(self, obj_filename, texture_size=2, adv_ver=False, adv_tex=False, rng_tex=False):
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
            normalization=False,
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

    def init_parameters(self, args):
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
                    torch.tensor([0,0.02,0.02], device="cuda") * torch.randn((3,), device='cuda')
                    + torch.tensor([
                        0.02,
                        5*args.scale0*np.cos(2 * np.pi * k / args.nobj),
                        5*args.scale0*np.sin(2 * np.pi * k / args.nobj)
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
            rotation_param = torch.randn((3,), requires_grad=True, device='cuda')
            parameters['rotation'] = rotation_param
        else:
            parameters['rotation'] = torch.zeros((3,), requires_grad=False, device='cuda')

        if args.scaling_clamp > 0:
            scaling_param = args.scale0 * (torch.ones((3.,), requires_grad=False, device='cuda'))
            scaling_param.requires_grad_(True)
            parameters['scaling'] = scaling_param
        else:
            parameters['scaling'] = torch.ones((3.,), requires_grad=False, device='cuda') * args.scale0


def prep_stop_sign(stop_sign):
    stop_sign_translation = torch.tensor([0.0, -1.5, 0.0]).cuda()
    stop_sign.vertices += stop_sign_translation
    return stop_sign


def prep_mug(base_object):
    base_object.vertices -= base_object.vertices.mean(1)
    base_object.vertices /= 6.0 #0.5 #2.0 #
    base_object.vertices += torch.tensor([-0.5,0.0,-0.5], device="cuda")
    return base_object


objects_dict = {
    'stopsign': ['adversarial_objects/data/custom_stop_sign.obj', prep_stop_sign],
    'coffeemug': ['adversarial_objects/data/coffeemug.obj', prep_mug],
    'cube': ['adversarial_objects/data/evil_cube_1.obj', None],
    'small_icosphere': ['adversarial_objects/data/obj2.obj', None],
    'big_icosphere': ['adversarial_objects/data/obj3.obj', None],
    'cylinder': ['adversarial_objects/data/obj4.obj', None],
    'divcube': ['adversarial_objects/data/subdivided_cube.obj', None],
}


def load_obj(obj_name, prep_fn_=None, *args, **kwargs):
    path, prep_fn = *objects_dict[obj_name]

    if prep_fn_ is not None:
        prep_fn = prep_fn_

    if prep_fn is None:
        prep_fn = lambda x: x

    with torch.no_grad():
        obj = Object(path, *args, **kwargs)
    return prep_fn(obj)

import math

import torch
import numpy

import neural_renderer as nr


class Renderer():
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0,0,1],
                 near=0.1, far=100,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0]):
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.P = P
            if isinstance(self.P, numpy.ndarray):
                self.P = torch.from_numpy(self.P).cuda()
            if self.P is None or P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
                raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(P.shape[0], 1)
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')


        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction 

        # rasterization
        self.rasterizer_eps = 1e-3

    def render_silhouettes(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            vertices = nr.projection(vertices, self.P, self.dist_coeffs, self.orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images

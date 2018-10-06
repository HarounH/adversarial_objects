""" File that loads model, scene, creates adversary
Unlike attack.py, it doesn't bother using a model
"""
from __future__ import print_function, absolute_import
import os
import argparse
import torch
import numpy as np
import tqdm
import imageio
import pdb
import neural_renderer as nr
from skimage.io import imread, imsave

# directory set up
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
output_dir = os.path.join(current_dir, 'output')
try:
    os.makedirs(output_dir)
except:
    pass

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("-bg", "--background", dest="background", type=str, default="", help="Path to background file (image)")
parser.add_argument("-obj", "--object", dest="object", type=str, default="custom_stop_sign.obj", help="Name of .obj file containing stop sign")
parser.add_argument("-o", "--output", dest="output_filename", type=str, default="custom_stop_sign.png", help="Filename for output image")
parser.add_argument("-angle", "--azimuth", dest="azimuth", type=float, default=90.0, help="Azimuth angle to use for rendering")
parser.add_argument("--camera_distance", dest="camera_distance", type=float, default=2.732, help="Camera distance to use for rendering")
parser.add_argument("--image_size", type=int, default=256, help="Size of square image")
args = parser.parse_args()



class Object:
    def __init__(self, obj_filename, texture_wrapping=None, use_bilinear=False):
        assert torch.cuda.is_available()
        vertices, faces, textures = nr.load_obj(
            obj_filename,
            load_texture=True,
            texture_size=4,
            texture_wrapping=texture_wrapping,
            use_bilinear=use_bilinear,
        )
        # TODO: Figure out how to load correctly.
        self.vertices = vertices[None, :, :].cuda()
        self.faces = faces[None, :, :].cuda()
        self.textures = textures.unsqueeze(0).cuda()

if __name__ == '__main__':
    # TODO: Load background
    # Load stop-sign
    for wrapping in ['REPEAT', 'MIRRORED_REPEAT', 'CLAMP_TO_EDGE', 'CLAMP_TO_BORDER']:
        for bilinear in [True, False]:
            print("Starting {}".format(os.path.join(output_dir, wrapping + str(bilinear) + args.output_filename)))
            stop_sign = Object(
                os.path.join(data_dir, args.object),
                texture_wrapping=wrapping,
                use_bilinear=bilinear
            )
            # Create adversary
            # Render into image
            renderer = nr.Renderer(camera_mode='look_at', image_size=args.image_size)
            camera_distance = args.camera_distance
            elevation = 30.0
            azimuth = args.azimuth
            renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
            obj_image = renderer(stop_sign.vertices, stop_sign.faces, stop_sign.textures)
            obj_image = obj_image.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]

            # Output
            # pdb.set_trace()
            imsave(os.path.join(output_dir, wrapping + str(bilinear) + args.output_filename), obj_image)

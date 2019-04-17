"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio
import pdb
import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument("-lt", "--load_textures", type=bool, default=False, help="Whether or not there is a texture file to be loaded")
    parser.add_argument('-ts', '--texture_size', type=int, default=4, help="Size of texture file")
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    # texture_size = 2

    # load .obj
    vertices, faces, textures = nr.load_obj(args.filename_input, load_texture=args.load_textures, texture_size=args.texture_size)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    textures = textures.unsqueeze(0).cuda()
    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    # correct_textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    pdb.set_trace()
    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    writer = imageio.get_writer(args.filename_output, mode='I')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()

import torch
from torch import nn
from torch.nn import functional as F
import neural_renderer as nr


rendering_parameters = {
    'coffeemug': {
        'camera_distance': 2.72-0.75,
        'elevation': 5.0,
        'azimuth': 90.0,
    },
    'stopsign': {
        'camera_distance': 2.72-0.75,
        'elevation': 0.0,
        'azimuth': 90.0,
    },
}


def get_renderer(image_size, camera_mode='look_at', base_object=None):
    renderer = nr.Renderer(camera_mode=camera_mode, image_size=image_size)
    if base_object is not None:
        d = rendering_parameters[base_object]
        camera_distance, elevation, azimuth = d['camera_distance'], d['elevation'], d['azimuth']
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    return renderer, camera_distance, elevation, azimuth

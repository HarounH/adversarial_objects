import torch
from torch import nn
from torch.nn import functional as F
import neural_renderer as nr
import json

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

with open('prepared_shapenet_info.json','r') as json_file:  
    data = json.load(json_file)
    for k,v in data.items():
        rendering_parameters[k] = v['rendering']

DEFAULT_CAMERA_MODE = 'look_at'
def get_renderer(image_size, camera_mode=DEFAULT_CAMERA_MODE, base_object=None):
    renderer = nr.Renderer(camera_mode=camera_mode, image_size=image_size)
    if base_object is not None:
        d = rendering_parameters[base_object]
        camera_distance, elevation, azimuth = d['camera_distance'], d['elevation'], d['azimuth']
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    return renderer, camera_distance, elevation, azimuth

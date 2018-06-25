import torch
import torch.nn as nn

import neural_renderer as nr

class Mesh:

    def __init__(self, vertices, faces, textures, texture_size=4):
        self.vertices = nn.Parameter(vertices)
        self.faces = faces
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]

        # create textures
        if textures is None:
            shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
            self.textures = nn.Parameter(0.05*torch.randn(*shape))
            self.texture_size = texture_size
        else:
            self.texture_size = textures.shape[0]

    def cuda(self):
        self.vertices = self.vertices.cuda()
        self.faces = self.faces.cuda()
        self.textures = self.textures.cuda()

    def cpu(self):
        self.vertices = self.vertices.cpu()
        self.faces = self.faces.cpu()
        self.textures = self.textures.cpu()

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.faces = self.faces.to(device)
        self.textures = self.textures.to(device)

    @classmethod
    def fromobj(cls, filename_obj, load_textures=False, normalization=True, texture_size=4, load_texture=False):
        if load_textures:
            vertices, faces, textures = nr.load_obj(filename_obj,
                                                    normalization=normalization,
                                                    texture_size=texture_size,
                                                    load_texture=True)
        else:
            vertices, faces = nr.load_obj(filename_obj,
                                          normalization=normalization,
                                          texture_size=texture_size,
                                          load_texture=False)
            textures = None
        return cls(vertices, faces, textures, texture_size)


    def forward(self):
        return self.vertices, self.faces, self.textures.sigmoid()


""" File that loads model, scene, creates adversary
Then trains the adversary too.
"""

import os
import json
import argparse
import numpy as np
import scipy
from random import shuffle
from time import time
import sys
import pdb
import pickle as pk
from collections import defaultdict
import itertools
# pytorch imports
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data.sampler as samplers
import tqdm
import imageio
from skimage.io import imread, imsave
import pdb
from utils import LossHandler
from utils import ImagenetReader
import regularization
import pretrainedmodels
from torch.utils.data import DataLoader
from torchvision import utils as vutils, datasets, transforms
from torch import nn
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
# parameters
parser = argparse.ArgumentParser()
# Input output specifications
parser.add_argument("--image_size", default=299, type=int, help="Square Image size that neural renderer should create for attacking.")
parser.add_argument("--victim_name", default="inceptionv3", help="Path relative current_dir to attack model.")
parser.add_argument("--imagenet_path", default="imagenet/imagenet_labels.csv", help="Path where the imagenet_labels.csv is located.")

args = parser.parse_args()
args.cuda = True

np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == '__main__':
    # Load imagenet_labels
    imagenet_labels = ImagenetReader(args.imagenet_path)
    # Load imagenet model
    victim = pretrainedmodels.__dict__[args.victim_name](num_classes=1000, pretrained='imagenet').cuda()  # nn.Module
    victim.eval()
    # Load dataset
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomRotation((270,270)),
        transforms.ToTensor(),
        ])
    dataset = ImageFolderWithPaths('adversarial_objects/data/real_world_4/', transform=data_transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    coffee_mug_count = [0,0]
    counts = [0,0]
    for i,x in enumerate(loader):
        image = x[0].cuda()
        clss = x[1]
        ytrue = (F.softmax(victim(image),dim=1))
        ytrue_label = int(torch.argmax(ytrue).detach().cpu().numpy())
        ytopk = torch.topk(ytrue,1)[1].detach().cpu().numpy()
        print("{}th Image, Class {},  classified by the classifier as: {} with p: {}".format(x[2][0].split('/')[-1], clss, imagenet_labels[ytrue_label], ytrue[0][ytrue_label]))
        # print("{}th Image, Class {}, coffee mug probability {}".format(i, clss, ytrue[0][504]))
        if 504 in ytopk:
            coffee_mug_count[clss]+=1
        counts[clss]+=1
    print(counts)
    print(coffee_mug_count)
    
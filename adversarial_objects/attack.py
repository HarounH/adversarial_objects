""" File that loads model, scene, creates adversary
Then trains the adversary too.
"""
import os
import argparse
import torch
import numpy as np
import tqdm
import imageio
import pdb
import neural_renderer as nr

# constants
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
output_dir = os.poth.join(current_dir, 'output')

# parameters
parser = argparse.ArgumentParser()
raise NotImplementedError("Create parser")
args = parser.parse_args()


if __name__ == '__main__':
    # Load background
    # Load stop-sign
    # Create adversary
    # Render into image
    # Load model
    # Ensure that adversary is adversarial
    # Optimize loss function
    # Output

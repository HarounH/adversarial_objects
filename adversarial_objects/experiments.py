""" File that runs the experiments required
"""
import os
import argparse
import pdb

from utils import SignReader
def mkdir_p(path):
    try:
        os.makedirs(path)
    except:
    	pass

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")

args = parser.parse_args()

def create_cmd(arguments):
	cmd = 'python adversarial_objects/refac_atk.py --bs 64 --victim_path victim_0/working_model_97.01223438506116.chk --seed {}'.format(args.seed)
	for arg in arguments:
		cmd += ' {}'.format(arg)
	return cmd


def varying_regularization():
	expt_dir = 'regularization_study_test_rand_tex/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	base_arguments = '--lr 0.05 -iter 500 --adv_tex --adv_ver --output_dir {}'.format(output_dir)
	regs = ['surface_area','aabb_volume','radius_volume','edge_length']
	for x,i in enumerate(regs):
		cmd = create_cmd([base_arguments,'--reg {} --output {}.png --tensorboard_dir tensorboard_{} '.format(i, i, i)])
		os.system(cmd)
	i = "fna_ad"
	cmd = create_cmd([base_arguments,'--fna_ad --output {}.png --tensorboard_dir tensorboard_{} '.format(i, i)])
	os.system(cmd)
	i = "fna_ad_edge_length"
	cmd = create_cmd([base_arguments,'--fna_ad --reg edge_length --output {}.png --tensorboard_dir tensorboard_{} '.format(i, i)])
	os.system(cmd)
	i = "fna_ad_edge_length_surface_area"
	cmd = create_cmd([base_arguments,'--fna_ad --reg edge_length surface_area --output {}.png --tensorboard_dir tensorboard_{} '.format(i, i)])
	os.system(cmd)
	i = "fna_ad_edge_length_nps"
	cmd = create_cmd([base_arguments,'--fna_ad --reg edge_length --nps --output {}.png --tensorboard_dir tensorboard_{} '.format(i, i)])
	os.system(cmd)

def varying_targets():
	base_arguments = '--lr 0.05 -iter 1000 --adv_tex --adv_ver'
	for x in range(1,43):
		cmd = create_cmd([base_arguments,'--reg {} --fna_ad --nobj 1 --target_class {} --output targeted_{}.png '.format('edge_length',x,x)])
		os.system(cmd)


def basic():
	base_arguments = '--lr 0.05 -iter 300 --adv_tex --adv_ver'
	cmd = create_cmd([base_arguments])
	os.system(cmd)
if __name__ == '__main__':
    varying_regularization()

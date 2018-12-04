""" File that runs the experiments required
"""
import os
import argparse
import pdb

from utils import SignReader


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1337, type=int, help="Seed for numpy and pytorch")

args = parser.parse_args()

def create_cmd(arguments):
	cmd = 'python adversarial_objects/refac_atk.py --bs 64 --victim_path victim_0/working_model_97.01223438506116.chk --seed {}'.format(args.seed)
	for arg in arguments:
		cmd += ' {}'.format(arg)
	return cmd


def varying_regularization():
	base_arguments = '--lr 0.05 -iter 300 --adv_tex --adv_ver'
	regs = ['surface_area','aabb_volume','radius_volume','surface_area aabb_volume']
	for x,i in enumerate(regs):
		cmd = create_cmd([base_arguments,'--reg {} --output {}.png'.format(i,x)])
		os.system(cmd)

def varying_targets():
	base_arguments = '--lr 0.01 -iter 1000 --adv_tex --adv_ver'
	for x in range(1,43):
		cmd = create_cmd([base_arguments,'--reg {} --nobj 1 --target_class {} --output targeted_{}.png'.format('surface_area aabb_volume',x,x)])
		os.system(cmd)
def basic():
	base_arguments = '--lr 0.05 -iter 300 --adv_tex --adv_ver'
	cmd = create_cmd([base_arguments])
	os.system(cmd)
if __name__ == '__main__':
    # varying_regularization()
    # basic()
    varying_targets()

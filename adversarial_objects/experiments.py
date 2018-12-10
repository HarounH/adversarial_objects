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

seeds = [42, 53345, 1337, 80085, 8008]
args = parser.parse_args()

def create_cmd(arguments):
	cmd = 'python adversarial_objects/refac_atk.py --bs 64 --victim_path victim_0/working_model_97.01223438506116.chk --seed {}'.format(args.seed)
	for arg in arguments:
		cmd += ' {}'.format(arg)
	return cmd


def varying_regularization():
	expt_dir = 'varying_regularization_largest_w/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	for seed in seeds:
		base_arguments = '--seed {} --reg_w 0.05 --lr 0.005 -iter 500 --adv_tex --adv_ver '.format(seed)
		regs = ['surface_area','aabb_volume','radius_volume','edge_length']

		i = 'no_regularization'
		opdir = output_dir+'/'+i
		mkdir_p('adversarial_objects/'+opdir)
		cmd = create_cmd([base_arguments,'--output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{} '.format(seed, opdir, i, seed)])
		os.system(cmd)


		i = "fna_ad_edge_length_nps"
		opdir = output_dir+'/'+i
		mkdir_p('adversarial_objects/'+opdir)
		cmd = create_cmd([base_arguments,'--fna_ad --reg edge_length --nps --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{} '.format(seed, opdir, i, seed)])
		os.system(cmd)


		for x,i in enumerate(regs):
			opdir = output_dir+'/'+i
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd([base_arguments,'--reg {} --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{} '.format(i, seed, opdir, i, seed)])
			os.system(cmd)
		i = "fna_ad"
		opdir = output_dir+'/'+i
		mkdir_p('adversarial_objects/'+opdir)
		cmd = create_cmd([base_arguments,'--fna_ad --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{} '.format(seed, opdir, i, seed)])
		os.system(cmd)
		i = "fna_ad_edge_length"
		opdir = output_dir+'/'+i
		mkdir_p('adversarial_objects/'+opdir)
		cmd = create_cmd([base_arguments,'--fna_ad --reg edge_length --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{} '.format(seed, opdir, i, seed)])
		os.system(cmd)
		i = "fna_ad_edge_length_surface_area"
		opdir = output_dir+'/'+i
		mkdir_p('adversarial_objects/'+opdir)
		cmd = create_cmd([base_arguments,'--fna_ad --reg edge_length surface_area --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{} '.format(seed, opdir, i, seed)])
		os.system(cmd)

def varying_nobj():
	expt_dir = 'varying_nobj/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	for seed in seeds:
		base_arguments = '--seed {} --lr 0.005 -iter 500 --adv_tex --adv_ver --fna_ad --reg edge_length '.format(seed)
		for j in [1,2,4,8,16]:
			opdir = output_dir+'/'+str(j)
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd([base_arguments,'--nobj {} --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{}'.format(j, seed, opdir, seed, j)])
			os.system(cmd)

def varying_training_range():
	expt_dir = 'varying_training_range/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	for seed in seeds:
		base_arguments = '--seed {} --lr 0.005 -iter 500 --adv_tex --adv_ver --fna_ad --reg edge_length '.format(seed)
		for j in [35,45]:
			opdir = output_dir+'/'+str(j)
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd([base_arguments,'--training_range {} --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{}'.format(j, seed, opdir, seed, j)])
			os.system(cmd)

def varying_ts():
	expt_dir = 'varying_tex/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	for seed in seeds:
		base_arguments = '--seed {} --lr 0.005 -iter 500 --adv_tex --adv_ver --fna_ad --reg edge_length '.format(seed)
		for j in [2,3,4,5,6,7,8]:
			opdir = output_dir+'/'+str(j)
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd([base_arguments,'--ts {} --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{}'.format(j, seed, opdir, seed, j)])
			os.system(cmd)

def varying_rand_ts():
	expt_dir = 'varying_rand_tex/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	for seed in seeds:
		base_arguments = '--seed {} --rng_tex --lr 0.005 -iter 500 --adv_tex --adv_ver --fna_ad --reg edge_length '.format(seed)
		for j in [2,3,4,5,6,7,8]:
			opdir = output_dir+'/'+str(j)
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd([base_arguments,'--ts {} --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{}'.format(j, seed, opdir, seed, j)])
			os.system(cmd)

def varying_shapes():
	expt_dir = 'varying_shapes/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	for seed in seeds:
		base_arguments = '--seed {} --lr 0.005 -iter 500 --adv_tex --adv_ver --fna_ad --reg edge_length '.format(seed)
		for j in (['obj3.obj', 'obj2.obj', 'obj4.obj','evil_cube_1.obj']):
			opdir = output_dir+'/'+j
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd([base_arguments,'--attacker_path {} --output {}.png --output_dir {} --tensorboard_dir tensorboard_{}_{}'.format(j, seed, opdir, seed, j)])
			os.system(cmd)

def varying_targets():
	expt_dir = 'varying_targets_super_good/'
	output_dir = 'output/{}'.format(expt_dir)
	mkdir_p('adversarial_objects/'+output_dir)
	for seed in seeds:
		base_arguments = '--seed {} --lr 0.005 -iter 1000 --nobj 2 --ts 4 --training_range 70 --rng_tex --adv_tex --adv_ver --fna_ad --reg edge_length '.format(seed)

		for i in range(42):
			j = "fna_ad_edge_length_nps"
			opdir = output_dir+'/'+j
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd([base_arguments,'--nps --output {}_{}_{}.png --target_class {} --output_dir {} --tensorboard_dir tensorboard_{}_{}_{}'.format(seed, j, i, i, opdir, j, seed, i)])
			os.system(cmd)

		for i in range(42):
			j = "no_regularization"
			opdir = output_dir+'/'+j
			mkdir_p('adversarial_objects/'+opdir)
			cmd = create_cmd(['--seed {} --lr 0.001 -iter 500 --attacker_path obj4.obj --nobj 2 --ts 4 --rng_tex --adv_tex --adv_ver '.format(seed),' --output {}_{}.png --target_class {} --output_dir {} --tensorboard_dir tensorboard_{}_{}_{}'.format(seed, j, i, opdir, j, seed, i)])
			os.system(cmd)


def basic():
	base_arguments = '--lr 0.05 -iter 300 --adv_tex --adv_ver'
	cmd = create_cmd([base_arguments])
	os.system(cmd)


if __name__ == '__main__':
	varying_targets()

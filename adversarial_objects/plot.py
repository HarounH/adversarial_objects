from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import argparse
import pdb
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import SignReader
# signnames = SignReader("victim_0/signnames.csv")
def mkdir_p(path):
    try:
        os.makedirs(path)
    except:
    	pass

# def plot_imagenet(
# 	expt_dir = None,
# 	):
# 	logging_dir = 'adversarial_objects/output/{}/'.format(expt_dir)
# 	event_paths = glob.glob(os.path.join(logging_dir, "*", "tensorboard"))
# 	tf.logging.set_verbosity(tf.logging.ERROR)
# 	for metric in ['surface_area','fna_ad','edge_length','edge_variance', 'attack_accuracy']:
# 		for path in event_paths:
# 			event_acc = EventAccumulator(path)
# 			event_acc.Reload()
# 			try:
# 				print(metric, event_acc.Scalars('{}'.format(metric))[-1].value, event_acc.Scalars('{}'.format(metric))[-1].step,  path.split('/')[-2], sep = ', ')
# 			except:
# 				print("Didnt find")


# plot_imagenet(
# 	expt_dir = 'regularization_experiments_adv_tex_2',
# )



#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

def plot(
	expt_dir = None,
	metric = None,
	expt_list = None,
	expt_list_x_label = None,
	wildcards = [''],
	legend_labels = ['Pass some label for each wildcard'],
	xlabel = 'Pass some label pls',
	ylabel = 'Pass some label pls',
	title = "Somethign please",
	plotname = 'default.png',
	dpi = 500,
	matcher = 1,
	):
	logging_dir = 'adversarial_objects/output/course_expts/{}/'.format(expt_dir)
	metric_dict = {}
	seeds = [42, 53345, 1337, 80085, 8008]
	plt.figure()
	for widx, wildcard in enumerate(wildcards):
		x = []
		y = []
		yerr = []
		for jidx, j in enumerate(expt_list):
			if matcher == 1:
				event_paths = glob.glob(os.path.join(logging_dir, "*{}".format(wildcard), "tensorboard_*_{}".format(j)))
			elif matcher == 2:
				event_paths = glob.glob(os.path.join(logging_dir, "*{}".format(wildcard), "tensorboard_{}*".format(j)))
			elif matcher == 3:
				event_paths = glob.glob(os.path.join(logging_dir, "{}".format(j), "tensorboard_{}*".format(j)))
			# pdb.set_trace()	
			if isinstance(j,int) and j==14:
				continue
			metric_dict[j] = []
			for path in event_paths:
				print(path)
				event_acc = EventAccumulator(path)
				event_acc.Reload()
				try:
					metric_dict[j].append((event_acc.Scalars('{}'.format(metric))[0].value))
				except:
					print("Didnt find")
			if j > 9 and j!=16:
				x.append(jidx)
				y.append(np.mean(metric_dict[j]))
				yerr.append(np.std(metric_dict[j]))
			else:
				colors = ['-','--','-.',':']
				colors2 = ['r','g','m','c']
				plt.axhline(y=np.mean(metric_dict[j]), label = "{}".format(expt_list_x_label[jidx]), linewidth = 1, color = colors2[jidx-6], linestyle=colors[jidx-6])
		if legend_labels is not None:
			plt.errorbar(x, y, yerr=yerr, fmt = 'o', label=legend_labels[widx])
		else:
			plt.errorbar(x, y, yerr=yerr, fmt = 'o')
	plt.xticks(x, expt_list_x_label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	# plt.grid()
	# plt.yscale("log")
	plt.title(title)
	# if legend_labels is not None:
	plt.legend(loc=2) # 'upper left')
	mkdir_p('adversarial_objects/plots/{}'.format(expt_dir))
	plt.savefig('adversarial_objects/plots/{}/{}'.format(expt_dir,plotname),dpi=dpi)

def plot_lines(
	expt_dir = None,
	metric = None,
	expt_list = None,
	expt_list_x_label = None,
	wildcards = [''],
	legend_labels = ['Pass some label for each wildcard'],
	xlabel = 'Pass some label pls',
	ylabel = 'Pass some label pls',
	title = "Somethign please",
	plotname = 'default.png',
	dpi = 500,
	matcher = 1,
	):
	logging_dir = 'adversarial_objects/output/{}/'.format(expt_dir)
	seeds = [42, 53345, 1337, 80085, 8008]
	plt.figure()
	for widx, wildcard in enumerate(wildcards):
		x = []
		y = []
		yerr = []

		for jidx, j in enumerate(expt_list):
			if matcher == 1:
				event_paths = glob.glob(os.path.join(logging_dir, "*{}".format(wildcard), "tensorboard_*_{}".format(j)))
			elif matcher == 2:
				event_paths = glob.glob(os.path.join(logging_dir, "*{}".format(wildcard), "tensorboard_{}*".format(j)))
			elif matcher == 3:
				event_paths = glob.glob(os.path.join(logging_dir, "{}".format(j), "tensorboard_{}*".format(j)))
			# pdb.set_trace()
			if isinstance(j,int) and j==14:
				continue

			metric_dict = np.zeros((len(event_paths),500))
			for pidx, path in enumerate(event_paths):
				print(path)
				event_acc = EventAccumulator(path)
				event_acc.Reload()
				# pdb.set_trace()
				try:
					x = [i.step for i in (event_acc.Scalars('{}'.format(metric)))]
					ytmp = [i.value for i in (event_acc.Scalars('{}'.format(metric)))]
					metric_dict[pidx,:] = np.array(ytmp)
				except:
					print("Didnt find")
			y = np.mean(metric_dict,axis=0)
			yerr = np.std(metric_dict,axis=0)
			# pdb.set_trace()
			skip=1
			if matcher==3:
				x = x[-500:]
			if legend_labels is not None:
				plt.plot(x, y, label=legend_labels[jidx])
			else:
				plt.plot(x, y)
	if expt_list_x_label is not None:
		plt.xticks(x, expt_list_x_label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid()
	# plt.yscale("log")
	plt.title(title)
	if legend_labels is not None:
		plt.legend()
	mkdir_p('adversarial_objects/plots/{}'.format(expt_dir))
	plt.savefig('adversarial_objects/plots/{}/{}'.format(expt_dir,plotname),dpi=dpi)


seeds = [42, 53345, 1337, 80085, 8008]
expt_list = [2,3,4,5,6,7,8]
expt_list = [i for i in range(42)]



matplotlib.rcParams.update({'font.size': 8})
metrics = ['nps','fna_ad','edge_length','edge_variance','surface_area','stop_sign_probability']
ylabels = ['NPS','Surface Normal Deviation', 'Mean Edge Length','Variance of Edge Length', 'Total Surface Area', 'P(Stop Sign)']
		


# What to plot
plot_regularization = False
plot_nobj = False
plot_shapes = False
plot_tex = False
plot_training_range = True

if False:
	plot(
		expt_dir = 'varying_targets',
		metric = 'targeted_attack',
		expt_list = [i for i in range(42)],
		expt_list_x_label = [i for i in range(42)],
		wildcards = ['fna_ad_edge_length','fna_ad_edge_length_nps'],
		legend_labels = ['No NPS regularization','NPS regularization'],
		xlabel = 'Target Class IDs',
		ylabel = 'Targeted Attack Success Rate',
		title = "Targeted Attack",
		plotname = 'targeted.png',
		dpi = 500,
	)
	
if plot_training_range:
	training_range_expt_list = [10,20,30,45,50,60,0,3,6,16]
	training_range_expt_labels = ['10','20','30','40','50','60','1 fixed angle','3 fixed angles','5 fixed angles','15 fixed angles']
	for expt_dir in ['varying_training_range']:
		titles = ['Attack Success Rate' for i in range(len(ylabels))]
		# for idx in range(len(metrics)):
		# 	plot_lines(
		# 		expt_dir = expt_dir,
		# 		metric = metrics[idx],
		# 		expt_list = training_range_expt_list,
		# 		expt_list_x_label = None,
		# 		wildcards = [''],
		# 		legend_labels = training_range_expt_labels,
		# 		xlabel = 'Iterations',
		# 		ylabel = ylabels[idx],
		# 		title = titles[idx],
		# 		plotname = '{}.png'.format(metrics[idx]),
		# 		dpi = 500,
		# 	)
		plot(
			expt_dir = expt_dir,
			metric = 'attack_accuracy',
			expt_list = training_range_expt_list,
			expt_list_x_label = training_range_expt_labels,
			wildcards = ['*'],
			legend_labels = None,
			xlabel = 'Training Range',
			ylabel = 'Attack Success Rate',
			title = "Effect of varying training range",
			plotname = 'training_range_attack_acc.png',
			dpi = 500,
		)

if plot_regularization:
	reg_expt_list = ['no_regularization', 'surface_area','aabb_volume','radius_volume','edge_length',  "fna_ad", "fna_ad_edge_length_nps", "fna_ad_edge_length", "fna_ad_edge_length_surface_area"]
	reg_expt_labels = ['No \n Regular-\nization','Surface\nArea \n (SA)','AABB\nVolume \n(AV)',
	'Radius\nVolume\n(RV)','Edge\nLength\n(EL)', "Surface Normal\n (SN)",
	"SN\nEL\nNPS","SN\nEL","SN\nEL\nSA"]
	expt_dir = 'varying_regularization_largest_w'
	titles = ['Effect of Varying Regularization' for i in range(len(ylabels))]
	for idx in range(len(metrics)):
		plot_lines(
			expt_dir = expt_dir,
			metric = metrics[idx],
			expt_list = reg_expt_list,
			expt_list_x_label = None,
			wildcards = [''],
			legend_labels = reg_expt_labels,
			xlabel = 'Iterations',
			ylabel = ylabels[idx],
			title = titles[idx],
			plotname = '{}.png'.format(metrics[idx]),
			dpi = 500,
			matcher=3,
		)

	plot(
		expt_dir = expt_dir,
		metric = 'attack_accuracy',
		expt_list = reg_expt_list,
		expt_list_x_label = reg_expt_labels,
		wildcards = [''],
		legend_labels = None,
		xlabel = '',
		ylabel = titles[0],
		title = "Attack Success Rate for Varying Regularization",
		plotname = 'attack_accuracy.png',
		dpi = 500,
		matcher=3,
	)

if plot_tex:
	tex_expt_list = [2,3,4,5,6,7,8]
	tex_expt_labels = [2,3,4,5,6,7,8]
	for expt_dir in ['varying_tex','varying_rand_tex']:
		titles = ['Effect of Number of Textures on Face' for i in range(len(ylabels))]
		for idx in range(len(metrics)):
			plot_lines(
				expt_dir = expt_dir,
				metric = metrics[idx],
				expt_list = tex_expt_list,
				expt_list_x_label = None,
				wildcards = [''],
				legend_labels = tex_expt_labels,
				xlabel = 'Iterations',
				ylabel = ylabels[idx],
				title = titles[idx],
				plotname = '{}.png'.format(metrics[idx]),
				dpi = 500,
			)
		plot(
			expt_dir = expt_dir,
			metric = 'attack_accuracy',
			expt_list = tex_expt_list,
			expt_list_x_label = tex_expt_labels,
			wildcards = ['*'],
			legend_labels = None,
			xlabel = 'Objects',
			ylabel = titles[0],
			title = titles[0],
			plotname = 'tex_attack_acc.png',
			dpi = 500,
		)

if plot_shapes:
	shapes_expt_list = ['obj3.obj', 'obj2.obj', 'obj4.obj','evil_cube_1.obj']
	# shapes_expt_list = [1,2,4,8,16]
	shapes_expt_labels = ['Icosphere with\n42 vertices', 'Icosphere with \n12 vertices', 'Cylinder', 'Cube']
	expt_dir = 'varying_shapes'
	titles = ['Effect of Shapes Used' for i in range(len(ylabels))]
	for idx in range(len(metrics)):
		plot_lines(
			expt_dir = expt_dir,
			metric = metrics[idx],
			expt_list = shapes_expt_list,
			expt_list_x_label = None,
			wildcards = [''],
			legend_labels = shapes_expt_labels,
			xlabel = 'Iterations',
			ylabel = ylabels[idx],
			title = titles[idx],
			plotname = '{}.png'.format(metrics[idx]),
			dpi = 500,
		)
	plot(
		expt_dir = expt_dir,
		metric = 'attack_accuracy',
		expt_list = shapes_expt_list,
		expt_list_x_label = shapes_expt_labels,
		wildcards = ['*'],
		legend_labels = None,
		xlabel = 'Objects',
		ylabel = 'Attack Success Rate',
		title = titles[0],
		plotname = 'shapes_attack_acc.png',
		dpi = 500,
	)

if plot_nobj:
	nobj_expt_list = [1,2,4,8,16]
	nobj_expt_labels = ['1','2','4','8','16']
	expt_dir = 'varying_nobj'
	titles = ['Effect of Number of Objects Used' for i in range(len(ylabels))]
	for idx in range(len(metrics)):
		plot_lines(
			expt_dir = expt_dir,
			metric = metrics[idx],
			expt_list = nobj_expt_list,
			expt_list_x_label = None,
			wildcards = [''],
			legend_labels = nobj_expt_labels,
			xlabel = 'Iterations',
			ylabel = ylabels[idx],
			title = titles[idx],
			plotname = '{}.png'.format(metrics[idx]),
			dpi = 500,
		)
	plot(
		expt_dir = expt_dir,
		metric = 'attack_accuracy',
		expt_list = [1,2,4,8,16],
		expt_list_x_label = [1,2,4,8,16],
		wildcards = ['*'],
		legend_labels = None,
		xlabel = 'Number of objects',
		ylabel = 'Attack Success Rate',
		title = titles[0],
		plotname = 'nobj_attack_acc.png',
		dpi = 500,
	)







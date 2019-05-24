import os
import sys
import subprocess
import json


gpu_id = int(sys.argv[1])
print('Using gpu ', gpu_id)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

ALL_SCENES_START_IDX = 0
ALL_SCENES_END_IDX = 200
# Bag
# shapenet.bag.0: Failure.
# shapenet.bag.10
todo_scenes_by_gpu_id = {
    0: [],
    1: [],
    2: [],
    3: [],
}

with open('prepared_shapenet_info.json', 'r') as f:
    shapenet_info = json.load(f)
    all_scenes = sorted(list(shapenet_info.keys()))[ALL_SCENES_START_IDX: ALL_SCENES_END_IDX]

for i, name in enumerate(all_scenes):
    todo_scenes_by_gpu_id[i % 4].append(name)

todo_scenes = todo_scenes_by_gpu_id[gpu_id]
print('Running experiments on ', todo_scenes)
attacker_names = ['cube', 'slab']

good_targets = {
    k: [632] for k in all_scenes
}

# Training ranges
untargeted_training_ranges = [0, 15, 30, 45, 60]
untargeted_num_azimuths_per_training_range = {x: [-1] for x in untargeted_training_ranges}
untargeted_num_azimuths_per_training_range[60].extend([5, 15])

targeted_training_ranges = [60]
targeted_num_azimuths_per_training_range = {x: [-1] for x in targeted_training_ranges}
targeted_num_azimuths_per_training_range[60].extend([5, 15])

for scene_name in todo_scenes:
    for attacker_name in attacker_names:
        # Untargeted attacks
        # for training_range in untargeted_training_ranges:
        #     for naz in untargeted_num_azimuths_per_training_range[training_range]:
        #         run = "untargeted_{}_{}_{}_{}".format(scene_name, attacker_name, training_range, naz)
        #         cmdstr = "python adversarial_objects/attack_classifier.py -s {} inceptionv3 -a {} --training_range {} -naz {} --adv_ver --adv_tex -iter 200 -r {} --scale0 0.2 --lr 0.01 --no_gif".format(scene_name, attacker_name, training_range, naz, run)
        #         print('Running', cmdstr)
        #         cmd = cmdstr.split(' ')
        #         outputdir = 'adversarial_objects/new_output/{}'.format(run)
        #         os.makedirs(outputdir, exist_ok=True)
        #         with open(os.path.join(outputdir, 'stdout.txt'), 'w') as f:
        #             f.write(cmdstr)
        #             ret = subprocess.run(cmd, stdout=f, env=os.environ)
        # Targeted attacks
        for target in good_targets[scene_name]:
            for training_range in targeted_training_ranges:
                for naz in targeted_num_azimuths_per_training_range[training_range]:
                    run = "target{}_{}_{}_{}_{}".format(target, scene_name, attacker_name, training_range, naz)
                    cmdstr = "python adversarial_objects/attack_classifier.py -s {} inceptionv3 -a {} --training_range {} -naz {} --target_class {} --adv_ver --adv_tex -iter 200 -r {} --scale0 0.2 --lr 0.005 --no_gif".format(scene_name, attacker_name, training_range, naz, target, run)
                    print('Running', cmdstr)
                    cmd = cmdstr.split(' ')
                    outputdir = 'adversarial_objects/new_output/{}'.format(run)
                    os.makedirs(outputdir, exist_ok=True)
                    with open(os.path.join(outputdir, 'stdout.txt'),'w') as f:
                        f.write(cmdstr)
                        ret = subprocess.run(cmd, stdout=f, env=os.environ)

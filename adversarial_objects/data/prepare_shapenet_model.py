import os
import json
import argparse
import pickle
import glob
import csv
import copy

RELATION_TO_PROJECT_ROOT = 'adversarial_objects/data'

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('shapenet_dir', help='Location that shapenet')
parser.add_argument('-o', '--output_file', default='/data/adversarial_objects/adversarial_objects/data/shapenet_model_list.json')
parser.add_argument('-il', '--imagenet_labels', default='../../imagenet/imagenet_labels.csv', help='Path to imagenet labels')
parser.add_argument('-nm', '--num_models_per_synset', default=30, help='Number of models to use per synset class')
args = parser.parse_args()

# Parse taxonomy
taxonomy_file = os.path.join(args.shapenet_dir, 'taxonomy.json')
with open(taxonomy_file, 'r') as f:
    taxonomy = json.load(f)
synset_ids = [x['synsetId'] for x in taxonomy]
synset_id2details = {x['synsetId']: x for x in taxonomy}

# Parse shapenet directory

output = {}
imagenet_id2names = {}
imagenet_name2id = {}

with open(args.imagenet_labels, 'r') as f:
    reader = csv.reader(f)
    for i, classname in reader:
        imagenet_id2names[int(i)] = classname
        for potential_name in classname.split(' '):
            imagenet_name2id[potential_name] = int(i)


non_class_count = 0
for synset_id in synset_ids:
    tax_i = synset_id2details[synset_id]
    found = False
    # Figure out imagenet class
    for _name in tax_i['name'].split(','):
        if _name in imagenet_name2id:
            name = _name
            found = True
            break

    if not found:
        non_class_count += 1
        print('synset_id {}: {} doesn\'t have an imagenet class'.format(
            synset_id, tax_i['name']))
        continue

    for model_idx, model_folder in enumerate(sorted(glob.glob('{}/{}/*'.format(args.shapenet_dir, synset_id)))):
        if model_idx == args.num_models_per_synset:
            break
        model_folder_suffix = os.path.basename(model_folder)
        # Relative to data/
        obj_file = os.path.join(RELATION_TO_PROJECT_ROOT, os.path.join(model_folder, 'models', 'model_normalized.obj'))
        _codename = 'shapenet.{}.{}'.format(name, model_idx)

        output_i = copy.deepcopy(tax_i)
        output_i['wavefront_file'] = obj_file
        output_i['model_folder_name'] = model_folder_suffix

        # Assuming k objects are being placed
        output_i['translation_param_init'] = {
            'group': [0.0, 0.0, 0.0],
            'random_multiplier': [0.02, 0.02, 0.02],
            'circle_radius': [0.0, 0.9, 0.9],
        }
        output_i['scaling_param_init'] = {
            'multiplier': [1.0, 1.0, 1.0],
        }
        output_i['rotation_param_init'] = {
            'absolute': [0.0, 0.0, 0.0],
        }

        output_i['rendering'] = {
            'camera_distance': 2.72,
            'elevation': 5.0,
            'azimuth': 90.0,
        }

        output_i['base_object_init'] = {
            # Translate first then scale.
            'translation': [0.0, 0.0, 0.0],
            'scale': 1.0,
        }
        output[_codename] = output_i

print('{} / {} synset-ids were not processed'.format(non_class_count, len(synset_ids)))
with open(args.output_file, 'w') as outfile:
    json.dump(output, outfile, indent=4)
print('Dumped {} object descriptions to {}'.format(len(output), args.output_file))

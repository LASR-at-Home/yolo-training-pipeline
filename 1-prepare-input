#!/usr/bin/env python3

import os
import json
import yaml
import subprocess

from src.data import list_inputs
from ordered_set import OrderedSet

FRAME_RATE = 0.5
TARGET_FRAMES = 25
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/image-seq"

CLASSES = OrderedSet()

# convert all videos
for id, cls in list_inputs():
    CLASSES.add(cls)
    output_subdir = os.path.join(OUTPUT_DIR, id)

    if os.path.exists(output_subdir):
        print(id, cls, 'has been skipped as frames already exist!')
        continue

    os.makedirs(output_subdir)

    input_path = os.path.join(INPUT_DIR, ','.join([id,cls]) + '.mp4')
    output_path = os.path.join(output_subdir, "output_%04d.png")

    cmd = f'ffmpeg -i "{input_path}" -r {FRAME_RATE} -vframes {TARGET_FRAMES} "{output_path}"'
    subprocess.run(cmd, shell=True)

# output class mappings
CLASS_MAP = {}
NAME_MAP = {}
counter = 0
for cls in CLASSES:
    CLASS_MAP[cls] = counter
    NAME_MAP[counter] = cls
    counter += 1

with open('data/classes.json', 'w') as f:
    f.write(json.dumps(CLASS_MAP))

# create dataset.yaml file for training
DATASET = {
    'path': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'dataset'),
    'train': 'train/images',
    'val': 'train/images',
    'test': 'test/images',
    'names': NAME_MAP
}

with open('data/dataset.yaml', 'w') as f:
    f.write(yaml.dump(DATASET))

#!/usr/bin/env python3
import os
import sys
import shutil

if len(sys.argv) < 2:
    print('Usage:')
    print('./0-clean-up [masks|objects|dataset|all]')
    exit()

target = sys.argv[1]

def delete_excl_gitkeep(path):
    files = os.listdir(path)
    for file in files:
        if file == '.gitkeep':
            continue
        
        fn = os.path.join(path, file)
        if os.path.isfile(fn):
            os.remove(fn)
        else:
            shutil.rmtree(fn)

# delete objects
if target == 'objects' or target == 'all':
    delete_excl_gitkeep('data/object-cutouts')

# delete masks
if target == 'masks' or target == 'all':
    delete_excl_gitkeep('data/masks')

# delete dataset
if target == 'dataset' or target == 'all':
    delete_excl_gitkeep('data/dataset/train/images')
    delete_excl_gitkeep('data/dataset/train/labels')

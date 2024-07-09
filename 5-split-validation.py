#!/usr/bin/env python3
import os

MIN_I = 2001
MAX_I = 2500

OUTPUT_IMAGES_DIR = 'data/dataset/train/images'
OUTPUT_LABELS_DIR = 'data/dataset/train/labels'

VAL_IMAGES_DIR = 'data/dataset/val/images'
VAL_LABELS_DIR = 'data/dataset/val/labels'

for i in range(MIN_I, MAX_I + 1):
    img_fn = os.path.join(OUTPUT_IMAGES_DIR, f'{i}.png')
    label_fn = os.path.join(OUTPUT_LABELS_DIR, f'{i}.txt')

    if os.path.exists(img_fn):
        os.rename(img_fn, os.path.join(VAL_IMAGES_DIR, f'{i}.png'))

    if os.path.exists(label_fn):
        os.rename(label_fn, os.path.join(VAL_LABELS_DIR, f'{i}.txt'))

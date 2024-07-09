#!/usr/bin/env python3
import os
import cv2
import math
import random
import numpy as np

from scipy import ndimage

# debug
import matplotlib.pyplot as plt

# configuration
SEED = 48543
GEN_IMAGES = 5
DEBUG = 'DEBUG' in os.environ and os.environ['DEBUG'] == '1'
MAX_SIZE = 512
random.seed(SEED)

META_DIR = 'data/input'
INPUT_DIR = 'data/masks'
INPUT_SEQ_DIR = 'data/image-seq'
OUTPUT_DIR = 'data/object-cutouts'

# load metadata
classes = [
    file[:-4].split(',')
        for file
        in os.listdir(META_DIR)
        if file.endswith('.mp4')
]

# load masks
masks = os.listdir(INPUT_DIR)
for mask_file in masks:
    if mask_file == '.gitkeep':
        continue

    # find the image we want to crop using mask
    fn, ext = mask_file.split('.', 1)
    id, image_file = fn.split('-')
    image_ext, _ = ext.rsplit('.', 1)

    # find the class
    cls = None
    for known_id, known_class in classes:
        if id == known_id:
            cls = known_class

    if cls is not None:
        print(cls, image_file)
    else:
        raise Exception(f'Unknown class for {id}')

    # image
    image_path = os.path.join(INPUT_SEQ_DIR, id, f'{image_file}.{image_ext}')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _channels = image.shape

    # mask
    mask_image = np.zeros((height, width), np.uint8)
    xyn = np.load(os.path.join(INPUT_DIR, mask_file))
    xy = np.array([[x * width, y * height] for x, y in xyn])

    try:
        cv2.fillPoly(mask_image, pts = np.int32([xy]), color = (255,255,255))
    except Exception:
        continue

    # full object image
    object_image = np.zeros((height, width, 4), dtype=np.uint8)
    object_image[:,:,0:3] = image
    object_image[:, :, 3] = mask_image

    # image boundary
    y,x = object_image[:,:,3].nonzero()
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)

    # cropped images
    object_image = object_image[miny:maxy, minx:maxx]
    height, width, _channels = object_image.shape

    # transform the mask
    xy = xy - [minx, miny]

    # scale the image down a bit to avoid long processing time
    sf = None
    if height > width:
        if height > MAX_SIZE:
            sf = MAX_SIZE / height
    elif width > MAX_SIZE:
        sf = MAX_SIZE / width

    if sf is not None:
        object_image = cv2.resize(object_image, (round(width * sf), round(height * sf)))
        height, width, _channels = object_image.shape
        xy = xy * sf

    # generate a bunch of rotations
    for i in range(0, GEN_IMAGES):
        DEG = random.randint(0, 360)
        print(DEG, 'degrees')

        rid = random.randint(10000, 99999)
        out_file = f'{cls},{i},{rid}-{image_file}'
        out_filepath = os.path.join(OUTPUT_DIR, f'{out_file}.png')

        if os.path.exists(out_filepath) and not DEBUG:
            continue

        theta = math.radians(DEG)
        rotate = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        translate = [
            width / 2,
            height / 2
        ]

        mask = xy - translate
        mask = mask @ rotate
        mask = mask + translate

        # transform image
        old_height, old_width, _ = object_image.shape
        rotated_object_image = ndimage.rotate(object_image, DEG)
        new_height, new_width, _ = rotated_object_image.shape

        mask = mask + [
            (new_width - old_width) / 2,
            (new_height - old_height) / 2
        ]

        try:
            if DEBUG:
                cv2.drawContours(rotated_object_image, np.int32([mask]), -1, (255,0,0,255), 4)
                plt.imshow(rotated_object_image)
                plt.show()
            else:
                image_to_write = cv2.cvtColor(rotated_object_image, cv2.COLOR_BGRA2RGBA)
                cv2.imwrite(out_filepath, image_to_write)
                np.save(os.path.join(OUTPUT_DIR, out_file), mask)
        except Exception as e:
            print('SKIP! Ran into exception', e)

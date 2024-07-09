#!/usr/bin/env python3
import os
import cv2
import json
import random
import numpy as np

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# debug
import matplotlib.pyplot as plt

SEED = 48543
GEN_IMAGES = 2500
MIN_PER_IMAGE = 2
MAX_PER_IMAGE = 7
MIN_COVER = 0.1
MAX_COVER = 0.6
IMGSZ = 640
DEBUG = 'DEBUG' in os.environ and os.environ['DEBUG'] == '1'
INPUT_DIR = 'data/object-cutouts'
INPUT_BACKGROUND_DIR = 'data/backgrounds'
OUTPUT_IMAGES_DIR = 'data/dataset/train/images'
OUTPUT_LABELS_DIR = 'data/dataset/train/labels'
random.seed(SEED)

# load class data
with open('data/classes.json', 'r') as f:
    CLASS = json.loads(f.read())

# create bucketed file source list
# this is really shit code but it works :)
cutouts = {}
cutout_index = {}
for cls in CLASS.keys():
    cutouts[cls] = []
    cutout_index[cls] = 0

for file in os.listdir(INPUT_DIR):
    if file.endswith('.png'):
        cls, id, name = file.split(',', 2)
        cutouts[cls].append(file)

def shuffle_cutout(cls):
    random.shuffle(cutouts[cls])

    # we prioritise using all unique images before rotations
    cutouts[cls].sort(key=lambda x:int(x.split(',', 2)[1]))

for cls in CLASS.keys():
    shuffle_cutout(cls)

queue = [a for a in CLASS.keys()]
random.shuffle(queue)

def next_in_queue():
    global queue

    if len(queue) == 0:
        queue = [a for a in CLASS.keys()]
        random.shuffle(queue)
    
    return queue.pop()

def next_cutout():
    cls = next_in_queue()
    file = cutouts[cls][cutout_index[cls]]

    # increment index and reshuffle
    cutout_index[cls] += 1
    if cutout_index[cls] >= len(cutouts[cls]):
        cutout_index[cls] = 0
        shuffle_cutout(cls)

    # return file
    return file

def load_bg(file):
    print('Loading background...', file)
    img = cv2.imread(os.path.join(INPUT_BACKGROUND_DIR, file))
    if IMGSZ is not None:
        height, width, _ = img.shape
        img = cv2.resize(img,
            (round(width * (IMGSZ / height)), round(IMGSZ)))
    return img

# load backgrounds
backgrounds = [
    file for file in
        [load_bg(file)
            for file
            in os.listdir(INPUT_BACKGROUND_DIR)
            if file != '.gitkeep']
        if file is not None
]

random.shuffle(backgrounds)

# generate required number of composite images
for i in range(1, GEN_IMAGES + 1):
    print(f'[{i} / {GEN_IMAGES}]')

    background = backgrounds.pop(0)
    backgrounds.append(background)
    background_height, background_width, _ = background.shape

    plot = np.array(background, copy=True)
    GEN_OBJECTS = random.randint(MIN_PER_IMAGE, MAX_PER_IMAGE)
    SEGMENTS = []

    for _ in range(0, GEN_OBJECTS):
        cutout = next_cutout()
        print('Loading cutout', cutout)
        cls,_ = cutout.split(',', 1)

        cutout_image = cv2.imread(os.path.join(INPUT_DIR, cutout), cv2.IMREAD_UNCHANGED)
        cutout_height, cutout_width, _ = cutout_image.shape
        
        target_cover = random.uniform(MIN_COVER, MAX_COVER)
        try_width, try_height = (
            background_width * target_cover,
            background_height * target_cover
        )

        acceptable_width, acceptable_height = (
            min(cutout_width, min(background_width, (try_height / cutout_height) * cutout_width)),
            min(cutout_height, min(background_height, (try_width / cutout_width) * cutout_height)),
        )

        target_width, target_height = (
            min(acceptable_width, acceptable_height * (cutout_width / cutout_height)),
            min(acceptable_height, acceptable_width * (cutout_height / cutout_width)),
        )

        sf = target_width / cutout_width

        cutout_image = cv2.resize(cutout_image,
            (round(target_width), round(target_height)))
        cutout_height, cutout_width, _ = cutout_image.shape

        # target placement
        x = random.randint(0, background_width - cutout_width)
        y = random.randint(0, background_height - cutout_height)

        # create the overlay image
        overlay_image = np.zeros((background_height, background_width, 4), dtype=np.uint8)
        overlay_image[y:y+cutout_height,x:x+cutout_width] = cutout_image
        
        # extract alpha channel and use it as a mask
        alpha = overlay_image[:,:,3] / 255.0
        mask = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        # composite the images together
        plot = plot * (1.0 - mask) \
            + overlay_image[:,:,:3] * mask
        
        # load segment data
        segment = np.load(os.path.join(INPUT_DIR, cutout[:-3] + 'npy'))
        segment = np.array([
            [(x + px * sf) / background_width, (y + py * sf) / background_height]
                for px, py
                in segment
        ])

        try:
            # occlude existing masks
            polygon = Polygon(segment)
            for j in range(0, len(SEGMENTS)):
                # remove any points that are within the mask we just created
                cls_id, mask = SEGMENTS[j]
                keep = [
                    not polygon.contains(Point([x, y]))
                        for x, y
                        in mask
                ]
                
                new_mask = mask[keep]

                try:
                    # try to merge points into the object
                    merge_index = next(i for i, x in enumerate(keep) if not x)

                    new_polygon = Polygon(mask)
                    new_points = np.array([
                        [x, y]
                            for x, y
                            in segment
                            if new_polygon.contains(Point([x, y]))
                    ]) # NOTE: we assume there's only one segment of overlap
                    # this will not work properly with more but this is more
                    # of a bandaid fix than anything.

                    # FIXME: handle each segment of points and do some funny math
                    # to determine which segment from the object needs to be pulled out

                    # try to roughly match the direction of the two merged segments
                    isx, isy = new_mask[merge_index - 1]
                    iex, iey = new_mask[merge_index]

                    jsx, jsy = new_points[0]
                    jex, jey = new_points[len(new_points) - 1]

                    flip_x, flip_y = False, False

                    if (iex > isx and jsx > jex) or (isx > iex and jex > jsx):
                        flip_x = True
                    
                    if (iey > isy and jsy > jey) or (isy > iey and jey > jsy):
                        flip_y = True
                    
                    # apply operation
                    if flip_x or flip_y:
                        new_points = new_points[::-1]

                    SEGMENTS[j] = (cls_id, np.concatenate((new_mask[:merge_index], new_points, new_mask[merge_index:]), axis=0))
                except StopIteration:
                    pass
                except IndexError:
                    # give up if we can't read points
                    SEGMENTS[j] = (cls_id, new_mask)

            # add new mask
            SEGMENTS.append((CLASS[cls], segment))
        except Exception as e:
            print('Skipping object due to error')
            print(e)
            continue
    
    plot = plot.astype(np.uint8)

    if DEBUG:
        # plot segmentation masks
        i, j, k = 0, 0, 0
        for _, segment in SEGMENTS:
            segxy = np.array([
                [x * background_width, y * background_height]
                    for x, y
                    in segment
            ])

            cv2.drawContours(plot, np.int32([segxy]), -1, (i,j,k), 8)

            i += 128
            if i > 255:
                i = 0
                j += 128
                if j > 255:
                    j = 0
                    k += 128
                    if k > 255:
                        k = 0

        # show debug output
        plotted_image = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(plotted_image)
        plt.show()
    else:
        # write to the dataset
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f'{i}.png'), plot)
        with open(os.path.join(OUTPUT_LABELS_DIR, f'{i}.txt'), 'w') as f:
            f.write('\n'.join([
                str(cls) + ' ' + ' '.join([str(a) for a in segment.flatten().tolist()])
                    for cls, segment
                    in SEGMENTS
            ]))

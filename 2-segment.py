#!/usr/bin/env python3

import os
import cv2
import json
import math
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from src.data import list_inputs

from ultralytics import SAM
model = SAM('mobile_sam.pt')

TARGET_HEIGHT = 1024
IMAGE_DIR = "data/image-seq"
OUTPUT_DIR = "data/masks"
TEMPFILE = os.path.join(tempfile.mkdtemp(), 'tmp.png')

# the_number = 0

class Annotate:
    def __init__(self, initial_image, initial_bbox):
        self.ax = plt.gca()
        self.image = initial_image
        self.mask_image = None
        self.result = None
        self.bbox = None

        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(TEMPFILE, self.image)

        self.rect = Rectangle((0,0), 1, 1, facecolor='none', alpha=0.5, edgecolor='red', linewidth=1)

        if initial_bbox is None:
            self.x0 = None
            self.y0 = None
            self.x1 = None
            self.y1 = None
        else:
            x0, y0, x1, y1 = initial_bbox
            self.x0 = x0
            self.y0 = y0
            self.x1 = x1
            self.y1 = y1

            self.bbox = [self.x0, self.y0, self.x1, self.y1]
            self.inference()
    
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

        self.draw()
    
    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
    
    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata

        if self.x0 is None or self.x1 is None \
            or self.y0 is None or self.y1 is None:
            return

        if self.x0 > self.x1:
            tmp = self.x0
            self.x0 = self.x1
            self.x1 = tmp

        if self.y0 > self.y1:
            tmp = self.y0
            self.y0 = self.y1
            self.y1 = tmp

        self.bbox = [self.x0, self.y0, self.x1, self.y1]

        self.inference()
        self.draw()

    def draw(self):
        self.ax.cla()
        self.ax.imshow(self.image)

        if self.mask_image is not None:
            self.ax.imshow(self.mask_image)

        if self.x1 is not None:
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))

        self.ax.add_patch(self.rect)

        self.ax.figure.canvas.draw()
    
    def inference(self):
        self.result = model.predict(TEMPFILE, bboxes=self.bbox)[0]
        self.mask_image = self.result.plot()

        # also draw lines because they're easier to see
        cv2.drawContours(self.mask_image, np.int32([self.result.masks.xy[0]]), -1, (255,0,0,255), 4)

with open('data/classes.json', 'r') as f:
    CLASS = json.loads(f.read())

for id, cls in list_inputs():
    # find the image sequence directory
    image_subdir = os.path.join(IMAGE_DIR, id)

    if not os.path.exists(image_subdir):
        print(cls, 'has not been generated yet or is missing frames!')
        continue

    print('Processing [' + id + ']', '"' + cls + '"')
    files = sorted(os.listdir(image_subdir))

    last_bbox = None

    # read all images
    for image_name in files:
        # check if we've already created this mask
        mask_file = os.path.join(OUTPUT_DIR, f'{id}-{image_name}')
        if os.path.exists(f'{mask_file}.npy'):
            print('Skipping', f'{id}-{image_name}', 'as mask already exists!')
            continue

        # load image metadata
        filename = os.path.join(image_subdir, image_name)
        image = cv2.imread(filename)

        height, width, _ = image.shape
        scale_factor = TARGET_HEIGHT / max(height, width)
        width, height = (round(width * scale_factor), round(height * scale_factor))
        image = cv2.resize(image, (width, height))

        # annotate
        result = None
        while result is None:
            a = Annotate(initial_image=image, initial_bbox=last_bbox)
            plt.title(f'{cls}: {image_name}')
            plt.show()

            last_bbox = a.bbox
            result = a.result

        np.save(mask_file, result.masks.xyn[0])            

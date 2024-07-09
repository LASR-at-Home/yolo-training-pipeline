## Preparing Environment

This guide comes with multiple scripts used to automate the process of labeling, creating, and training the dataset. Begin by cloning the repository into a new directory:

```bash
git clone https://github.com/insertish/yolov8-auto-trainer
cd yolov8-auto-trainer
```

If you're on NixOS, you just need to run:

```bash
nix-shell
```

Otherwise, configure your Python environment:

```bash
python3.10 -m venv venv # must have Python 3.9+ installed
source venv/bin/activate
pip install -r requirements.txt

# .. you may need to install tk:
sudo apt-get install python3.10-tk
```

Alternatively, use Conda to setup your environment:

```bash
conda create --name yolo python=3.11
conda activate yolo
pip install -r requirements.txt
```

You can now validate everything works by running:

```bash
python 0-validate-env
```

## Gather Source Images

To begin you want to collect images or videos for each object you want to classify in your dataset:

- You'll want to capture the object from a wide range of angles or at least the ones you expect to find the object in.
- It is preferable to use a variety of lighting conditions or the lighting conditions you expect to infer on.
- You do not need to care about the background as long as its not too similar to the object you're trying to segment.
- If you expect to be running the model on a specific camera, you should source your images from the camera itself to best reflect image quality and object appearance.

Next, we prepare the data directory with all the images we want to use.

You will be either copying or creating empty video files in the `data/input` directory which provides the required metadata to the program, the filename format is `id,class.mp4` where:

- The `id` is a unique number used by the program to identify this video / set of images / etc.
- The `class` is the desired class that this object is detected as by the model.

For example:

```
0,izzy.mp4
1,pen.mp4
2,screwdriver.mp4
3,screwdriver.mp4
.. etc
```

### I have videos of my objects

You should copy your videos into the `data/input` directory using the filename format described earlier.

### I have a bunch of images of my objects

Given a class `object` with the ID `4`, you would first create a dummy video file:

```bash
touch data/input/4.mp4
```

Then create the corresponding image sequence folder:

```bash
mkdir data/image-seq/4
```

Now copy all of your images into the `data/image-seq/4` directory.

### I want to scrape images of my desired object off the internet

If you want more data to train on, you can try to find more images online by pulling them off a search engine such as Bing:

```bash
pip install bing_image_downloader
```

Then drop into a Python shell and run:

```python
from bing_image_downloader import downloader

downloader.download(
    "Costa Coffee",
    limit=128,
    output_dir="bing_images",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
    verbose=True,
)
```

## Gather Backgrounds For Synthetic Images

For creating the synthetic datasets, we need a collection of backgrounds.

TEST: 200 bgs for 1000 images vs. 1000 bgs for 1000 images

https://unsample.net/

## Create Segmentation Masks For Training

Firstly, convert any videos we still have into image sequences, generate the class mappings, and create the model information file:

```bash
python 1-prepare-input
```

Now we're ready to segment the actual objects themselves:

```bash
python 2-segment
```

A pyplot window should open in which you can draw a bounding box around the object, a segmentation mask should appear and you can either redraw or press `q` to confirm the mask.

It will then carry on to the next image. If it is the same object, then the bounding box will be kept and it'll automatically run segmentation again. In most cases you can just hit `q` again.

If you make a mistake at any point and save the wrong mask, you can delete the latest file from `data/masks` and then re-run the segmentation program after you finish.

## Create And Train Synthetic Dataset

After segmenting everything, we are ready to create our model.

### Object Cutouts

We begin by extracting objects and their corresponding masks from the source images.

TODO: options available for change

```bash
# create all object cutouts
python 4-generate-cutouts

# .. or first preview what they'll look like:
DEBUG=1 python 4-generate-cutouts
```

[grid image of cutouts]

### Synthetic Images

Then we can generate the synthetic images which will be used to train our model.

TODO: options available for change

```bash
# generate synthetic images
python 5-generate-composites

# .. or first preview what they'll look like:
DEBUG=1 python 5-generate-composites
```

[grid image of composites]

### Train Model

Finally, we can train the model.

TODO: options available for change

```bash
# train on the synthetic dataset
python 6-train
```

The output directory will be listed after it concludes running.

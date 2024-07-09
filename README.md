# YOLOv8 Auto Trainer

## Prerequisites

NixOS:

```bash
nix-shell
```

Other systems:

```bash
python3 -m venv venv # must have Python 3.9+ installed
source venv/bin/activate
pip install -r requirements-no-nix.txt
```

BASICALLY:

- gather videos of spinning objects / or otherwise a camera moving around an object and put them in 'data/input'

  > they should be mp4 because im lazy

  the filenames should be `id,class.mp4`, e.g.

  ```
  0,izzy.mp4
  1,pen.mp4
  2,screwdriver.mp4
  3,screwdriver.mp4
  .. etc
  ```

  the id is for internal use by the program, the class is for training

  > NOTE: you can also just gather images of the object, to bypass this step, simply do:
  >
  > touch data/input/154,my_class.mp4
  > mkdir data/image-seq/154
  >
  > now add images to data/image-seq!

  > when it comes to training on specific objcets, you want to include adjacent objects in your dataset
  >
  > e.g. you may want to also train on people, example datasets: https://github.com/VikramShenoy97/Human-Segmentation-Dataset https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset
  >
  > you may want to add similar looking objects, or just other objects that may appear in your scene that you don't want to mis categorise as something you're looking for, e.g. phones, etc

  > you may want to also just download images from the internet
  >
  > this is pretty simple
  >
  > pip install bing_image_downloader

```
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

- also download background images, i just use https://unsample.net/ and download like 100-200 1024x images to put into `data/backgrounds`

- ensure your environment is ready to go

  `./0-validate-env`

  you want CUDA for the segmentation task otherwise it will be tedious

- prepare your inputs

  `./1-prepare-input`

  this splits any unsplit videos into N frames at a given frame rate

  then creates the label and dataset information

- segment your data

  `./2-segment`

  use mouse to create bounding box
  press q to confirm segmentation mask

  you can re-create bounding box at any point

  simplescreenrecorder-2023-08-31_18.11.29.mkv

- generate images and train

  `./3-generate-cutouts`
  `./4-generate-composites`
  `./5-train`

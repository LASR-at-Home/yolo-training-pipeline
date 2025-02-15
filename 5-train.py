#!/usr/bin/env python3
import os
from ultralytics import YOLO

kwargs = {}
if "KCL" in os.environ and os.environ["KCL"] == "1":
    # saturate RTX A5000 GPU
    kwargs["batch"] = 128

kwargs["batch"] = 7
model = YOLO("yolov8x-seg.pt")
results = model.train(
    data="data/dataset.yaml", epochs=150, imgsz=640, save_period=1, **kwargs
)

#!/usr/bin/env python3

import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

cuda = torch.cuda.is_available()
print("CUDA is available:", cuda)

if not cuda:
    print("WARN: GPU acceleration is not available!")

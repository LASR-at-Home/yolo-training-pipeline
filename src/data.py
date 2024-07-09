import os

INPUT_DIR = "data/input"

def list_inputs():
    inputs = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".mp4"):
            inputs.append(os.path.splitext(filename)[0].split(','))
    
    return sorted(inputs, key=lambda x: int(x[0]))

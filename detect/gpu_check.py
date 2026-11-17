"""
gpu_check.py

smallest possible Modal job: starts a container with a T4 GPU, confirms
PyTorch can see the GPU, and returns the GPU's name to the local machine.

"""

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")
app = modal.App("av-log-search-gpu-check", image=image)


@app.function(gpu="T4")
def check_gpu():
    import torch

    print("cuda available:", torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0)
    print("gpu name:", gpu_name)
    return gpu_name


@app.local_entrypoint()
def main():
    gpu_name = check_gpu.remote()
    print("returned to my Mac:", gpu_name)
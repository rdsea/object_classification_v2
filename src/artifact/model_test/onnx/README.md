# Base image

- We need different base image as we have three generations of jetson edge and each have its different jetpack
- Base image l4t-CUDA only have CUDA preinstalled, if we need to use cuDNN, we need to installed it manually inside the image or use TensorRT image

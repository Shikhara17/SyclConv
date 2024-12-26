# SYCL-Based Depthwise Separable Convolution Evaluation

## Overview
This project evaluates the performance of SYCL-based depthwise separable convolution against the standard PyTorch `conv2d` implementation on NVIDIA GPUs within a high-performance computing (HPC) context. The primary focus is to compare execution times and analyze the performance benefits of using SYCL for deep learning convolution operations.

## Installation

### Prerequisites
- A C++ Compiler with SYCL support, such as ComputeCpp or DPC++.
- Python 3.x
- PyTorch
- NVIDIA GPU with CUDA support

### Setup
Clone the repository to your local machine and set up the environment:

### Experimental Setup
The evaluation was conducted on an HPC system using NVIDIA’s GPUs. The experiment involved the following steps:

- Image Loading: A couple of images were initially loaded and preprocessed to convert them into tensors suitable for convolution operations.
Model Setup: A Sequential model containing two convolutional layers (one depthwise and one pointwise) was set up.
- Benchmarking: The execution time for processing the images was recorded. The experiment compared:
A serial version of depthwise convolution , The standard PyTorch conv2d and A parallelized SYCL-based depthwise convolution.
- Model Swapping: The ai3_swapmodel() function was utilized to switch between the standard and the SYCL-based models to ensure the same input was processed by both models for a fair comparison.

## Performance Results

The following table summarizes the execution times for the different methods used in the evaluation:

| Method Used        | Execution Time (sec) |
|--------------------|----------------------|
| Serial Depthwise   | 0.23                 |
| PyTorch conv2d     | 0.11                 |
| SYCL Depthwise     | 0.084                |



The SYCL-based implementation provided a significant improvement in execution time over both the serial implementation and the PyTorch conv2d. The parallelized SYCL model achieved approximately 1.3x the speed of PyTorch's conv2d.


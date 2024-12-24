# SyclConv

# SYCLConv Implementation

## Introduction
**SYCLConv** is a pioneering project that implements depthwise separable convolution using SYCL to enhance computational efficiency within the AI3 framework. This project is inspired by the innovative techniques described in the MobileNets paper and is specifically designed to optimize performance for devices with limited computational resources.

## Project Goals
- **Enhance Computational Efficiency**: Replace PyTorch's standard conv2d implementation with a more efficient depthwise separable convolution, reducing computational overhead.
- **Optimize for Limited Resources**: Tailor the convolution process for devices with less processing power, ensuring faster inference times without sacrificing accuracy.
- **Improve Inference Time**: Significantly decrease the time it takes to process input data through the model, making it feasible for real-time applications.

## Tech Stack
- **SYCL**: Utilized for its ability to write single-source code that can run on multiple types of hardware accelerators.
- **AI3 Framework**: The core platform for integrating the new convolution method. 
- **MobileNets Principles**: Employ strategies from the MobileNets architecture to ensure lightweight and efficient neural networks.

## Installation
To get started with EfficientNet SYCL, follow these steps to set up the environment and run the application:

1. **Clone the Repository**
   ```bash
   Go through the documentation for custom cunfiguration : https://github.com/KLab-AI3/ai3.git
   git clone https://github.com/Shikhara17/SyclConv.git
   cd SyclConv
   

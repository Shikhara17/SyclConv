import time
import torch
import torch.nn as nn
from torchvision import transforms
import ai3
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Define a CNN model with Depthwise and Pointwise convolution
class DepthwiseSeparableConvNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_dim, stride=1, padding=0):
        super(DepthwiseSeparableConvNet, self).__init__()

        # Depthwise convolution applies a single kernel per input channel
        self.depthwise_layer = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_dim,
                                         stride=stride, padding=padding, groups=1)

        # Pointwise convolution combines depthwise outputs to produce final feature maps
        self.pointwise_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, input_tensor):
        x = self.depthwise_layer(input_tensor)  # Apply depthwise convolution
        x = self.pointwise_layer(x)  # Apply pointwise convolution
        return x


def load_pickle_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        train_images, train_labels, test_images, test_labels = pickle.load(f)
    return train_images, train_labels, test_images, test_labels


def preprocess_images(image_data):
    return [torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() for img in image_data]


def evaluate_model(model, image_set, parallelism=None, use_custom_impl=False, verbose=False):
    """
    Measure inference times for a given model and dataset.
    Logs the first output tensor values for verification.
    """
    if use_custom_impl:
        print(f"Applying ai3.swap_conv2d to implement custom implementation with parallelism = {parallelism}")
        global dynamic_thread_parallelism
        dynamic_thread_parallelism = parallelism
        ai3.swap_conv2d(model)

    execution_times = []
    first_output_logged = False  # To ensure we log the tensor only once

    for index, img_tensor in enumerate(image_set):
        if verbose and index == 0:
            print(f"First processed image has shape: {img_tensor.shape}")
        start_time = time.time()
        model_output = model(img_tensor)
        end_time = time.time()
        execution_times.append(end_time - start_time)

        # Log the first output tensor values for verification
        if not first_output_logged:
            print(f"Output Tensor Shape: {model_output.shape}")
            print(f"Output Tensor Values: {model_output.cpu().detach().numpy()}")
            first_output_logged = True

    if verbose:
        print(f"Processed {len(image_set)} images in total.")
    return execution_times


def main():
    # Load preprocessed data
    pickle_file_path = 'preprocessed_subset_data.pkl'
    train_data, train_labels, test_data, test_labels = load_pickle_data(pickle_file_path)

    # Prepare data for PyTorch
    test_data = preprocess_images(test_data)

    if not test_data:
        print("No test data available.")
        return

    # Initialize the Separable CNN model
    cnn_model = DepthwiseSeparableConvNet(input_channels=3, output_channels=16, kernel_dim=3, stride=1, padding=1)

    # Define parallelism levels to benchmark
    parallelism_levels = [8, 16, 32, 64, 128, 256, 1024]

    # Collect benchmark results
    pytorch_times = []
    custom_times = []

    for parallelism in parallelism_levels:
        # Measure time for PyTorch implementation
        print("\nEvaluating PyTorch Conv2D...")
        torch_results = evaluate_model(cnn_model, test_data, use_custom_impl=False, verbose=True)
        pytorch_times.append(sum(torch_results) / len(torch_results))

        # Measure time for custom SYCL implementation
        print("\nEvaluating Custom Depthwise Separable Conv2D...")
        custom_results = evaluate_model(cnn_model, test_data, parallelism=parallelism, use_custom_impl=True, verbose=True)
        custom_times.append(sum(custom_results) / len(custom_results))

    # Compile results into a DataFrame
    results_summary = pd.DataFrame({
        "Parallelism Levels": parallelism_levels,
        "PyTorch Avg Time (s)": pytorch_times,
        "Custom Avg Time (s)": custom_times
    })
    print("\nBenchmark Results:")
    print(results_summary)

    # Generate and save plot
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))

    # Plot PyTorch results
    plt.plot(parallelism_levels, pytorch_times, marker='o', label='PyTorch Conv2D', linestyle='-', linewidth=2)

    # Plot Custom implementation results
    plt.plot(parallelism_levels, custom_times, marker='s', label='Custom Depthwise Conv2D', linestyle='--', linewidth=2)

    # Add plot details
    plt.xlabel('Parallelism Levels', fontsize=12)
    plt.ylabel('Average Inference Time (seconds)', fontsize=12)
    plt.title('Inference Time Comparison: PyTorch vs Custom SYCL', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Save and display plot
    plt.tight_layout()
    plt.savefig("performance_comparison_plot.png")
    plt.show()


if __name__ == "__main__":
    main()


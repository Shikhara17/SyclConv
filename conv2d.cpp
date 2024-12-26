#pragma once

#include <CL/sycl.hpp>
#include <vector>
#include <optional>

/**
 * @CUSTOM_OP{Conv2D,conv2d}
 */
const bool DEFAULT_CONV2D = false;
template <typename dtype>
Tensor conv2d_custom(Tensor input, const Tensor &kernel,
                     const std::optional<const Tensor> &bias = std::nullopt,
                     const uint padding_height = 0, const uint padding_width = 0,
                     const uint stride_height = 1, const uint stride_width = 1,
                     const uint dilation_height = 1, const uint dilation_width = 1,
                     const PaddingMode padding_mode = PaddingMode::Zeros,
                     uint groups = 1, int thread_parallelism = 64) {
    // Ensure compatibility
    ensure_same_type(input, kernel, bias);
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "Padding mode must be zeroes");
    errs::bail_if(groups != 1, "Groups must be 1");

    // Extract dimensions
    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();
    const uint depthwise_kernel_height = kernel.height();
    const uint depthwise_kernel_width = kernel.width();
    const uint output_channels = kernel.output_channels();
    const uint kernel_channels = kernel.input_channels();

    const uint output_height =
        (input_height - depthwise_kernel_height + 2 * padding_height) / stride_height + 1;
    const uint output_width =
        (input_width - depthwise_kernel_width + 2 * padding_width) / stride_width + 1;

    // Create output tensor
    Tensor output({output_channels, output_height, output_width}, input.scalar_type);

    // SYCL queue for computation
    sycl::queue compute_queue(sycl::default_selector{});

    // Allocate device memory
    auto device_input = sycl::malloc_device<dtype>(input.count(), compute_queue);
    auto device_kernel = sycl::malloc_device<dtype>(kernel.count(), compute_queue);
    auto device_output = sycl::malloc_device<dtype>(output.count(), compute_queue);

    float *device_bias = nullptr;
    if (bias.has_value()) {
        device_bias = static_cast<float *>(bias->data);
    }

    // Copy data to device
    compute_queue.memcpy(device_input, input.data, input.count() * sizeof(dtype)).wait();
    compute_queue.memcpy(device_kernel, kernel.data, kernel.count() * sizeof(dtype)).wait();

    // Launch SYCL kernel for combined depthwise and pointwise convolution
    compute_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(output_channels, output_height, output_width),
            sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> work_item) {
            const uint oc = work_item.get_global_id(0); // Output channel
            const uint oh = work_item.get_global_id(1); // Output height
            const uint ow = work_item.get_global_id(2); // Output width

            dtype depthwise_result = 0.0;

            // Depthwise convolution logic
            for (uint ic = 0; ic < input_channels; ++ic) {
                for (uint kh = 0; kh < depthwise_kernel_height; ++kh) {
                    for (uint kw = 0; kw < depthwise_kernel_width; ++kw) {
                        int ih = oh * stride_height - padding_height + kh * dilation_height;
                        int iw = ow * stride_width - padding_width + kw * dilation_width;

                        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                            depthwise_result += device_input[ic * input_height * input_width +
                                                              ih * input_width + iw] *
                                                device_kernel[ic * depthwise_kernel_height * depthwise_kernel_width +
                                                              kh * depthwise_kernel_width + kw];
                        }
                    }
                }
            }

            // Pointwise convolution logic (channel mixing)
            dtype pointwise_result = 0.0;
            for (uint ic = 0; ic < input_channels; ++ic) {
                pointwise_result += depthwise_result * device_kernel[oc * input_channels + ic];
            }

            // Add bias if provided
            if (device_bias != nullptr) {
                pointwise_result += device_bias[oc];
            }

            device_output[oc * output_height * output_width + oh * output_width + ow] = pointwise_result;
        });

    // Copy results back to host
    compute_queue.memcpy(output.data, device_output, output.count() * sizeof(dtype)).wait();

    // Free device memory
    sycl::free(device_input, compute_queue);
    sycl::free(device_kernel, compute_queue);
    sycl::free(device_output, compute_queue);
    if (device_bias) {
        sycl::free(device_bias, compute_queue);
    }

    return output;
}


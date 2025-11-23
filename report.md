# OpenCL Image Processing Assignment Report

## 1. Host Program Description

The host program (`main.c`) is written in C and orchestrates the OpenCL execution.

### Configuration and Strategy
- **Image Loading**: The program uses the `stb_image` library to load JPEG images from the `input_images` directory. It iterates through the directory using standard POSIX `dirent.h` functions.
- **OpenCL Setup**:
    - It initializes the OpenCL platform and attempts to find a GPU device. If a GPU is not available, it falls back to the CPU.
    - A context and command queue are created for the selected device.
    - The kernel source code is loaded from `grayscale.cl` and compiled at runtime.
- **Memory Management**:
    - For each image, memory buffers are allocated on the device (`clCreateBuffer`).
    - The image data is transferred from host to device.
    - The kernel is executed.
    - The result is read back to the host.
    - Device memory is released immediately after processing each image to prevent memory exhaustion when processing large datasets (10,000+ images).
- **Parallelization**: The global work size is set to the total number of pixels in the image. This allows each pixel to be processed in parallel by a work-item.

## 2. Grayscale Conversion Algorithm

### Algorithm
The conversion from RGB (Red, Green, Blue) to Grayscale (Y) is performed using the standard NTSC formula:

$$ Y = 0.299 \times R + 0.587 \times G + 0.114 \times B $$

This formula accounts for the human eye's varying sensitivity to different colors (more sensitive to green, less to blue).

### OpenCL Kernel Design
- **Kernel Name**: `rgba_to_grayscale`
- **Arguments**:
    - `__global uchar4* input`: The input image buffer. We treat the input as an array of `uchar4` vectors (Red, Green, Blue, Alpha) to ensure memory alignment and efficient access.
    - `__global uchar* output`: The output image buffer (single channel grayscale).
    - `int width`, `int height`: Image dimensions.
- **Execution**:
    - Each work-item calculates its global ID (`get_global_id(0)`).
    - If the ID is within the image bounds, it reads the pixel at that index.
    - It applies the formula and writes the result to the output buffer.

## 3. Sample Results

*(Place 5 pairs of Original and Grayscale images here)*

1. **Image 1**:
   - Original: [Insert Image]
   - Grayscale: [Insert Image]

2. **Image 2**:
   - Original: [Insert Image]
   - Grayscale: [Insert Image]

...

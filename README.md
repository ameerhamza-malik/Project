# OpenCL Image Grayscale Converter

This project converts a dataset of JPEG images to grayscale using OpenCL for parallel processing.

## Prerequisites

- Linux Operating System
- GCC Compiler
- OpenCL Drivers and SDK (e.g., NVIDIA CUDA Toolkit, AMD ROCm, or Intel Compute Runtime)
- Input images in `input_images/` directory.

## Compilation

To compile the project, run:

```bash
make
```

This will produce an executable named `grayscale_converter`.

## Usage

1. Ensure you have a directory named `input_images` in the same folder as the executable.
2. Place your JPEG images inside `input_images`.
3. Run the program:

```bash
./grayscale_converter
```

4. The processed grayscale images will be saved in the `output_images` directory.

## Troubleshooting

- **OpenCL Library Not Found**: If `make` fails with linker errors, ensure `libOpenCL.so` is in your library path or adjust the `Makefile` to point to your OpenCL SDK installation (e.g., `-L/usr/local/cuda/lib64`).
- **Header Not Found**: If `CL/cl.h` is not found, add the include path to `CFLAGS` in `Makefile` (e.g., `-I/usr/local/cuda/include`).

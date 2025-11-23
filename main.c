#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// For directory traversal on Linux
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_SOURCE_SIZE (0x100000)

// Function to load OpenCL kernel source
char* load_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);
    return source_str;
}

void check_error(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during %s: %d\n", operation, err);
        exit(1);
    }
}

int main() {
    // 1. Setup OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    check_error(ret, "clGetPlatformIDs");

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS) {
        printf("GPU not found, trying CPU...\n");
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
    }
    check_error(ret, "clGetDeviceIDs");

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    check_error(ret, "clCreateContext");

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    check_error(ret, "clCreateCommandQueue");

    char* kernelSource = load_kernel_source("grayscale.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &ret);
    check_error(ret, "clCreateProgramWithSource");

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("Build Log: %s\n", buffer);
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "rgba_to_grayscale", &ret);
    check_error(ret, "clCreateKernel");

    free(kernelSource);

    // 2. Process Images
    const char* input_dir = "input_images";
    const char* output_dir = "output_images";

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        #ifdef _WIN32
        _mkdir(output_dir);
        #else
        mkdir(output_dir, 0700);
        #endif
    }

    DIR* d;
    struct dirent* dir;
    d = opendir(input_dir);
    if (!d) {
        fprintf(stderr, "Could not open input directory: %s\n", input_dir);
        return 1;
    }

    printf("Processing images from %s...\n", input_dir);

    while ((dir = readdir(d)) != NULL) {
        if (dir->d_type == DT_REG || dir->d_type == DT_UNKNOWN) { // Regular file
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s/%s", input_dir, dir->d_name);

            // Check extension (simple check)
            if (strstr(filepath, ".jpg") || strstr(filepath, ".jpeg") || strstr(filepath, ".JPG")) {
                int width, height, channels;
                // Load as 4 channels (RGBA) to align with OpenCL vector types if desired, or just 3.
                // Using 4 channels (RGBA) is often safer for alignment and we used uchar4 in kernel.
                unsigned char* img_data = stbi_load(filepath, &width, &height, &channels, 4); 
                
                if (!img_data) {
                    printf("Failed to load image: %s\n", filepath);
                    continue;
                }

                size_t image_size = width * height * 4 * sizeof(unsigned char);
                size_t gray_size = width * height * sizeof(unsigned char);

                // Create Memory Buffers
                cl_mem input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size, img_data, &ret);
                cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gray_size, NULL, &ret);

                // Set Arguments
                ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_mem_obj);
                ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_mem_obj);
                ret |= clSetKernelArg(kernel, 2, sizeof(int), (void*)&width);
                ret |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&height);
                check_error(ret, "clSetKernelArg");

                // Execute Kernel
                size_t global_item_size = width * height;
                size_t local_item_size = 64; // Adjust based on device capabilities
                // Ensure global size is multiple of local size
                if (global_item_size % local_item_size != 0) {
                    global_item_size = (global_item_size / local_item_size + 1) * local_item_size;
                }

                ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
                check_error(ret, "clEnqueueNDRangeKernel");

                // Read Result
                unsigned char* gray_data = (unsigned char*)malloc(gray_size);
                ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0, gray_size, gray_data, 0, NULL, NULL);
                check_error(ret, "clEnqueueReadBuffer");

                // Save Image
                char out_filepath[512];
                snprintf(out_filepath, sizeof(out_filepath), "%s/%s", output_dir, dir->d_name);
                stbi_write_jpg(out_filepath, width, height, 1, gray_data, 100);

                printf("Processed: %s\n", dir->d_name);

                // Cleanup per image
                stbi_image_free(img_data);
                free(gray_data);
                clReleaseMemObject(input_mem_obj);
                clReleaseMemObject(output_mem_obj);
            }
        }
    }
    closedir(d);

    // Cleanup OpenCL
    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    printf("Done.\n");
    return 0;
}

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
#define BATCH_SIZE 8

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

typedef struct {
    char input_path[512];
    char output_path[512];
    unsigned char* rgba_data;
    unsigned char* gray_data;
    cl_mem input_mem_obj;
    cl_mem output_mem_obj;
    cl_event kernel_event;
    cl_event read_event;
    size_t image_size;
    size_t gray_size;
    int width;
    int height;
} ImageJob;

size_t adjust_global_size(size_t total_pixels, size_t local_item_size) {
    if (total_pixels == 0) {
        return 0;
    }
    if (local_item_size == 0) {
        return total_pixels;
    }
    size_t remainder = total_pixels % local_item_size;
    if (remainder == 0) {
        return total_pixels;
    }
    return total_pixels + (local_item_size - remainder);
}

void cleanup_job(ImageJob* job) {
    if (job->rgba_data) {
        stbi_image_free(job->rgba_data);
        job->rgba_data = NULL;
    }
    if (job->gray_data) {
        free(job->gray_data);
        job->gray_data = NULL;
    }
    if (job->input_mem_obj) {
        clReleaseMemObject(job->input_mem_obj);
        job->input_mem_obj = NULL;
    }
    if (job->output_mem_obj) {
        clReleaseMemObject(job->output_mem_obj);
        job->output_mem_obj = NULL;
    }
    if (job->kernel_event) {
        clReleaseEvent(job->kernel_event);
        job->kernel_event = NULL;
    }
    if (job->read_event) {
        clReleaseEvent(job->read_event);
        job->read_event = NULL;
    }
}

int has_supported_extension(const char* filepath) {
    const char* extensions[] = {".jpg", ".JPG", ".jpeg", ".JPEG"};
    size_t ext_count = sizeof(extensions) / sizeof(extensions[0]);
    for (size_t i = 0; i < ext_count; ++i) {
        if (strstr(filepath, extensions[i]) != NULL) {
            return 1;
        }
    }
    return 0;
}

void process_batch(ImageJob* batch, int batch_size, cl_context context, cl_command_queue command_queue, cl_kernel kernel) {
    const size_t local_item_size = 64;

    for (int i = 0; i < batch_size; ++i) {
        ImageJob* job = &batch[i];
        cl_int ret;

        job->input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, job->image_size, job->rgba_data, &ret);
        check_error(ret, "clCreateBuffer(input)");

        job->output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, job->gray_size, NULL, &ret);
        check_error(ret, "clCreateBuffer(output)");

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&job->input_mem_obj);
        ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&job->output_mem_obj);
        ret |= clSetKernelArg(kernel, 2, sizeof(int), (void*)&job->width);
        ret |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&job->height);
        check_error(ret, "clSetKernelArg(batch)");

        size_t total_pixels = (size_t)job->width * (size_t)job->height;
        size_t global_item_size = adjust_global_size(total_pixels, local_item_size);

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &job->kernel_event);
        check_error(ret, "clEnqueueNDRangeKernel(batch)");

        ret = clEnqueueReadBuffer(command_queue, job->output_mem_obj, CL_FALSE, 0, job->gray_size, job->gray_data, 1, &job->kernel_event, &job->read_event);
        check_error(ret, "clEnqueueReadBuffer(batch)");
    }

    for (int i = 0; i < batch_size; ++i) {
        ImageJob* job = &batch[i];
        clWaitForEvents(1, &job->read_event);

        stbi_write_jpg(job->output_path, job->width, job->height, 1, job->gray_data, 100);
        printf("Processed (batch): %s\n", job->input_path);

        cleanup_job(job);
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
    const char* input_dir = "/home/kali/Downloads/input_images";
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

    ImageJob batch[BATCH_SIZE];
    memset(batch, 0, sizeof(batch));
    int batch_count = 0;

    while ((dir = readdir(d)) != NULL) {
        if (dir->d_type == DT_REG || dir->d_type == DT_UNKNOWN) {
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s/%s", input_dir, dir->d_name);

            if (!has_supported_extension(filepath)) {
                continue;
            }

            int width = 0;
            int height = 0;
            int channels = 0;
            unsigned char* img_data = stbi_load(filepath, &width, &height, &channels, 4);

            if (!img_data) {
                printf("Failed to load image: %s\n", filepath);
                continue;
            }

            ImageJob* job = &batch[batch_count];
            memset(job, 0, sizeof(ImageJob));
            snprintf(job->input_path, sizeof(job->input_path), "%s", dir->d_name);
            snprintf(job->output_path, sizeof(job->output_path), "%s/%s", output_dir, dir->d_name);
            job->rgba_data = img_data;
            job->width = width;
            job->height = height;
            job->image_size = (size_t)width * (size_t)height * 4 * sizeof(unsigned char);
            job->gray_size = (size_t)width * (size_t)height * sizeof(unsigned char);
            job->gray_data = (unsigned char*)malloc(job->gray_size);
            if (!job->gray_data) {
                fprintf(stderr, "Allocation failed for %s\n", filepath);
                stbi_image_free(job->rgba_data);
                job->rgba_data = NULL;
                continue;
            }

            batch_count++;

            if (batch_count == BATCH_SIZE) {
                process_batch(batch, batch_count, context, command_queue, kernel);
                memset(batch, 0, sizeof(batch));
                batch_count = 0;
            }
        }
    }

    if (batch_count > 0) {
        process_batch(batch, batch_count, context, command_queue, kernel);
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

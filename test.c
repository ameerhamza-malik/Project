#include <stdio.h>
#include <CL/cl.h>

int main(void) {
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    printf("Platforms: %u\n", numPlatforms);
    cl_platform_id platforms[4];
    clGetPlatformIDs(4, platforms, NULL);
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        char name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        printf("[%u] Platform: %s\n", i, name);
        cl_uint numDevices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        cl_device_id devices[8];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 8, devices, NULL);
        for (cl_uint j = 0; j < numDevices; ++j) {
            char devName[256];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(devName), devName, NULL);
            printf("    Device: %s\n", devName);
        }
    }
    return 0;
}
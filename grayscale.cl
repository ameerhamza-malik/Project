__kernel void rgba_to_grayscale(__global uchar4* input, __global uchar* output, int width, int height) {
    int gid = get_global_id(0);
    int total_pixels = width * height;

    if (gid < total_pixels) {
        uchar4 pixel = input[gid];
        // Standard NTSC conversion formula
        float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
        output[gid] = (uchar)gray;
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BLOCK_SIZE 16

__global__ void conv2D(float* output, float* input, float* kernel, int input_width, int input_height)
{
// 2D thread index
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
// Check if we are within the bounds of the input image
if (x < input_width && y < input_height)
{
    // Compute the output pixel value
    float sum = 0;
    for (int ky = 0; ky < 3; ++ky)
    {
        for (int kx = 0; kx < 3; ++kx)
        {
            // Compute the index of the input pixel
            int px = x - 1 + kx;
            int py = y - 1 + ky;

            // Check if the pixel is within the bounds of the input image
            if (px >= 0 && px < input_width && py >= 0 && py < input_height)
            {
                // Perform the convolution
                sum += input[py * input_width + px] * kernel[ky * 3 + kx];
            }
        }
    }

    // Set the output pixel value
    output[y * input_width + x] = sum;
}
}

int main()
{
// Set the values of the kernel
float h_kernel[3 * 3] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
// Allocate device memory
float* d_input;
float* d_kernel;
float* d_output;
int input_size = 256 * 256;
int kernel_size = 3 * 3;
int output_size = 256 * 256;
cudaMalloc((void**)&d_input, input_size * sizeof(float));
cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));
cudaMalloc((void**)&d_output, output_size * sizeof(float));

// Allocate host memory for the input image, kernel, and output image
float* h_input = new float[input_size];
float* h_output = new float[output_size];

// Load and process 10 different images
for (int i = 1; i <= 10; i++)
{
    // Load the input image
    int width, height, componentCount;
    char filename[256];
    sprintf(filename, "foto%d.png", i);
    unsigned char *input_image = stbi_load(filename,&width, &height, &componentCount, 4);
        // Get the image dimensions
    int input_width = width;
    int input_height = height;

    // Check if the image dimensions are correct
    if (input_width != 256 || input_height != 256)
    {
        printf("Error: Invalid image dimensions\n");
        return 1;
    }

    // Copy the input image data to the host
    for (int y = 0; y < input_height; ++y)
    {
        for (int x = 0; x < input_width; ++x)
        {
            h_input[y * input_width + x] = static_cast<float>(input_image[(y * input_width + x) * 4]);
        }
    }

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    conv2D<<<grid, block>>>(d_output, d_input, d_kernel, input_width, input_height);

    // Copy data from device to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Save the output image
    unsigned char *output_image = new unsigned char[output_size * 4];
    for (int y = 0; y < input_height; ++y)
    {
        for (int x = 0; x < input_width; ++x)
        {
            output_image[(y * input_width + x) * 4] = static_cast<unsigned char>(h_output[y * input_width + x]);
            output_image[(y * input_width + x) * 4 + 1] = static_cast<unsigned char>(h_output[y * input_width + x]);
            output_image[(y * input_width + x) * 4 + 2] = static_cast<unsigned char>(h_output[y * input_width + x]);
            output_image[(y * input_width + x) * 4 + 3] = 255;
        }
    }

    // Save the output image
    char output_filename[256];
    sprintf(output_filename, "output%d.png", i);
    stbi_write_png(output_filename, input_width, input_height, 4, output_image, input_width * 4);

    // Free memory
    delete[] input_image;
    delete[] output_image;
}

// Free device memory
cudaFree(d_input);
cudaFree(d_kernel);
cudaFree(d_output);

// Free host memory
delete[] h_input;
delete[] h_output;

return 0;

}




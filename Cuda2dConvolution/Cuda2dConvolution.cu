#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <time.h>

#define BLOCK_SIZE 16

__global__ void conv2D(float* output, float* input, float* kernel, int input_width, int input_height)
{
    // 2D thread index
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // Check if we are within the bounds of the input image
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

__global__ void minPoolFilter(float* output, float* input, int input_width, int input_height, int pool_size)
{
    // 2D thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if we are within the bounds of the input image
    {
        // Initialize the min value to the first input pixel value
        float min_val = input[y * input_width + x];

        // Iterate through the pooling window and find the minimum value
        for (int py = y; py < y + pool_size; py+2)
        {
            for (int px = x; px < x + pool_size; px+2)
            {
                // Check if the pixel is within the bounds of the input image
                if (px >= 0 && px < input_width && py >= 0 && py < input_height)
                {
                    // Update the min value if necessary
                    min_val = min(min_val, input[py * input_width + px]);
                }
            }
        }

        // Set the output pixel value to the min value
        output[y * input_width + x] = min_val;
}
}

__global__ void maxPoolFilter(float* output, float* input, int input_width, int input_height, int pool_size)
{
    // 2D thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if we are within the bounds of the input image
    {
        /// Initialize the max value to the minimum possible float value
          float max_val = input[y * input_width + x];

        // Iterate through the pooling window and find the minimum value
        for (int py = y; py < y + pool_size; py+2)
        {
            for (int px = x; px < x + pool_size; px+2)
            {
                // Check if the pixel is within the bounds of the input image
                if (px >= 0 && px < input_width && py >= 0 && py < input_height)
                {
                    // Update the min value if necessary
                    max_val = max(max_val, input[py * input_width + px]);
                }
            }
        }

        // Set the output pixel value to the min value
        output[y * input_width + x] = max_val;
}
}


int main()
{
   // Print the elapsed time
    printf("Starting the program...\n");
  clock_t start_time = clock();
// Set the values of the kernel
float h_kernel[3 * 3] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
// Set the pooling size
int pool_size = 2;
// Allocate device memory
float* d_input;
float* d_kernel;
float* d_conv_output;
float* d_min_pool_output;
float* d_max_pool_output;

int input_size = 256 * 256;
int kernel_size = 3 * 3;
int conv_output_size = 256 * 256;
int min_pool_output_size = 256 * 256 / (pool_size * pool_size);
int max_pool_output_size = 256 * 256 / (pool_size * pool_size);
cudaMalloc((void**)&d_input, input_size * sizeof(float));
cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));
cudaMalloc((void**)&d_conv_output, conv_output_size * sizeof(float));
cudaMalloc((void**)&d_min_pool_output, min_pool_output_size * sizeof(float));
cudaMalloc((void**)&d_max_pool_output, max_pool_output_size * sizeof(float));

// Allocate host memory for the input image, kernel, and output images
float* h_input = new float[input_size];
float* h_conv_output = new float[conv_output_size];
float* h_min_pool_output = new float[min_pool_output_size];
float* h_max_pool_output = new float[max_pool_output_size];


// Load and process 10 different images
for (int i = 1; i <= 10; i++)
{
    // Load the input image
    int width, height, componentCount;
    char folder_path_input[100] = "/content/input_images/";
    char filename[256];
    sprintf(filename, "%sfoto%d.png", folder_path_input, i);
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
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((input_width + blockSize.x - 1) / blockSize.x, (input_height + blockSize.y - 1) / blockSize.y);
conv2D<<<gridSize, blockSize>>>(d_conv_output, d_input, d_kernel, input_width, input_height);

    // Launch the min pool kernel
    dim3 minPoolBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 minPoolGridSize((input_width + minPoolBlockSize.x - 1) / minPoolBlockSize.x, (input_height + minPoolBlockSize.y - 1) / minPoolBlockSize.y);
    minPoolFilter<<<minPoolGridSize, minPoolBlockSize>>>(d_min_pool_output, d_input, input_width, input_height, pool_size);

    // Launch the max pool kernel
    dim3 maxPoolBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 maxPoolGridSize((input_width + maxPoolBlockSize.x - 1) / maxPoolBlockSize.x, (input_height + maxPoolBlockSize.y - 1) / maxPoolBlockSize.y);
    maxPoolFilter<<<minPoolGridSize, minPoolBlockSize>>>(d_max_pool_output, d_input, input_width, input_height, pool_size);

    // Copy data from device to host
    cudaMemcpy(h_conv_output, d_conv_output, conv_output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_pool_output, d_min_pool_output, min_pool_output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_pool_output, d_max_pool_output, max_pool_output_size * sizeof(float), cudaMemcpyDeviceToHost);

   // Save the output image 2D CONV
    unsigned char *conv_output_image = new unsigned char[conv_output_size * 4];
    for (int y = 0; y < input_height; ++y)
    {
        for (int x = 0; x < input_width; ++x)
        {
            conv_output_image[(y * input_width + x) * 4] = static_cast<unsigned char>(h_conv_output[y * input_width + x]);
            conv_output_image[(y * input_width + x) * 4 + 1] = static_cast<unsigned char>(h_conv_output[y * input_width + x]);
            conv_output_image[(y * input_width + x) * 4 + 2] = static_cast<unsigned char>(h_conv_output[y * input_width + x]);
            conv_output_image[(y * input_width + x) * 4 + 3] = 255;
        }
    }

    // Save the output images to files
    char folder_path_conv[100] = "/content/conv_images/";
    char conv_output_filename[256];
    sprintf(conv_output_filename, "%sconv_output_%d.png",folder_path_conv, i);
    stbi_write_png(conv_output_filename, input_width, input_height, 4, conv_output_image, input_width * 4);

    char folder_path_minpool[100] = "/content/minpool_images/";
    char min_pool_output_filename[256];
    sprintf(min_pool_output_filename, "%smin_pool_output_%d.png",folder_path_minpool, i);
    stbi_write_png(min_pool_output_filename, input_width / pool_size, input_height / pool_size, 4, h_min_pool_output, input_width / pool_size * 4);

    char folder_path_maxpool[100] = "/content/maxpool_images/";
    char max_pool_output_filename[256];
    sprintf(max_pool_output_filename, "%smax_pool_output_%d.png",folder_path_maxpool, i);
    stbi_write_png(max_pool_output_filename, input_width / pool_size, input_height / pool_size, 4, h_max_pool_output, input_width / pool_size * 4);

    // Clean up
    stbi_image_free(input_image);

    
}

// Free device memory
cudaFree(d_input);
cudaFree(d_kernel);
cudaFree(d_conv_output);
cudaFree(d_min_pool_output);
cudaFree(d_max_pool_output);

// Free host memory
delete[] h_input;
delete[] h_conv_output;
delete[] h_min_pool_output;
delete[] h_max_pool_output;

 // Stop the timer
    clock_t end_time = clock();

  // Calculate the elapsed time in seconds
  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  // Print the elapsed time
    printf("Elapsed time: %f seconds\n", elapsed_time);
    printf("Finished program!\n");
  

return 0;
}


   




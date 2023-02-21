#Introduction
Image processing is an important field in computer vision and artificial intelligence, with many applications in areas such as object recognition, feature extraction, and image analysis. The development of high-performance computing has revolutionized image processing algorithms, enabling faster and more accurate analysis of images. In this project, we present three image processing functions - maxpooling, minpooling, and convolution - implemented in CUDA programming language to achieve better performance.

#Maxpooling
Maxpooling is a simple yet powerful function that replaces the pixel values in a given window with the maximum value. The purpose of maxpooling is to reduce the size of the image by aggregating the pixel values in a given window. We have implemented maxpooling using CUDA to take advantage of the parallelism offered by modern GPUs.

#Minpooling
Minpooling is similar to maxpooling, but instead of replacing the pixel values with the maximum value, we replace them with the minimum value. The goal of minpooling is to highlight the smallest features in an image. The implementation of minpooling in CUDA is similar to maxpooling, and we have also included it in this project.

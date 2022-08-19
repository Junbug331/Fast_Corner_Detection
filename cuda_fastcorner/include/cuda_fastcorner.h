#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h> // __syncthreads()
#include <cuda_runtime.h> // __global__
#include <device_launch_parameters.h> // blockIdx, threadIdx

typedef unsigned char uchar;

__global__ void reduceImgKernel(uchar *input, int width, int height, float scaleX, float scaleY, uchar *output);
__global__ void fastCornerDetectorKernel(uchar *low_img, int low_w, int low_h, uchar *input_img, int input_w, int input_h, int T1, int T2, uchar* result);
__host__ void cudaFastCornerDetectorHost(uchar * img, float* result, int width, int height, int T1, int T2, float scaleX, float scaleY);

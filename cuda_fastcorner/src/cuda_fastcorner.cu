#include <stdio.h>
#include <string>
#include <memory>
#include <stdexcept>

#include "cuda_fastcorner.h"

template<typename ...Args>
std::string string_format(const std::string& format, Args ...args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    if (size <= 0) { throw std::runtime_error("Error during formatting"); }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);
}
#define cudaErrChk(code) { gpuAssert(code, __FILE__, __LINE__, true);  }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        throw std::runtime_error(string_format("GpuAssert: %s, File: %s, line: %d\n", cudaGetErrorString(code), file, line));
    }
}

__global__ void reduceImgKernel(uchar *input, int width, int height, float scaleX, float scaleY, uchar *output)
{
    extern __shared__ uchar mem[];
    __shared__ uchar val;
    int shared_w = int(scaleX+0.5);
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < 0 || row >= height || col < 0 || col >= width)
        return;

    mem[threadIdx.y + shared_w + threadIdx.x] = input[row*width + col];
    val = 0;
    __syncthreads();
    //atomicAdd(&val, input[row*width+col]);

}

__global__ void fastCornerDetectorKernel(uchar *low_img, int low_w, int low_h, uchar *input_img, int input_w, int input_h, int T1, int T2, float* result)
{

}

__host__ void cudaFastCornerDetectorHost(uchar * img, float* result, int width, int height, int T1, int T2, float scaleX, float scaleY)
{
    uchar *d_input_img, *d_low_img;
    float *d_result;

    int low_w = int(width/scaleX + 0.5);
    int low_h = int (height/scaleY + 0.5);
    size_t low_size = low_w * low_h * sizeof(uchar);
    size_t input_size = width * height * sizeof(uchar);

    cudaErrChk(cudaMalloc((void **)&d_input_img, input_size))
    cudaErrChk(cudaMalloc((void **)&d_result, width*height*sizeof(float)))
    cudaErrChk(cudaMalloc((void **)&d_low_img, low_size))

    cudaErrChk(cudaMemcpy(d_input_img, img, input_size, cudaMemcpyHostToDevice))

    // Generate low resolution image
    int threadsPerBlock = int(scaleX+0.5) * int(scaleY+0.5);
    dim3 block(threadsPerBlock, threadsPerBlock);
    dim3 grid(low_w, low_h);
    reduceImgKernel<<<grid, block, sizeof(uchar)*threadsPerBlock*threadsPerBlock >>>(d_input_img, width, height, scaleX, scaleY, d_low_img);
    cudaErrChk(cudaGetLastError())

    int blocksPerGridX = (low_w + threadsPerBlock-1)/threadsPerBlock;
    int blocksPerGridY = (low_h + threadsPerBlock-1)/threadsPerBlock;
    grid = dim3(blocksPerGridX, blocksPerGridY);

    // Call fast corner detection kernel
    threadsPerBlock = 32;
    fastCornerDetectorKernel<<<grid, block>>>(d_low_img, low_w, low_h, d_input_img, width, height, T1, T2, d_result);
    cudaErrChk(cudaMemcpy(result, d_result, input_size, cudaMemcpyDeviceToHost))

    cudaErrChk(cudaFree(d_input_img))
    cudaErrChk(cudaFree(d_low_img))
    cudaErrChk(cudaFree(d_result))
}

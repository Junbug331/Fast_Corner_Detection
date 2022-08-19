cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(cudafastCorner CUDA)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_FAST_SRC_FILES
src/cuda_fastcorner.cu)

add_library(cuda_fastcorner ${CUDA_FAST_SRC_FILES})

target_include_directories(cuda_fastcorner PUBLIC
	include
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(cuda_fastcorner PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)

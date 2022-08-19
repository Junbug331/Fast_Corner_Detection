cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(baseProject CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	message(STATUS "OpenCV found - ${OpenCV_DIR}")
	include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_subdirectory(cuda_fastcorner)
add_subdirectory(fastcorner)

set(SRC_FILES src/main.cpp)

add_executable(example ${SRC_FILES})

target_link_libraries(example PRIVATE
	${OpenCV_LIBS}
	cuda_fastcorner
	fastcorner
)

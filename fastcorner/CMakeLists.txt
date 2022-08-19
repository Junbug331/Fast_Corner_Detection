cmake_minimum_required(VERSION 3.15)
project(fastcorner)
set(CMAKE_CXX_STANDARD 17)
set(FASTCORNER_SRC_FILES src/fastcorner.cpp)

add_library(fastcorner ${FASTCORNER_SRC_FILES})
target_include_directories(fastcorner PUBLIC
	include
)

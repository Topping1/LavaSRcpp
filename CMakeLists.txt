cmake_minimum_required(VERSION 3.14)
project(LavaSR_CPP)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Automatically find the installed ONNXRuntime package
find_package(onnxruntime REQUIRED)

# 2. Define the executable
add_executable(lavasr main.cpp)

# 3. Include the current directory (so it can find dr_wav.h)
target_include_directories(lavasr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 4. Link the ONNXRuntime library
target_link_libraries(lavasr PRIVATE onnxruntime::onnxruntime)
cmake_minimum_required(VERSION 3.30)

# Project definition with CUDA and C++ languages
project(imagehash_cuda CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 50)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Check if cuda-samples directory exists
if (NOT EXISTS "${CMAKE_SOURCE_DIR}/cuda-samples")
    message(FATAL_ERROR "Required directory 'cuda-samples' not found.")
endif ()

# Set module path and include directories
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_config")
include_directories(cuda-samples/Common)

# Check and set CUDA host compiler
if (DEFINED ENV{NVCC_CCBIN})
    message(STATUS "Using CUDA host compiler from NVCC_CCBIN: $ENV{NVCC_CCBIN}")
    set(CMAKE_CUDA_HOST_COMPILER "$ENV{NVCC_CCBIN}")
else ()
    message(WARNING "NVCC_CCBIN environment variable is not set. Using default host compiler.")
endif ()

# Define options for external dependencies
option(USE_FREEIMAGE "Enable FreeImage support" OFF)
option(USE_OPENGL "Enable OpenGL support" OFF)
option(USE_GLEW "Enable GLEW support" OFF)
option(USE_GLUT "Enable GLUT support" OFF)

# List of libraries and include dirs to be populated
set(EXTERNAL_LIBRARIES "")
set(EXTERNAL_INCLUDE_DIRS "")

# Conditionally find and configure FreeImage
if (USE_FREEIMAGE)
    find_package(FreeImage)
    if (FREEIMAGE_FOUND)
        message(STATUS "FreeImage found:")
        message(STATUS "  Include dirs: ${FREEIMAGE_INCLUDE_DIRS}")
        message(STATUS "  Libraries: ${FREEIMAGE_LIBRARIES}")
        list(APPEND EXTERNAL_INCLUDE_DIRS ${FREEIMAGE_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${FREEIMAGE_LIBRARIES})
        add_compile_definitions(HAS_FREEIMAGE)
    else ()
        message(FATAL_ERROR "FreeImage requested but not found, please install it or disable with USE_FREEIMAGE=OFF")
    endif ()
endif ()

# Conditionally find and configure OpenGL
if (USE_OPENGL)
    find_package(OpenGL)
    if (OPENGL_FOUND)
        message(STATUS "OpenGL found")
        list(APPEND EXTERNAL_INCLUDE_DIRS ${OPENGL_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${OPENGL_LIBRARIES})
        add_compile_definitions(HAS_OPENGL)
    else ()
        message(FATAL_ERROR "OpenGL requested but not found, please install it or disable with USE_OPENGL=OFF")
    endif ()
endif ()

# Conditionally find and configure GLEW
if (USE_GLEW)
    find_package(GLEW)
    if (GLEW_FOUND)
        message(STATUS "GLEW found")
        list(APPEND EXTERNAL_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${GLEW_LIBRARIES})
        add_compile_definitions(HAS_GLEW)
    else ()
        message(FATAL_ERROR "GLEW requested but not found, please install it or disable with USE_GLEW=OFF")
    endif ()
endif ()

# Conditionally find and configure GLUT
if (USE_GLUT)
    find_package(GLUT)
    if (GLUT_FOUND)
        message(STATUS "GLUT found")
        list(APPEND EXTERNAL_INCLUDE_DIRS ${GLUT_INCLUDE_DIR})
        list(APPEND EXTERNAL_LIBRARIES ${GLUT_LIBRARIES})
        add_compile_definitions(HAS_GLUT)
    else ()
        message(FATAL_ERROR "GLUT requested but not found, please install it or disable with USE_GLUT=OFF")
    endif ()
endif ()

# Common headers
set(COMMON_HEADERS
        common/pg/cDefs.h
        common/pg/cudaDefs.h
        common/pg/imageManager.h
        config/constants_common_decl.h
        config/constants_host_decl.h
        config/constants_device_decl.h
        src/phash/phash_kern.cuh
        src/helpers/FreeImageBitmap.h
        src/helpers/CudaCheck.cuh
        src/phash/phash.cuh
        src/lsh/lsh.cuh
)

# Define common sources
set(COMMON_SOURCES
        config/constants_host_def.cpp
        config/constants_device_def.cu
        src/phash/phash_kern.cu
        src/phash/phash.cu
        src/lsh/lsh.cu

)

# Define executable with main source and common headers
add_executable(imagehash_cuda
        src/main.cu
        ${COMMON_HEADERS}
        ${COMMON_SOURCES}
)

# Add include directories
target_include_directories(imagehash_cuda PRIVATE
        config
        ${EXTERNAL_INCLUDE_DIRS})

# Link libraries
target_link_libraries(imagehash_cuda PRIVATE ${EXTERNAL_LIBRARIES})
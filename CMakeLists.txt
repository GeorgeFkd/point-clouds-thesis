cmake_minimum_required(VERSION 3.20)
project(PointCloudApp)

# Use C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile_commands.json for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(PCL 1.12 REQUIRED COMPONENTS
    common
    io
    filters
    visualization
    search
)

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    dnn
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel processing enabled")
else()
    message(WARNING "OpenMP not found - parallel processing disabled")
endif()

# Include directories
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${PCL_LIBRARY_DIRS}
    ${Boost_LIBRARY_DIRS}
)

# Compiler definitions
add_definitions(${PCL_DEFINITIONS})

# Common libraries
set(COMMON_LIBS
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    list(APPEND COMMON_LIBS OpenMP::OpenMP_CXX)
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optimization for Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# ===== Executables =====

# 1. 2D Object Detection
add_executable(2d_object_detection src/2d_object_detection.cpp)
target_link_libraries(2d_object_detection ${COMMON_LIBS})

# 2. 3D Object Detection
add_executable(3d_object_detection src/3d_object_detection.cpp)
target_link_libraries(3d_object_detection ${COMMON_LIBS})

# 3. 2D Filterer
add_executable(2d_filterer src/2d_filterer.cpp)
target_link_libraries(2d_filterer ${COMMON_LIBS})

# 4. 3D Filterer
add_executable(3d_filterer src/3d_filterer.cpp)
target_link_libraries(3d_filterer ${COMMON_LIBS})

# 5. Point Cloud Renderer
add_executable(pc_renderer src/pc_renderer.cpp)
target_link_libraries(pc_renderer ${COMMON_LIBS})

# ===== Installation =====
install(TARGETS 
    2d_object_detection
    3d_object_detection
    2d_filterer
    3d_filterer
    pc_renderer
    DESTINATION bin
)

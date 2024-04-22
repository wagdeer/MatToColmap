cmake_minimum_required(VERSION 3.10)

project(MatToColmap)

find_package(colmap REQUIRED)
# or to require a specific version: find_package(colmap 3.4 REQUIRED)

find_package(OpenCV 4 REQUIRED)

add_executable(mat2colmap mat2colmap.cc)
target_link_libraries(mat2colmap colmap::colmap ${OpenCV_LIBS})
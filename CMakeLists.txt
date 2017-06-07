cmake_minimum_required(VERSION 3.6)
project(hsvVideo0607)

FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(hsvVideo0607 ${SOURCE_FILES})

target_link_libraries(hsvVideo0607 ${OpenCV_LIBS})
cmake_minimum_required(VERSION 4.1)
project(ws_area1)

set(CMAKE_CXX_STANDARD 20)

add_executable(ws_area1 main.cpp
        vision/src/apriltag.h)

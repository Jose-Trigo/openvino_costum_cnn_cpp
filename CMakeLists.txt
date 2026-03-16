cmake_minimum_required(VERSION 3.10)
project(TrafficSignClassifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenVINO
find_package(OpenVINO REQUIRED)
# OpenCV
find_package(OpenCV REQUIRED)

add_executable(classifier main.cpp)

target_link_libraries(classifier
    openvino::runtime
    ${OpenCV_LIBS}
)

set_property(TARGET classifier PROPERTY CXX_STANDARD 17)

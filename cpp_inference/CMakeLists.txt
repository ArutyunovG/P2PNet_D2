cmake_minimum_required(VERSION 3.8)

project(p2p_demo)

find_package(OpenCV REQUIRED)

add_executable(p2p_demo ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)
target_include_directories(p2p_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(p2p_demo ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

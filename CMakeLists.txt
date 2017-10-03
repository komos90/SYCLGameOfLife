project(GameOfLifeSycl)
cmake_minimum_required(VERSION 3.2.2)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
set(COMPUTECPP_PACKAGE_ROOT_DIR /opt/computecpp/)
include(FindComputeCpp)
find_package(SDL2 REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(SYSTEM ${COMPUTECPP_INCLUDE_DIRECTORY} ${SDL2_INCLUDE_DIRS})

add_executable(game_of_life main.cpp)
add_sycl_to_target(game_of_life ${CMAKE_CURRENT_BINARY_DIR} main.cpp)
target_link_libraries(game_of_life PUBLIC ${SDL2_LIBRARIES})

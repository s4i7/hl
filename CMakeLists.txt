cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hl LANGUAGES CXX HIP)

set(CMAKE_HIP_ARCHITECTURES "gfx1100")
find_package(HIP REQUIRED)

enable_language(HIP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

option(BUILD_TESTS "Build tests" OFF)

# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )
# 
# FetchContent_MakeAvailable(googletest)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hl")

add_library(hl)

target_sources(hl
  PRIVATE
    "${SOURCE_DIR}/device.cpp"
    "${SOURCE_DIR}/gemm.cpp"
)

set_source_files_properties(
    "${SOURCE_DIR}/device.cpp"
    "${SOURCE_DIR}/gemm.cpp"
    PROPERTIES LANGUAGE HIP
)

target_link_libraries(hl PUBLIC hip::host hip::device)
target_include_directories(hl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_DIR})

add_subdirectory(pipelines)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

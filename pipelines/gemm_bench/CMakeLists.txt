add_executable(gemm_bench
  gemm_bench.hip
)

target_link_libraries(gemm_bench PRIVATE hl)
target_include_directories(gemm_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_DIR})

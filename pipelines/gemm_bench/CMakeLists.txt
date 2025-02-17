add_executable(gemm_bench
  gemm_bench.hip
)

target_link_libraries(gemm_bench PRIVATE hl)

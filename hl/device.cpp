#include "device.hpp"
#include "macros.hpp"

#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

auto device_prop(int idx) -> hipDeviceProp_t {
  hipDeviceProp_t prop;
  HIP_CALL(hipGetDeviceProperties(&prop, idx));
  return prop;
}

auto init_device(int idx, bool list_devices) -> void {
  int deviceCount;
  HIP_CALL(hipGetDeviceCount(&deviceCount));

  if (list_devices) {
    std::printf("Found %d device(s)\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      auto prop = device_prop(i);
      std::printf("Device %d: %s\n", i, prop.name);
    }
  }

  HIP_CALL(hipSetDevice(0));
  auto prop = device_prop(idx);
  std::printf("Using device: %s\n", prop.name);
}

#pragma once
#include <iostream>

#define HIP_CALL(call) do {                                  \
    hipError_t err = call;                                   \
    if (err != hipSuccess) {                                 \
        std::cerr << "HIP error at " << __FILE__ << ":"      \
                  << __LINE__ << " - " << hipGetErrorString(err) \
                  << std::endl;                              \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)


// cuda_bf16.h - CUDA bfloat16 names on HIP for a gfx906 (STRATA_HIP_GFX906) build.
//
// The Inkling CUDA backend includes <cuda_bf16.h>.  Every use of the hardware-conversion intrinsics
// (__float2bfloat16_rn / __floats2bfloat162_rn) is inside `#if __CUDA_ARCH__ >= 800`, which never holds on HIP;
// the tree's own software rounding branch is taken there.  The unconditional uses are only type names
// (__nv_bfloat16 as a 2-byte element in reinterpret_cast), which map 1:1 onto HIP's __hip_bfloat16.
// The _rn spellings are provided regardless so the guarded code still parses if an arch macro ever gets defined.
#pragma once
#include "strata_hip.h"
#include <hip/hip_bf16.h>

typedef __hip_bfloat16  __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
static __device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}
static __device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return __hip_bfloat162(__float2bfloat16(a), __float2bfloat16(b));
}
#endif

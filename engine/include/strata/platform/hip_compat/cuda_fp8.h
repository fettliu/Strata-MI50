// cuda_fp8.h - CUDA FP8 names mapped onto HIP's OCP FP8 types for a gfx906 (STRATA_HIP_GFX906) build.
//
// The Inkling CUDA backend includes <cuda_fp8.h> and uses __nv_fp8_e4m3 (the saturating OCP E4M3FN class,
// float ctor + the __x raw byte) to store FP8 KV.  HIP's amd_hip_fp8.h has exactly that type as
// __hip_fp8_e4m3 (OCP E4M3 = e4m3fn, __HIP_SATFINITE, float ctor host+device), so the NVIDIA name is an alias.
// The hardware conversion intrinsics (__nv_cvt_fp8_to_halfraw, __nv_cvt_fp8x2_to_halfraw2) are only used under
// `#if __CUDA_ARCH__ >= 890`, which is never true on HIP; the software branch of those call sites is taken.
#pragma once
#include "strata_hip.h"
#include <hip/hip_fp8.h>

typedef __hip_fp8_storage_t   __nv_fp8_storage_t;
typedef __hip_fp8x2_storage_t __nv_fp8x2_storage_t;
typedef __hip_fp8_e4m3        __nv_fp8_e4m3;
typedef __hip_fp8_e5m2        __nv_fp8_e5m2;

// interpretation tags (only referenced inside never-taken __CUDA_ARCH__ >= 890/1000 branches; defined for clarity)
#ifndef __NV_E4M3
#define __NV_E4M3 __HIP_E4M3
#endif
#ifndef __NV_E5M2
#define __NV_E5M2 __HIP_E5M2
#endif

// cuda_fp4.h - CUDA FP4 names for a gfx906 (STRATA_HIP_GFX906) build.
//
// The Inkling CUDA backend includes <cuda_fp4.h>, but every FP4 symbol (__nv_fp4x2_storage_t,
// __nv_cvt_fp4x2_to_halfraw2, __NV_E2M1, __nv_fp4x2_e2m1) is used either under `#if __CUDA_ARCH__ >= 1000`
// (Blackwell hardware FP4, never on gfx906) or in the sm_120a-only translation unit that a gfx906 build never
// compiles.  The include must merely resolve, so this header is intentionally empty apart from the storage
// typedef the guarded branches name.
#pragma once
#include "strata_hip.h"

typedef unsigned char __nv_fp4x2_storage_t;
#ifndef __NV_E2M1
#define __NV_E2M1 0
#endif

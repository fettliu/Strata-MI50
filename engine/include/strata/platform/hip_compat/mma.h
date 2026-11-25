// mma.h - <mma.h> stub for a gfx906 (STRATA_HIP_GFX906) build.
//
// The Inkling CUDA backend includes <mma.h> for nvcuda::wmma, but every wmma use is under
// `#if __CUDA_ARCH__ >= 700` (Volta+ tensor cores, never true on HIP).  This stub exists only so the include
// resolves and any stray unguarded reference still parses as an empty namespace.
#pragma once
namespace nvcuda { namespace wmma { } }

# engine/src/inkling2/gfx906/CMakeLists.txt - AMD gfx906 (MI50, wave64) backend for `strata-inkling`.
#
# Upstream Inkling ships a CUDA backend (sm_60..sm_121a) and a wave32 HIP backend (RDNA2-4).  gfx906 is wave64 and
# has no matrix cores, so neither path applies as written.  This lane compiles the *CUDA* backend's sources as HIP
# through Strata's gfx906 compat layer (include/strata/platform/hip_compat, STRATA_HIP_GFX906 in the root build):
# every tensor-core path in the backend is behind `#if __CUDA_ARCH__ >= N`, which is never defined on HIP, so the
# upstream Pascal/Volta SIMT path (sm_60/sm_70) is what gets compiled.  Compiled with the ROCm clang++ directly
# (-x hip), the same way the wave32 HIP backend (../hip/CMakeLists.txt) is built.
#
# Enabled only by the root build with -DSTRATA_HIP_GFX906=ON (which forces STRATA_ENABLE_CUDA so the backend copy
# of the source list is the one used, and makes CUDA::cudart / CUDA::cublas stand-ins for hip::host / roc::hipblas).

set(_ink_gfx_src "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_cuda_backend.cu")
set(_ink_gfx_obj "${CMAKE_CURRENT_BINARY_DIR}/ink_cuda_backend_gfx906${CMAKE_CXX_OUTPUT_EXTENSION}")
# Link-time stand-ins for the sm_120a (Blackwell) MoE TU the backend references but a gfx906 build never compiles
set(_ink_gfx_stub_src "${CMAKE_CURRENT_SOURCE_DIR}/ink_sm120_stub.cpp")
set(_ink_gfx_stub_obj "${CMAKE_CURRENT_BINARY_DIR}/ink_sm120_stub_gfx906${CMAKE_CXX_OUTPUT_EXTENSION}")

get_filename_component(_ink_engine_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(_ink_compat "${_ink_engine_root}/include/strata/platform/hip_compat")

set(_ink_gfx_inc
  "-I${_ink_compat}"
  "-I${_ink_engine_root}/include"
  "-I${CMAKE_CURRENT_SOURCE_DIR}/.."
  "-I${CMAKE_CURRENT_SOURCE_DIR}/../cuda"
  "-I${_ink_engine_root}/third_party/stb"
  "-I${_ink_engine_root}/third_party/zstd-dec"
  "-I${_ink_engine_root}/src/hy4flash/run"
)

# The HIP compiler the root toolchain found (clang++ with the ROCm driver)
set(_ink_gfx_cxx "${CMAKE_HIP_COMPILER}")
if(NOT _ink_gfx_cxx)
  find_program(_ink_gfx_cxx NAMES clang++ HINTS "$ENV{ROCM_PATH}/llvm/bin" "/opt/rocm/llvm/bin" "/opt/rocm/bin")
endif()

set(_ink_gfx_extra_flags "")
if(DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT AND NOT "${CMAKE_HIP_COMPILER_ROCM_ROOT}" STREQUAL "")
  list(APPEND _ink_gfx_extra_flags "--rocm-path=${CMAKE_HIP_COMPILER_ROCM_ROOT}")
endif()

add_custom_command(
  OUTPUT "${_ink_gfx_obj}"
  COMMAND "${_ink_gfx_cxx}" -x hip -c -O3 -std=c++20 --offload-arch=gfx906
          ${_ink_gfx_extra_flags} ${_ink_gfx_inc}
          -DINKLING_ENABLE_CUDA=1 -DSTRATA_HIP_GFX906=1 -D__HIP_PLATFORM_AMD__=1 -D__CUDACC__=1
          -w
          "${_ink_gfx_src}" -o "${_ink_gfx_obj}"
  DEPENDS "${_ink_gfx_src}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_cuda.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_moe_cuda.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_attn.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_hybrid.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_runner.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_spec.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_topk.cuh"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_dspark.cuh"
          "${_ink_compat}/cuda_fp8.h"
          "${_ink_compat}/cuda_bf16.h"
          "${_ink_compat}/cuda_fp4.h"
          "${_ink_compat}/mma.h"
          "${_ink_compat}/strata_hip.h"
          "${_ink_compat}/cublas_v2.h"
  COMMENT "Compiling strata-inkling gfx906 backend (CUDA backend as HIP, wave64 compat)"
  VERBATIM
)

add_custom_command(
  OUTPUT "${_ink_gfx_stub_obj}"
  COMMAND "${_ink_gfx_cxx}" -x hip -c -O2 -std=c++20 --offload-arch=gfx906
          ${_ink_gfx_extra_flags} ${_ink_gfx_inc}
          -DINKLING_ENABLE_CUDA=1 -DSTRATA_HIP_GFX906=1 -D__HIP_PLATFORM_AMD__=1 -D__CUDACC__=1
          -w
          "${_ink_gfx_stub_src}" -o "${_ink_gfx_stub_obj}"
  DEPENDS "${_ink_gfx_stub_src}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../cuda/ink_moe_fp4_sm120.cuh"
  COMMENT "Compiling strata-inkling gfx906 sm120 link-time stubs"
  VERBATIM
)

add_library(strata_inkling_cuda_gfx906 STATIC "${_ink_gfx_obj}" "${_ink_gfx_stub_obj}")
set_target_properties(strata_inkling_cuda_gfx906 PROPERTIES LINKER_LANGUAGE CXX)

# hip::host brings amdhip64; roc::hipblas is the cuBLAS stand-in the backend calls through CUDA::cublas.
if(TARGET hip::host)
  target_link_libraries(strata_inkling_cuda_gfx906 PUBLIC strata_quant hip::host)
else()
  target_link_libraries(strata_inkling_cuda_gfx906 PUBLIC strata_quant)
endif()
if(TARGET roc::hipblas)
  target_link_libraries(strata_inkling_cuda_gfx906 PUBLIC roc::hipblas)
elseif(TARGET hip::hipblas)
  target_link_libraries(strata_inkling_cuda_gfx906 PUBLIC hip::hipblas)
endif()
if(UNIX)
  target_link_libraries(strata_inkling_cuda_gfx906 PUBLIC ${CMAKE_DL_LIBS})
endif()

# The backend is the CUDA one, so the host TUs take the INKLING_ENABLE_CUDA branch (make_cuda_backend()).
target_compile_definitions(strata-inkling PRIVATE INKLING_ENABLE_CUDA=1)
target_link_libraries(strata-inkling PRIVATE strata_inkling_cuda_gfx906)

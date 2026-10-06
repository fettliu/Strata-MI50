#pragma once
// HIPBLAS_V2 selects rocBLAS 2's API: hipblasDatatype_t becomes hipDataType (so the HIP_R_* enums below feed
// hipblasGemmEx_v2 directly) and the Ex entry points map to their _v2 forms. ROCm 6.3's hipblas carries no
// SetWorkspace at all (rocBLAS sizes its own workspace; strata's real workspace routing is the hipBLASLt state).
#ifndef HIPBLAS_V2
#define HIPBLAS_V2 1
#endif
#include <hipblas/hipblas.h>
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode hipblasSetMathMode
#define cublasSetStream hipblasSetStream
#define cublasStatus_t hipblasStatus_t

inline hipblasStatus_t cublas_set_workspace_no_op(hipblasHandle_t, void*, size_t) { return HIPBLAS_STATUS_SUCCESS; }
#define cublasSetWorkspace(h, buf, bytes) (::cublas_set_workspace_no_op((h), (buf), (bytes)))

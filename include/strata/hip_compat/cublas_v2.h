#pragma once
// HIPBLAS_V2 selects rocBLAS 2's API: hipblasDatatype_t becomes hipDataType (so the HIP_R_* enums below feed
// hipblasGemmEx_v2 directly) and the Ex entry points map to their _v2 forms. ROCm 6.3's hipblas carries no
// SetWorkspace at all (rocBLAS sizes its own workspace; strata's real workspace routing is the hipBLASLt state).
#ifndef HIPBLAS_V2
#define HIPBLAS_V2 1
#endif
#include <hipblas/hipblas.h>
#include <hipblas/hipblas-version.h>

#if hipblasVersionMajor == 0
// Ubuntu's hipBLAS 0.54 has no workspace API or compute-type enum. Use its
// underlying rocBLAS API directly, including ownership of the real handle,
// so the engine's explicit workspace contract is preserved.
#include <rocblas/rocblas.h>
#define CUBLAS_COMPUTE_32F rocblas_datatype_f32_r
#define CUBLAS_DEFAULT_MATH 0
#define CUBLAS_GEMM_DEFAULT rocblas_gemm_algo_standard
#define CUBLAS_OP_N rocblas_operation_none
#define CUBLAS_OP_T rocblas_operation_transpose
#define CUBLAS_STATUS_SUCCESS rocblas_status_success
#define CUDA_R_16BF rocblas_datatype_bf16_r
#define CUDA_R_16F rocblas_datatype_f16_r
#define CUDA_R_32F rocblas_datatype_f32_r
#define cublasCreate rocblas_create_handle
#define cublasDestroy rocblas_destroy_handle
#define cublasHandle_t rocblas_handle
#define cublasSetStream rocblas_set_stream
#define cublasSetWorkspace rocblas_set_workspace
#define cublasStatus_t rocblas_status

inline rocblas_status cublasSetMathMode(rocblas_handle h, int mode) {
    // This API has only its default arithmetic; GEMM below explicitly selects
    // FP32 accumulation. Reject any request for a different math mode.
    if (!h) return rocblas_status_invalid_handle;
    return mode == CUBLAS_DEFAULT_MATH ? rocblas_status_success : rocblas_status_invalid_value;
}
inline rocblas_status cublasGemmEx(rocblas_handle h, rocblas_operation trans_a,
    rocblas_operation trans_b, int m, int n, int k, const void* alpha,
    const void* a, rocblas_datatype a_type, int lda,
    const void* b, rocblas_datatype b_type, int ldb, const void* beta,
    void* c, rocblas_datatype c_type, int ldc,
    rocblas_datatype compute_type, rocblas_gemm_algo algo) {
    return rocblas_gemm_ex(h, trans_a, trans_b, m, n, k, alpha,
        a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,
        c, c_type, ldc, compute_type, algo, 0, 0);
}
#else
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
#if hipblasVersionMajor < 1
// hipBLAS 1.0 (ROCm 6.x) carries no SetWorkspace at all (rocBLAS sizes its own; strata's real
// workspace routing is the hipBLASLt state), so the CUDA call maps to a no-op there. Newer hipBLAS
// has hipblasSetWorkspace and keeps the direct mapping.
inline hipblasStatus_t cublas_set_workspace_no_op(hipblasHandle_t, void*, size_t) { return HIPBLAS_STATUS_SUCCESS; }
#define cublasSetWorkspace(h, buf, bytes) (::cublas_set_workspace_no_op((h), (buf), (bytes)))
#else
#define cublasSetWorkspace hipblasSetWorkspace
#endif
#endif

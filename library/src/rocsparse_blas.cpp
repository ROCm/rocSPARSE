/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse_blas.h"

#define RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                 \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)            \
        {                                                             \
            return rocblas2rocsparse_status(TMP_STATUS_FOR_CHECK);    \
        }                                                             \
    }

rocsparse_status rocblas2rocsparse_status(rocblas_status status)
{
    switch(status)
    {
    case rocblas_status_success:
        return rocsparse_status_success;
    case rocblas_status_invalid_handle:
        return rocsparse_status_invalid_handle;
    case rocblas_status_not_implemented:
        return rocsparse_status_not_implemented;
    case rocblas_status_invalid_pointer:
        return rocsparse_status_invalid_handle;
    case rocblas_status_invalid_size:
        return rocsparse_status_invalid_size;
    case rocblas_status_memory_error:
        return rocsparse_status_memory_error;
    case rocblas_status_internal_error:
        return rocsparse_status_internal_error;
    case rocblas_status_invalid_value:
        return rocsparse_status_invalid_value;
    case rocblas_status_perf_degraded:
    case rocblas_status_size_query_mismatch:
    case rocblas_status_size_increased:
    case rocblas_status_size_unchanged:
    case rocblas_status_continue:
    case rocblas_status_check_numerics_fail:
        return rocsparse_status_internal_error;
    }
}

rocblas_pointer_mode rocsparse2rocblas_pointer_mode(rocsparse_pointer_mode mode)
{
    switch(mode)
    {
    case rocsparse_pointer_mode_host:
        return rocblas_pointer_mode_host;
    case rocsparse_pointer_mode_device:
        return rocblas_pointer_mode_device;
    }
}

rocblas_operation rocsparse2rocblas_operation(rocsparse_operation op)
{
    switch(op)
    {
    case rocsparse_operation_none:
        return rocblas_operation_none;
    case rocsparse_operation_transpose:
        return rocblas_operation_transpose;
    case rocsparse_operation_conjugate_transpose:
        return rocblas_operation_conjugate_transpose;
    }
}

rocblas_datatype rocsparse2rocblas_datatype(rocsparse_datatype type)
{
    switch(type)
    {
    case rocsparse_datatype_f32_r:
        return rocblas_datatype_f32_r;
    case rocsparse_datatype_f64_r:
        return rocblas_datatype_f64_r;
    case rocsparse_datatype_f32_c:
        return rocblas_datatype_f32_c;
    case rocsparse_datatype_f64_c:
        return rocblas_datatype_f64_c;
    case rocsparse_datatype_i8_r:
        return rocblas_datatype_i8_r;
    case rocsparse_datatype_u8_r:
        return rocblas_datatype_u8_r;
    case rocsparse_datatype_i32_r:
        return rocblas_datatype_i32_r;
    case rocsparse_datatype_u32_r:
        return rocblas_datatype_u32_r;
    }
}

rocblas_gemm_algo rocsparse2rocblas_gemm_algo(rocsparse_blas_gemm_alg alg)
{
    switch(alg)
    {
    case rocsparse_blas_gemm_alg_standard:
        return rocblas_gemm_algo_standard;
    case rocsparse_blas_gemm_alg_solution_index:
        return rocblas_gemm_algo_solution_index;
    }
}

rocsparse_status rocsparse_blas_create_handle(rocsparse_blas_handle* handle)
{
    RETURN_IF_ROCBLAS_ERROR(rocblas_create_handle((rocblas_handle*)handle));
    return rocsparse_status_success;
}

rocsparse_status rocsparse_blas_destroy_handle(rocsparse_blas_handle handle)
{
    RETURN_IF_ROCBLAS_ERROR(rocblas_destroy_handle((rocblas_handle)handle));
    return rocsparse_status_success;
}

rocsparse_status rocsparse_blas_set_stream(rocsparse_blas_handle handle, hipStream_t stream)
{
    RETURN_IF_ROCBLAS_ERROR(rocblas_set_stream((rocblas_handle)handle, stream));
    return rocsparse_status_success;
}

rocsparse_status rocsparse_blas_set_pointer_mode(rocsparse_blas_handle  handle,
                                                 rocsparse_pointer_mode pointer_mode)
{
    RETURN_IF_ROCBLAS_ERROR(rocblas_set_pointer_mode((rocblas_handle)handle,
                                                     rocsparse2rocblas_pointer_mode(pointer_mode)));
    return rocsparse_status_success;
}

rocsparse_status rocsparse_blas_gemm_ex(rocsparse_blas_handle   handle,
                                        rocsparse_operation     transA,
                                        rocsparse_operation     transB,
                                        rocsparse_int           m,
                                        rocsparse_int           n,
                                        rocsparse_int           k,
                                        const void*             alpha,
                                        const void*             a,
                                        rocsparse_datatype      a_type,
                                        rocsparse_int           lda,
                                        const void*             b,
                                        rocsparse_datatype      b_type,
                                        rocsparse_int           ldb,
                                        const void*             beta,
                                        const void*             c,
                                        rocsparse_datatype      c_type,
                                        rocsparse_int           ldc,
                                        void*                   d,
                                        rocsparse_datatype      d_type,
                                        rocsparse_int           ldd,
                                        rocsparse_datatype      compute_type,
                                        rocsparse_blas_gemm_alg algo,
                                        int32_t                 solution_index,
                                        uint32_t                flags)
{
    RETURN_IF_ROCBLAS_ERROR(rocblas_gemm_ex((rocblas_handle)handle,
                                            rocsparse2rocblas_operation(transA),
                                            rocsparse2rocblas_operation(transB),
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            a,
                                            rocsparse2rocblas_datatype(a_type),
                                            lda,
                                            b,
                                            rocsparse2rocblas_datatype(b_type),
                                            ldb,
                                            beta,
                                            c,
                                            rocsparse2rocblas_datatype(c_type),
                                            ldc,
                                            d,
                                            rocsparse2rocblas_datatype(d_type),
                                            ldd,
                                            rocsparse2rocblas_datatype(compute_type),
                                            rocsparse2rocblas_gemm_algo(algo),
                                            solution_index,
                                            flags));
    return rocsparse_status_success;
}

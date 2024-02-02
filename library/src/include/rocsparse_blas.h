/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief rocsparse_blas.h provides rocsparse BLAS interface.
 */

#pragma once

#include "rocsparse-types.h"

typedef enum rocsparse_blas_impl_
{
    rocsparse_blas_impl_none,
    rocsparse_blas_impl_default,
    rocsparse_blas_impl_rocblas
} rocsparse_blas_impl;

const char* to_string(rocsparse_blas_impl value);

typedef struct _rocsparse_blas_rocblas_handle* rocsparse_blas_rocblas_handle;

struct _rocsparse_blas_handle
{
    rocsparse_blas_impl           blas_impl{rocsparse_blas_impl_none};
    rocsparse_blas_rocblas_handle blas_rocblas_handle{};
};

typedef struct _rocsparse_blas_handle* rocsparse_blas_handle;

/*!
*   \brief List of BLAS gemm algorithms
*/
typedef enum rocsparse_blas_gemm_alg_
{
    rocsparse_blas_gemm_alg_standard,
    rocsparse_blas_gemm_alg_solution_index
} rocsparse_blas_gemm_alg;

/*!
*   \brief Create handle.
*/
rocsparse_status rocsparse_blas_create_handle(rocsparse_blas_handle* handle,
                                              rocsparse_blas_impl    blas_impl);

/*!
*   \brief Destroy handle.
*/
rocsparse_status rocsparse_blas_destroy_handle(rocsparse_blas_handle handle);

/*!
*   \brief Set stream for handle.
*/
rocsparse_status rocsparse_blas_set_stream(rocsparse_blas_handle handle, hipStream_t stream);

/*!
*   \brief Set pointer mode handle.
*/
rocsparse_status rocsparse_blas_set_pointer_mode(rocsparse_blas_handle  handle,
                                                 rocsparse_pointer_mode pointer_mode);

/*!
*   \brief Dense gemm operation.
*/
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
                                        uint32_t                flags);

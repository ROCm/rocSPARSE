/*! \file */
/* ************************************************************************
* Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include "definitions.h"
#include "utility.h"

#include "rocsparse_coo2dense.hpp"

#include "coo2dense_device.h"

#include <rocprim/rocprim.hpp>

template <typename I, typename T>
rocsparse_status rocsparse_coo2dense_template(rocsparse_handle          handle,
                                              I                         m,
                                              I                         n,
                                              I                         nnz,
                                              const rocsparse_mat_descr descr,
                                              const T*                  coo_val,
                                              const I*                  coo_row_ind,
                                              const I*                  coo_col_ind,
                                              T*                        A,
                                              I                         lda,
                                              rocsparse_order           order)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcoo2dense"),
              m,
              n,
              nnz,
              descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)A,
              lda);

    log_bench(handle, "./rocsparse-bench -f coo2dense -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check matrix descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0 || lda < (order == rocsparse_order_column ? m : n))
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    I mn = order == rocsparse_order_column ? m : n;
    I nm = order == rocsparse_order_column ? n : m;

    // Set memory to zero.
    RETURN_IF_HIP_ERROR(
        hipMemset2DAsync(A, sizeof(T) * lda, 0, sizeof(T) * mn, nm, handle->stream));

    if(nnz > 0)
    {
#define COO2DENSE_DIM 512
        dim3 blocks((nnz - 1) / COO2DENSE_DIM + 1);
        dim3 threads(COO2DENSE_DIM);

        hipLaunchKernelGGL((coo2dense_kernel<COO2DENSE_DIM>),
                           blocks,
                           threads,
                           0,
                           stream,
                           m,
                           n,
                           nnz,
                           lda,
                           descr->base,
                           coo_val,
                           coo_row_ind,
                           coo_col_ind,
                           A,
                           order);
#undef COO2DENSE_DIM
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                         \
    template rocsparse_status rocsparse_coo2dense_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                 \
        ITYPE                     m,                                      \
        ITYPE                     n,                                      \
        ITYPE                     nnz,                                    \
        const rocsparse_mat_descr descr,                                  \
        const TTYPE*              coo_val,                                \
        const ITYPE*              coo_row_ind,                            \
        const ITYPE*              coo_col_ind,                            \
        TTYPE*                    A,                                      \
        ITYPE                     lda,                                    \
        rocsparse_order           order);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_scoo2dense(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              coo_val,
                                                 const rocsparse_int*      coo_row_ind,
                                                 const rocsparse_int*      coo_col_ind,
                                                 float*                    A,
                                                 rocsparse_int             ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld, rocsparse_order_column);
}

extern "C" rocsparse_status rocsparse_dcoo2dense(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 rocsparse_int             nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             coo_val,
                                                 const rocsparse_int*      coo_row_ind,
                                                 const rocsparse_int*      coo_col_ind,
                                                 double*                   A,
                                                 rocsparse_int             ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld, rocsparse_order_column);
}

extern "C" rocsparse_status rocsparse_ccoo2dense(rocsparse_handle               handle,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  n,
                                                 rocsparse_int                  nnz,
                                                 const rocsparse_mat_descr      descr,
                                                 const rocsparse_float_complex* coo_val,
                                                 const rocsparse_int*           coo_row_ind,
                                                 const rocsparse_int*           coo_col_ind,
                                                 rocsparse_float_complex*       A,
                                                 rocsparse_int                  ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld, rocsparse_order_column);
}

extern "C" rocsparse_status rocsparse_zcoo2dense(rocsparse_handle                handle,
                                                 rocsparse_int                   m,
                                                 rocsparse_int                   n,
                                                 rocsparse_int                   nnz,
                                                 const rocsparse_mat_descr       descr,
                                                 const rocsparse_double_complex* coo_val,
                                                 const rocsparse_int*            coo_row_ind,
                                                 const rocsparse_int*            coo_col_ind,
                                                 rocsparse_double_complex*       A,
                                                 rocsparse_int                   ld)
{
    return rocsparse_coo2dense_template(
        handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld, rocsparse_order_column);
}

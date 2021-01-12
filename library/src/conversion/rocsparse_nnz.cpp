/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_nnz.hpp"
#include "definitions.h"
#include "rocsparse_nnz_impl.hpp"
#include "utility.h"

#include "nnz_device.h"
#include <rocprim/rocprim.hpp>

template <typename I, typename J, typename T>
rocsparse_status rocsparse_nnz_kernel_row(
    rocsparse_handle handle, J m, J n, const T* A, I ld, rocsparse_order order, I* nnz_per_rows)
{
    hipStream_t stream = handle->stream;

    static constexpr rocsparse_int NNZ_DIM_X = 64;
    static constexpr rocsparse_int NNZ_DIM_Y = 16;
    rocsparse_int                  blocks    = (m - 1) / (NNZ_DIM_X * 4) + 1;
    if(std::is_same<T, rocsparse_double_complex>{})
        blocks = (m - 1) / (NNZ_DIM_X) + 1;
    dim3 k_grid(blocks);
    dim3 k_threads(NNZ_DIM_X, NNZ_DIM_Y);
    hipLaunchKernelGGL((nnz_kernel_row<NNZ_DIM_X, NNZ_DIM_Y>),
                       k_grid,
                       k_threads,
                       0,
                       stream,
                       order,
                       m,
                       n,
                       A,
                       ld,
                       nnz_per_rows);

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_nnz_kernel_col(
    rocsparse_handle handle, J m, J n, const T* A, I ld, rocsparse_order order, I* nnz_per_columns)
{
    hipStream_t stream = handle->stream;

    static constexpr rocsparse_int NB = 256;
    dim3                           kernel_blocks(n);
    dim3                           kernel_threads(NB);
    hipLaunchKernelGGL((nnz_kernel_col<NB>),
                       kernel_blocks,
                       kernel_threads,
                       0,
                       stream,
                       order,
                       m,
                       n,
                       A,
                       ld,
                       nnz_per_columns);

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_nnz_template(rocsparse_handle    handle,
                                        rocsparse_direction dir,
                                        rocsparse_order     order,
                                        J                   m,
                                        J                   n,
                                        const T*            A,
                                        I                   ld,
                                        I*                  nnz_per_row_columns)
{

    if(0 == m || 0 == n)
    {
        return rocsparse_status_success;
    }

    rocsparse_status status = rocsparse_status_invalid_value;

    switch(dir)
    {

    case rocsparse_direction_row:
    {
        status = rocsparse_nnz_kernel_row(handle, m, n, A, ld, order, nnz_per_row_columns);
        break;
    }

    case rocsparse_direction_column:
    {
        status = rocsparse_nnz_kernel_col(handle, m, n, A, ld, order, nnz_per_row_columns);
        break;
    }
    }

    return status;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                               \
    template rocsparse_status rocsparse_nnz_impl<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                              \
        rocsparse_direction       dir,                                 \
        rocsparse_order           order,                               \
        JTYPE                     m,                                   \
        JTYPE                     n,                                   \
        const rocsparse_mat_descr descr,                               \
        const TTYPE*              A,                                   \
        ITYPE                     ld,                                  \
        ITYPE*                    nnz_per_row_columns,                 \
        ITYPE*                    nnz_total_dev_host_ptr);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

extern "C" {

//
// Check if the macro CAPI_IMPL already exists.
//
#ifdef CAPI_IMPL
#error macro CAPI_IMPL is already defined.
#endif

//
// Definition of the C-implementation.
//
#define CAPI_IMPL(name_, type_)                                              \
    rocsparse_status name_(rocsparse_handle          handle,                 \
                           rocsparse_direction       dir,                    \
                           rocsparse_int             m,                      \
                           rocsparse_int             n,                      \
                           const rocsparse_mat_descr descr,                  \
                           const type_*              A,                      \
                           rocsparse_int             ld,                     \
                           rocsparse_int*            nnz_per_row_columns,    \
                           rocsparse_int*            nnz_total_dev_host_ptr) \
    {                                                                        \
        try                                                                  \
        {                                                                    \
            return rocsparse_nnz_impl(handle,                                \
                                      dir,                                   \
                                      rocsparse_order_column,                \
                                      m,                                     \
                                      n,                                     \
                                      descr,                                 \
                                      A,                                     \
                                      ld,                                    \
                                      nnz_per_row_columns,                   \
                                      nnz_total_dev_host_ptr);               \
        }                                                                    \
        catch(...)                                                           \
        {                                                                    \
            return exception_to_rocsparse_status();                          \
        }                                                                    \
    }

//
// C-implementations.
//
CAPI_IMPL(rocsparse_snnz, float);
CAPI_IMPL(rocsparse_dnnz, double);
CAPI_IMPL(rocsparse_cnnz, rocsparse_float_complex);
CAPI_IMPL(rocsparse_znnz, rocsparse_double_complex);

//
// Undefine the macro CAPI_IMPL.
//
#undef CAPI_IMPL
}

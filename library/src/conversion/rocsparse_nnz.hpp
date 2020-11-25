/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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
#include "nnz_device.h"

template <typename T>
rocsparse_status rocsparse_nnz_kernel_row(rocsparse_handle handle,
                                          rocsparse_int    m,
                                          rocsparse_int    n,
                                          const T*         A,
                                          rocsparse_int    ld,
                                          rocsparse_int*   nnz_per_rows)
{

    hipStream_t stream = handle->stream;
    {
        static constexpr int NNZ_DIM_X = 64;
        static constexpr int NNZ_DIM_Y = 16;
        rocsparse_int        blocks    = (m - 1) / (NNZ_DIM_X * 4) + 1;
        if(std::is_same<T, rocsparse_double_complex>{})
            blocks = (m - 1) / (NNZ_DIM_X) + 1;
        dim3 k_grid(blocks);
        dim3 k_threads(NNZ_DIM_X, NNZ_DIM_Y);
        hipLaunchKernelGGL((nnz_kernel_row<NNZ_DIM_X, NNZ_DIM_Y, T>),
                           k_grid,
                           k_threads,
                           0,
                           stream,
                           m,
                           n,
                           A,
                           ld,
                           nnz_per_rows);
        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse_nnz_kernel_col(rocsparse_handle handle,
                                          rocsparse_int    m,
                                          rocsparse_int    n,
                                          const T*         A,
                                          rocsparse_int    ld,
                                          rocsparse_int*   nnz_per_columns)
{

    hipStream_t stream = handle->stream;

    {
        static constexpr rocsparse_int NB = 256;
        dim3                           kernel_blocks(n);
        dim3                           kernel_threads(NB);
        hipLaunchKernelGGL((nnz_kernel_col<NB, T>),
                           kernel_blocks,
                           kernel_threads,
                           0,
                           stream,
                           m,
                           n,
                           A,
                           ld,
                           nnz_per_columns);
    }
    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_nnz_template(rocsparse_handle    handle,
                                        rocsparse_direction dir,
                                        rocsparse_int       m,
                                        rocsparse_int       n,
                                        const T*            A,
                                        rocsparse_int       ld,
                                        rocsparse_int*      nnz_per_row_columns)
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
        status = rocsparse_nnz_kernel_row(handle, m, n, A, ld, nnz_per_row_columns);
        break;
    }

    case rocsparse_direction_column:
    {
        status = rocsparse_nnz_kernel_col(handle, m, n, A, ld, nnz_per_row_columns);
        break;
    }
    }

    return status;
}

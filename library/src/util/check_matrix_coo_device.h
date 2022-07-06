/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef CHECK_MATRIX_COO_DEVICE_H
#define CHECK_MATRIX_COO_DEVICE_H

#include "common.h"

ROCSPARSE_DEVICE_ILF void record_data_status(rocsparse_data_status* data_status,
                                             rocsparse_data_status  status)
{
    if(status != rocsparse_data_status_success)
    {
        *data_status = status;
    }
}

template <unsigned int BLOCKSIZE, typename T, typename I, typename J>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void check_matrix_coo_device(J m,
                                 J n,
                                 I nnz,
                                 const T* __restrict__ coo_val,
                                 const I* __restrict__ coo_row_ind,
                                 const J* __restrict__ coo_col_ind,
                                 rocsparse_index_base   idx_base,
                                 rocsparse_matrix_type  matrix_type,
                                 rocsparse_fill_mode    uplo,
                                 rocsparse_storage_mode storage,
                                 rocsparse_data_status* data_status)
{
    I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

    if(gid >= nnz)
    {
        return;
    }

    // Check actual COO arrays for valid data
    I row = coo_row_ind[gid] - idx_base;
    I col = coo_col_ind[gid] - idx_base;

    if(row < 0 || row >= m)
    {
        record_data_status(data_status, rocsparse_data_status_invalid_index);
        return;
    }

    if(col < 0 || col >= n)
    {
        record_data_status(data_status, rocsparse_data_status_invalid_index);
        return;
    }

    // check if values are inf or nan
    T val = coo_val[gid];
    if(rocsparse_is_inf(val))
    {
        record_data_status(data_status, rocsparse_data_status_inf);
        return;
    }

    if(rocsparse_is_nan(val))
    {
        record_data_status(data_status, rocsparse_data_status_nan);
        return;
    }

    // Check matrix type and fill mode is correct
    if(matrix_type != rocsparse_matrix_type_general)
    {
        if(uplo == rocsparse_fill_mode_lower)
        {
            if(row < col)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_fill);
                return;
            }
        }
        else
        {
            if(row > col)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_fill);
                return;
            }
        }
    }

    // Check sorting is correct
    if(storage == rocsparse_storage_mode_sorted)
    {
        if(gid < nnz - 1)
        {
            I next_row = coo_row_ind[gid + 1] - idx_base;

            if(row == next_row)
            {
                I next_col = coo_col_ind[gid + 1] - idx_base;

                if(col > next_col && (next_col >= 0 && next_col < n))
                {
                    record_data_status(data_status, rocsparse_data_status_invalid_sorting);
                    return;
                }
            }
        }
    }
}

#endif // CHECK_MATRIX_COO_DEVICE_H

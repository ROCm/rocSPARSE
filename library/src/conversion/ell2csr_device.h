/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef ELL2CSR_DEVICE_H
#define ELL2CSR_DEVICE_H

#include "handle.h"

#include <hip/hip_runtime.h>

__global__ void ell2csr_index_base(rocsparse_int* __restrict__ nnz) { --(*nnz); }

__global__ void ell2csr_nnz_per_row(rocsparse_int m,
                                    rocsparse_int n,
                                    rocsparse_int ell_width,
                                    const rocsparse_int* __restrict__ ell_col_ind,
                                    rocsparse_index_base ell_base,
                                    rocsparse_int* __restrict__ csr_row_ptr,
                                    rocsparse_index_base csr_base)
{
    rocsparse_int ai = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    if(ai == 0)
    {
        csr_row_ptr[0] = csr_base;
    }

    rocsparse_int nnz = 0;

    for(rocsparse_int p = 0; p < ell_width; ++p)
    {
        rocsparse_int idx = ELL_IND(ai, p, m, ell_width);
        rocsparse_int col = ell_col_ind[idx] - ell_base;

        if(col >= 0 && col < n)
        {
            ++nnz;
        }
        else
        {
            break;
        }
    }

    csr_row_ptr[ai + 1] = nnz;
}

template <typename T>
__global__ void ell2csr_fill(rocsparse_int m,
                             rocsparse_int n,
                             rocsparse_int ell_width,
                             const rocsparse_int* __restrict__ ell_col_ind,
                             const T* __restrict__ ell_val,
                             rocsparse_index_base ell_base,
                             const rocsparse_int* __restrict__ csr_row_ptr,
                             rocsparse_int* __restrict__ csr_col_ind,
                             T* __restrict__ csr_val,
                             rocsparse_index_base csr_base)
{
    rocsparse_int ai = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    rocsparse_int csr_idx = csr_row_ptr[ai] - csr_base;

    for(rocsparse_int p = 0; p < ell_width; ++p)
    {
        rocsparse_int ell_idx = ELL_IND(ai, p, m, ell_width);
        rocsparse_int ell_col = ell_col_ind[ell_idx] - ell_base;

        if(ell_col >= 0 && ell_col < n)
        {
            csr_col_ind[csr_idx] = ell_col + csr_base;
            csr_val[csr_idx]     = ell_val[ell_idx];
            ++csr_idx;
        }
        else
        {
            break;
        }
    }
}

#endif // ELL2CSR_DEVICE_H

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
#ifndef CSR2ELL_DEVICE_H
#define CSR2ELL_DEVICE_H

#include "handle.h"
#include "common.h"

#include <hip/hip_runtime.h>

// Compute non-zero entries per CSR row and do a block reduction over the maximum
// Store result in a workspace for final reduction on part2
template <rocsparse_int NB>
__global__ void
ell_width_kernel_part1(rocsparse_int m, const rocsparse_int* csr_row_ptr, rocsparse_int* workspace)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ rocsparse_int sdata[NB];
    sdata[tid] = 0;

    for(rocsparse_int idx = gid; idx < m; idx += hipGridDim_x * hipBlockDim_x)
    {
        sdata[tid] = max(sdata[tid], csr_row_ptr[idx + 1] - csr_row_ptr[idx]);
    }

    __syncthreads();

    rocsparse_blockreduce_max<rocsparse_int, NB>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

// Part2 kernel for final reduction over the maximum CSR nnz row entries
template <rocsparse_int NB>
__global__ void ell_width_kernel_part2(rocsparse_int m, rocsparse_int* workspace)
{
    rocsparse_int tid = hipThreadIdx_x;

    __shared__ rocsparse_int sdata[NB];
    sdata[tid] = 0;

    for(rocsparse_int i = tid; i < m; i += NB)
    {
        sdata[tid] = max(sdata[tid], workspace[i]);
    }

    __syncthreads();

    rocsparse_blockreduce_max<rocsparse_int, NB>(tid, sdata);

    if(tid == 0)
    {
        workspace[0] = sdata[0];
    }
}

// CSR to ELL format conversion kernel
template <typename T>
__global__ void csr2ell_kernel(rocsparse_int m,
                               const T* csr_val,
                               const rocsparse_int* csr_row_ptr,
                               const rocsparse_int* csr_col_ind,
                               rocsparse_index_base csr_idx_base,
                               rocsparse_int ell_width,
                               rocsparse_int* ell_col_ind,
                               T* ell_val,
                               rocsparse_index_base ell_idx_base)
{
    rocsparse_int ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    rocsparse_int p = 0;

    rocsparse_int row_begin = csr_row_ptr[ai] - csr_idx_base;
    rocsparse_int row_end   = csr_row_ptr[ai + 1] - csr_idx_base;

    // Fill ELL matrix
    for(rocsparse_int aj = row_begin; aj < row_end; ++aj)
    {
        if(p >= ell_width)
        {
            break;
        }

        rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx]  = csr_col_ind[aj] - csr_idx_base + ell_idx_base;
        ell_val[idx]      = csr_val[aj];
    }

    // Pad remaining ELL structure
    for(rocsparse_int aj = row_end - row_begin; aj < ell_width; ++aj)
    {
        rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx]  = -1;
        ell_val[idx]      = static_cast<T>(0);
    }
}

#endif // CSR2ELL_DEVICE_H

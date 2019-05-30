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
#ifndef CSR2COO_DEVICE_H
#define CSR2COO_DEVICE_H

#include <hip/hip_runtime.h>

// CSR to COO matrix conversion kernel
template <rocsparse_int WF_SIZE>
__global__ void csr2coo_kernel(rocsparse_int        m,
                               const rocsparse_int* csr_row_ptr,
                               rocsparse_int*       coo_row_ind,
                               rocsparse_index_base idx_base)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int lid = tid & (WF_SIZE - 1);
    rocsparse_int nwf = hipGridDim_x * hipBlockDim_x / WF_SIZE;

    for(rocsparse_int row = gid / WF_SIZE; row < m; row += nwf)
    {
        for(rocsparse_int aj = csr_row_ptr[row] + lid; aj < csr_row_ptr[row + 1]; aj += WF_SIZE)
        {
            coo_row_ind[aj - idx_base] = row + idx_base;
        }
    }
}

#endif // CSR2COO_DEVICE_H

/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, typename I, typename J>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csr2coo_kernel(J m, const I* csr_row_ptr, J* coo_row_ind, rocsparse_index_base idx_base)
{
    J tid = hipThreadIdx_x;
    J gid = hipBlockIdx_x * BLOCKSIZE + tid;
    J lid = tid & (WF_SIZE - 1);
    J nwf = hipGridDim_x * BLOCKSIZE / WF_SIZE;

    for(J row = gid / WF_SIZE; row < m; row += nwf)
    {
        for(I aj = csr_row_ptr[row] + lid; aj < csr_row_ptr[row + 1]; aj += WF_SIZE)
        {
            coo_row_ind[aj - idx_base] = row + idx_base;
        }
    }
}

#endif // CSR2COO_DEVICE_H

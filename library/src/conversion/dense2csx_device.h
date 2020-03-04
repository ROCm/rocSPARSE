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
#ifndef DENSE2CSX_DEVICE_H
#define DENSE2CSX_DEVICE_H

#include "handle.h"
#include <hip/hip_runtime.h>

template <rocsparse_int DIM_X, typename T>
__global__ void dense2csc_kernel(rocsparse_int  base,
                                 rocsparse_int  m,
                                 rocsparse_int  n,
                                 const T*       A,
                                 rocsparse_int  lda,
                                 T*             cscValA,
                                 rocsparse_int* cscColPtrA,
                                 rocsparse_int* cscRowIndA)
{
    rocsparse_int col = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(col < n)
    {
        rocsparse_int shift = cscColPtrA[col] - base;
        for(rocsparse_int row = 0; row < m; ++row)
        {
            if(A[row + col * lda] != 0)
            {
                cscValA[shift]      = A[row + col * lda];
                cscRowIndA[shift++] = row + base;
            }
        }
    }
}

template <rocsparse_int DIM_X, typename T>
__global__ void dense2csr_kernel(rocsparse_int base,
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 const T* __restrict__ A,
                                 rocsparse_int lda,
                                 T* __restrict__ csrValA,
                                 rocsparse_int* __restrict__ csrRowPtrA,
                                 rocsparse_int* __restrict__ csrColIndA)
{
    rocsparse_int row = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(row < m)
    {
        rocsparse_int shift = csrRowPtrA[row] - base;
        for(rocsparse_int col = 0; col < n; ++col)
        {
            if(A[row + col * lda] != 0)
            {
                csrValA[shift]      = A[row + col * lda];
                csrColIndA[shift++] = col + base;
            }
        }
    }
}

#endif

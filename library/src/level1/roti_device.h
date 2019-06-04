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
#ifndef ROTI_DEVICE_H
#define ROTI_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T>
__device__ void roti_device(rocsparse_int        nnz,
                            T*                   x_val,
                            const rocsparse_int* x_ind,
                            T*                   y,
                            T                    c,
                            T                    s,
                            rocsparse_index_base idx_base)
{
    rocsparse_int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(idx >= nnz)
    {
        return;
    }

    rocsparse_int i = x_ind[idx] - idx_base;

    T xr = x_val[idx];
    T yr = y[i];

    x_val[idx] = c * xr + s * yr;
    y[i]       = c * yr - s * xr;
}

#endif // ROTI_DEVICE_H

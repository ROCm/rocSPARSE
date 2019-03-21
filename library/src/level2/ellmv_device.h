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
#ifndef ELLMV_DEVICE_H
#define ELLMV_DEVICE_H

#include "handle.h"
#include "common.h"

#include <hip/hip_runtime.h>

// ELL SpMV for general, non-transposed matrices
template <typename T>
static __device__ void ellmvn_device(rocsparse_int m,
                                     rocsparse_int n,
                                     rocsparse_int ell_width,
                                     T alpha,
                                     const rocsparse_int* ell_col_ind,
                                     const T* ell_val,
                                     const T* x,
                                     T beta,
                                     T* y,
                                     rocsparse_index_base idx_base)
{
    rocsparse_int ai = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    T sum = static_cast<T>(0);
    for(rocsparse_int p = 0; p < ell_width; ++p)
    {
        rocsparse_int idx = ELL_IND(ai, p, m, ell_width);
        rocsparse_int col = rocsparse_nontemporal_load(ell_col_ind + idx) - idx_base;

        if(col >= 0 && col < n)
        {
            sum = rocsparse_fma(
                rocsparse_nontemporal_load(ell_val + idx), rocsparse_ldg(x + col), sum);
        }
        else
        {
            break;
        }
    }

    if(beta != static_cast<T>(0))
    {
        T yv = rocsparse_nontemporal_load(y + ai);
        rocsparse_nontemporal_store(rocsparse_fma(beta, yv, alpha * sum), y + ai);
    }
    else
    {
        rocsparse_nontemporal_store(alpha * sum, y + ai);
    }
}

#endif // ELLMV_DEVICE_H

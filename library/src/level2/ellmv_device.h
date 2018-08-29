/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef ELLMV_DEVICE_H
#define ELLMV_DEVICE_H

#include "handle.h"

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
        rocsparse_int col = ell_col_ind[idx] - idx_base;

        if(col >= 0 && col < n)
        {
            sum += ell_val[idx] * __ldg(x + col);
        }
        else
        {
            break;
        }
    }

    if(beta != static_cast<T>(0))
    {
        y[ai] = beta * y[ai] + alpha * sum;
    }
    else
    {
        y[ai] = alpha * sum;
    }
}

#endif // ELLMV_DEVICE_H

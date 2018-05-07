/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef CSR2HYB_DEVICE_H
#define CSR2HYB_DEVICE_H

#include "handle.h"

#include <hip/hip_runtime.h>

template <typename T>
__device__
void csr2ell_device(rocsparse_int m,
                    const T *csr_val,
                    const rocsparse_int *csr_row_ptr,
                    const rocsparse_int *csr_col_ind,
                    rocsparse_int ell_width,
                    rocsparse_int *ell_col_ind,
                    T *ell_val)
{
    rocsparse_int ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (ai >= m)
    {
        return;
    }

    rocsparse_int p = 0;
    rocsparse_int aj = csr_row_ptr[ai];

    // Fill ELL matrix
    for (; aj<csr_row_ptr[ai+1]; ++aj)
    {
        if (p >= ell_width)
        {
            break;
        }

        rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx] = csr_col_ind[aj];
        ell_val[idx] = csr_val[aj];
    }

    // TODO store rownnz

    // Pad remaining ELL structure
    for (; aj<ell_width; ++aj)
    {
        rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx] = -1;
        ell_val[idx] = static_cast<T>(0);
    }
}

#endif // CSR2HYB_DEVICE_H

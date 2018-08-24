/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef CSR2COO_DEVICE_H
#define CSR2COO_DEVICE_H

#include <hip/hip_runtime.h>

// CSR to COO matrix conversion kernel
template <rocsparse_int WF_SIZE>
__global__ void csr2coo_kernel(rocsparse_int m,
                               const rocsparse_int* csr_row_ptr,
                               rocsparse_int* coo_row_ind,
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

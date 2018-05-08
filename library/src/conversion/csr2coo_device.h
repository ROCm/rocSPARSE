/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef CSR2COO_DEVICE_H
#define CSR2COO_DEVICE_H

#include <hip/hip_runtime.h>

template <rocsparse_int THREADS>
__global__
void csr2coo_kernel(rocsparse_int m,
                    const rocsparse_int *csr_row_ptr,
                    rocsparse_int *coo_row_ind)
{
    rocsparse_int gid  = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocsparse_int lid  = hipThreadIdx_x % THREADS;
    rocsparse_int vid  = gid / THREADS;
    rocsparse_int nvec = hipGridDim_x * hipBlockDim_x / THREADS;

    for(rocsparse_int ai=vid; ai<m; ai+=nvec)
    {
        for(rocsparse_int aj=csr_row_ptr[ai]+lid; aj<csr_row_ptr[ai+1]; aj+=THREADS)
        {
            coo_row_ind[aj] = ai;
        }
    }
}

#endif // CSR2COO_DEVICE_H

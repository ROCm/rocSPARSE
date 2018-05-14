/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef CSR2HYB_DEVICE_H
#define CSR2HYB_DEVICE_H

#include "handle.h"

#include <hip/hip_runtime.h>

template <rocsparse_int NB>
__device__
void ell_width_reduce(rocsparse_int tid, rocsparse_int *data)
{
    __syncthreads();

    for (int i=NB>>1; i>0; i>>=1)
    {
        if (tid < i)
        {
            data[tid] = max(data[tid], data[tid+i]);
        }

        __syncthreads();
    }
}

template <rocsparse_int NB>
__global__
void ell_width_kernel_part1(rocsparse_int m,
                            const rocsparse_int *csr_row_ptr,
                            rocsparse_int *workspace)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ rocsparse_int sdata[NB];

    if (gid < m)
    {
        sdata[tid] = csr_row_ptr[gid+1] - csr_row_ptr[gid];
    }
    else
    {
        sdata[tid] = 0;
    }

    ell_width_reduce<NB>(tid, sdata);

    if (tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

template <rocsparse_int NB>
__global__
void ell_width_kernel_part2(rocsparse_int m, rocsparse_int *workspace)
{
    rocsparse_int tid = hipThreadIdx_x;

    __shared__ rocsparse_int sdata[NB];
    sdata[tid] = 0;

    for (rocsparse_int i=tid; i<m; i+=NB)
    {
        sdata[tid] = (workspace[i] > sdata[tid]) ? workspace[i] : sdata[tid];
    }

    __syncthreads();

    if (m < 32)
    {
        if (tid == 0)
        {
            for (rocsparse_int i=1; i<m; ++i)
            {
                sdata[0] = (sdata[i] > sdata[0]) ? sdata[i] : sdata[0];
            }
        }
    }
    else
    {
        ell_width_reduce<NB>(tid, sdata);
    }

    if (tid == 0)
    {
        workspace[0] = sdata[0];
    }
}

template <typename T>
__global__
void csr2ell_kernel(rocsparse_int m,
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

    rocsparse_int p  = 0;
    rocsparse_int aj = csr_row_ptr[ai];

    // Fill ELL matrix
    for (; aj<csr_row_ptr[ai+1]; ++aj)
    {
        if (p >= ell_width)
        {
            break;
        }

        rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx]  = csr_col_ind[aj];
        ell_val[idx]      = csr_val[aj];
    }

    // TODO store rownnz

    // Pad remaining ELL structure
    for (; aj<ell_width; ++aj)
    {
        rocsparse_int idx = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx]  = -1;
        ell_val[idx]      = static_cast<T>(0);
    }
}

#endif // CSR2HYB_DEVICE_H

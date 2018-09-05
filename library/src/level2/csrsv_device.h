/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef CSRSV_DEVICE_H
#define CSRSV_DEVICE_H

#include <hip/hip_runtime.h>

template <rocsparse_int WF_SIZE>
static __device__ __inline__ void two_reduce(int* local_max, int *local_spin)
{
#if defined(__HIP_PLATFORM_HCC__)
    int max_depth = *local_max;

    if(WF_SIZE > 1)
    {
        // row_shr = 1
        max_depth = __hip_move_dpp(*local_max, 0x111, 0xf, 0xf, 0);
        *local_spin += __hip_move_dpp(*local_spin, 0x111, 0xf, 0xf, 0);
        *local_max = (max_depth > *local_max) ? max_depth : *local_max;
    }

    if(WF_SIZE > 2)
    {
        // row_shr = 2
        max_depth = __hip_move_dpp(*local_max, 0x112, 0xf, 0xf, 0);
        *local_spin += __hip_move_dpp(*local_spin, 0x112, 0xf, 0xf, 0);
        *local_max = (max_depth > *local_max) ? max_depth : *local_max;
    }

    if(WF_SIZE > 4)
    {
        // row_shr = 4 ; bank_mask = 0xe
        max_depth = __hip_move_dpp(*local_max, 0x114, 0xf, 0xe, 0);
        *local_spin += __hip_move_dpp(*local_spin, 0x114, 0xf, 0xe, 0);
        *local_max = (max_depth > *local_max) ? max_depth : *local_max;
    }

    if(WF_SIZE > 8)
    {
        // row_shr = 8 ; bank_mask = 0xc
        max_depth = __hip_move_dpp(*local_max, 0x118, 0xf, 0xc, 0);
        *local_spin += __hip_move_dpp(*local_spin, 0x118, 0xf, 0xc, 0);
        *local_max = (max_depth > *local_max) ? max_depth : *local_max;
    }

    if(WF_SIZE > 16)
    {
        // row_bcast = 15 ; row_mask = 0xa
        max_depth = __hip_move_dpp(*local_max, 0x142, 0xa, 0xf, 0);
        *local_spin += __hip_move_dpp(*local_spin, 0x142, 0xa, 0xf, 0);
        *local_max = (max_depth > *local_max) ? max_depth : *local_max;
    }

    if(WF_SIZE > 32)
    {
        // row_bcast = 31 ; row_mask = 0xc
        max_depth = __hip_move_dpp(*local_max, 0x143, 0xc, 0xf, 0);
        *local_spin += __hip_move_dpp(*local_spin, 0x143, 0xc, 0xf, 0);
        *local_max = (max_depth > *local_max) ? max_depth : *local_max;
    }
#elif defined(__HIP_PLATFORM_NVCC__)
    for(int i = WF_SIZE >> 1; i >= 1; i >>= 1)
    {
        *local_max = max(*local_max, __shfl_down_sync(0xffffffff, *local_max, i));
        *local_spin += __shfl_down_sync(0xffffffff, *local_spin, i);
    }
#endif
}

template <rocsparse_int WF_SIZE, rocsparse_fill_mode FILL_MODE>
__global__ void csrsv_analysis_kernel(rocsparse_int m,
                                      const rocsparse_int* __restrict__ csr_row_ptr,
                                      const rocsparse_int* __restrict__ csr_col_ind,
                                      rocsparse_int* __restrict__ csr_diag_ind,
                                      rocsparse_int* __restrict__ done_array,
                                      rocsparse_int* __restrict__ rows_per_level,
                                      rocsparse_int* __restrict__ max_depth,
                                      unsigned long long* __restrict__ total_spin,
                                      rocsparse_int* __restrict__ max_nnz,
                                      rocsparse_index_base idx_base)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int lid = tid & (WF_SIZE - 1);
    rocsparse_int row = gid / WF_SIZE;

    if(row >= m)
    {
        return;
    }

    if(FILL_MODE == rocsparse_fill_mode_upper)
    {
        // Processing upper triangular matrix
        row = m - 1 - row;
    }

    // Initialize matrix diagonal index
    if(lid == 0)
    {
        csr_diag_ind[row] = -1;
    }

    rocsparse_int local_max = 0;
    rocsparse_int local_spin = 0;

    int row_begin = csr_row_ptr[row] - idx_base;
    int row_end   = csr_row_ptr[row + 1] - idx_base;

    // This wavefront operates on a single row, from its beginning to end.
    for(int j = row_begin + lid; j < row_end; j += WF_SIZE)
    {
        // local_col will tell us, for this iteration of the above for loop
        // (i.e. for this entry in this row), which columns contain the
        // non-zero values. We must then ensure that the output from the row
        // associated with the local_col is complete to ensure that we can
        // calculate the right answer.
        int local_col = csr_col_ind[j] - idx_base;

        // Store diagonal index
        if(local_col == row)
        {
            csr_diag_ind[row] = j;
        }

        if(FILL_MODE == rocsparse_fill_mode_upper)
        {
            if(local_col <= row)
            {
                continue;
            }
        }
        else if(FILL_MODE == rocsparse_fill_mode_lower)
        {
            // Diagonal and above, skip this.
            if (local_col >= row)
            {
                break;
            }
        }

        int local_done = 0;

        // While there are threads in this workgroup that have been unable to
        // get their input, loop and wait for the flag to exist.
        while (!local_done)
        {
#if defined(__HIP_PLATFORM_HCC__)
            local_done = __atomic_load_n(&done_array[local_col], __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
            local_done = atomicOr(&done_array[local_col], 0);
#endif
            ++local_spin;
        }

        local_max = max(local_done, local_max);
    }

    // Determine maximum local depth and local spin loops
    two_reduce<WF_SIZE>(&local_max, &local_spin);
    ++local_max;

    if (lid == WF_SIZE - 1)
    {
#if defined(__HIP_PLATFORM_HCC__)
        __atomic_store_n(&done_array[row], local_max, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        atomicOr(&done_array[row], local_max);
#endif

        // Must atomic these next three, since other WGs are doing the same thing
        // We're sending out "local_max - 1" because of 0-based indexing.
        // However, we needed to put a non-zero value into the done_array up above
        // when we crammed local_depth in, so these two will be off by one.
        atomicAdd(&rows_per_level[local_max-1], 1);
        atomicMax(max_depth, local_max);
        atomicAdd(total_spin, local_spin);
        atomicMax(max_nnz, row_end - row_begin);
    }
}






#if defined(__HIP_PLATFORM_HCC__)
// While HIP does not contain llvm intrinsics
__device__ int __llvm_amdgcn_readlane(int index, int offset) __asm("llvm.amdgcn.readlane");

static __device__ __inline__ float wf_reduce(float temp_sum)
{
    typedef union flt_b32 {
        float val;
        int b32;
    } flt_b32_t;
    flt_b32_t upper_sum, t_temp_sum;

    t_temp_sum.val = temp_sum;
    upper_sum.b32 = __hip_ds_swizzle(t_temp_sum.b32, 0x80b1);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32 = __hip_ds_swizzle(t_temp_sum.b32, 0x804e);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32 = __hip_ds_swizzle(t_temp_sum.b32, 0x101f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32 = __hip_ds_swizzle(t_temp_sum.b32, 0x201f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32 = __hip_ds_swizzle(t_temp_sum.b32, 0x401f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32 = __llvm_amdgcn_readlane(t_temp_sum.b32, 32);
    t_temp_sum.val += upper_sum.val;
    temp_sum = t_temp_sum.val;

    return temp_sum;
}

static __device__ __inline__ double wf_reduce(double temp_sum)
{
    typedef union dbl_b32 {
        double val;
        int b32[2];
    } dbl_b32_t;
    dbl_b32_t upper_sum, t_temp_sum;

    t_temp_sum.val = temp_sum;
    upper_sum.b32[0] = __hip_ds_swizzle(t_temp_sum.b32[0], 0x80b1);
    upper_sum.b32[1] = __hip_ds_swizzle(t_temp_sum.b32[1], 0x80b1);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32[0] = __hip_ds_swizzle(t_temp_sum.b32[0], 0x804e);
    upper_sum.b32[1] = __hip_ds_swizzle(t_temp_sum.b32[1], 0x804e);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32[0] = __hip_ds_swizzle(t_temp_sum.b32[0], 0x101f);
    upper_sum.b32[1] = __hip_ds_swizzle(t_temp_sum.b32[1], 0x101f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32[0] = __hip_ds_swizzle(t_temp_sum.b32[0], 0x201f);
    upper_sum.b32[1] = __hip_ds_swizzle(t_temp_sum.b32[1], 0x201f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32[0] = __hip_ds_swizzle(t_temp_sum.b32[0], 0x401f);
    upper_sum.b32[1] = __hip_ds_swizzle(t_temp_sum.b32[1], 0x401f);
    t_temp_sum.val += upper_sum.val;
    upper_sum.b32[0] = __llvm_amdgcn_readlane(t_temp_sum.b32[0], 32);
    upper_sum.b32[1] = __llvm_amdgcn_readlane(t_temp_sum.b32[1], 32);
    t_temp_sum.val += upper_sum.val;
    temp_sum = t_temp_sum.val;

    return temp_sum;
}
#elif defined(__HIP_PLATFORM_NVCC__)
template <typename T>
static __device__ __inline__ T wf_reduce(T temp_sum)
{
    for(int i = 16; i >= 1; i >>= 1)
    {
        temp_sum += __shfl_down_sync(0xffffffff, temp_sum, i);
    }

    return temp_sum;
}
#endif

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__device__ void csrsv_device(rocsparse_int m,
                             T alpha,
                             const rocsparse_int* __restrict__ csr_row_ptr,
                             const rocsparse_int* __restrict__ csr_col_ind,
                             const T* __restrict__ csr_val,
                             const T* __restrict__ x,
                             T* __restrict__ y,
                             rocsparse_int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int offset,
                             rocsparse_index_base idx_base,
                             rocsparse_fill_mode fill_mode,
                             rocsparse_diag_type diag_type)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * BLOCKSIZE + tid;
    rocsparse_int lid = tid & (WF_SIZE - 1);
    rocsparse_int wid = tid / WF_SIZE;
    rocsparse_int idx = gid / WF_SIZE;

    __shared__ T diagonal[BLOCKSIZE / WF_SIZE];

    if(idx >= m)
    {
        return;
    }

    rocsparse_int row       = map[idx + offset];
    rocsparse_int row_begin = csr_row_ptr[row] - idx_base;
    rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;

    T local_sum = static_cast<T>(0);

    if(lid == 0)
    {
        local_sum = alpha * x[row];
    }

    for(rocsparse_int j = row_begin + lid; j < row_end; j += WF_SIZE)
    {
        rocsparse_int local_col = csr_col_ind[j] - idx_base;
        T local_val = csr_val[j];

        if(fill_mode == rocsparse_fill_mode_upper)
        {
            // Processing upper triangular
            if(local_col < row)
            {
                continue;
            }

            if(local_col == row)
            {
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    diagonal[wid] = static_cast<T>(1) / local_val;
                }
                
                continue;
            }
        }
        else if(fill_mode == rocsparse_fill_mode_lower)
        {
            // Processing lower triangular
            if(local_col > row)
            {
                break;
            }

            if(local_col == row)
            {
                if(diag_type == rocsparse_diag_type_non_unit)
                {
                    diagonal[wid] = static_cast<T>(1) / local_val;
                }

                break;
            }
        }

#if defined(__HIP_PLATFORM_HCC__)
        while(!__atomic_load_n(&done_array[local_col], __ATOMIC_RELAXED));
#elif defined(__HIP_PLATFORM_NVCC__)
        while(!atomicOr(&done_array[local_col], 0));
#endif

#if defined(__HIP_PLATFORM_HCC__)
        T out_val;
        __atomic_load(&y[local_col], &out_val, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        T out_val = y[local_col];
#endif

        local_sum -= local_val * out_val;
    }

    local_sum = wf_reduce(local_sum);

    if(diag_type == rocsparse_diag_type_non_unit)
    {
        local_sum *= diagonal[wid];
    }

    if (lid == 0)
    {
#if defined(__HIP_PLATFORM_HCC__)
        __atomic_store(&y[row], &local_sum, __ATOMIC_RELAXED);
        __atomic_store_n(&done_array[row], 1, __ATOMIC_RELAXED);
#elif defined(__HIP_PLATFORM_NVCC__)
        y[row] = local_sum;
        atomicOr(&done_array[row], 1);
#endif
    }
}

#endif // CSRSV_DEVICE_H

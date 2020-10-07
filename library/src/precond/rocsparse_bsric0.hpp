/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_BSRIC0_HPP
#define ROCSPARSE_BSRIC0_HPP

#include "../level2/rocsparse_csrsv.hpp"
#include "bsric0_device.h"
#include "definitions.h"
#include "rocsparse.h"
#include "rocsparse_csric0.hpp"
#include "utility.h"

#include <hip/hip_runtime.h>

#define launch_bsric0_small_maxnnzb_unrolled8x8_kernel(                          \
    T, block_size, wf_size, maz_nnzb, bsr_block_dim)                             \
    hipLaunchKernelGGL((bsric0_small_maxnnzb_unrolled8x8_kernel<T,               \
                                                                block_size,      \
                                                                wf_size,         \
                                                                maz_nnzb,        \
                                                                bsr_block_dim>), \
                       bsric0_blocks,                                            \
                       bsric0_threads,                                           \
                       0,                                                        \
                       stream,                                                   \
                       dir,                                                      \
                       mb,                                                       \
                       block_dim,                                                \
                       bsr_row_ptr,                                              \
                       bsr_col_ind,                                              \
                       bsr_val,                                                  \
                       info->bsric0_info->trm_diag_ind,                          \
                       d_done_array,                                             \
                       info->bsric0_info->row_map,                               \
                       info->zero_pivot,                                         \
                       descr->base);

#define launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(                        \
    T, block_size, wf_size, maz_nnzb, bsr_block_dim, blk_size_y)                       \
    hipLaunchKernelGGL((bsric0_hash_small_maxnnzb_small_blockdim_kernel<T,             \
                                                                        block_size,    \
                                                                        wf_size,       \
                                                                        maz_nnzb,      \
                                                                        bsr_block_dim, \
                                                                        blk_size_y>),  \
                       bsric0_blocks,                                                  \
                       bsric0_threads,                                                 \
                       0,                                                              \
                       stream,                                                         \
                       dir,                                                            \
                       mb,                                                             \
                       block_dim,                                                      \
                       bsr_row_ptr,                                                    \
                       bsr_col_ind,                                                    \
                       bsr_val,                                                        \
                       info->bsric0_info->trm_diag_ind,                                \
                       d_done_array,                                                   \
                       info->bsric0_info->row_map,                                     \
                       info->zero_pivot,                                               \
                       descr->base);

#define launch_bsric0_hash_large_maxnnzb_small_blockdim_kernel(                        \
    T, block_size, wf_size, maz_nnzb, bsr_block_dim, blk_size_y)                       \
    hipLaunchKernelGGL((bsric0_hash_large_maxnnzb_small_blockdim_kernel<T,             \
                                                                        block_size,    \
                                                                        wf_size,       \
                                                                        maz_nnzb,      \
                                                                        bsr_block_dim, \
                                                                        blk_size_y>),  \
                       bsric0_blocks,                                                  \
                       bsric0_threads,                                                 \
                       0,                                                              \
                       stream,                                                         \
                       dir,                                                            \
                       mb,                                                             \
                       block_dim,                                                      \
                       bsr_row_ptr,                                                    \
                       bsr_col_ind,                                                    \
                       bsr_val,                                                        \
                       info->bsric0_info->trm_diag_ind,                                \
                       d_done_array,                                                   \
                       info->bsric0_info->row_map,                                     \
                       info->zero_pivot,                                               \
                       descr->base);

#define launch_bsric0_hash_large_blockdim_kernel(T, block_size, wf_size, maz_nnzb, bsr_block_dim) \
    hipLaunchKernelGGL(                                                                           \
        (bsric0_hash_large_blockdim_kernel<T, block_size, wf_size, maz_nnzb, bsr_block_dim>),     \
        bsric0_blocks,                                                                            \
        bsric0_threads,                                                                           \
        0,                                                                                        \
        stream,                                                                                   \
        dir,                                                                                      \
        mb,                                                                                       \
        block_dim,                                                                                \
        bsr_row_ptr,                                                                              \
        bsr_col_ind,                                                                              \
        bsr_val,                                                                                  \
        info->bsric0_info->trm_diag_ind,                                                          \
        d_done_array,                                                                             \
        info->bsric0_info->row_map,                                                               \
        info->zero_pivot,                                                                         \
        descr->base);

#define launch_bsric0_binsearch_kernel(T, block_size, wf_size, threads_per_row, sleep)            \
    hipLaunchKernelGGL((bsric0_binsearch_kernel<T, block_size, wf_size, threads_per_row, sleep>), \
                       bsric0_blocks,                                                             \
                       bsric0_threads,                                                            \
                       0,                                                                         \
                       stream,                                                                    \
                       dir,                                                                       \
                       mb,                                                                        \
                       block_dim,                                                                 \
                       bsr_row_ptr,                                                               \
                       bsr_col_ind,                                                               \
                       bsr_val,                                                                   \
                       info->bsric0_info->trm_diag_ind,                                           \
                       d_done_array,                                                              \
                       info->bsric0_info->row_map,                                                \
                       info->zero_pivot,                                                          \
                       descr->base);

template <typename T>
rocsparse_status rocsparse_bsric0_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nnzb,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  bsr_val,
                                                    const rocsparse_int*      bsr_row_ptr,
                                                    const rocsparse_int*      bsr_col_ind,
                                                    rocsparse_int             block_dim,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_analysis_policy analysis,
                                                    rocsparse_solve_policy    solve,
                                                    void*                     temp_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsric0_analysis"),
              dir,
              mb,
              nnzb,
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)info,
              solve,
              analysis);

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check analysis policy
    if(analysis != rocsparse_analysis_policy_reuse && analysis != rocsparse_analysis_policy_force)
    {
        return rocsparse_status_invalid_value;
    }

    // Check solve policy
    if(solve != rocsparse_solve_policy_auto)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(mb < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnzb < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nnzb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Differentiate the analysis policies
    if(analysis == rocsparse_analysis_policy_reuse)
    {
        // We try to re-use already analyzed lower part, if available.
        // It is the user's responsibility that this data is still valid,
        // since he passed the 'reuse' flag.

        // If bsric0 meta data is already available, do nothing
        if(info->bsric0_info != nullptr)
        {
            return rocsparse_status_success;
        }

        // Check for other lower analysis meta data
        // Note: Uncomment when bsrilu0 is complete
        // if(info->bsrilu0_info != nullptr)
        // {
        //     // bsrilu0 meta data
        //     info->bsric0_info = info->bsrilu0_info;
        //     return rocsparse_status_success;
        // }

        if(info->bsrsv_lower_info != nullptr)
        {
            info->bsric0_info = info->bsrsv_lower_info;
            return rocsparse_status_success;
        }
    }

    // User is explicitly asking to force a re-analysis, or no valid data has been
    // found to be re-used.

    // Clear bsric0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsric0_info));

    // Create bsric0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&info->bsric0_info));

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                     rocsparse_operation_none,
                                                     mb,
                                                     nnzb,
                                                     descr,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     info->bsric0_info,
                                                     &info->zero_pivot,
                                                     temp_buffer));

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_bsric0_template(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_int             mb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           T*                        bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_solve_policy    policy,
                                           void*                     temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsric0"),
              mb,
              nnzb,
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              (const void*&)info,
              policy,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f bsric0 -r", replaceX<T>("X"), "--mtx <matrix.mtx> ");

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || nnzb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nnzb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for analysis call
    if(info->bsric0_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    ptr += 256;

    // done array
    int* d_done_array = reinterpret_cast<int*>(ptr);

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_done_array, 0, sizeof(int) * mb, stream));

    // Max nnz blocks per row
    rocsparse_int max_nnzb = info->bsric0_info->max_nnz;

    // Max nnz per row
    rocsparse_int max_nnz = max_nnzb * block_dim;

    // Determine gcnArch
    int gcnArch = handle->properties.gcnArch;
    int asicRev = handle->asic_rev;

    rocsparse_int m = mb * block_dim;

    if(gcnArch == 908 && asicRev < 2)
    {
#define BSRIC0_DIM 64
        if(block_dim <= 2)
        {
            dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
            dim3 bsric0_threads(64);
            launch_bsric0_binsearch_kernel(T, 64, 64, 32, true);
        }
        else if(block_dim <= 4)
        {
            dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
            dim3 bsric0_threads(64);
            launch_bsric0_binsearch_kernel(T, 64, 64, 16, true);
        }
        else if(block_dim <= 8)
        {
            dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
            dim3 bsric0_threads(64);
            launch_bsric0_binsearch_kernel(T, 64, 64, 8, true);
        }
        else if(block_dim <= 16)
        {
            dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
            dim3 bsric0_threads(64);
            launch_bsric0_binsearch_kernel(T, 64, 64, 4, true);
        }
        else if(block_dim <= 32)
        {
            dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
            dim3 bsric0_threads(64);
            launch_bsric0_binsearch_kernel(T, 64, 64, 2, true);
        }
        else
        {
            dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
            dim3 bsric0_threads(64);
            launch_bsric0_binsearch_kernel(T, 64, 64, 1, true);
        }
#undef BSRIC0_DIM

        return rocsparse_status_success;
    }

    if(handle->wavefront_size == 32)
    {
#define BSRIC0_DIM 32
        dim3 bsric0_blocks((mb * 32 - 1) / 32 + 1);
        dim3 bsric0_threads(32);
        launch_bsric0_binsearch_kernel(T, 32, 32, 1, false);
#undef BSRIC0_DIM

        return rocsparse_status_success;
    }

    if(handle->wavefront_size == 64)
    {
#define BSRIC0_DIM 64
        if(max_nnzb <= 8)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 8, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 8, 4, 4);
            }
            else if(block_dim == 5)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(5, 1, 5);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 8, 5);
            }
            else if(block_dim == 6)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(6, 1, 6);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 8, 6);
            }
            else if(block_dim == 7)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(7, 1, 7);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 8, 7);
            }
            else if(block_dim == 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(8, 1, 8);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 8, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 8, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 8, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else if(max_nnzb <= 12)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 12, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 12, 4, 4);
            }
            else if(block_dim == 5)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(5, 1, 5);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 12, 5);
            }
            else if(block_dim == 6)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(6, 1, 6);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 12, 6);
            }
            else if(block_dim == 7)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(7, 1, 7);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 12, 7);
            }
            else if(block_dim == 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(8, 1, 8);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 12, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 12, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 12, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else if(max_nnzb <= 16)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 16, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 16, 4, 4);
            }
            else if(block_dim == 5)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(5, 1, 5);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 16, 5);
            }
            else if(block_dim == 6)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(6, 1, 6);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 16, 6);
            }
            else if(block_dim == 7)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(7, 1, 7);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 16, 7);
            }
            else if(block_dim == 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(8, 1, 8);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 16, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 16, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 16, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else if(max_nnzb <= 20)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 20, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 20, 4, 4);
            }
            else if(block_dim == 5)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(5, 1, 5);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 20, 5);
            }
            else if(block_dim == 6)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(6, 1, 6);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 20, 6);
            }
            else if(block_dim == 7)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(7, 1, 7);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 20, 7);
            }
            else if(block_dim == 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(8, 1, 8);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 20, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 20, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 20, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else if(max_nnzb <= 24)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 24, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 24, 4, 4);
            }
            else if(block_dim == 5)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(5, 1, 5);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 24, 5);
            }
            else if(block_dim == 6)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(6, 1, 6);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 24, 6);
            }
            else if(block_dim == 7)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(7, 1, 7);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 24, 7);
            }
            else if(block_dim == 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(8, 1, 8);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 24, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 24, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 24, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else if(max_nnzb <= 28)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 28, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_small_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 28, 4, 4);
            }
            else if(block_dim == 5)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(5, 1, 5);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 28, 5);
            }
            else if(block_dim == 6)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(6, 1, 6);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 28, 6);
            }
            else if(block_dim == 7)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(7, 1, 7);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 28, 7);
            }
            else if(block_dim == 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(8, 1, 8);
                launch_bsric0_small_maxnnzb_unrolled8x8_kernel(T, BSRIC0_DIM, 64, 28, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 28, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 28, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else if(max_nnzb <= 64)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_large_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 64, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_large_maxnnzb_small_blockdim_kernel(T, BSRIC0_DIM, 64, 64, 4, 4);
            }
            else if(block_dim <= 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 128, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 64, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 64, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else if(max_nnzb <= 128)
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(2, 16, 2);
                launch_bsric0_hash_large_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 128, 2, 16);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(4, 4, 4);
                launch_bsric0_hash_large_maxnnzb_small_blockdim_kernel(
                    T, BSRIC0_DIM, 64, 128, 4, 4);
            }
            else if(block_dim <= 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 128, 8);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 128, 16);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / BSRIC0_DIM + 1);
                dim3 bsric0_threads(BSRIC0_DIM);
                launch_bsric0_hash_large_blockdim_kernel(T, BSRIC0_DIM, 64, 128, 32);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
        else
        {
            if(block_dim <= 2)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 32, false);
            }
            else if(block_dim <= 4)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 16, false);
            }
            else if(block_dim <= 8)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 8, false);
            }
            else if(block_dim <= 16)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 4, false);
            }
            else if(block_dim <= 32)
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 2, false);
            }
            else
            {
                dim3 bsric0_blocks((mb * 64 - 1) / 64 + 1);
                dim3 bsric0_threads(64);
                launch_bsric0_binsearch_kernel(T, 64, 64, 1, false);
            }
        }
#undef BSRIC0_DIM

        return rocsparse_status_success;
    }

    return rocsparse_status_arch_mismatch;
}

#endif // ROCSPARSE_BSRIC0_HPP

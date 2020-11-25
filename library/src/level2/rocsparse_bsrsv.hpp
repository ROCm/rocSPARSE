/*! \file */
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
#ifndef ROCSPARSE_BSRSV_HPP
#define ROCSPARSE_BSRSV_HPP

#include "../level2/rocsparse_csrsv.hpp"
#include "bsrsv_device.h"

#define LAUNCH_BSRSV_GTHR_DIM(bsize, wfsize, dim)                      \
    hipLaunchKernelGGL((bsrsv_gather<T, wfsize, bsize / wfsize, dim>), \
                       dim3((wfsize * nnzb - 1) / bsize + 1),          \
                       dim3(wfsize, bsize / wfsize),                   \
                       0,                                              \
                       stream,                                         \
                       nnzb,                                           \
                       bsrsv->trmt_perm,                               \
                       bsr_val,                                        \
                       bsrt_val,                                       \
                       bsr_dim);

#define LAUNCH_BSRSV_GTHR(bsize, wfsize, dim) \
    if(dim <= 2)                              \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 4, 2)    \
    }                                         \
    else if(dim <= 4)                         \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 16, 4)   \
    }                                         \
    else if(wfsize == 32)                     \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 16, 4)   \
    }                                         \
    else                                      \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 64, 8)   \
    }

#define LAUNCH_BSRSV_SHARED(fill, ptr, bsize, wfsize, dim, arch, asic)        \
    if(fill == rocsparse_fill_mode_lower)                                     \
    {                                                                         \
        if(ptr == rocsparse_pointer_mode_host)                                \
        {                                                                     \
            if(arch == 908 && asic < 2)                                       \
            {                                                                 \
                LAUNCH_BSRSV_LOWER_SHARED_HOSTPTR(bsize, wfsize, dim, true);  \
            }                                                                 \
            else                                                              \
            {                                                                 \
                LAUNCH_BSRSV_LOWER_SHARED_HOSTPTR(bsize, wfsize, dim, false); \
            }                                                                 \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            if(arch == 908 && asic < 2)                                       \
            {                                                                 \
                LAUNCH_BSRSV_LOWER_SHARED_DEVPTR(bsize, wfsize, dim, true);   \
            }                                                                 \
            else                                                              \
            {                                                                 \
                LAUNCH_BSRSV_LOWER_SHARED_DEVPTR(bsize, wfsize, dim, false);  \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    else                                                                      \
    {                                                                         \
        if(ptr == rocsparse_pointer_mode_host)                                \
        {                                                                     \
            if(arch == 908 && asic < 2)                                       \
            {                                                                 \
                LAUNCH_BSRSV_UPPER_SHARED_HOSTPTR(bsize, wfsize, dim, true);  \
            }                                                                 \
            else                                                              \
            {                                                                 \
                LAUNCH_BSRSV_UPPER_SHARED_HOSTPTR(bsize, wfsize, dim, false); \
            }                                                                 \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            if(arch == 908 && asic < 2)                                       \
            {                                                                 \
                LAUNCH_BSRSV_UPPER_SHARED_DEVPTR(bsize, wfsize, dim, true);   \
            }                                                                 \
            else                                                              \
            {                                                                 \
                LAUNCH_BSRSV_UPPER_SHARED_DEVPTR(bsize, wfsize, dim, false);  \
            }                                                                 \
        }                                                                     \
    }

#define LAUNCH_BSRSV_LOWER_SHARED_HOSTPTR(bsize, wfsize, dim, arch)                    \
    hipLaunchKernelGGL((bsrsv_lower_shared_host_pointer<T, bsize, wfsize, dim, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                            \
                       dim3(bsize),                                                    \
                       0,                                                              \
                       stream,                                                         \
                       mb,                                                             \
                       *alpha,                                                         \
                       local_bsr_row_ptr,                                              \
                       local_bsr_col_ind,                                              \
                       local_bsr_val,                                                  \
                       bsr_dim,                                                        \
                       x,                                                              \
                       y,                                                              \
                       done_array,                                                     \
                       bsrsv->row_map,                                                 \
                       info->zero_pivot,                                               \
                       descr->base,                                                    \
                       descr->diag_type,                                               \
                       dir)

#define LAUNCH_BSRSV_LOWER_SHARED_DEVPTR(bsize, wfsize, dim, arch)                       \
    hipLaunchKernelGGL((bsrsv_lower_shared_device_pointer<T, bsize, wfsize, dim, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                              \
                       dim3(bsize),                                                      \
                       0,                                                                \
                       stream,                                                           \
                       mb,                                                               \
                       alpha,                                                            \
                       local_bsr_row_ptr,                                                \
                       local_bsr_col_ind,                                                \
                       local_bsr_val,                                                    \
                       bsr_dim,                                                          \
                       x,                                                                \
                       y,                                                                \
                       done_array,                                                       \
                       bsrsv->row_map,                                                   \
                       info->zero_pivot,                                                 \
                       descr->base,                                                      \
                       descr->diag_type,                                                 \
                       dir)

#define LAUNCH_BSRSV_UPPER_SHARED_HOSTPTR(bsize, wfsize, dim, arch)                    \
    hipLaunchKernelGGL((bsrsv_upper_shared_host_pointer<T, bsize, wfsize, dim, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                            \
                       dim3(bsize),                                                    \
                       0,                                                              \
                       stream,                                                         \
                       mb,                                                             \
                       *alpha,                                                         \
                       local_bsr_row_ptr,                                              \
                       local_bsr_col_ind,                                              \
                       local_bsr_val,                                                  \
                       bsr_dim,                                                        \
                       x,                                                              \
                       y,                                                              \
                       done_array,                                                     \
                       bsrsv->row_map,                                                 \
                       info->zero_pivot,                                               \
                       descr->base,                                                    \
                       descr->diag_type,                                               \
                       dir)

#define LAUNCH_BSRSV_UPPER_SHARED_DEVPTR(bsize, wfsize, dim, arch)                       \
    hipLaunchKernelGGL((bsrsv_upper_shared_device_pointer<T, bsize, wfsize, dim, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                              \
                       dim3(bsize),                                                      \
                       0,                                                                \
                       stream,                                                           \
                       mb,                                                               \
                       alpha,                                                            \
                       local_bsr_row_ptr,                                                \
                       local_bsr_col_ind,                                                \
                       local_bsr_val,                                                    \
                       bsr_dim,                                                          \
                       x,                                                                \
                       y,                                                                \
                       done_array,                                                       \
                       bsrsv->row_map,                                                   \
                       info->zero_pivot,                                                 \
                       descr->base,                                                      \
                       descr->diag_type,                                                 \
                       dir)

#define LAUNCH_BSRSV_GENERAL(fill, ptr, bsize, wfsize, arch, asic)        \
    if(fill == rocsparse_fill_mode_lower)                                 \
    {                                                                     \
        if(ptr == rocsparse_pointer_mode_host)                            \
        {                                                                 \
            if(arch == 908 && asic < 2)                                   \
            {                                                             \
                LAUNCH_BSRSV_LOWER_GENERAL_HOSTPTR(bsize, wfsize, true);  \
            }                                                             \
            else                                                          \
            {                                                             \
                LAUNCH_BSRSV_LOWER_GENERAL_HOSTPTR(bsize, wfsize, false); \
            }                                                             \
        }                                                                 \
        else                                                              \
        {                                                                 \
            if(arch == 908 && asic < 2)                                   \
            {                                                             \
                LAUNCH_BSRSV_LOWER_GENERAL_DEVPTR(bsize, wfsize, true);   \
            }                                                             \
            else                                                          \
            {                                                             \
                LAUNCH_BSRSV_LOWER_GENERAL_DEVPTR(bsize, wfsize, false);  \
            }                                                             \
        }                                                                 \
    }                                                                     \
    else if(ptr == rocsparse_pointer_mode_host)                           \
    {                                                                     \
        if(arch == 908 && asic < 2)                                       \
        {                                                                 \
            LAUNCH_BSRSV_UPPER_GENERAL_HOSTPTR(bsize, wfsize, true);      \
        }                                                                 \
        else                                                              \
        {                                                                 \
            LAUNCH_BSRSV_UPPER_GENERAL_HOSTPTR(bsize, wfsize, false);     \
        }                                                                 \
    }                                                                     \
    else                                                                  \
    {                                                                     \
        if(arch == 908 && asic < 2)                                       \
        {                                                                 \
            LAUNCH_BSRSV_UPPER_GENERAL_DEVPTR(bsize, wfsize, true);       \
        }                                                                 \
        else                                                              \
        {                                                                 \
            LAUNCH_BSRSV_UPPER_GENERAL_DEVPTR(bsize, wfsize, false);      \
        }                                                                 \
    }

#define LAUNCH_BSRSV_LOWER_GENERAL_HOSTPTR(bsize, wfsize, arch)                    \
    hipLaunchKernelGGL((bsrsv_lower_general_host_pointer<T, bsize, wfsize, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                        \
                       dim3(bsize),                                                \
                       0,                                                          \
                       stream,                                                     \
                       mb,                                                         \
                       *alpha,                                                     \
                       local_bsr_row_ptr,                                          \
                       local_bsr_col_ind,                                          \
                       local_bsr_val,                                              \
                       bsr_dim,                                                    \
                       x,                                                          \
                       y,                                                          \
                       done_array,                                                 \
                       bsrsv->row_map,                                             \
                       info->zero_pivot,                                           \
                       descr->base,                                                \
                       descr->diag_type,                                           \
                       dir)

#define LAUNCH_BSRSV_LOWER_GENERAL_DEVPTR(bsize, wfsize, arch)                       \
    hipLaunchKernelGGL((bsrsv_lower_general_device_pointer<T, bsize, wfsize, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                          \
                       dim3(bsize),                                                  \
                       0,                                                            \
                       stream,                                                       \
                       mb,                                                           \
                       alpha,                                                        \
                       local_bsr_row_ptr,                                            \
                       local_bsr_col_ind,                                            \
                       local_bsr_val,                                                \
                       bsr_dim,                                                      \
                       x,                                                            \
                       y,                                                            \
                       done_array,                                                   \
                       bsrsv->row_map,                                               \
                       info->zero_pivot,                                             \
                       descr->base,                                                  \
                       descr->diag_type,                                             \
                       dir)

#define LAUNCH_BSRSV_UPPER_GENERAL_HOSTPTR(bsize, wfsize, arch)                    \
    hipLaunchKernelGGL((bsrsv_upper_general_host_pointer<T, bsize, wfsize, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                        \
                       dim3(bsize),                                                \
                       0,                                                          \
                       stream,                                                     \
                       mb,                                                         \
                       *alpha,                                                     \
                       local_bsr_row_ptr,                                          \
                       local_bsr_col_ind,                                          \
                       local_bsr_val,                                              \
                       bsr_dim,                                                    \
                       x,                                                          \
                       y,                                                          \
                       done_array,                                                 \
                       bsrsv->row_map,                                             \
                       info->zero_pivot,                                           \
                       descr->base,                                                \
                       descr->diag_type,                                           \
                       dir)

#define LAUNCH_BSRSV_UPPER_GENERAL_DEVPTR(bsize, wfsize, arch)                       \
    hipLaunchKernelGGL((bsrsv_upper_general_device_pointer<T, bsize, wfsize, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),                          \
                       dim3(bsize),                                                  \
                       0,                                                            \
                       stream,                                                       \
                       mb,                                                           \
                       alpha,                                                        \
                       local_bsr_row_ptr,                                            \
                       local_bsr_col_ind,                                            \
                       local_bsr_val,                                                \
                       bsr_dim,                                                      \
                       x,                                                            \
                       y,                                                            \
                       done_array,                                                   \
                       bsrsv->row_map,                                               \
                       info->zero_pivot,                                             \
                       descr->base,                                                  \
                       descr->diag_type,                                             \
                       dir)

template <typename T>
rocsparse_status rocsparse_bsrsv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  bsr_val,
                                                   const rocsparse_int*      bsr_row_ptr,
                                                   const rocsparse_int*      bsr_col_ind,
                                                   rocsparse_int             bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_analysis_policy analysis,
                                                   rocsparse_solve_policy    solve,
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
              replaceX<T>("rocsparse_Xbsrsv_analysis"),
              dir,
              trans,
              mb,
              nnzb,
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              bsr_dim,
              (const void*&)info,
              solve,
              analysis,
              (const void*&)temp_buffer);

    // Check operation type
    if(trans != rocsparse_operation_none && trans != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
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
    else if(bsr_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nnzb == 0 || bsr_dim == 0)
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

    // Switch between lower and upper triangular analysis
    if(descr->fill_mode == rocsparse_fill_mode_upper)
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed upper part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If bsrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->bsrsv_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->bsrsvt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }

            //            // Check for other upper analysis meta data that could be used
            //            if(trans == rocsparse_operation_none && info->bsrsm_upper_info != nullptr)
            //            {
            //                // bsrsm meta data
            //                info->bsrsv_upper_info = info->bsrsm_upper_info;
            //                return rocsparse_status_success;
            //            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear bsrsv
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans == rocsparse_operation_none)
                                                                 ? info->bsrsv_upper_info
                                                                 : info->bsrsvt_upper_info));

        // Create bsrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans == rocsparse_operation_none)
                                                                ? &info->bsrsv_upper_info
                                                                : &info->bsrsvt_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(
            handle,
            trans,
            mb,
            nnzb,
            descr,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind,
            (trans == rocsparse_operation_none) ? info->bsrsv_upper_info : info->bsrsvt_upper_info,
            &info->zero_pivot,
            temp_buffer));
    }
    else
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed lower part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If bsrsv meta data is already available, do nothing
            if(trans == rocsparse_operation_none && info->bsrsv_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_transpose && info->bsrsvt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data that could be used
            if(trans == rocsparse_operation_none && info->bsric0_info != nullptr)
            {
                // bsric0 meta data
                info->bsrsv_lower_info = info->bsric0_info;
                return rocsparse_status_success;
            }
            else if(trans == rocsparse_operation_none && info->bsrilu0_info != nullptr)
            {
                // bsrilu0 meta data
                info->bsrsv_lower_info = info->bsrilu0_info;
                return rocsparse_status_success;
            }
            // else if(trans == rocsparse_operation_none && info->bsrsm_lower_info != nullptr)
            // {
            //     // bsrsm meta data
            //     info->bsrsv_lower_info = info->bsrsm_lower_info;
            //     return rocsparse_status_success;
            // }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used.

        // Clear bsrsv
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info((trans == rocsparse_operation_none)
                                                                 ? info->bsrsv_lower_info
                                                                 : info->bsrsvt_lower_info));

        // Create bsrsv info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info((trans == rocsparse_operation_none)
                                                                ? &info->bsrsv_lower_info
                                                                : &info->bsrsvt_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(
            handle,
            trans,
            mb,
            nnzb,
            descr,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind,
            (trans == rocsparse_operation_none) ? info->bsrsv_lower_info : info->bsrsvt_lower_info,
            &info->zero_pivot,
            temp_buffer));
    }

    return rocsparse_status_success;
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, rocsparse_int BSRDIM, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_lower_shared_host_pointer(rocsparse_int mb,
                                         T             alpha,
                                         const rocsparse_int* __restrict__ bsr_row_ptr,
                                         const rocsparse_int* __restrict__ bsr_col_ind,
                                         const T* __restrict__ bsr_val,
                                         rocsparse_int bsr_dim,
                                         const T* __restrict__ x,
                                         T* __restrict__ y,
                                         int* __restrict__ done_array,
                                         rocsparse_int* __restrict__ map,
                                         rocsparse_int* __restrict__ zero_pivot,
                                         rocsparse_index_base idx_base,
                                         rocsparse_diag_type  diag_type,
                                         rocsparse_direction  dir)
{
    bsrsv_lower_shared_device<T, BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                   alpha,
                                                                   bsr_row_ptr,
                                                                   bsr_col_ind,
                                                                   bsr_val,
                                                                   bsr_dim,
                                                                   x,
                                                                   y,
                                                                   done_array,
                                                                   map,
                                                                   zero_pivot,
                                                                   idx_base,
                                                                   diag_type,
                                                                   dir);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, rocsparse_int BSRDIM, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_lower_shared_device_pointer(rocsparse_int mb,
                                           const T* __restrict__ alpha,
                                           const rocsparse_int* __restrict__ bsr_row_ptr,
                                           const rocsparse_int* __restrict__ bsr_col_ind,
                                           const T* __restrict__ bsr_val,
                                           rocsparse_int bsr_dim,
                                           const T* __restrict__ x,
                                           T* __restrict__ y,
                                           int* __restrict__ done_array,
                                           rocsparse_int* __restrict__ map,
                                           rocsparse_int* __restrict__ zero_pivot,
                                           rocsparse_index_base idx_base,
                                           rocsparse_diag_type  diag_type,
                                           rocsparse_direction  dir)
{
    bsrsv_lower_shared_device<T, BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                   *alpha,
                                                                   bsr_row_ptr,
                                                                   bsr_col_ind,
                                                                   bsr_val,
                                                                   bsr_dim,
                                                                   x,
                                                                   y,
                                                                   done_array,
                                                                   map,
                                                                   zero_pivot,
                                                                   idx_base,
                                                                   diag_type,
                                                                   dir);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, rocsparse_int BSRDIM, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_upper_shared_host_pointer(rocsparse_int mb,
                                         T             alpha,
                                         const rocsparse_int* __restrict__ bsr_row_ptr,
                                         const rocsparse_int* __restrict__ bsr_col_ind,
                                         const T* __restrict__ bsr_val,
                                         rocsparse_int bsr_dim,
                                         const T* __restrict__ x,
                                         T* __restrict__ y,
                                         int* __restrict__ done_array,
                                         rocsparse_int* __restrict__ map,
                                         rocsparse_int* __restrict__ zero_pivot,
                                         rocsparse_index_base idx_base,
                                         rocsparse_diag_type  diag_type,
                                         rocsparse_direction  dir)
{
    bsrsv_upper_shared_device<T, BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                   alpha,
                                                                   bsr_row_ptr,
                                                                   bsr_col_ind,
                                                                   bsr_val,
                                                                   bsr_dim,
                                                                   x,
                                                                   y,
                                                                   done_array,
                                                                   map,
                                                                   zero_pivot,
                                                                   idx_base,
                                                                   diag_type,
                                                                   dir);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, rocsparse_int BSRDIM, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_upper_shared_device_pointer(rocsparse_int mb,
                                           const T* __restrict__ alpha,
                                           const rocsparse_int* __restrict__ bsr_row_ptr,
                                           const rocsparse_int* __restrict__ bsr_col_ind,
                                           const T* __restrict__ bsr_val,
                                           rocsparse_int bsr_dim,
                                           const T* __restrict__ x,
                                           T* __restrict__ y,
                                           int* __restrict__ done_array,
                                           rocsparse_int* __restrict__ map,
                                           rocsparse_int* __restrict__ zero_pivot,
                                           rocsparse_index_base idx_base,
                                           rocsparse_diag_type  diag_type,
                                           rocsparse_direction  dir)
{
    bsrsv_upper_shared_device<T, BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                   *alpha,
                                                                   bsr_row_ptr,
                                                                   bsr_col_ind,
                                                                   bsr_val,
                                                                   bsr_dim,
                                                                   x,
                                                                   y,
                                                                   done_array,
                                                                   map,
                                                                   zero_pivot,
                                                                   idx_base,
                                                                   diag_type,
                                                                   dir);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_lower_general_host_pointer(rocsparse_int mb,
                                          T             alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          rocsparse_int bsr_dim,
                                          const T* __restrict__ x,
                                          T* __restrict__ y,
                                          int* __restrict__ done_array,
                                          rocsparse_int* __restrict__ map,
                                          rocsparse_int* __restrict__ zero_pivot,
                                          rocsparse_index_base idx_base,
                                          rocsparse_diag_type  diag_type,
                                          rocsparse_direction  dir)
{
    bsrsv_lower_general_device<T, BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                            alpha,
                                                            bsr_row_ptr,
                                                            bsr_col_ind,
                                                            bsr_val,
                                                            bsr_dim,
                                                            x,
                                                            y,
                                                            done_array,
                                                            map,
                                                            zero_pivot,
                                                            idx_base,
                                                            diag_type,
                                                            dir);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_lower_general_device_pointer(rocsparse_int mb,
                                            const T* __restrict__ alpha,
                                            const rocsparse_int* __restrict__ bsr_row_ptr,
                                            const rocsparse_int* __restrict__ bsr_col_ind,
                                            const T* __restrict__ bsr_val,
                                            rocsparse_int bsr_dim,
                                            const T* __restrict__ x,
                                            T* __restrict__ y,
                                            int* __restrict__ done_array,
                                            rocsparse_int* __restrict__ map,
                                            rocsparse_int* __restrict__ zero_pivot,
                                            rocsparse_index_base idx_base,
                                            rocsparse_diag_type  diag_type,
                                            rocsparse_direction  dir)
{
    bsrsv_lower_general_device<T, BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                            *alpha,
                                                            bsr_row_ptr,
                                                            bsr_col_ind,
                                                            bsr_val,
                                                            bsr_dim,
                                                            x,
                                                            y,
                                                            done_array,
                                                            map,
                                                            zero_pivot,
                                                            idx_base,
                                                            diag_type,
                                                            dir);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_upper_general_host_pointer(rocsparse_int mb,
                                          T             alpha,
                                          const rocsparse_int* __restrict__ bsr_row_ptr,
                                          const rocsparse_int* __restrict__ bsr_col_ind,
                                          const T* __restrict__ bsr_val,
                                          rocsparse_int bsr_dim,
                                          const T* __restrict__ x,
                                          T* __restrict__ y,
                                          int* __restrict__ done_array,
                                          rocsparse_int* __restrict__ map,
                                          rocsparse_int* __restrict__ zero_pivot,
                                          rocsparse_index_base idx_base,
                                          rocsparse_diag_type  diag_type,
                                          rocsparse_direction  dir)
{
    bsrsv_upper_general_device<T, BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                            alpha,
                                                            bsr_row_ptr,
                                                            bsr_col_ind,
                                                            bsr_val,
                                                            bsr_dim,
                                                            x,
                                                            y,
                                                            done_array,
                                                            map,
                                                            zero_pivot,
                                                            idx_base,
                                                            diag_type,
                                                            dir);
}

template <typename T, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrsv_upper_general_device_pointer(rocsparse_int mb,
                                            const T* __restrict__ alpha,
                                            const rocsparse_int* __restrict__ bsr_row_ptr,
                                            const rocsparse_int* __restrict__ bsr_col_ind,
                                            const T* __restrict__ bsr_val,
                                            rocsparse_int bsr_dim,
                                            const T* __restrict__ x,
                                            T* __restrict__ y,
                                            int* __restrict__ done_array,
                                            rocsparse_int* __restrict__ map,
                                            rocsparse_int* __restrict__ zero_pivot,
                                            rocsparse_index_base idx_base,
                                            rocsparse_diag_type  diag_type,
                                            rocsparse_direction  dir)
{
    bsrsv_upper_general_device<T, BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                            *alpha,
                                                            bsr_row_ptr,
                                                            bsr_col_ind,
                                                            bsr_val,
                                                            bsr_dim,
                                                            x,
                                                            y,
                                                            done_array,
                                                            map,
                                                            zero_pivot,
                                                            idx_base,
                                                            diag_type,
                                                            dir);
}

template <typename T>
rocsparse_status rocsparse_bsrsv_solve_template(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             bsr_dim,
                                                rocsparse_mat_info        info,
                                                const T*                  x,
                                                T*                        y,
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
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xbsrsv"),
                  dir,
                  trans,
                  mb,
                  nnzb,
                  *alpha,
                  (const void*&)descr,
                  (const void*&)bsr_val,
                  (const void*&)bsr_row_ptr,
                  (const void*&)bsr_col_ind,
                  bsr_dim,
                  (const void*&)info,
                  (const void*&)x,
                  (const void*&)y,
                  policy,
                  (const void*&)temp_buffer);

        log_bench(handle,
                  "./rocsparse-bench -f bsrsv -r",
                  replaceX<T>("X"),
                  "--mtx <matrix.mtx> ",
                  "--blockdim",
                  bsr_dim,
                  "--alpha",
                  *alpha);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xbsrsv"),
                  dir,
                  trans,
                  mb,
                  nnzb,
                  (const void*&)alpha,
                  (const void*&)descr,
                  (const void*&)bsr_val,
                  (const void*&)bsr_row_ptr,
                  (const void*&)bsr_col_ind,
                  bsr_dim,
                  (const void*&)info,
                  (const void*&)x,
                  (const void*&)y,
                  policy,
                  (const void*&)temp_buffer);
    }

    // Check operation type
    if(trans != rocsparse_operation_none && trans != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
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
    else if(bsr_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nnzb == 0 || bsr_dim == 0)
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
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    ptr += 256;

    // done array
    int* done_array = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * ((mb - 1) / 256 + 1) * 256;

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * mb, stream));

    rocsparse_trm_info bsrsv
        = (descr->fill_mode == rocsparse_fill_mode_upper)
              ? ((trans == rocsparse_operation_none) ? info->bsrsv_upper_info
                                                     : info->bsrsvt_upper_info)
              : ((trans == rocsparse_operation_none) ? info->bsrsv_lower_info
                                                     : info->bsrsvt_lower_info);

    if(bsrsv == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // If diag type is unit, re-initialize zero pivot to remove structural zeros
    if(descr->diag_type == rocsparse_diag_type_unit)
    {
        rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            info->zero_pivot, &max, sizeof(rocsparse_int), hipMemcpyHostToDevice, stream));

        // Wait for device transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }

    // Pointers to differentiate between transpose mode
    const rocsparse_int* local_bsr_row_ptr = bsr_row_ptr;
    const rocsparse_int* local_bsr_col_ind = bsr_col_ind;
    const T*             local_bsr_val     = bsr_val;

    rocsparse_fill_mode fill_mode = descr->fill_mode;

    // When computing transposed triangular solve, we first need to update the
    // transposed matrix values
    if(trans == rocsparse_operation_transpose)
    {
        T* bsrt_val = reinterpret_cast<T*>(ptr);

        // Gather transposed values
        LAUNCH_BSRSV_GTHR(256, 64, bsr_dim);

        local_bsr_row_ptr = bsrsv->trmt_row_ptr;
        local_bsr_col_ind = bsrsv->trmt_col_ind;
        local_bsr_val     = bsrt_val;

        fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                             : rocsparse_fill_mode_lower;
    }

    // Determine gcnArch and ASIC revision
    int gcnArch = handle->properties.gcnArch;
    int asicRev = handle->asic_rev;

    if(handle->wavefront_size == 64)
    {
        if(bsr_dim <= 8)
        {
            // Launch shared memory based kernel for small BSR block dimensions
            LAUNCH_BSRSV_SHARED(fill_mode, handle->pointer_mode, 128, 64, 8, gcnArch, asicRev);
        }
        else if(bsr_dim <= 16)
        {
            // Launch shared memory based kernel for small BSR block dimensions
            LAUNCH_BSRSV_SHARED(fill_mode, handle->pointer_mode, 128, 64, 16, gcnArch, asicRev);
        }
        else if(bsr_dim <= 32)
        {
            // Launch shared memory based kernel for small BSR block dimensions
            LAUNCH_BSRSV_SHARED(fill_mode, handle->pointer_mode, 128, 64, 32, gcnArch, asicRev);
        }
        else
        {
            // Launch general algorithm for large BSR block dimensions (> 32x32)
            LAUNCH_BSRSV_GENERAL(fill_mode, handle->pointer_mode, 128, 64, gcnArch, asicRev);
        }
    }
    else
    {
        // Launch general algorithm
        LAUNCH_BSRSV_GENERAL(fill_mode, handle->pointer_mode, 128, 32, gcnArch, asicRev);
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_BSRSV_HPP

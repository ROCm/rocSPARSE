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
#ifndef ROCSPARSE_BSRILU0_HPP
#define ROCSPARSE_BSRILU0_HPP

#include <type_traits>

#include "../level2/rocsparse_csrsv.hpp"
#include "bsrilu0_device.h"
#include "definitions.h"
#include "rocsparse.h"
#include "utility.h"

#define LAUNCH_BSRILU28()                                                            \
    if(handle->pointer_mode == rocsparse_pointer_mode_device)                        \
    {                                                                                \
        hipLaunchKernelGGL((bsrilu0_2_8_device_pointer<T, U, 64, 64, 8>),            \
                           dim3(mb),                                                 \
                           dim3(8, 8),                                               \
                           0,                                                        \
                           handle->stream,                                           \
                           dir,                                                      \
                           mb,                                                       \
                           bsr_row_ptr,                                              \
                           bsr_col_ind,                                              \
                           bsr_val,                                                  \
                           info->bsrilu0_info->trm_diag_ind,                         \
                           block_dim,                                                \
                           done_array,                                               \
                           info->bsrilu0_info->row_map,                              \
                           info->zero_pivot,                                         \
                           base,                                                     \
                           info->boost_enable,                                       \
                           reinterpret_cast<const U*>(info->boost_tol),              \
                           reinterpret_cast<const T*>(info->boost_val));             \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        hipLaunchKernelGGL(                                                          \
            (bsrilu0_2_8_host_pointer<T, U, 64, 64, 8>),                             \
            dim3(mb),                                                                \
            dim3(8, 8),                                                              \
            0,                                                                       \
            handle->stream,                                                          \
            dir,                                                                     \
            mb,                                                                      \
            bsr_row_ptr,                                                             \
            bsr_col_ind,                                                             \
            bsr_val,                                                                 \
            info->bsrilu0_info->trm_diag_ind,                                        \
            block_dim,                                                               \
            done_array,                                                              \
            info->bsrilu0_info->row_map,                                             \
            info->zero_pivot,                                                        \
            base,                                                                    \
            info->boost_enable,                                                      \
            (info->boost_enable != 0) ? *reinterpret_cast<const U*>(info->boost_tol) \
                                      : static_cast<U>(0),                           \
            (info->boost_enable != 0) ? *reinterpret_cast<const T*>(info->boost_val) \
                                      : static_cast<T>(0));                          \
    }

#define LAUNCH_BSRILU932(dim)                                                        \
    if(handle->pointer_mode == rocsparse_pointer_mode_device)                        \
    {                                                                                \
        hipLaunchKernelGGL((bsrilu0_9_32_device_pointer<T, U, 64, 64, dim>),         \
                           dim3(mb),                                                 \
                           dim3(dim, 64 / dim),                                      \
                           0,                                                        \
                           handle->stream,                                           \
                           dir,                                                      \
                           mb,                                                       \
                           bsr_row_ptr,                                              \
                           bsr_col_ind,                                              \
                           bsr_val,                                                  \
                           info->bsrilu0_info->trm_diag_ind,                         \
                           block_dim,                                                \
                           done_array,                                               \
                           info->bsrilu0_info->row_map,                              \
                           info->zero_pivot,                                         \
                           base,                                                     \
                           info->boost_enable,                                       \
                           reinterpret_cast<const U*>(info->boost_tol),              \
                           reinterpret_cast<const T*>(info->boost_val));             \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        hipLaunchKernelGGL(                                                          \
            (bsrilu0_9_32_host_pointer<T, U, 64, 64, dim>),                          \
            dim3(mb),                                                                \
            dim3(dim, 64 / dim),                                                     \
            0,                                                                       \
            handle->stream,                                                          \
            dir,                                                                     \
            mb,                                                                      \
            bsr_row_ptr,                                                             \
            bsr_col_ind,                                                             \
            bsr_val,                                                                 \
            info->bsrilu0_info->trm_diag_ind,                                        \
            block_dim,                                                               \
            done_array,                                                              \
            info->bsrilu0_info->row_map,                                             \
            info->zero_pivot,                                                        \
            base,                                                                    \
            info->boost_enable,                                                      \
            (info->boost_enable != 0) ? *reinterpret_cast<const U*>(info->boost_tol) \
                                      : static_cast<U>(0),                           \
            (info->boost_enable != 0) ? *reinterpret_cast<const T*>(info->boost_val) \
                                      : static_cast<T>(0));                          \
    }

#define LAUNCH_BSRILU3364()                                                          \
    if(handle->pointer_mode == rocsparse_pointer_mode_device)                        \
    {                                                                                \
        hipLaunchKernelGGL((bsrilu0_33_64_device_pointer<T, U, 64, 64, 64>),         \
                           dim3(mb),                                                 \
                           dim3(64),                                                 \
                           0,                                                        \
                           handle->stream,                                           \
                           dir,                                                      \
                           mb,                                                       \
                           bsr_row_ptr,                                              \
                           bsr_col_ind,                                              \
                           bsr_val,                                                  \
                           info->bsrilu0_info->trm_diag_ind,                         \
                           block_dim,                                                \
                           done_array,                                               \
                           info->bsrilu0_info->row_map,                              \
                           info->zero_pivot,                                         \
                           base,                                                     \
                           info->boost_enable,                                       \
                           reinterpret_cast<const U*>(info->boost_tol),              \
                           reinterpret_cast<const T*>(info->boost_val));             \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        hipLaunchKernelGGL(                                                          \
            (bsrilu0_33_64_host_pointer<T, U, 64, 64, 64>),                          \
            dim3(mb),                                                                \
            dim3(64),                                                                \
            0,                                                                       \
            handle->stream,                                                          \
            dir,                                                                     \
            mb,                                                                      \
            bsr_row_ptr,                                                             \
            bsr_col_ind,                                                             \
            bsr_val,                                                                 \
            info->bsrilu0_info->trm_diag_ind,                                        \
            block_dim,                                                               \
            done_array,                                                              \
            info->bsrilu0_info->row_map,                                             \
            info->zero_pivot,                                                        \
            base,                                                                    \
            info->boost_enable,                                                      \
            (info->boost_enable != 0) ? *reinterpret_cast<const U*>(info->boost_tol) \
                                      : static_cast<U>(0),                           \
            (info->boost_enable != 0) ? *reinterpret_cast<const T*>(info->boost_val) \
                                      : static_cast<T>(0));                          \
    }

#define LAUNCH_BSRILU65inf(sleep)                                                    \
    if(handle->pointer_mode == rocsparse_pointer_mode_device)                        \
    {                                                                                \
        hipLaunchKernelGGL((bsrilu0_general_device_pointer<T, U, 128, 64, sleep>),   \
                           dim3((64 * mb - 1) / 128 + 1),                            \
                           dim3(128),                                                \
                           0,                                                        \
                           handle->stream,                                           \
                           dir,                                                      \
                           mb,                                                       \
                           bsr_row_ptr,                                              \
                           bsr_col_ind,                                              \
                           bsr_val,                                                  \
                           info->bsrilu0_info->trm_diag_ind,                         \
                           block_dim,                                                \
                           done_array,                                               \
                           info->bsrilu0_info->row_map,                              \
                           info->zero_pivot,                                         \
                           base,                                                     \
                           info->boost_enable,                                       \
                           reinterpret_cast<const U*>(info->boost_tol),              \
                           reinterpret_cast<const T*>(info->boost_val));             \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        hipLaunchKernelGGL(                                                          \
            (bsrilu0_general_host_pointer<T, U, 128, 64, false>),                    \
            dim3((64 * mb - 1) / 128 + 1),                                           \
            dim3(128),                                                               \
            0,                                                                       \
            handle->stream,                                                          \
            dir,                                                                     \
            mb,                                                                      \
            bsr_row_ptr,                                                             \
            bsr_col_ind,                                                             \
            bsr_val,                                                                 \
            info->bsrilu0_info->trm_diag_ind,                                        \
            block_dim,                                                               \
            done_array,                                                              \
            info->bsrilu0_info->row_map,                                             \
            info->zero_pivot,                                                        \
            base,                                                                    \
            info->boost_enable,                                                      \
            (info->boost_enable != 0) ? *reinterpret_cast<const U*>(info->boost_tol) \
                                      : static_cast<U>(0),                           \
            (info->boost_enable != 0) ? *reinterpret_cast<const T*>(info->boost_val) \
                                      : static_cast<T>(0));                          \
    }

template <typename T, typename U>
rocsparse_status rocsparse_bsrilu0_numeric_boost_template(rocsparse_handle   handle,
                                                          rocsparse_mat_info info,
                                                          int                enable_boost,
                                                          const U*           boost_tol,
                                                          const T*           boost_val)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrilu0_numeric_boost"),
              (const void*&)info,
              enable_boost,
              (const void*&)boost_tol,
              (const void*&)boost_val);

    // Reset boost
    info->boost_enable        = 0;
    info->use_double_prec_tol = 0;

    // Numeric boost
    if(enable_boost)
    {
        // Check pointer arguments
        if(boost_tol == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        else if(boost_val == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }

        info->boost_enable        = enable_boost;
        info->use_double_prec_tol = std::is_same<U, double>();
        info->boost_tol           = reinterpret_cast<const void*>(boost_tol);
        info->boost_val           = reinterpret_cast<const void*>(boost_val);
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_bsrilu0_analysis_template(rocsparse_handle          handle,
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
              replaceX<T>("rocsparse_Xbsrilu0_analysis"),
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

        // If bsrilu0 meta data is already available, do nothing
        if(info->bsrilu0_info != nullptr)
        {
            return rocsparse_status_success;
        }

        // Check for other lower analysis meta data
        if(info->bsric0_info != nullptr)
        {
            // bsric0 meta data
            info->bsric0_info = info->bsrilu0_info;
            return rocsparse_status_success;
        }

        if(info->bsrsv_lower_info != nullptr)
        {
            // bsrsv meta data
            info->bsrilu0_info = info->bsrsv_lower_info;
            return rocsparse_status_success;
        }
    }

    // User is explicitly asking to force a re-analysis, or no valid data has been
    // found to be re-used.

    // Clear bsrilu0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrilu0_info));

    // Create bsrilu0 info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&info->bsrilu0_info));

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_analysis(handle,
                                                     rocsparse_operation_none,
                                                     mb,
                                                     nnzb,
                                                     descr,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     info->bsrilu0_info,
                                                     &info->zero_pivot,
                                                     temp_buffer));

    return rocsparse_status_success;
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int BSRDIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_2_8_device_pointer(rocsparse_direction dir,
                                    rocsparse_int       mb,
                                    const rocsparse_int* __restrict__ bsr_row_ptr,
                                    const rocsparse_int* __restrict__ bsr_col_ind,
                                    T* __restrict__ bsr_val,
                                    const rocsparse_int* __restrict__ bsr_diag_ind,
                                    rocsparse_int bsr_dim,
                                    int* __restrict__ done_array,
                                    const rocsparse_int* __restrict__ map,
                                    rocsparse_int* __restrict__ zero_pivot,
                                    rocsparse_index_base idx_base,
                                    int                  enable_boost,
                                    const U* __restrict__ boost_tol,
                                    const T* __restrict__ boost_val)
{
    bsrilu0_2_8_device<T, U, BLOCKSIZE, WFSIZE, BSRDIM>(
        dir,
        mb,
        bsr_row_ptr,
        bsr_col_ind,
        bsr_val,
        bsr_diag_ind,
        bsr_dim,
        done_array,
        map,
        zero_pivot,
        idx_base,
        enable_boost,
        enable_boost ? *boost_tol : static_cast<U>(0),
        enable_boost ? *boost_val : static_cast<T>(0));
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int BSRDIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_2_8_host_pointer(rocsparse_direction  dir,
                                  rocsparse_int        mb,
                                  const rocsparse_int* bsr_row_ptr,
                                  const rocsparse_int* bsr_col_ind,
                                  T*                   bsr_val,
                                  const rocsparse_int* bsr_diag_ind,
                                  rocsparse_int        bsr_dim,
                                  int*                 done_array,
                                  const rocsparse_int* map,
                                  rocsparse_int*       zero_pivot,
                                  rocsparse_index_base idx_base,
                                  int                  enable_boost,
                                  U                    boost_tol,
                                  T                    boost_val)
{
    bsrilu0_2_8_device<T, U, BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                        mb,
                                                        bsr_row_ptr,
                                                        bsr_col_ind,
                                                        bsr_val,
                                                        bsr_diag_ind,
                                                        bsr_dim,
                                                        done_array,
                                                        map,
                                                        zero_pivot,
                                                        idx_base,
                                                        enable_boost,
                                                        boost_tol,
                                                        boost_val);
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int BSRDIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_9_32_device_pointer(rocsparse_direction dir,
                                     rocsparse_int       mb,
                                     const rocsparse_int* __restrict__ bsr_row_ptr,
                                     const rocsparse_int* __restrict__ bsr_col_ind,
                                     T* __restrict__ bsr_val,
                                     const rocsparse_int* __restrict__ bsr_diag_ind,
                                     rocsparse_int bsr_dim,
                                     int* __restrict__ done_array,
                                     const rocsparse_int* __restrict__ map,
                                     rocsparse_int* __restrict__ zero_pivot,
                                     rocsparse_index_base idx_base,
                                     int                  enable_boost,
                                     const U* __restrict__ boost_tol,
                                     const T* __restrict__ boost_val)
{
    bsrilu0_9_32_device<T, U, BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                         mb,
                                                         bsr_row_ptr,
                                                         bsr_col_ind,
                                                         bsr_val,
                                                         bsr_diag_ind,
                                                         bsr_dim,
                                                         done_array,
                                                         map,
                                                         zero_pivot,
                                                         idx_base,
                                                         enable_boost,
                                                         enable_boost ? *boost_tol : 0.0,
                                                         enable_boost ? *boost_val
                                                                      : static_cast<T>(0));
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int BSRDIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_9_32_host_pointer(rocsparse_direction  dir,
                                   rocsparse_int        mb,
                                   const rocsparse_int* bsr_row_ptr,
                                   const rocsparse_int* bsr_col_ind,
                                   T*                   bsr_val,
                                   const rocsparse_int* bsr_diag_ind,
                                   rocsparse_int        bsr_dim,
                                   int*                 done_array,
                                   const rocsparse_int* map,
                                   rocsparse_int*       zero_pivot,
                                   rocsparse_index_base idx_base,
                                   int                  enable_boost,
                                   U                    boost_tol,
                                   T                    boost_val)
{
    bsrilu0_9_32_device<T, U, BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                         mb,
                                                         bsr_row_ptr,
                                                         bsr_col_ind,
                                                         bsr_val,
                                                         bsr_diag_ind,
                                                         bsr_dim,
                                                         done_array,
                                                         map,
                                                         zero_pivot,
                                                         idx_base,
                                                         enable_boost,
                                                         boost_tol,
                                                         boost_val);
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int BSRDIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_33_64_device_pointer(rocsparse_direction dir,
                                      rocsparse_int       mb,
                                      const rocsparse_int* __restrict__ bsr_row_ptr,
                                      const rocsparse_int* __restrict__ bsr_col_ind,
                                      T* __restrict__ bsr_val,
                                      const rocsparse_int* __restrict__ bsr_diag_ind,
                                      rocsparse_int bsr_dim,
                                      int* __restrict__ done_array,
                                      const rocsparse_int* __restrict__ map,
                                      rocsparse_int* __restrict__ zero_pivot,
                                      rocsparse_index_base idx_base,
                                      int                  enable_boost,
                                      const U* __restrict__ boost_tol,
                                      const T* __restrict__ boost_val)
{
    bsrilu0_33_64_device<T, U, BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                          mb,
                                                          bsr_row_ptr,
                                                          bsr_col_ind,
                                                          bsr_val,
                                                          bsr_diag_ind,
                                                          bsr_dim,
                                                          done_array,
                                                          map,
                                                          zero_pivot,
                                                          idx_base,
                                                          enable_boost,
                                                          enable_boost ? *boost_tol : 0.0,
                                                          enable_boost ? *boost_val
                                                                       : static_cast<T>(0));
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int BSRDIM>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_33_64_host_pointer(rocsparse_direction  dir,
                                    rocsparse_int        mb,
                                    const rocsparse_int* bsr_row_ptr,
                                    const rocsparse_int* bsr_col_ind,
                                    T*                   bsr_val,
                                    const rocsparse_int* bsr_diag_ind,
                                    rocsparse_int        bsr_dim,
                                    int*                 done_array,
                                    const rocsparse_int* map,
                                    rocsparse_int*       zero_pivot,
                                    rocsparse_index_base idx_base,
                                    int                  enable_boost,
                                    U                    boost_tol,
                                    T                    boost_val)
{
    bsrilu0_33_64_device<T, U, BLOCKSIZE, WFSIZE, BSRDIM>(dir,
                                                          mb,
                                                          bsr_row_ptr,
                                                          bsr_col_ind,
                                                          bsr_val,
                                                          bsr_diag_ind,
                                                          bsr_dim,
                                                          done_array,
                                                          map,
                                                          zero_pivot,
                                                          idx_base,
                                                          enable_boost,
                                                          boost_tol,
                                                          boost_val);
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_general_device_pointer(rocsparse_direction  dir,
                                        rocsparse_int        mb,
                                        const rocsparse_int* bsr_row_ptr,
                                        const rocsparse_int* bsr_col_ind,
                                        T*                   bsr_val,
                                        const rocsparse_int* bsr_diag_ind,
                                        rocsparse_int        bsr_dim,
                                        int*                 done_array,
                                        const rocsparse_int* map,
                                        rocsparse_int*       zero_pivot,
                                        rocsparse_index_base idx_base,
                                        int                  enable_boost,
                                        const U*             boost_tol,
                                        const T*             boost_val)
{
    bsrilu0_general_device<T, U, BLOCKSIZE, WFSIZE, SLEEP>(
        dir,
        mb,
        bsr_row_ptr,
        bsr_col_ind,
        bsr_val,
        bsr_diag_ind,
        bsr_dim,
        done_array,
        map,
        zero_pivot,
        idx_base,
        enable_boost,
        enable_boost ? *boost_tol : static_cast<U>(0),
        enable_boost ? *boost_val : static_cast<T>(0));
}

template <typename T, typename U, unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP>
__launch_bounds__(BLOCKSIZE) __global__
    void bsrilu0_general_host_pointer(rocsparse_direction  dir,
                                      rocsparse_int        mb,
                                      const rocsparse_int* bsr_row_ptr,
                                      const rocsparse_int* bsr_col_ind,
                                      T*                   bsr_val,
                                      const rocsparse_int* bsr_diag_ind,
                                      rocsparse_int        bsr_dim,
                                      int*                 done_array,
                                      const rocsparse_int* map,
                                      rocsparse_int*       zero_pivot,
                                      rocsparse_index_base idx_base,
                                      int                  enable_boost,
                                      U                    boost_tol,
                                      T                    boost_val)
{
    bsrilu0_general_device<T, U, BLOCKSIZE, WFSIZE, SLEEP>(dir,
                                                           mb,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           bsr_val,
                                                           bsr_diag_ind,
                                                           bsr_dim,
                                                           done_array,
                                                           map,
                                                           zero_pivot,
                                                           idx_base,
                                                           enable_boost,
                                                           boost_tol,
                                                           boost_val);
}

template <typename T,
          typename U,
          typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value
                                      || std::is_same<T, rocsparse_float_complex>::value,
                                  int>::type
          = 0>
inline void bsrilu0_launcher(rocsparse_handle     handle,
                             rocsparse_direction  dir,
                             rocsparse_int        mb,
                             rocsparse_index_base base,
                             T*                   bsr_val,
                             const rocsparse_int* bsr_row_ptr,
                             const rocsparse_int* bsr_col_ind,
                             rocsparse_int        block_dim,
                             rocsparse_mat_info   info,
                             int*                 done_array)
{
    if(handle->properties.gcnArch == 908 && handle->asic_rev < 2)
    {
        LAUNCH_BSRILU65inf(true);
    }
    else
    {
        if(block_dim <= 8)
        {
            LAUNCH_BSRILU28();
        }
        else if(block_dim <= 16)
        {
            LAUNCH_BSRILU932(16);
        }
        else if(block_dim <= 32)
        {
            LAUNCH_BSRILU932(32);
        }
        else if(block_dim <= 64)
        {
            LAUNCH_BSRILU3364();
        }
        else
        {
            LAUNCH_BSRILU65inf(false);
        }
    }
}

template <typename T,
          typename U,
          typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type = 0>
inline void bsrilu0_launcher(rocsparse_handle     handle,
                             rocsparse_direction  dir,
                             rocsparse_int        mb,
                             rocsparse_index_base base,
                             T*                   bsr_val,
                             const rocsparse_int* bsr_row_ptr,
                             const rocsparse_int* bsr_col_ind,
                             rocsparse_int        block_dim,
                             rocsparse_mat_info   info,
                             int*                 done_array)
{
    if(handle->properties.gcnArch == 908 && handle->asic_rev < 2)
    {
        LAUNCH_BSRILU65inf(true);
    }
    else
    {
        if(block_dim <= 8)
        {
            LAUNCH_BSRILU28();
        }
        else if(block_dim <= 16)
        {
            LAUNCH_BSRILU932(16);
        }
        else if(block_dim <= 32)
        {
            LAUNCH_BSRILU932(32);
        }
        else
        {
            LAUNCH_BSRILU65inf(false);
        }
    }
}

template <typename T, typename U>
rocsparse_status rocsparse_bsrilu0_template(rocsparse_handle          handle,
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
              replaceX<T>("rocsparse_Xbsrilu0"),
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

    log_bench(handle, "./rocsparse-bench -f bsrilu0 -r", replaceX<T>("X"), "--mtx <matrix.mtx> ");

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
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
    if(info->bsrilu0_info == nullptr)
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

    bsrilu0_launcher<T, U>(handle,
                           dir,
                           mb,
                           descr->base,
                           bsr_val,
                           bsr_row_ptr,
                           bsr_col_ind,
                           block_dim,
                           info,
                           d_done_array);

    return rocsparse_status_success;
}

#endif // ROCSPARSE_BSRILU0_HPP

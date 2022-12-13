/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "definitions.h"
#include "rocsparse_bsrsv.hpp"
#include "utility.h"

#include "bsrsv_device.h"

#define LAUNCH_BSRSV_GTHR_DIM(bsize, wfsize, dim)                 \
    hipLaunchKernelGGL((bsr_gather<wfsize, bsize / wfsize, dim>), \
                       dim3((wfsize * nnzb - 1) / bsize + 1),     \
                       dim3(wfsize, bsize / wfsize),              \
                       0,                                         \
                       stream,                                    \
                       dir,                                       \
                       nnzb,                                      \
                       (rocsparse_int*)bsrsv->trmt_perm,          \
                       bsr_val,                                   \
                       bsrt_val,                                  \
                       block_dim)

#define LAUNCH_BSRSV_GTHR(bsize, wfsize, dim) \
    if(dim <= 2)                              \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 4, 2);   \
    }                                         \
    else if(dim <= 4)                         \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else if(wfsize == 32)                     \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else                                      \
    {                                         \
        LAUNCH_BSRSV_GTHR_DIM(bsize, 64, 8);  \
    }

#define LAUNCH_BSRSV_SHARED(fill, ptr, bsize, wfsize, dim, arch, asic) \
    if(fill == rocsparse_fill_mode_lower)                              \
    {                                                                  \
        if(arch == 908 && asic < 2)                                    \
        {                                                              \
            LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, true);       \
        }                                                              \
        else                                                           \
        {                                                              \
            LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, false);      \
        }                                                              \
    }                                                                  \
    else                                                               \
    {                                                                  \
        if(arch == 908 && asic < 2)                                    \
        {                                                              \
            LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, true);       \
        }                                                              \
        else                                                           \
        {                                                              \
            LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, false);      \
        }                                                              \
    }

#define LAUNCH_BSRSV_LOWER_SHARED(bsize, wfsize, dim, arch)            \
    hipLaunchKernelGGL((bsrsv_lower_shared<bsize, wfsize, dim, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),            \
                       dim3(bsize),                                    \
                       0,                                              \
                       stream,                                         \
                       mb,                                             \
                       alpha_device_host,                              \
                       local_bsr_row_ptr,                              \
                       local_bsr_col_ind,                              \
                       local_bsr_val,                                  \
                       block_dim,                                      \
                       x,                                              \
                       y,                                              \
                       done_array,                                     \
                       (rocsparse_int*)bsrsv->row_map,                 \
                       (rocsparse_int*)info->zero_pivot,               \
                       descr->base,                                    \
                       descr->diag_type,                               \
                       dir)

#define LAUNCH_BSRSV_UPPER_SHARED(bsize, wfsize, dim, arch)            \
    hipLaunchKernelGGL((bsrsv_upper_shared<bsize, wfsize, dim, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),            \
                       dim3(bsize),                                    \
                       0,                                              \
                       stream,                                         \
                       mb,                                             \
                       alpha_device_host,                              \
                       local_bsr_row_ptr,                              \
                       local_bsr_col_ind,                              \
                       local_bsr_val,                                  \
                       block_dim,                                      \
                       x,                                              \
                       y,                                              \
                       done_array,                                     \
                       (rocsparse_int*)bsrsv->row_map,                 \
                       (rocsparse_int*)info->zero_pivot,               \
                       descr->base,                                    \
                       descr->diag_type,                               \
                       dir)

#define LAUNCH_BSRSV_GENERAL(fill, ptr, bsize, wfsize, arch, asic) \
    if(fill == rocsparse_fill_mode_lower)                          \
    {                                                              \
        if(arch == 908 && asic < 2)                                \
        {                                                          \
            LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, true);       \
        }                                                          \
        else                                                       \
        {                                                          \
            LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, false);      \
        }                                                          \
    }                                                              \
    else                                                           \
    {                                                              \
        if(arch == 908 && asic < 2)                                \
        {                                                          \
            LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, true);       \
        }                                                          \
        else                                                       \
        {                                                          \
            LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, false);      \
        }                                                          \
    }

#define LAUNCH_BSRSV_LOWER_GENERAL(bsize, wfsize, arch)            \
    hipLaunchKernelGGL((bsrsv_lower_general<bsize, wfsize, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),        \
                       dim3(bsize),                                \
                       0,                                          \
                       stream,                                     \
                       mb,                                         \
                       alpha_device_host,                          \
                       local_bsr_row_ptr,                          \
                       local_bsr_col_ind,                          \
                       local_bsr_val,                              \
                       block_dim,                                  \
                       x,                                          \
                       y,                                          \
                       done_array,                                 \
                       (rocsparse_int*)bsrsv->row_map,             \
                       (rocsparse_int*)info->zero_pivot,           \
                       descr->base,                                \
                       descr->diag_type,                           \
                       dir)

#define LAUNCH_BSRSV_UPPER_GENERAL(bsize, wfsize, arch)            \
    hipLaunchKernelGGL((bsrsv_upper_general<bsize, wfsize, arch>), \
                       dim3((wfsize * mb - 1) / bsize + 1),        \
                       dim3(bsize),                                \
                       0,                                          \
                       stream,                                     \
                       mb,                                         \
                       alpha_device_host,                          \
                       local_bsr_row_ptr,                          \
                       local_bsr_col_ind,                          \
                       local_bsr_val,                              \
                       block_dim,                                  \
                       x,                                          \
                       y,                                          \
                       done_array,                                 \
                       (rocsparse_int*)bsrsv->row_map,             \
                       (rocsparse_int*)info->zero_pivot,           \
                       descr->base,                                \
                       descr->diag_type,                           \
                       dir)

template <unsigned int  BLOCKSIZE,
          unsigned int  WFSIZE,
          rocsparse_int BSRDIM,
          bool          SLEEP,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrsv_lower_shared(rocsparse_int mb,
                            U             alpha_device_host,
                            const rocsparse_int* __restrict__ bsr_row_ptr,
                            const rocsparse_int* __restrict__ bsr_col_ind,
                            const T* __restrict__ bsr_val,
                            rocsparse_int block_dim,
                            const T* __restrict__ x,
                            T* __restrict__ y,
                            int* __restrict__ done_array,
                            rocsparse_int* __restrict__ map,
                            rocsparse_int* __restrict__ zero_pivot,
                            rocsparse_index_base idx_base,
                            rocsparse_diag_type  diag_type,
                            rocsparse_direction  dir)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    bsrsv_lower_shared_device<BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                alpha,
                                                                bsr_row_ptr,
                                                                bsr_col_ind,
                                                                bsr_val,
                                                                block_dim,
                                                                x,
                                                                y,
                                                                done_array,
                                                                map,
                                                                zero_pivot,
                                                                idx_base,
                                                                diag_type,
                                                                dir);
}

template <unsigned int  BLOCKSIZE,
          unsigned int  WFSIZE,
          rocsparse_int BSRDIM,
          bool          SLEEP,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrsv_upper_shared(rocsparse_int mb,
                            U             alpha_device_host,
                            const rocsparse_int* __restrict__ bsr_row_ptr,
                            const rocsparse_int* __restrict__ bsr_col_ind,
                            const T* __restrict__ bsr_val,
                            rocsparse_int block_dim,
                            const T* __restrict__ x,
                            T* __restrict__ y,
                            int* __restrict__ done_array,
                            rocsparse_int* __restrict__ map,
                            rocsparse_int* __restrict__ zero_pivot,
                            rocsparse_index_base idx_base,
                            rocsparse_diag_type  diag_type,
                            rocsparse_direction  dir)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    bsrsv_upper_shared_device<BLOCKSIZE, WFSIZE, BSRDIM, SLEEP>(mb,
                                                                alpha,
                                                                bsr_row_ptr,
                                                                bsr_col_ind,
                                                                bsr_val,
                                                                block_dim,
                                                                x,
                                                                y,
                                                                done_array,
                                                                map,
                                                                zero_pivot,
                                                                idx_base,
                                                                diag_type,
                                                                dir);
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrsv_lower_general(rocsparse_int mb,
                             U             alpha_device_host,
                             const rocsparse_int* __restrict__ bsr_row_ptr,
                             const rocsparse_int* __restrict__ bsr_col_ind,
                             const T* __restrict__ bsr_val,
                             rocsparse_int block_dim,
                             const T* __restrict__ x,
                             T* __restrict__ y,
                             int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int* __restrict__ zero_pivot,
                             rocsparse_index_base idx_base,
                             rocsparse_diag_type  diag_type,
                             rocsparse_direction  dir)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    bsrsv_lower_general_device<BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                         alpha,
                                                         bsr_row_ptr,
                                                         bsr_col_ind,
                                                         bsr_val,
                                                         block_dim,
                                                         x,
                                                         y,
                                                         done_array,
                                                         map,
                                                         zero_pivot,
                                                         idx_base,
                                                         diag_type,
                                                         dir);
}

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, bool SLEEP, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void bsrsv_upper_general(rocsparse_int mb,
                             U             alpha_device_host,
                             const rocsparse_int* __restrict__ bsr_row_ptr,
                             const rocsparse_int* __restrict__ bsr_col_ind,
                             const T* __restrict__ bsr_val,
                             rocsparse_int block_dim,
                             const T* __restrict__ x,
                             T* __restrict__ y,
                             int* __restrict__ done_array,
                             rocsparse_int* __restrict__ map,
                             rocsparse_int* __restrict__ zero_pivot,
                             rocsparse_index_base idx_base,
                             rocsparse_diag_type  diag_type,
                             rocsparse_direction  dir)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    bsrsv_upper_general_device<BLOCKSIZE, WFSIZE, SLEEP>(mb,
                                                         alpha,
                                                         bsr_row_ptr,
                                                         bsr_col_ind,
                                                         bsr_val,
                                                         block_dim,
                                                         x,
                                                         y,
                                                         done_array,
                                                         map,
                                                         zero_pivot,
                                                         idx_base,
                                                         diag_type,
                                                         dir);
}

template <typename T, typename U>
rocsparse_status rocsparse_bsrsv_solve_dispatch(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                U                         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                const T*                  x,
                                                T*                        y,
                                                rocsparse_solve_policy    policy,
                                                void*                     temp_buffer)
{

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
        static const rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
        RETURN_IF_HIP_ERROR(hipMemcpyAsync((rocsparse_int*)info->zero_pivot,
                                           &max,
                                           sizeof(rocsparse_int),
                                           hipMemcpyHostToDevice,
                                           stream));

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
        LAUNCH_BSRSV_GTHR(256, 64, block_dim);

        local_bsr_row_ptr = (rocsparse_int*)bsrsv->trmt_row_ptr;
        local_bsr_col_ind = (rocsparse_int*)bsrsv->trmt_col_ind;
        local_bsr_val     = (T*)bsrt_val;

        fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                             : rocsparse_fill_mode_lower;
    }

    // Determine gcnArch and ASIC revision
    int gcnArch = handle->properties.gcnArch;
    int asicRev = handle->asic_rev;

    if(handle->wavefront_size == 64)
    {
        if(block_dim <= 8)
        {
            // Launch shared memory based kernel for small BSR block dimensions
            LAUNCH_BSRSV_SHARED(fill_mode, handle->pointer_mode, 128, 64, 8, gcnArch, asicRev);
        }
        else if(block_dim <= 16)
        {
            // Launch shared memory based kernel for small BSR block dimensions
            LAUNCH_BSRSV_SHARED(fill_mode, handle->pointer_mode, 128, 64, 16, gcnArch, asicRev);
        }
        else if(block_dim <= 32)
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
        //
        // This is wavefront 32, let's exclude it.
        //
        // LCOV_EXCL_START;

        // Launch general algorithm
        LAUNCH_BSRSV_GENERAL(fill_mode, handle->pointer_mode, 128, 32, gcnArch, asicRev);

        // LCOV_EXCL_STOP;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_bsrsv_solve_template(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
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

    if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbsrsv"),
              dir,
              trans,
              mb,
              nnzb,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
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
              block_dim,
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha_device_host));

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(policy))
    {
        return rocsparse_status_invalid_value;
    }

    // Check operation type
    if(trans != rocsparse_operation_none && trans != rocsparse_operation_transpose)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || nnzb < 0 || block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || block_dim == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr || alpha_device_host == nullptr || x == nullptr || y == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb != 0 && (bsr_val == nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_bsrsv_solve_dispatch(handle,
                                              dir,
                                              trans,
                                              mb,
                                              nnzb,
                                              alpha_device_host,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              block_dim,
                                              info,
                                              x,
                                              y,
                                              policy,
                                              temp_buffer);
    }
    else
    {
        return rocsparse_bsrsv_solve_dispatch(handle,
                                              dir,
                                              trans,
                                              mb,
                                              nnzb,
                                              *alpha_device_host,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              block_dim,
                                              info,
                                              x,
                                              y,
                                              policy,
                                              temp_buffer);
    }
    return rocsparse_status_success;
}

// bsrsv_solve
#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nnzb,        \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     rocsparse_mat_info        info,        \
                                     const TYPE*               x,           \
                                     TYPE*                     y,           \
                                     rocsparse_solve_policy    policy,      \
                                     void*                     temp_buffer) \
    {                                                                       \
        return rocsparse_bsrsv_solve_template(handle,                       \
                                              dir,                          \
                                              trans,                        \
                                              mb,                           \
                                              nnzb,                         \
                                              alpha,                        \
                                              descr,                        \
                                              bsr_val,                      \
                                              bsr_row_ptr,                  \
                                              bsr_col_ind,                  \
                                              block_dim,                    \
                                              info,                         \
                                              x,                            \
                                              y,                            \
                                              policy,                       \
                                              temp_buffer);                 \
    }

C_IMPL(rocsparse_sbsrsv_solve, float);
C_IMPL(rocsparse_dbsrsv_solve, double);
C_IMPL(rocsparse_cbsrsv_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsv_solve, rocsparse_double_complex);
#undef C_IMPL

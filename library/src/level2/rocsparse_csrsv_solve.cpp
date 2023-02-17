/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../level1/rocsparse_gthr.hpp"
#include "csrsv_device.h"

#include "rocsparse_csrsv.hpp"
#include "utility.h"

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         SLEEP,
          typename I,
          typename J,
          typename T,
          typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrsv_kernel(J m,
                  U alpha_device_host,
                  const I* __restrict__ csr_row_ptr,
                  const J* __restrict__ csr_col_ind,
                  const T* __restrict__ csr_val,
                  const T* __restrict__ x,
                  T* __restrict__ y,
                  int* __restrict__ done_array,
                  J* __restrict__ map,
                  int offset,
                  J* __restrict__ zero_pivot,
                  rocsparse_index_base idx_base,
                  rocsparse_fill_mode  fill_mode,
                  rocsparse_diag_type  diag_type)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    csrsv_device<BLOCKSIZE, WF_SIZE, SLEEP>(m,
                                            alpha,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            x,
                                            y,
                                            done_array,
                                            map,
                                            offset,
                                            zero_pivot,
                                            idx_base,
                                            fill_mode,
                                            diag_type);
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrsv_solve_dispatch(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                J                         m,
                                                I                         nnz,
                                                U                         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
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
    ptr += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * m, stream));

    rocsparse_trm_info csrsv
        = (descr->fill_mode == rocsparse_fill_mode_upper)
              ? ((trans == rocsparse_operation_none) ? info->csrsv_upper_info
                                                     : info->csrsvt_upper_info)
              : ((trans == rocsparse_operation_none) ? info->csrsv_lower_info
                                                     : info->csrsvt_lower_info);

    if(csrsv == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // If diag type is unit, re-initialize zero pivot to remove structural zeros
    if(descr->diag_type == rocsparse_diag_type_unit)
    {
        RETURN_IF_HIP_ERROR(rocsparse_assign_async(
            static_cast<J*>(info->zero_pivot), std::numeric_limits<J>::max(), stream));
    }

    // Pointers to differentiate between transpose mode
    const I* local_csr_row_ptr = csr_row_ptr;
    const J* local_csr_col_ind = csr_col_ind;
    const T* local_csr_val     = csr_val;

    rocsparse_fill_mode fill_mode = descr->fill_mode;

    // When computing transposed triangular solve, we first need to update the
    // transposed matrix values
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        T* csrt_val = reinterpret_cast<T*>(ptr);

        // Gather values
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_gthr_template(
            handle, nnz, csr_val, csrt_val, (const I*)csrsv->trmt_perm, rocsparse_index_base_zero));

        if(trans == rocsparse_operation_conjugate_transpose)
        {
            // conjugate csrt_val
            hipLaunchKernelGGL((conjugate<256, I, T>),
                               dim3((nnz - 1) / 256 + 1),
                               dim3(256),
                               0,
                               stream,
                               nnz,
                               csrt_val);
        }

        local_csr_row_ptr = (const I*)csrsv->trmt_row_ptr;
        local_csr_col_ind = (const J*)csrsv->trmt_col_ind;
        local_csr_val     = (const T*)csrt_val;

        fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                             : rocsparse_fill_mode_lower;
    }

    // Determine gcnArch
    int gcnArch = handle->properties.gcnArch;
    int asicRev = handle->asic_rev;

#define CSRSV_DIM 1024
    dim3 csrsv_blocks(((int64_t)handle->wavefront_size * m - 1) / CSRSV_DIM + 1);
    dim3 csrsv_threads(CSRSV_DIM);

    // gfx908
    if(gcnArch == 908 && asicRev < 2)
    {
        // LCOV_EXCL_START
        hipLaunchKernelGGL((csrsv_kernel<CSRSV_DIM, 64, true>),
                           csrsv_blocks,
                           csrsv_threads,
                           0,
                           stream,
                           m,
                           alpha_device_host,
                           local_csr_row_ptr,
                           local_csr_col_ind,
                           local_csr_val,
                           x,
                           y,
                           done_array,
                           (J*)csrsv->row_map,
                           0,
                           (J*)info->zero_pivot,
                           descr->base,
                           fill_mode,
                           descr->diag_type);
        // LCOV_EXCL_STOP
    }
    else
    {
        // rocsparse_pointer_mode_device
        if(handle->wavefront_size == 32)
        {
            // LCOV_EXCL_START
            hipLaunchKernelGGL((csrsv_kernel<CSRSV_DIM, 32, false>),
                               csrsv_blocks,
                               csrsv_threads,
                               0,
                               stream,
                               m,
                               alpha_device_host,
                               local_csr_row_ptr,
                               local_csr_col_ind,
                               local_csr_val,
                               x,
                               y,
                               done_array,
                               (J*)csrsv->row_map,
                               0,
                               (J*)info->zero_pivot,
                               descr->base,
                               fill_mode,
                               descr->diag_type);
            // LCOV_EXCL_STOP
        }
        else
        {
            assert(handle->wavefront_size == 64);
            hipLaunchKernelGGL((csrsv_kernel<CSRSV_DIM, 64, false>),
                               csrsv_blocks,
                               csrsv_threads,
                               0,
                               stream,
                               m,
                               alpha_device_host,
                               local_csr_row_ptr,
                               local_csr_col_ind,
                               local_csr_val,
                               x,
                               y,
                               done_array,
                               (J*)csrsv->row_map,
                               0,
                               (J*)info->zero_pivot,
                               descr->base,
                               fill_mode,
                               descr->diag_type);
        }
    }
#undef CSRSV_DIM

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrsv_solve_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                J                         m,
                                                I                         nnz,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
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
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrsv"),
              trans,
              m,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              (const void*&)x,
              (const void*&)y,
              policy,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f csrsv -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha_device_host));

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(policy))
    {
        return rocsparse_status_invalid_value;
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
    if(m < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || alpha_device_host == nullptr || x == nullptr || y == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csr_col_ind == nullptr && csr_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_csrsv_solve_dispatch(handle,
                                              trans,
                                              m,
                                              nnz,
                                              alpha_device_host,
                                              descr,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              x,
                                              y,
                                              policy,
                                              temp_buffer);
    }
    else
    {
        return rocsparse_csrsv_solve_dispatch(handle,
                                              trans,
                                              m,
                                              nnz,
                                              *alpha_device_host,
                                              descr,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              x,
                                              y,
                                              policy,
                                              temp_buffer);
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                           \
    template rocsparse_status rocsparse_csrsv_solve_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                          \
        rocsparse_operation       trans,                                           \
        JTYPE                     m,                                               \
        ITYPE                     nnz,                                             \
        const TTYPE*              alpha_device_host,                               \
        const rocsparse_mat_descr descr,                                           \
        const TTYPE*              csr_val,                                         \
        const ITYPE*              csr_row_ptr,                                     \
        const JTYPE*              csr_col_ind,                                     \
        rocsparse_mat_info        info,                                            \
        const TTYPE*              x,                                               \
        TTYPE*                    y,                                               \
        rocsparse_solve_policy    policy,                                          \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans,       \
                                     rocsparse_int             m,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_mat_info        info,        \
                                     const TYPE*               x,           \
                                     TYPE*                     y,           \
                                     rocsparse_solve_policy    policy,      \
                                     void*                     temp_buffer) \
    {                                                                       \
        return rocsparse_csrsv_solve_template(handle,                       \
                                              trans,                        \
                                              m,                            \
                                              nnz,                          \
                                              alpha,                        \
                                              descr,                        \
                                              csr_val,                      \
                                              csr_row_ptr,                  \
                                              csr_col_ind,                  \
                                              info,                         \
                                              x,                            \
                                              y,                            \
                                              policy,                       \
                                              temp_buffer);                 \
    }

C_IMPL(rocsparse_scsrsv_solve, float);
C_IMPL(rocsparse_dcsrsv_solve, double);
C_IMPL(rocsparse_ccsrsv_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsv_solve, rocsparse_double_complex);

#undef C_IMPL

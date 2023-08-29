/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "common.h"
#include "definitions.h"
#include "rocsparse_csritsv.hpp"
#include "rocsparse_csrmv.hpp"
#include "utility.h"

template <unsigned int BLOCKSIZE, typename I, typename J>
__launch_bounds__(BLOCKSIZE) static __global__
    void kernel_count_missing_diagonal(J m,
                                       const I* __restrict__ ptr_diag_,
                                       J ptr_shift_,
                                       const J* __restrict__ ind_,
                                       rocsparse_index_base base_,
                                       J* __restrict__ count,
                                       rocsparse_int* __restrict__ position)
{
    const J tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    if(tid < m)
    {
        const J c = (((ind_[ptr_diag_[tid] - base_ + ptr_shift_] - base_) != tid) ? 1 : 0);
        if(c > 0)
        {
            const rocsparse_int p = (tid + base_);
            rocsparse_atomic_min(position, p);
            rocsparse_atomic_add(count, c);
        }
    }
}

template <rocsparse_fill_mode FILL_MODE, unsigned int BLOCKSIZE, typename I, typename J>
__launch_bounds__(BLOCKSIZE) static __global__
    void kernel_count_missing_diagonal2(J m,
                                        const I* __restrict__ ptr_,
                                        const J* __restrict__ ind_,
                                        rocsparse_index_base base_,
                                        J* __restrict__ count,
                                        rocsparse_int* __restrict__ position)
{
    const J tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    if(tid < m)
    {
        static constexpr int shift = (FILL_MODE == rocsparse_fill_mode_lower) ? 1 : 0;
        const J c = (((ind_[ptr_[tid + shift] - shift - base_] - base_) != tid) ? 1 : 0);
        if(c > 0)
        {
            const rocsparse_int p = (tid + base_);
            rocsparse_atomic_min(position, p);
            rocsparse_atomic_add(count, c);
        }
    }
}

template <rocsparse_fill_mode FILL_MODE, unsigned int BLOCKSIZE, typename I, typename J>
__launch_bounds__(BLOCKSIZE) static __global__
    void kernel_count_diagonal_triangular(J m,
                                          const I* __restrict__ ptr_,
                                          const J* __restrict__ ind_,
                                          rocsparse_index_base base_,
                                          J* __restrict__ count)
{
    const J tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    if(tid < m)
    {
        static constexpr int shift = (FILL_MODE == rocsparse_fill_mode_lower) ? 1 : 0;
        const J c = (((ind_[ptr_[tid + shift] - shift - base_] - base_) == tid) ? 1 : 0);
        if(c > 0)
        {
            rocsparse_atomic_add(count, c);
        }
    }
}

template <unsigned int BLOCKSIZE, typename I, typename J>
__launch_bounds__(BLOCKSIZE) static __global__ void kernel_ptr_end_unit(J m,
                                                                        const I* __restrict__ ptr_,
                                                                        const J* __restrict__ ind_,
                                                                        I* __restrict__ ptr_end,
                                                                        rocsparse_index_base base)
{
    const J tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    if(tid < m)
    {
        ptr_end[tid] = ptr_[tid + 1];
        for(I k = ptr_[tid] - base; k < ptr_[tid + 1] - base; ++k)
        {
            if(ind_[k] - base >= tid)
            {
                ptr_end[tid] = k + base;
                break;
            }
        }
    }
}

template <unsigned int BLOCKSIZE, typename I, typename J>
__launch_bounds__(BLOCKSIZE) static __global__
    void kernel_ptr_end_non_unit(J m,
                                 const I* __restrict__ ptr_,
                                 const J* __restrict__ ind_,
                                 I* __restrict__ ptr_end,
                                 rocsparse_index_base base)
{
    unsigned int tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    if(tid < m)
    {
        ptr_end[tid] = ptr_[tid + 1];
        for(I k = ptr_[tid] - base; k < ptr_[tid + 1] - base; ++k)
        {
            if(ind_[k] - base > tid)
            {
                ptr_end[tid] = k + base;
                break;
            }
        }
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csritsv_info_analysis(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 J                         m,
                                                 I                         nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  csr_val,
                                                 const I*                  csr_row_ptr,
                                                 const J*                  csr_col_ind,
                                                 rocsparse_csritsv_info    info,
                                                 rocsparse_int**           zero_pivot,
                                                 void*                     temp_buffer)
{
    // Allocate buffer to hold zero pivot
    if(zero_pivot[0] == nullptr)
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync((void**)zero_pivot, sizeof(rocsparse_int), handle->stream));
    }

    // Initialize zero pivot
    rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        zero_pivot[0], &max, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));

    const rocsparse_fill_mode fill_mode = descr->fill_mode;
    const rocsparse_diag_type diag_type = descr->diag_type;

    if(nnz == 0)
    {
        if(diag_type == rocsparse_diag_type_non_unit)
        {
            const rocsparse_int b = (rocsparse_int)descr->base;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                zero_pivot[0], &b, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));
            return rocsparse_status_success;
        }
    }

    static constexpr unsigned int BLOCKSIZE = 1024;
    switch(descr->type)
    {
    case rocsparse_matrix_type_general:
    {
        //
        // we need to compute ptr_end and store it in info.
        //
        info->ptr_end_indextype = rocsparse_indextype_i32;
        if(sizeof(I) > sizeof(int32_t))
        {
            info->ptr_end_indextype = rocsparse_indextype_i64;
        }
        info->ptr_end_size = m;
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&info->ptr_end, sizeof(I) * m, handle->stream));
        info->is_submatrix = true;

        //
        // Compute ptr_end.
        //

        if((fill_mode == rocsparse_fill_mode_lower && diag_type == rocsparse_diag_type_unit)
           || (fill_mode == rocsparse_fill_mode_upper && diag_type == rocsparse_diag_type_non_unit))
        {

            dim3 blocks((m - 1) / BLOCKSIZE + 1);
            dim3 threads(BLOCKSIZE);
            hipLaunchKernelGGL((kernel_ptr_end_unit<1024, I, J>),
                               blocks,
                               threads,
                               0,
                               handle->stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               (I*)info->ptr_end,
                               descr->base);
        }
        else if((fill_mode == rocsparse_fill_mode_lower
                 && diag_type == rocsparse_diag_type_non_unit)
                || (fill_mode == rocsparse_fill_mode_upper
                    && diag_type == rocsparse_diag_type_unit))
        {

            dim3 blocks((m - 1) / BLOCKSIZE + 1);
            dim3 threads(BLOCKSIZE);
            hipLaunchKernelGGL((kernel_ptr_end_non_unit<1024, I, J>),
                               blocks,
                               threads,
                               0,
                               handle->stream,
                               m,
                               csr_row_ptr,
                               csr_col_ind,
                               (I*)info->ptr_end,
                               descr->base);
        }

        break;
    }

    case rocsparse_matrix_type_triangular:
    {
        info->ptr_end_indextype = rocsparse_indextype_i32;
        if(sizeof(I) > sizeof(int32_t))
        {
            info->ptr_end_indextype = rocsparse_indextype_i64;
        }
        info->ptr_end_size = m;
        info->ptr_end      = (void*)(csr_row_ptr + 1);
        info->is_submatrix = false;
        break;
    }

    case rocsparse_matrix_type_symmetric:
    case rocsparse_matrix_type_hermitian:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    }

    //
    // count missing diagonal.
    //
    if(diag_type == rocsparse_diag_type_non_unit)
    {

        RETURN_IF_HIP_ERROR(hipMemsetAsync(temp_buffer, 0, sizeof(J), handle->stream));
        if(info->is_submatrix)
        {
            const J ptr_shift = (fill_mode == rocsparse_fill_mode_upper) ? 0 : -1;
            dim3    blocks((m - 1) / BLOCKSIZE + 1);
            dim3    threads(BLOCKSIZE);
            hipLaunchKernelGGL((kernel_count_missing_diagonal<1024, I, J>),
                               blocks,
                               threads,
                               0,
                               handle->stream,
                               m,
                               (const I*)info->ptr_end,
                               ptr_shift,
                               csr_col_ind,
                               descr->base,
                               (J*)temp_buffer,
                               zero_pivot[0]);
        }
        else
        {
            dim3 blocks((m - 1) / BLOCKSIZE + 1);
            dim3 threads(BLOCKSIZE);
            if(fill_mode == rocsparse_fill_mode_lower)
            {
                hipLaunchKernelGGL(
                    (kernel_count_missing_diagonal2<rocsparse_fill_mode_lower, 1024, I, J>),
                    blocks,
                    threads,
                    0,
                    handle->stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    descr->base,
                    (J*)temp_buffer,
                    zero_pivot[0]);
            }
            else
            {
                hipLaunchKernelGGL(
                    (kernel_count_missing_diagonal2<rocsparse_fill_mode_upper, 1024, I, J>),
                    blocks,
                    threads,
                    0,
                    handle->stream,
                    m,
                    csr_row_ptr,
                    csr_col_ind,
                    descr->base,
                    (J*)temp_buffer,
                    zero_pivot[0]);
            }
        }
        J count_missing_diagonal;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&count_missing_diagonal,
                                           temp_buffer,
                                           sizeof(J),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        if(count_missing_diagonal > 0)
        {
            //
            // zero pivot.
            //
            return rocsparse_status_success;
        }
    }
    else
    {
        if(descr->type == rocsparse_matrix_type_triangular)
        {
            if(false == info->is_submatrix)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(temp_buffer, 0, sizeof(J), handle->stream));
                J count_diagonal = 0;
                if(nnz > 0)
                {
                    //
                    // We nned to check diagonal element are not present.
                    //
                    dim3 blocks((m - 1) / BLOCKSIZE + 1);
                    dim3 threads(BLOCKSIZE);
                    if(fill_mode == rocsparse_fill_mode_lower)
                    {
                        hipLaunchKernelGGL(
                            (kernel_count_diagonal_triangular<rocsparse_fill_mode_lower,
                                                              1024,
                                                              I,
                                                              J>),
                            blocks,
                            threads,
                            0,
                            handle->stream,
                            m,
                            csr_row_ptr,
                            csr_col_ind,
                            descr->base,
                            (J*)temp_buffer);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (kernel_count_diagonal_triangular<rocsparse_fill_mode_upper,
                                                              1024,
                                                              I,
                                                              J>),
                            blocks,
                            threads,
                            0,
                            handle->stream,
                            m,
                            csr_row_ptr,
                            csr_col_ind,
                            descr->base,
                            (J*)temp_buffer);
                    }

                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&count_diagonal,
                                                       temp_buffer,
                                                       sizeof(J),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                }

                if(count_diagonal > 0)
                {
                    std::cout << "The matrix is specified as unit triangular but contains "
                              << count_diagonal << " diagonal element(s)." << std::endl;
                    return rocsparse_status_internal_error;
                }
            }
        }
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csritsv_analysis_template(rocsparse_handle          handle,
                                                     rocsparse_operation       trans,
                                                     J                         m,
                                                     I                         nnz,
                                                     const rocsparse_mat_descr descr,
                                                     const T*                  csr_val,
                                                     const I*                  csr_row_ptr,
                                                     const J*                  csr_col_ind,
                                                     rocsparse_mat_info        info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy    solve,
                                                     void*                     temp_buffer)
{
    // Quick return if possible
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    // Differentiate the analysis policies
    if(analysis == rocsparse_analysis_policy_reuse)
    {

        // We try to re-use already analyzed lower part, if available.
        // It is the user's responsibility that this data is still valid,
        // since he passed the 'reuse' flag.

        // If csrsv meta data is already available, do nothing
        if(info->csritsv_info != nullptr)
        {
            return rocsparse_status_success;
        }
    }

    // User is explicitly asking to force a re-analysis, or no valid data has been
    // found to be re-used.

    // Clear csritsv info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csritsv_info(info->csritsv_info));

    // Create csritsv info
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csritsv_info(&info->csritsv_info));

    // Analyze the structure.
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csritsv_info_analysis(handle,
                                                              trans,
                                                              m,
                                                              nnz,
                                                              descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              info->csritsv_info,
                                                              (rocsparse_int**)&info->zero_pivot,
                                                              temp_buffer));

    //
    // Now, in case data are contiguous we can call csrmnv_analysis.
    //

    if(false == info->csritsv_info->is_submatrix)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrmv_analysis_template(
            handle, trans, m, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info));
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csritsv_analysis_impl(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 J                         m,
                                                 I                         nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  csr_val,
                                                 const I*                  csr_row_ptr,
                                                 const J*                  csr_col_ind,
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
              replaceX<T>("rocsparse_Xcsritsv_analysis"),
              trans,
              m,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              solve,
              analysis,
              (const void*&)temp_buffer);
    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(analysis))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(solve))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general
       && descr->type != rocsparse_matrix_type_triangular)
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

    // Check pointer arguments
    if(m > 0 && csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(nnz > 0 && csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(nnz > 0 && csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(m > 0 && temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    return rocsparse_csritsv_analysis_template(handle,
                                               trans,
                                               m,
                                               nnz,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               analysis,
                                               solve,
                                               temp_buffer);
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                           \
    template rocsparse_status rocsparse_csritsv_analysis_template( \
        rocsparse_handle          handle,                          \
        rocsparse_operation       trans,                           \
        JTYPE                     m,                               \
        ITYPE                     nnz,                             \
        const rocsparse_mat_descr descr,                           \
        const TTYPE*              csr_val,                         \
        const ITYPE*              csr_row_ptr,                     \
        const JTYPE*              csr_col_ind,                     \
        rocsparse_mat_info        info,                            \
        rocsparse_analysis_policy analysis,                        \
        rocsparse_solve_policy    solve,                           \
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
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     rocsparse_mat_info        info,        \
                                     rocsparse_analysis_policy analysis,    \
                                     rocsparse_solve_policy    solve,       \
                                     void*                     temp_buffer) \
    {                                                                       \
        return rocsparse_csritsv_analysis_impl(handle,                      \
                                               trans,                       \
                                               m,                           \
                                               nnz,                         \
                                               descr,                       \
                                               csr_val,                     \
                                               csr_row_ptr,                 \
                                               csr_col_ind,                 \
                                               info,                        \
                                               analysis,                    \
                                               solve,                       \
                                               temp_buffer);                \
    }

C_IMPL(rocsparse_scsritsv_analysis, float);
C_IMPL(rocsparse_dcsritsv_analysis, double);
C_IMPL(rocsparse_ccsritsv_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zcsritsv_analysis, rocsparse_double_complex);

#undef C_IMPL

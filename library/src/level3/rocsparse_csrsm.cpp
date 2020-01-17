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

#include "rocsparse_csrsm.hpp"
#include "definitions.h"
#include "rocsparse.h"
#include "utility.h"

#include <limits>

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_csrsm_zero_pivot(rocsparse_handle   handle,
                                                       rocsparse_mat_info info,
                                                       rocsparse_int*     position)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle, "rocsparse_csrsm_zero_pivot", (const void*&)info, (const void*&)position);

    // Check pointer arguments
    if(position == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // If m == 0 || nrhs == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->zero_pivot == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 255, sizeof(rocsparse_int), stream));
        }
        else
        {
            *position = -1;
        }

        return rocsparse_status_success;
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        rocsparse_int zero_pivot;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &zero_pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(zero_pivot == std::numeric_limits<rocsparse_int>::max())
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, 255, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                               info->zero_pivot,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToDevice,
                                               stream));

            return rocsparse_status_zero_pivot;
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(
            hipMemcpy(position, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // If no zero pivot is found, set -1
        if(*position == std::numeric_limits<rocsparse_int>::max())
        {
            *position = -1;
        }
        else
        {
            return rocsparse_status_zero_pivot;
        }
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrsm_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_operation       trans_A,
                                                         rocsparse_operation       trans_B,
                                                         rocsparse_int             m,
                                                         rocsparse_int             nrhs,
                                                         rocsparse_int             nnz,
                                                         const float*              alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const float*              csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         const float*              B,
                                                         rocsparse_int             ldb,
                                                         rocsparse_mat_info        info,
                                                         rocsparse_solve_policy    policy,
                                                         size_t*                   buffer_size)
{
    return rocsparse_csrsm_buffer_size_template(handle,
                                                trans_A,
                                                trans_B,
                                                m,
                                                nrhs,
                                                nnz,
                                                alpha,
                                                descr,
                                                csr_val,
                                                csr_row_ptr,
                                                csr_col_ind,
                                                B,
                                                ldb,
                                                info,
                                                policy,
                                                buffer_size);
}

extern "C" rocsparse_status rocsparse_dcsrsm_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_operation       trans_A,
                                                         rocsparse_operation       trans_B,
                                                         rocsparse_int             m,
                                                         rocsparse_int             nrhs,
                                                         rocsparse_int             nnz,
                                                         const double*             alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const double*             csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         const double*             B,
                                                         rocsparse_int             ldb,
                                                         rocsparse_mat_info        info,
                                                         rocsparse_solve_policy    policy,
                                                         size_t*                   buffer_size)
{
    return rocsparse_csrsm_buffer_size_template(handle,
                                                trans_A,
                                                trans_B,
                                                m,
                                                nrhs,
                                                nnz,
                                                alpha,
                                                descr,
                                                csr_val,
                                                csr_row_ptr,
                                                csr_col_ind,
                                                B,
                                                ldb,
                                                info,
                                                policy,
                                                buffer_size);
}

extern "C" rocsparse_status rocsparse_ccsrsm_buffer_size(rocsparse_handle               handle,
                                                         rocsparse_operation            trans_A,
                                                         rocsparse_operation            trans_B,
                                                         rocsparse_int                  m,
                                                         rocsparse_int                  nrhs,
                                                         rocsparse_int                  nnz,
                                                         const rocsparse_float_complex* alpha,
                                                         const rocsparse_mat_descr      descr,
                                                         const rocsparse_float_complex* csr_val,
                                                         const rocsparse_int*           csr_row_ptr,
                                                         const rocsparse_int*           csr_col_ind,
                                                         const rocsparse_float_complex* B,
                                                         rocsparse_int                  ldb,
                                                         rocsparse_mat_info             info,
                                                         rocsparse_solve_policy         policy,
                                                         size_t*                        buffer_size)
{
    return rocsparse_csrsm_buffer_size_template(handle,
                                                trans_A,
                                                trans_B,
                                                m,
                                                nrhs,
                                                nnz,
                                                alpha,
                                                descr,
                                                csr_val,
                                                csr_row_ptr,
                                                csr_col_ind,
                                                B,
                                                ldb,
                                                info,
                                                policy,
                                                buffer_size);
}

extern "C" rocsparse_status rocsparse_zcsrsm_buffer_size(rocsparse_handle                handle,
                                                         rocsparse_operation             trans_A,
                                                         rocsparse_operation             trans_B,
                                                         rocsparse_int                   m,
                                                         rocsparse_int                   nrhs,
                                                         rocsparse_int                   nnz,
                                                         const rocsparse_double_complex* alpha,
                                                         const rocsparse_mat_descr       descr,
                                                         const rocsparse_double_complex* csr_val,
                                                         const rocsparse_int* csr_row_ptr,
                                                         const rocsparse_int* csr_col_ind,
                                                         const rocsparse_double_complex* B,
                                                         rocsparse_int                   ldb,
                                                         rocsparse_mat_info              info,
                                                         rocsparse_solve_policy          policy,
                                                         size_t* buffer_size)
{
    return rocsparse_csrsm_buffer_size_template(handle,
                                                trans_A,
                                                trans_B,
                                                m,
                                                nrhs,
                                                nnz,
                                                alpha,
                                                descr,
                                                csr_val,
                                                csr_row_ptr,
                                                csr_col_ind,
                                                B,
                                                ldb,
                                                info,
                                                policy,
                                                buffer_size);
}

extern "C" rocsparse_status rocsparse_scsrsm_analysis(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      rocsparse_int             m,
                                                      rocsparse_int             nrhs,
                                                      rocsparse_int             nnz,
                                                      const float*              alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const float*              csr_val,
                                                      const rocsparse_int*      csr_row_ptr,
                                                      const rocsparse_int*      csr_col_ind,
                                                      const float*              B,
                                                      rocsparse_int             ldb,
                                                      rocsparse_mat_info        info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy    solve,
                                                      void*                     temp_buffer)
{
    return rocsparse_csrsm_analysis_template(handle,
                                             trans_A,
                                             trans_B,
                                             m,
                                             nrhs,
                                             nnz,
                                             alpha,
                                             descr,
                                             csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             B,
                                             ldb,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrsm_analysis(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      rocsparse_int             m,
                                                      rocsparse_int             nrhs,
                                                      rocsparse_int             nnz,
                                                      const double*             alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const double*             csr_val,
                                                      const rocsparse_int*      csr_row_ptr,
                                                      const rocsparse_int*      csr_col_ind,
                                                      const double*             B,
                                                      rocsparse_int             ldb,
                                                      rocsparse_mat_info        info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy    solve,
                                                      void*                     temp_buffer)
{
    return rocsparse_csrsm_analysis_template(handle,
                                             trans_A,
                                             trans_B,
                                             m,
                                             nrhs,
                                             nnz,
                                             alpha,
                                             descr,
                                             csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             B,
                                             ldb,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_ccsrsm_analysis(rocsparse_handle               handle,
                                                      rocsparse_operation            trans_A,
                                                      rocsparse_operation            trans_B,
                                                      rocsparse_int                  m,
                                                      rocsparse_int                  nrhs,
                                                      rocsparse_int                  nnz,
                                                      const rocsparse_float_complex* alpha,
                                                      const rocsparse_mat_descr      descr,
                                                      const rocsparse_float_complex* csr_val,
                                                      const rocsparse_int*           csr_row_ptr,
                                                      const rocsparse_int*           csr_col_ind,
                                                      const rocsparse_float_complex* B,
                                                      rocsparse_int                  ldb,
                                                      rocsparse_mat_info             info,
                                                      rocsparse_analysis_policy      analysis,
                                                      rocsparse_solve_policy         solve,
                                                      void*                          temp_buffer)
{
    return rocsparse_csrsm_analysis_template(handle,
                                             trans_A,
                                             trans_B,
                                             m,
                                             nrhs,
                                             nnz,
                                             alpha,
                                             descr,
                                             csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             B,
                                             ldb,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_zcsrsm_analysis(rocsparse_handle                handle,
                                                      rocsparse_operation             trans_A,
                                                      rocsparse_operation             trans_B,
                                                      rocsparse_int                   m,
                                                      rocsparse_int                   nrhs,
                                                      rocsparse_int                   nnz,
                                                      const rocsparse_double_complex* alpha,
                                                      const rocsparse_mat_descr       descr,
                                                      const rocsparse_double_complex* csr_val,
                                                      const rocsparse_int*            csr_row_ptr,
                                                      const rocsparse_int*            csr_col_ind,
                                                      const rocsparse_double_complex* B,
                                                      rocsparse_int                   ldb,
                                                      rocsparse_mat_info              info,
                                                      rocsparse_analysis_policy       analysis,
                                                      rocsparse_solve_policy          solve,
                                                      void*                           temp_buffer)
{
    return rocsparse_csrsm_analysis_template(handle,
                                             trans_A,
                                             trans_B,
                                             m,
                                             nrhs,
                                             nnz,
                                             alpha,
                                             descr,
                                             csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             B,
                                             ldb,
                                             info,
                                             analysis,
                                             solve,
                                             temp_buffer);
}

extern "C" rocsparse_status rocsparse_csrsm_clear(rocsparse_handle handle, rocsparse_mat_info info)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle, "rocsparse_csrsm_clear", (const void*&)info);

    // Clear csrsm meta data (this includes lower, upper and their transposed equivalents
    if(!rocsparse_check_csrtr_shared(info, info->csrsm_lower_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsm_lower_info));
    }
    if(!rocsparse_check_csrtr_shared(info, info->csrsm_upper_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsm_upper_info));
    }

    info->csrsm_lower_info = nullptr;
    info->csrsm_upper_info = nullptr;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrsm_solve(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nrhs,
                                                   rocsparse_int             nnz,
                                                   const float*              alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const float*              csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   float*                    B,
                                                   rocsparse_int             ldb,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   void*                     temp_buffer)
{
    return rocsparse_csrsm_solve_template(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          nrhs,
                                          nnz,
                                          alpha,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          B,
                                          ldb,
                                          info,
                                          policy,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrsm_solve(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nrhs,
                                                   rocsparse_int             nnz,
                                                   const double*             alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const double*             csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   double*                   B,
                                                   rocsparse_int             ldb,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   void*                     temp_buffer)
{
    return rocsparse_csrsm_solve_template(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          nrhs,
                                          nnz,
                                          alpha,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          B,
                                          ldb,
                                          info,
                                          policy,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_ccsrsm_solve(rocsparse_handle               handle,
                                                   rocsparse_operation            trans_A,
                                                   rocsparse_operation            trans_B,
                                                   rocsparse_int                  m,
                                                   rocsparse_int                  nrhs,
                                                   rocsparse_int                  nnz,
                                                   const rocsparse_float_complex* alpha,
                                                   const rocsparse_mat_descr      descr,
                                                   const rocsparse_float_complex* csr_val,
                                                   const rocsparse_int*           csr_row_ptr,
                                                   const rocsparse_int*           csr_col_ind,
                                                   rocsparse_float_complex*       B,
                                                   rocsparse_int                  ldb,
                                                   rocsparse_mat_info             info,
                                                   rocsparse_solve_policy         policy,
                                                   void*                          temp_buffer)
{
    return rocsparse_csrsm_solve_template(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          nrhs,
                                          nnz,
                                          alpha,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          B,
                                          ldb,
                                          info,
                                          policy,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_zcsrsm_solve(rocsparse_handle                handle,
                                                   rocsparse_operation             trans_A,
                                                   rocsparse_operation             trans_B,
                                                   rocsparse_int                   m,
                                                   rocsparse_int                   nrhs,
                                                   rocsparse_int                   nnz,
                                                   const rocsparse_double_complex* alpha,
                                                   const rocsparse_mat_descr       descr,
                                                   const rocsparse_double_complex* csr_val,
                                                   const rocsparse_int*            csr_row_ptr,
                                                   const rocsparse_int*            csr_col_ind,
                                                   rocsparse_double_complex*       B,
                                                   rocsparse_int                   ldb,
                                                   rocsparse_mat_info              info,
                                                   rocsparse_solve_policy          policy,
                                                   void*                           temp_buffer)
{
    return rocsparse_csrsm_solve_template(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          nrhs,
                                          nnz,
                                          alpha,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          B,
                                          ldb,
                                          info,
                                          policy,
                                          temp_buffer);
}

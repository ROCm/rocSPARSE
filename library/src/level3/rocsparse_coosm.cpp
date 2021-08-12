/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include "rocsparse_coosm.hpp"
#include "definitions.h"
#include "rocsparse_csrsm.hpp"
#include "utility.h"
#include <rocprim/rocprim.hpp>

#include "../conversion/rocsparse_coo2csr.hpp"

template <typename I, typename T>
rocsparse_status rocsparse_coosm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      I                         m,
                                                      I                         nrhs,
                                                      I                         nnz,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  coo_val,
                                                      const I*                  coo_row_ind,
                                                      const I*                  coo_col_ind,
                                                      const T*                  B,
                                                      I                         ldb,
                                                      rocsparse_mat_info        info,
                                                      rocsparse_solve_policy    policy,
                                                      size_t*                   buffer_size)
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
              replaceX<T>("rocsparse_Xcoosm_buffer_size"),
              trans_A,
              trans_B,
              m,
              nrhs,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)B,
              ldb,
              (const void*&)info,
              policy,
              (const void*&)buffer_size);

    // Check index base
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check operation type
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nrhs < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(trans_B == rocsparse_operation_none && ldb < m)
    {
        return rocsparse_status_invalid_size;
    }
    else if((trans_B == rocsparse_operation_transpose
             || trans_B == rocsparse_operation_conjugate_transpose)
            && ldb < nrhs)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid buffer_size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || nrhs == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(coo_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(coo_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(coo_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Call CSR buffer size
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsm_buffer_size_template(handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   m,
                                                                   nrhs,
                                                                   nnz,
                                                                   alpha,
                                                                   descr,
                                                                   coo_val,
                                                                   coo_row_ind,
                                                                   coo_col_ind,
                                                                   B,
                                                                   ldb,
                                                                   info,
                                                                   policy,
                                                                   buffer_size));

    // For coosm we first convert from COO to CSR format.
    *buffer_size += sizeof(I) * (m / 256 + 1) * 256;

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse_coosm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   I                         m,
                                                   I                         nrhs,
                                                   I                         nnz,
                                                   const T*                  alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  coo_val,
                                                   const I*                  coo_row_ind,
                                                   const I*                  coo_col_ind,
                                                   const T*                  B,
                                                   I                         ldb,
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
              replaceX<T>("rocsparse_Xcoosm_analysis"),
              trans_A,
              trans_B,
              m,
              nrhs,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)B,
              ldb,
              (const void*&)info,
              analysis,
              solve,
              (const void*&)temp_buffer);

    // Check operation type
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
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
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nrhs < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(coo_row_ind == nullptr || coo_col_ind == nullptr || coo_val == nullptr || B == nullptr
       || alpha == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // convert to csr
    I* csr_row_ptr = reinterpret_cast<I*>(ptr);
    ptr += sizeof(I) * (m / 256 + 1) * 256;

    const I* csr_col_ind = coo_col_ind;
    const T* csr_val     = coo_val;

    // Create column pointers
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_coo2csr_template(handle, coo_row_ind, nnz, m, csr_row_ptr, descr->base));

    // Call CSR analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsm_analysis_template(handle,
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
                                                                ptr));

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse_coosm_solve_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                I                         m,
                                                I                         nrhs,
                                                I                         nnz,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  coo_val,
                                                const I*                  coo_row_ind,
                                                const I*                  coo_col_ind,
                                                T*                        B,
                                                I                         ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_solve_policy    policy,
                                                void*                     temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr || info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcoosm_solve"),
              trans_A,
              trans_B,
              m,
              nrhs,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)B,
              ldb,
              (const void*&)info,
              policy,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f coosm -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha_device_host));

    // Check operation type
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || nrhs < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(coo_val == nullptr || coo_row_ind == nullptr || coo_col_ind == nullptr
       || alpha_device_host == nullptr || B == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // convert to csr
    I* csr_row_ptr = reinterpret_cast<I*>(ptr);
    ptr += sizeof(I) * (m / 256 + 1) * 256;

    const I* csr_col_ind = coo_col_ind;
    const T* csr_val     = coo_val;

    return rocsparse_csrsm_solve_template(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          nrhs,
                                          nnz,
                                          alpha_device_host,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          B,
                                          ldb,
                                          info,
                                          policy,
                                          ptr);
}

#define INSTANTIATE(ITYPE, TTYPE)                                   \
    template rocsparse_status rocsparse_coosm_buffer_size_template( \
        rocsparse_handle          handle,                           \
        rocsparse_operation       trans_A,                          \
        rocsparse_operation       trans_B,                          \
        ITYPE                     m,                                \
        ITYPE                     nrhs,                             \
        ITYPE                     nnz,                              \
        const TTYPE*              alpha,                            \
        const rocsparse_mat_descr descr,                            \
        const TTYPE*              coo_val,                          \
        const ITYPE*              coo_row_ind,                      \
        const ITYPE*              coo_col_ind,                      \
        const TTYPE*              B,                                \
        ITYPE                     ldb,                              \
        rocsparse_mat_info        info,                             \
        rocsparse_solve_policy    policy,                           \
        size_t*                   buffer_size);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                \
    template rocsparse_status rocsparse_coosm_analysis_template( \
        rocsparse_handle          handle,                        \
        rocsparse_operation       trans_A,                       \
        rocsparse_operation       trans_B,                       \
        ITYPE                     m,                             \
        ITYPE                     nrhs,                          \
        ITYPE                     nnz,                           \
        const TTYPE*              alpha,                         \
        const rocsparse_mat_descr descr,                         \
        const TTYPE*              coo_val,                       \
        const ITYPE*              coo_row_ind,                   \
        const ITYPE*              coo_col_ind,                   \
        const TTYPE*              B,                             \
        ITYPE                     ldb,                           \
        rocsparse_mat_info        info,                          \
        rocsparse_analysis_policy analysis,                      \
        rocsparse_solve_policy    solve,                         \
        void*                     temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                                                   \
    template rocsparse_status rocsparse_coosm_solve_template(rocsparse_handle          handle,      \
                                                             rocsparse_operation       trans_A,     \
                                                             rocsparse_operation       trans_B,     \
                                                             ITYPE                     m,           \
                                                             ITYPE                     nrhs,        \
                                                             ITYPE                     nnz,         \
                                                             const TTYPE*              alpha,       \
                                                             const rocsparse_mat_descr descr,       \
                                                             const TTYPE*              coo_val,     \
                                                             const ITYPE*              coo_row_ind, \
                                                             const ITYPE*              coo_col_ind, \
                                                             TTYPE*                    B,           \
                                                             ITYPE                     ldb,         \
                                                             rocsparse_mat_info        info,        \
                                                             rocsparse_solve_policy    policy,      \
                                                             void*                     temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

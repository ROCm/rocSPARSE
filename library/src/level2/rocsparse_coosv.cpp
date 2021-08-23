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

#include "rocsparse_coosv.hpp"
#include "definitions.h"
#include "rocsparse_csrsv.hpp"
#include "utility.h"
#include <rocprim/rocprim.hpp>

#include "../conversion/rocsparse_coo2csr.hpp"

template <typename I, typename T>
rocsparse_status rocsparse_coosv_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans,
                                                      I                         m,
                                                      I                         nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  coo_val,
                                                      const I*                  coo_row_ind,
                                                      const I*                  coo_col_ind,
                                                      rocsparse_mat_info        info,
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
              replaceX<T>("rocsparse_Xcoosv_buffer_size"),
              trans,
              m,
              nnz,
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)info,
              (const void*&)buffer_size);

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid buffer_size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Call CSR buffer size
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size_template(
        handle, trans, m, nnz, descr, coo_val, coo_row_ind, coo_col_ind, info, buffer_size));

    // For coosv we first convert from COO to CSR format.
    *buffer_size += sizeof(I) * (m / 256 + 1) * 256;

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                   \
    template rocsparse_status rocsparse_coosv_buffer_size_template( \
        rocsparse_handle          handle,                           \
        rocsparse_operation       trans,                            \
        ITYPE                     m,                                \
        ITYPE                     nnz,                              \
        const rocsparse_mat_descr descr,                            \
        const TTYPE*              coo_val,                          \
        const ITYPE*              coo_row_ind,                      \
        const ITYPE*              coo_col_ind,                      \
        rocsparse_mat_info        info,                             \
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

template <typename I, typename T>
rocsparse_status rocsparse_coosv_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   I                         m,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  coo_val,
                                                   const I*                  coo_row_ind,
                                                   const I*                  coo_col_ind,
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
              replaceX<T>("rocsparse_Xcoosv_analysis"),
              trans,
              m,
              nnz,
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
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
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
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
    if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
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
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsv_analysis_template(handle,
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
                                                                ptr));

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                \
    template rocsparse_status rocsparse_coosv_analysis_template( \
        rocsparse_handle          handle,                        \
        rocsparse_operation       trans,                         \
        ITYPE                     m,                             \
        ITYPE                     nnz,                           \
        const rocsparse_mat_descr descr,                         \
        const TTYPE*              coo_val,                       \
        const ITYPE*              coo_row_ind,                   \
        const ITYPE*              coo_col_ind,                   \
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

template <typename I, typename T>
rocsparse_status rocsparse_coosv_solve_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                I                         m,
                                                I                         nnz,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  coo_val,
                                                const I*                  coo_row_ind,
                                                const I*                  coo_col_ind,
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
              replaceX<T>("rocsparse_Xcoosv"),
              trans,
              m,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)coo_val,
              (const void*&)coo_row_ind,
              (const void*&)coo_col_ind,
              (const void*&)info,
              (const void*&)x,
              (const void*&)y,
              policy,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f coosv -r",
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

    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
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
    if(alpha_device_host == nullptr || x == nullptr || y == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    I* csr_row_ptr = reinterpret_cast<I*>(ptr);
    ptr += sizeof(I) * (m / 256 + 1) * 256;

    const I* csr_col_ind = coo_col_ind;
    const T* csr_val     = coo_val;

    return rocsparse_csrsv_solve_template(handle,
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
                                          ptr);
}

#define INSTANTIATE(ITYPE, TTYPE)                                           \
    template rocsparse_status rocsparse_coosv_solve_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans,                                    \
        ITYPE                     m,                                        \
        ITYPE                     nnz,                                      \
        const TTYPE*              alpha_device_host,                        \
        const rocsparse_mat_descr descr,                                    \
        const TTYPE*              coo_val,                                  \
        const ITYPE*              coo_row_ind,                              \
        const ITYPE*              coo_col_ind,                              \
        rocsparse_mat_info        info,                                     \
        const TTYPE*              x,                                        \
        TTYPE*                    y,                                        \
        rocsparse_solve_policy    policy,                                   \
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

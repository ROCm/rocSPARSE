/*! \file */
/* ************************************************************************
* Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

#include <algorithm>

#include "../conversion/rocsparse_csr2coo.hpp"
#include "rocsparse_csrmm.hpp"

#include "definitions.h"
#include "utility.h"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmm_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_spmm_alg        alg,
                                                      J                         m,
                                                      J                         n,
                                                      J                         k,
                                                      I                         nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
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

    // Logging
    log_trace(handle,
              "rocsparse_csrmm_buffer_size",
              trans_A,
              m,
              n,
              k,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)buffer_size);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check index base
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || csr_col_ind == nullptr || csr_val == nullptr
       || buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(alg == rocsparse_spmm_alg_csr_merge && trans_A == rocsparse_operation_none)
    {
        *buffer_size = sizeof(J) * ((nnz - 1) / 256 + 1) * 256;
    }
    else
    {
        *buffer_size = 4;
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmm_analysis_template(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_spmm_alg        alg,
                                                   J                         m,
                                                   J                         n,
                                                   J                         k,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
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

    // Logging
    log_trace(handle,
              "rocsparse_csrmm_analysis",
              trans_A,
              m,
              n,
              k,
              nnz,
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check index base
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr || csr_col_ind == nullptr || csr_val == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(alg == rocsparse_spmm_alg_csr_merge && trans_A == rocsparse_operation_none)
    {
        char* ptr         = reinterpret_cast<char*>(temp_buffer);
        J*    csr_row_ind = reinterpret_cast<J*>(ptr);
        ptr += sizeof(J) * ((nnz - 1) / 256 + 1) * 256;

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo_template(handle, csr_row_ptr, nnz, m, csr_row_ind, descr->base));
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_general(rocsparse_handle          handle,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  rocsparse_order           order,
                                                  J                         m,
                                                  J                         n,
                                                  J                         k,
                                                  I                         nnz,
                                                  U                         alpha_device_host,
                                                  const rocsparse_mat_descr descr,
                                                  const T*                  csr_val,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  const T*                  B,
                                                  J                         ldb,
                                                  U                         beta_device_host,
                                                  T*                        C,
                                                  J                         ldc);

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_row_split(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_order           order,
                                                    J                         m,
                                                    J                         n,
                                                    J                         k,
                                                    I                         nnz,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    const T*                  B,
                                                    J                         ldb,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    J                         ldc);

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_merge(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_order           order,
                                                J                         m,
                                                J                         n,
                                                J                         k,
                                                I                         nnz,
                                                U                         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const T*                  B,
                                                J                         ldb,
                                                U                         beta_device_host,
                                                T*                        C,
                                                J                         ldc,
                                                void*                     temp_buffer);

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_dispatch(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order,
                                                   rocsparse_spmm_alg        alg,
                                                   J                         m,
                                                   J                         n,
                                                   J                         k,
                                                   I                         nnz,
                                                   U                         alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  B,
                                                   J                         ldb,
                                                   U                         beta_device_host,
                                                   T*                        C,
                                                   J                         ldc,
                                                   void*                     temp_buffer)
{
    if(alg == rocsparse_spmm_alg_csr || trans_A != rocsparse_operation_none)
    {
        return rocsparse_csrmm_template_general(handle,
                                                trans_A,
                                                trans_B,
                                                order,
                                                m,
                                                n,
                                                k,
                                                nnz,
                                                alpha_device_host,
                                                descr,
                                                csr_val,
                                                csr_row_ptr,
                                                csr_col_ind,
                                                B,
                                                ldb,
                                                beta_device_host,
                                                C,
                                                ldc);
    }
    else if(alg == rocsparse_spmm_alg_csr_row_split)
    {
        return rocsparse_csrmm_template_row_split(handle,
                                                  trans_A,
                                                  trans_B,
                                                  order,
                                                  m,
                                                  n,
                                                  k,
                                                  nnz,
                                                  alpha_device_host,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  B,
                                                  ldb,
                                                  beta_device_host,
                                                  C,
                                                  ldc);
    }
    else if(alg == rocsparse_spmm_alg_csr_merge)
    {
        return rocsparse_csrmm_template_merge(handle,
                                              trans_A,
                                              trans_B,
                                              order,
                                              m,
                                              n,
                                              k,
                                              nnz,
                                              alpha_device_host,
                                              descr,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              B,
                                              ldb,
                                              beta_device_host,
                                              C,
                                              ldc,
                                              temp_buffer);
    }

    return rocsparse_status_not_implemented;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrmm_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_order           order_B,
                                          rocsparse_order           order_C,
                                          rocsparse_spmm_alg        alg,
                                          J                         m,
                                          J                         n,
                                          J                         k,
                                          I                         nnz,
                                          const T*                  alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const T*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const T*                  B,
                                          J                         ldb,
                                          const T*                  beta_device_host,
                                          T*                        C,
                                          J                         ldc,
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

    // Logging TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrmm"),
              trans_A,
              trans_B,
              m,
              n,
              k,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)B,
              ldb,
              LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
              (const void*&)C,
              ldc);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    if(order_B != order_C)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m < 0 || n < 0 || k < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        return rocsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if(alpha_device_host == nullptr || beta_device_host == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_host
       && *alpha_device_host == static_cast<T>(0) && *beta_device_host == static_cast<T>(1))
    {
        return rocsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || B == nullptr
       || C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check leading dimension of B
    J one = 1;
    if(trans_B == rocsparse_operation_none)
    {
        if(ldb < std::max(one, order_B == rocsparse_order_column ? k : n))
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldb < std::max(one, order_B == rocsparse_order_column ? n : k))
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(ldc < std::max(one, order_C == rocsparse_order_column ? m : n))
    {
        return rocsparse_status_invalid_size;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_csrmm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 alg,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 alpha_device_host,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 B,
                                                 ldb,
                                                 beta_device_host,
                                                 C,
                                                 ldc,
                                                 temp_buffer);
    }
    else
    {
        return rocsparse_csrmm_template_dispatch(handle,
                                                 trans_A,
                                                 trans_B,
                                                 order_B,
                                                 alg,
                                                 m,
                                                 n,
                                                 k,
                                                 nnz,
                                                 *alpha_device_host,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 B,
                                                 ldb,
                                                 *beta_device_host,
                                                 C,
                                                 ldc,
                                                 temp_buffer);
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template rocsparse_status rocsparse_csrmm_buffer_size_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                                \
        rocsparse_operation       trans_A,                                               \
        rocsparse_spmm_alg        alg,                                                   \
        JTYPE                     m,                                                     \
        JTYPE                     n,                                                     \
        JTYPE                     k,                                                     \
        ITYPE                     nnz,                                                   \
        const rocsparse_mat_descr descr,                                                 \
        const TTYPE*              csr_val,                                               \
        const ITYPE*              csr_row_ptr,                                           \
        const JTYPE*              csr_col_ind,                                           \
        size_t*                   buffer_size);

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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                              \
    template rocsparse_status rocsparse_csrmm_analysis_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans_A,                                            \
        rocsparse_spmm_alg        alg,                                                \
        JTYPE                     m,                                                  \
        JTYPE                     n,                                                  \
        JTYPE                     k,                                                  \
        ITYPE                     nnz,                                                \
        const rocsparse_mat_descr descr,                                              \
        const TTYPE*              csr_val,                                            \
        const ITYPE*              csr_row_ptr,                                        \
        const JTYPE*              csr_col_ind,                                        \
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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                     \
    template rocsparse_status rocsparse_csrmm_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_operation       trans_A,                                   \
        rocsparse_operation       trans_B,                                   \
        rocsparse_order           order_B,                                   \
        rocsparse_order           order_C,                                   \
        rocsparse_spmm_alg        alg,                                       \
        JTYPE                     m,                                         \
        JTYPE                     n,                                         \
        JTYPE                     k,                                         \
        ITYPE                     nnz,                                       \
        const TTYPE*              alpha_device_host,                         \
        const rocsparse_mat_descr descr,                                     \
        const TTYPE*              csr_val,                                   \
        const ITYPE*              csr_row_ptr,                               \
        const JTYPE*              csr_col_ind,                               \
        const TTYPE*              B,                                         \
        JTYPE                     ldb,                                       \
        const TTYPE*              beta_device_host,                          \
        TTYPE*                    C,                                         \
        JTYPE                     ldc,                                       \
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
                                     rocsparse_operation       trans_A,     \
                                     rocsparse_operation       trans_B,     \
                                     rocsparse_int             m,           \
                                     rocsparse_int             n,           \
                                     rocsparse_int             k,           \
                                     rocsparse_int             nnz,         \
                                     const TYPE*               alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TYPE*               csr_val,     \
                                     const rocsparse_int*      csr_row_ptr, \
                                     const rocsparse_int*      csr_col_ind, \
                                     const TYPE*               B,           \
                                     rocsparse_int             ldb,         \
                                     const TYPE*               beta,        \
                                     TYPE*                     C,           \
                                     rocsparse_int             ldc)         \
    {                                                                       \
        return rocsparse_csrmm_template(handle,                             \
                                        trans_A,                            \
                                        trans_B,                            \
                                        rocsparse_order_column,             \
                                        rocsparse_order_column,             \
                                        rocsparse_spmm_alg_csr,             \
                                        m,                                  \
                                        n,                                  \
                                        k,                                  \
                                        nnz,                                \
                                        alpha,                              \
                                        descr,                              \
                                        csr_val,                            \
                                        csr_row_ptr,                        \
                                        csr_col_ind,                        \
                                        B,                                  \
                                        ldb,                                \
                                        beta,                               \
                                        C,                                  \
                                        ldc,                                \
                                        nullptr);                           \
    }

C_IMPL(rocsparse_scsrmm, float);
C_IMPL(rocsparse_dcsrmm, double);
C_IMPL(rocsparse_ccsrmm, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrmm, rocsparse_double_complex);

#undef C_IMPL

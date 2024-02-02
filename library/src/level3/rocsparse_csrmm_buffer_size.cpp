/*! \file */
/* ************************************************************************
* Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "utility.h"

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_csrmm_alg value_)
{
    switch(value_)
    {
    case rocsparse_csrmm_alg_default:
    case rocsparse_csrmm_alg_row_split:
    case rocsparse_csrmm_alg_merge:
    {
        return false;
    }
    }
    return true;
};

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_buffer_size_template_merge(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_csrmm_alg       alg,
                                                      J                         m,
                                                      J                         n,
                                                      J                         k,
                                                      I                         nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
                                                      size_t*                   buffer_size);

    template <typename T, typename I, typename J, typename A>
    static rocsparse_status csrmm_buffer_size_core(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_csrmm_alg       alg,
                                                   J                         m,
                                                   J                         n,
                                                   J                         k,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const A*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   size_t*                   buffer_size)
    {
        switch(alg)
        {
        case rocsparse_csrmm_alg_merge:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_template_merge<T>(handle,
                                                                                     trans_A,
                                                                                     alg,
                                                                                     m,
                                                                                     n,
                                                                                     k,
                                                                                     nnz,
                                                                                     descr,
                                                                                     csr_val,
                                                                                     csr_row_ptr,
                                                                                     csr_col_ind,
                                                                                     buffer_size));
            return rocsparse_status_success;
        }

        case rocsparse_csrmm_alg_default:
        case rocsparse_csrmm_alg_row_split:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    template <typename T, typename I, typename J, typename A>
    static rocsparse_status csrmm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_csrmm_alg       alg,
                                                          J                         m,
                                                          J                         n,
                                                          J                         k,
                                                          I                         nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const A*                  csr_val,
                                                          const I*                  csr_row_ptr,
                                                          const J*                  csr_col_ind,
                                                          size_t*                   buffer_size)
    {
        if(m == 0 || n == 0 || k == 0)
        {
            buffer_size[0] = 0;
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename T, typename I, typename J, typename A>
    static rocsparse_status csrmm_buffer_size_checkarg(rocsparse_handle          handle, //0
                                                       rocsparse_operation       trans_A, //1
                                                       rocsparse_csrmm_alg       alg, //2
                                                       J                         m, //3
                                                       J                         n, //4
                                                       J                         k, //5
                                                       I                         nnz, //6
                                                       const rocsparse_mat_descr descr, //7
                                                       const A*                  csr_val, //8
                                                       const I*                  csr_row_ptr, //9
                                                       const J*                  csr_col_ind, //10
                                                       size_t*                   buffer_size) //11
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, alg);
        ROCSPARSE_CHECKARG_POINTER(7, descr);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, n);
        ROCSPARSE_CHECKARG_SIZE(5, k);
        ROCSPARSE_CHECKARG_SIZE(6, nnz);

        ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, csr_col_ind);

        const rocsparse_status status = rocsparse::csrmm_buffer_size_quickreturn<T>(handle,
                                                                                    trans_A,
                                                                                    alg,
                                                                                    m,
                                                                                    n,
                                                                                    k,
                                                                                    nnz,
                                                                                    descr,
                                                                                    csr_val,
                                                                                    csr_row_ptr,
                                                                                    csr_col_ind,
                                                                                    buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(11, buffer_size);
        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J, typename A>
rocsparse_status rocsparse::csrmm_buffer_size_template(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_csrmm_alg       alg,
                                                       J                         m,
                                                       J                         n,
                                                       J                         k,
                                                       I                         nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const A*                  csr_val,
                                                       const I*                  csr_row_ptr,
                                                       const J*                  csr_col_ind,
                                                       size_t*                   buffer_size)
{

    const rocsparse_status status = rocsparse::csrmm_buffer_size_quickreturn<T>(
        handle, trans_A, alg, m, n, k, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_core<T>(
        handle, trans_A, alg, m, n, k, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size));

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_buffer_size_impl(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_csrmm_alg       alg,
                                            J                         m,
                                            J                         n,
                                            J                         k,
                                            I                         nnz,
                                            const rocsparse_mat_descr descr,
                                            const A*                  csr_val,
                                            const I*                  csr_row_ptr,
                                            const J*                  csr_col_ind,
                                            size_t*                   buffer_size)
    {

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

        const rocsparse_status status = rocsparse::csrmm_buffer_size_checkarg<T>(handle,
                                                                                 trans_A,
                                                                                 alg,
                                                                                 m,
                                                                                 n,
                                                                                 k,
                                                                                 nnz,
                                                                                 descr,
                                                                                 csr_val,
                                                                                 csr_row_ptr,
                                                                                 csr_col_ind,
                                                                                 buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_buffer_size_core<T>(handle,
                                                                       trans_A,
                                                                       alg,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       nnz,
                                                                       descr,
                                                                       csr_val,
                                                                       csr_row_ptr,
                                                                       csr_col_ind,
                                                                       buffer_size));

        return rocsparse_status_success;
    }
}

#define INSTANTIATE_BUFFER_SIZE(TTYPE, ITYPE, JTYPE, ATYPE)                 \
    template rocsparse_status rocsparse::csrmm_buffer_size_template<TTYPE>( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans_A,                                  \
        rocsparse_csrmm_alg       alg,                                      \
        JTYPE                     m,                                        \
        JTYPE                     n,                                        \
        JTYPE                     k,                                        \
        ITYPE                     nnz,                                      \
        const rocsparse_mat_descr descr,                                    \
        const ATYPE*              csr_val,                                  \
        const ITYPE*              csr_row_ptr,                              \
        const JTYPE*              csr_col_ind,                              \
        size_t*                   buffer_size);

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, int8_t);
#undef INSTANTIATE_BUFFER_SIZE

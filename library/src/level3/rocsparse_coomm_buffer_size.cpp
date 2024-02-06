/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "rocsparse_coomm.hpp"
#include "utility.h"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_coomm_alg value_)
{
    switch(value_)
    {
    case rocsparse_coomm_alg_default:
    case rocsparse_coomm_alg_atomic:
    case rocsparse_coomm_alg_segmented:
    case rocsparse_coomm_alg_segmented_atomic:
    {
        return false;
    }
    }
    return true;
};

namespace rocsparse
{
    template <typename T, typename I, typename A>
    rocsparse_status coomm_buffer_size_template_segmented(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          I                         m,
                                                          I                         n,
                                                          I                         k,
                                                          int64_t                   nnz,
                                                          I                         batch_count,
                                                          const rocsparse_mat_descr descr,
                                                          const A*                  coo_val,
                                                          const I*                  coo_row_ind,
                                                          const I*                  coo_col_ind,
                                                          size_t*                   buffer_size);

    template <typename T, typename I, typename A>
    static rocsparse_status coomm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_coomm_alg       alg,
                                                          I                         m,
                                                          I                         n,
                                                          I                         k,
                                                          int64_t                   nnz,
                                                          I                         batch_count,
                                                          const rocsparse_mat_descr descr,
                                                          const A*                  coo_val,
                                                          const I*                  coo_row_ind,
                                                          const I*                  coo_col_ind,
                                                          size_t*                   buffer_size)
    {
        // Quick return if possible
        if(m == 0 || n == 0 || k == 0)
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename T, typename I, typename A>
    static rocsparse_status coomm_buffer_size_checkarg(rocsparse_handle          handle, //0
                                                       rocsparse_operation       trans_A, //1
                                                       rocsparse_coomm_alg       alg, //2
                                                       I                         m, //3
                                                       I                         n, //4
                                                       I                         k, //5
                                                       int64_t                   nnz, //6
                                                       I                         batch_count, //7
                                                       const rocsparse_mat_descr descr, //8
                                                       const A*                  coo_val, //9
                                                       const I*                  coo_row_ind, //10
                                                       const I*                  coo_col_ind, //11
                                                       size_t*                   buffer_size) //12
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, alg);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, n);
        ROCSPARSE_CHECKARG_SIZE(5, k);
        ROCSPARSE_CHECKARG_SIZE(6, nnz);
        ROCSPARSE_CHECKARG_SIZE(7, batch_count);
        ROCSPARSE_CHECKARG_POINTER(8, descr);
        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_ARRAY(9, nnz, coo_val);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(11, nnz, coo_col_ind);

        const rocsparse_status status = rocsparse::coomm_buffer_size_quickreturn<T>(handle,
                                                                                    trans_A,
                                                                                    alg,
                                                                                    m,
                                                                                    n,
                                                                                    k,
                                                                                    nnz,
                                                                                    batch_count,
                                                                                    descr,
                                                                                    coo_val,
                                                                                    coo_row_ind,
                                                                                    coo_col_ind,
                                                                                    buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(12, buffer_size);
        return rocsparse_status_continue;
    }

    template <typename T, typename I, typename A>
    static rocsparse_status coomm_buffer_size_core(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_coomm_alg       alg,
                                                   I                         m,
                                                   I                         n,
                                                   I                         k,
                                                   int64_t                   nnz,
                                                   I                         batch_count,
                                                   const rocsparse_mat_descr descr,
                                                   const A*                  coo_val,
                                                   const I*                  coo_row_ind,
                                                   const I*                  coo_col_ind,
                                                   size_t*                   buffer_size)
    {

        switch(alg)
        {
        case rocsparse_coomm_alg_default:
        case rocsparse_coomm_alg_atomic:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        case rocsparse_coomm_alg_segmented:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::coomm_buffer_size_template_segmented<T>(handle,
                                                                   trans_A,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   nnz,
                                                                   batch_count,
                                                                   descr,
                                                                   coo_val,
                                                                   coo_row_ind,
                                                                   coo_col_ind,
                                                                   buffer_size));
            return rocsparse_status_success;
        }

        case rocsparse_coomm_alg_segmented_atomic:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
}

template <typename T, typename I, typename A>
rocsparse_status rocsparse::coomm_buffer_size_template(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_coomm_alg       alg,
                                                       I                         m,
                                                       I                         n,
                                                       I                         k,
                                                       int64_t                   nnz,
                                                       I                         batch_count,
                                                       const rocsparse_mat_descr descr,
                                                       const A*                  coo_val,
                                                       const I*                  coo_row_ind,
                                                       const I*                  coo_col_ind,
                                                       size_t*                   buffer_size)
{
    const rocsparse_status status = rocsparse::coomm_buffer_size_quickreturn<T>(handle,
                                                                                trans_A,
                                                                                alg,
                                                                                m,
                                                                                n,
                                                                                k,
                                                                                nnz,
                                                                                batch_count,
                                                                                descr,
                                                                                coo_val,
                                                                                coo_row_ind,
                                                                                coo_col_ind,
                                                                                buffer_size);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_buffer_size_core<T>(handle,
                                                                   trans_A,
                                                                   alg,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   nnz,
                                                                   batch_count,
                                                                   descr,
                                                                   coo_val,
                                                                   coo_row_ind,
                                                                   coo_col_ind,
                                                                   buffer_size));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I, typename A>
    rocsparse_status coomm_buffer_size_impl(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_coomm_alg       alg,
                                            I                         m,
                                            I                         n,
                                            I                         k,
                                            int64_t                   nnz,
                                            I                         batch_count,
                                            const rocsparse_mat_descr descr,
                                            const A*                  coo_val,
                                            const I*                  coo_row_ind,
                                            const I*                  coo_col_ind,
                                            size_t*                   buffer_size)
    {

        rocsparse::log_trace(handle,
                             "rocsparse_coomm_buffer_size",
                             trans_A,
                             alg,
                             m,
                             n,
                             k,
                             nnz,
                             (const void*&)descr,
                             (const void*&)coo_val,
                             (const void*&)coo_row_ind,
                             (const void*&)coo_col_ind,
                             (const void*&)buffer_size);

        const rocsparse_status status = rocsparse::coomm_buffer_size_checkarg<T>(handle,
                                                                                 trans_A,
                                                                                 alg,
                                                                                 m,
                                                                                 n,
                                                                                 k,
                                                                                 nnz,
                                                                                 batch_count,
                                                                                 descr,
                                                                                 coo_val,
                                                                                 coo_row_ind,
                                                                                 coo_col_ind,
                                                                                 buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_buffer_size_core<T>(handle,
                                                                       trans_A,
                                                                       alg,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       nnz,
                                                                       batch_count,
                                                                       descr,
                                                                       coo_val,
                                                                       coo_row_ind,
                                                                       coo_col_ind,
                                                                       buffer_size));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE_BUFFER_SIZE(TTYPE, ITYPE, ATYPE)                        \
    template rocsparse_status rocsparse::coomm_buffer_size_template<TTYPE>( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans_A,                                  \
        rocsparse_coomm_alg       alg,                                      \
        ITYPE                     m,                                        \
        ITYPE                     n,                                        \
        ITYPE                     k,                                        \
        int64_t                   nnz,                                      \
        ITYPE                     batch_count,                              \
        const rocsparse_mat_descr descr,                                    \
        const ATYPE*              coo_val,                                  \
        const ITYPE*              coo_row_ind,                              \
        const ITYPE*              coo_col_ind,                              \
        size_t*                   buffer_size);

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int8_t);
#undef INSTANTIATE_BUFFER_SIZE

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

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_analysis_template_merge(rocsparse_handle          handle,
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
                                                   void*                     temp_buffer);

    template <typename T, typename I, typename J, typename A>
    static rocsparse_status csrmm_analysis_core(rocsparse_handle          handle,
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
                                                void*                     temp_buffer)
    {
        switch(alg)
        {
        case rocsparse_csrmm_alg_merge:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis_template_merge<T>(handle,
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
                                                                                  temp_buffer));
            return rocsparse_status_success;
        }

        case rocsparse_csrmm_alg_default:
        case rocsparse_csrmm_alg_row_split:
        {
            return rocsparse_status_success;
        }
        }
    }

    template <typename T, typename I, typename J, typename A>
    static rocsparse_status csrmm_analysis_quickreturn(rocsparse_handle          handle,
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
                                                       void*                     temp_buffer)
    {
        if(m == 0 || n == 0 || k == 0)
        {
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    template <typename T, typename I, typename J, typename A>
    static rocsparse_status csrmm_analysis_checkarg(rocsparse_handle          handle, //0
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
                                                    void*                     temp_buffer) //11
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

        const rocsparse_status status = rocsparse::csrmm_analysis_quickreturn<T>(handle,
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
                                                                                 temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(11, temp_buffer);
        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J, typename A>
rocsparse_status rocsparse::csrmm_analysis_template(rocsparse_handle          handle,
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
                                                    void*                     temp_buffer)
{
    const rocsparse_status status = rocsparse::csrmm_analysis_quickreturn<T>(
        handle, trans_A, alg, m, n, k, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis_core<T>(
        handle, trans_A, alg, m, n, k, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_analysis_impl(rocsparse_handle          handle,
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
                                         void*                     temp_buffer)
    {
        const rocsparse_status status = rocsparse::csrmm_analysis_checkarg<T>(handle,
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
                                                                              temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmm_analysis_core<T>(handle,
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
                                                                    temp_buffer));

        return rocsparse_status_success;
    }
}

#define INSTANTIATE_ANALYSIS(TTYPE, ITYPE, JTYPE, ATYPE)                 \
    template rocsparse_status rocsparse::csrmm_analysis_template<TTYPE>( \
        rocsparse_handle          handle,                                \
        rocsparse_operation       trans_A,                               \
        rocsparse_csrmm_alg       alg,                                   \
        JTYPE                     m,                                     \
        JTYPE                     n,                                     \
        JTYPE                     k,                                     \
        ITYPE                     nnz,                                   \
        const rocsparse_mat_descr descr,                                 \
        const ATYPE*              csr_val,                               \
        const ITYPE*              csr_row_ptr,                           \
        const JTYPE*              csr_col_ind,                           \
        void*                     temp_buffer);

// Uniform precisions
INSTANTIATE_ANALYSIS(float, int32_t, int32_t, float);
INSTANTIATE_ANALYSIS(float, int64_t, int32_t, float);
INSTANTIATE_ANALYSIS(float, int64_t, int64_t, float);
INSTANTIATE_ANALYSIS(double, int32_t, int32_t, double);
INSTANTIATE_ANALYSIS(double, int64_t, int32_t, double);
INSTANTIATE_ANALYSIS(double, int64_t, int64_t, double);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_ANALYSIS(int32_t, int32_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int32_t, int64_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int32_t, int64_t, int64_t, int8_t);
INSTANTIATE_ANALYSIS(float, int32_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(float, int64_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(float, int64_t, int64_t, int8_t);
#undef INSTANTIATE_ANALYSIS

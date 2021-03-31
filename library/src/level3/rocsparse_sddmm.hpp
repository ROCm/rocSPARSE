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

#pragma once
#ifndef ROCSPARSE_SDDMM_HPP
#define ROCSPARSE_SDDMM_HPP

#include "handle.h"

template <rocsparse_format FORMAT, rocsparse_sddmm_alg ALG, typename I, typename J, typename T>
struct rocsparse_sddmm_st
{

    static rocsparse_status buffer_size(rocsparse_handle     handle,
                                        rocsparse_operation  trans_A,
                                        rocsparse_operation  trans_B,
                                        rocsparse_order      order_A,
                                        rocsparse_order      order_B,
                                        J                    m,
                                        J                    n,
                                        J                    k,
                                        I                    nnz,
                                        const T*             alpha,
                                        const T*             A_val,
                                        J                    A_ld,
                                        const T*             B_val,
                                        J                    B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size);

    static rocsparse_status preprocess(rocsparse_handle     handle,
                                       rocsparse_operation  trans_A,
                                       rocsparse_operation  trans_B,
                                       rocsparse_order      order_A,
                                       rocsparse_order      order_B,
                                       J                    m,
                                       J                    n,
                                       J                    k,
                                       I                    nnz,
                                       const T*             alpha,
                                       const T*             A_val,
                                       J                    A_ld,
                                       const T*             B_val,
                                       J                    B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer);

    static rocsparse_status compute(rocsparse_handle     handle,
                                    rocsparse_operation  trans_A,
                                    rocsparse_operation  trans_B,
                                    rocsparse_order      order_A,
                                    rocsparse_order      order_B,
                                    J                    m,
                                    J                    n,
                                    J                    k,
                                    I                    nnz,
                                    const T*             alpha,
                                    const T*             A_val,
                                    J                    A_ld,
                                    const T*             B_val,
                                    J                    B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer);

    static rocsparse_status buffer_size_template(rocsparse_handle            handle,
                                                 rocsparse_operation         trans_A,
                                                 rocsparse_operation         trans_B,
                                                 const void*                 alpha,
                                                 const rocsparse_dnmat_descr mat_A,
                                                 const rocsparse_dnmat_descr mat_B,
                                                 const void*                 beta,
                                                 const rocsparse_spmat_descr mat_C,
                                                 rocsparse_datatype          compute_type,
                                                 rocsparse_sddmm_alg         alg,
                                                 size_t*                     out_buffer_size)
    {
        switch(FORMAT)
        {
        case rocsparse_format_csr:
        case rocsparse_format_coo:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::buffer_size(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->row_data,
                (const J*)mat_C->col_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                out_buffer_size);
        }

        case rocsparse_format_csc:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::buffer_size(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->col_data,
                (const J*)mat_C->row_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                out_buffer_size);
        }
        case rocsparse_format_ell:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::buffer_size(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)nullptr,
                (const J*)mat_C->col_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                out_buffer_size);
        }

        case rocsparse_format_coo_aos:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::buffer_size(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->ind_data,
                (const J*)(((const I*)mat_C->ind_data) + 1),
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                out_buffer_size);
        }
        }
        return rocsparse_status_invalid_value;
    }

    static rocsparse_status preprocess_template(rocsparse_handle            handle,
                                                rocsparse_operation         trans_A,
                                                rocsparse_operation         trans_B,
                                                const void*                 alpha,
                                                const rocsparse_dnmat_descr mat_A,
                                                const rocsparse_dnmat_descr mat_B,
                                                const void*                 beta,
                                                const rocsparse_spmat_descr mat_C,
                                                rocsparse_datatype          compute_type,
                                                rocsparse_sddmm_alg         alg,
                                                void*                       buffer)
    {
        switch(FORMAT)
        {
        case rocsparse_format_csr:
        case rocsparse_format_coo:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::preprocess(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->row_data,
                (const J*)mat_C->col_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        case rocsparse_format_csc:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::preprocess(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->col_data,
                (const J*)mat_C->row_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        case rocsparse_format_ell:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::preprocess(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)nullptr,
                (const J*)mat_C->col_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        case rocsparse_format_coo_aos:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::preprocess(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->ind_data,
                (const J*)(((const I*)mat_C->ind_data) + 1),
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        }
        return rocsparse_status_invalid_value;
    }

    static rocsparse_status compute_template(rocsparse_handle            handle,
                                             rocsparse_operation         trans_A,
                                             rocsparse_operation         trans_B,
                                             const void*                 alpha,
                                             const rocsparse_dnmat_descr mat_A,
                                             const rocsparse_dnmat_descr mat_B,
                                             const void*                 beta,
                                             const rocsparse_spmat_descr mat_C,
                                             rocsparse_datatype          compute_type,
                                             rocsparse_sddmm_alg         alg,
                                             void*                       buffer)
    {
        switch(FORMAT)
        {
        case rocsparse_format_csr:
        case rocsparse_format_coo:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::compute(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->row_data,
                (const J*)mat_C->col_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        case rocsparse_format_csc:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::compute(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->col_data,
                (const J*)mat_C->row_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        case rocsparse_format_ell:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::compute(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)nullptr,
                (const J*)mat_C->col_data,
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        case rocsparse_format_coo_aos:
        {
            return rocsparse_sddmm_st<FORMAT, ALG, I, J, T>::compute(
                handle,
                trans_A,
                trans_B,
                mat_A->order,
                mat_B->order,
                mat_C->rows,
                mat_C->cols,
                (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                mat_C->nnz,
                (const T*)alpha,
                (const T*)mat_A->values,
                mat_A->ld,
                (const T*)mat_B->values,
                mat_B->ld,
                (const T*)beta,
                (const I*)mat_C->ind_data,
                (const J*)(((const I*)mat_C->ind_data) + 1),
                (T*)mat_C->val_data,
                mat_C->idx_base,
                alg,
                buffer);
        }
        }
        return rocsparse_status_invalid_value;
    }
};

#endif // ROCSPARSE_SDDMM_HPP

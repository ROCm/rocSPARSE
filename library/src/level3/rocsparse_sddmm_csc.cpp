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

#include "common.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "rocsparse_sddmm.hpp"
#include "utility.h"

template <typename I, typename J>
rocsparse_status rocsparse_csr2coo_template(rocsparse_handle     handle,
                                            const I*             csr_row_ptr,
                                            I                    nnz,
                                            J                    m,
                                            J*                   coo_row_ind,
                                            rocsparse_index_base idx_base);

template <typename I, typename J, typename T>
struct rocsparse_sddmm_st<rocsparse_format_csc, rocsparse_sddmm_alg_default, I, J, T>
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
                                        size_t*              buffer_size)
    {
        rocsparse_status status
            = rocsparse_sddmm_st<rocsparse_format_coo, rocsparse_sddmm_alg_default, J, J, T>::
                buffer_size(handle,
                            trans_A,
                            trans_B,
                            order_A,
                            order_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            A_val,
                            A_ld,
                            B_val,
                            B_ld,
                            beta,
                            (J*)nullptr,
                            (J*)C_col_data,
                            C_val_data,
                            C_base,
                            alg,
                            buffer_size);

        if(status != rocsparse_status_success)
        {
            return status;
        }

        buffer_size[0] += nnz * sizeof(J);
        return rocsparse_status_success;
    }

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
                                       const I*             C_ptr_data,
                                       const J*             C_ind_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {

        //
        // Compute
        //
        J*    col_data   = (J*)buffer;
        void* coo_buffer = col_data + nnz;

        rocsparse_status status
            = rocsparse_csr2coo_template(handle, C_ptr_data, nnz, n, col_data, C_base);
        if(status != rocsparse_status_success)
        {
            return status;
        }
        return rocsparse_sddmm_st<rocsparse_format_coo, rocsparse_sddmm_alg_default, J, J, T>::
            preprocess(handle,
                       trans_A,
                       trans_B,
                       order_A,
                       order_B,
                       m,
                       n,
                       k,
                       nnz,
                       alpha,
                       A_val,
                       A_ld,
                       B_val,
                       B_ld,
                       beta,
                       C_ind_data,
                       col_data,
                       C_val_data,
                       C_base,
                       alg,
                       coo_buffer);
    }

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
                                    const I*             C_ptr_data,
                                    const J*             C_ind_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        J*    col_data   = (J*)buffer;
        void* coo_buffer = col_data + nnz;
        return rocsparse_sddmm_st<rocsparse_format_coo, rocsparse_sddmm_alg_default, J, J, T>::
            compute(handle,
                    trans_A,
                    trans_B,
                    order_A,
                    order_B,
                    m,
                    n,
                    k,
                    (J)nnz,
                    alpha,
                    A_val,
                    A_ld,
                    B_val,
                    B_ld,
                    beta,
                    C_ind_data,
                    col_data,
                    C_val_data,
                    C_base,
                    alg,
                    coo_buffer);
    }
};

template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_double_complex>;

template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   rocsparse_double_complex>;

template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csc,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_double_complex>;

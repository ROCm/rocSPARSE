/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_CSRGEMM_HPP
#define ROCSPARSE_CSRGEMM_HPP

#include "handle.h"

#define CSRGEMM_MAXGROUPS 8
#define CSRGEMM_NNZ_HASH 79
#define CSRGEMM_FLL_HASH 137

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrgemm_buffer_size_template(rocsparse_handle          handle,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        J                         m,
                                                        J                         n,
                                                        J                         k,
                                                        const T*                  alpha,
                                                        const rocsparse_mat_descr descr_A,
                                                        I                         nnz_A,
                                                        const I*                  csr_row_ptr_A,
                                                        const J*                  csr_col_ind_A,
                                                        const rocsparse_mat_descr descr_B,
                                                        I                         nnz_B,
                                                        const I*                  csr_row_ptr_B,
                                                        const J*                  csr_col_ind_B,
                                                        const T*                  beta,
                                                        const rocsparse_mat_descr descr_D,
                                                        I                         nnz_D,
                                                        const I*                  csr_row_ptr_D,
                                                        const J*                  csr_col_ind_D,
                                                        rocsparse_mat_info        info_C,
                                                        size_t*                   buffer_size);

template <typename I, typename J>
rocsparse_status rocsparse_csrgemm_nnz_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                J                         m,
                                                J                         n,
                                                J                         k,
                                                const rocsparse_mat_descr descr_A,
                                                I                         nnz_A,
                                                const I*                  csr_row_ptr_A,
                                                const J*                  csr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                I                         nnz_B,
                                                const I*                  csr_row_ptr_B,
                                                const J*                  csr_col_ind_B,
                                                const rocsparse_mat_descr descr_D,
                                                I                         nnz_D,
                                                const I*                  csr_row_ptr_D,
                                                const J*                  csr_col_ind_D,
                                                const rocsparse_mat_descr descr_C,
                                                I*                        csr_row_ptr_C,
                                                I*                        nnz_C,
                                                const rocsparse_mat_info  info_C,
                                                void*                     temp_buffer);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrgemm_template(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         m,
                                            J                         n,
                                            J                         k,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            I                         nnz_A,
                                            const T*                  csr_val_A,
                                            const I*                  csr_row_ptr_A,
                                            const J*                  csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            I                         nnz_B,
                                            const T*                  csr_val_B,
                                            const I*                  csr_row_ptr_B,
                                            const J*                  csr_col_ind_B,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_D,
                                            I                         nnz_D,
                                            const T*                  csr_val_D,
                                            const I*                  csr_row_ptr_D,
                                            const J*                  csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const I*                  csr_row_ptr_C,
                                            J*                        csr_col_ind_C,
                                            const rocsparse_mat_info  info_C,
                                            void*                     temp_buffer);

template <typename I, typename J>
rocsparse_status rocsparse_csrgemm_symbolic_template(rocsparse_handle          handle,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     J                         m,
                                                     J                         n,
                                                     J                         k,
                                                     const rocsparse_mat_descr descr_A,
                                                     I                         nnz_A,
                                                     const I*                  csr_row_ptr_A,
                                                     const J*                  csr_col_ind_A,
                                                     const rocsparse_mat_descr descr_B,
                                                     I                         nnz_B,

                                                     const I* csr_row_ptr_B,
                                                     const J* csr_col_ind_B,

                                                     const rocsparse_mat_descr descr_D,
                                                     I                         nnz_D,

                                                     const I*                  csr_row_ptr_D,
                                                     const J*                  csr_col_ind_D,
                                                     const rocsparse_mat_descr descr_C,
                                                     I                         nnz_C,
                                                     const I*                  csr_row_ptr_C,
                                                     J*                        csr_col_ind_C,
                                                     const rocsparse_mat_info  info_C,
                                                     void*                     temp_buffer);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrgemm_numeric_template(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    J                         m,
                                                    J                         n,
                                                    J                         k,
                                                    const T*                  alpha,
                                                    const rocsparse_mat_descr descr_A,
                                                    I                         nnz_A,
                                                    const T*                  csr_val_A,
                                                    const I*                  csr_row_ptr_A,
                                                    const J*                  csr_col_ind_A,
                                                    const rocsparse_mat_descr descr_B,
                                                    I                         nnz_B,
                                                    const T*                  csr_val_B,
                                                    const I*                  csr_row_ptr_B,
                                                    const J*                  csr_col_ind_B,
                                                    const T*                  beta,
                                                    const rocsparse_mat_descr descr_D,
                                                    I                         nnz_D,
                                                    const T*                  csr_val_D,
                                                    const I*                  csr_row_ptr_D,
                                                    const J*                  csr_col_ind_D,
                                                    const rocsparse_mat_descr descr_C,
                                                    I                         nnz_C,
                                                    T*                        csr_val_C,
                                                    const I*                  csr_row_ptr_C,
                                                    const J*                  csr_col_ind_C,
                                                    const rocsparse_mat_info  info_C,
                                                    void*                     temp_buffer);

#endif // ROCSPARSE_CSRGEMM_HPP

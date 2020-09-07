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
#include "rocsparse.h"

#include "rocsparse_prune_dense2csr_by_percentage.hpp"

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const float*              A,
                                                         rocsparse_int             lda,
                                                         float                     percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const float*              csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
{
    return rocsparse_prune_dense2csr_by_percentage_buffer_size_template(handle,
                                                                        m,
                                                                        n,
                                                                        A,
                                                                        lda,
                                                                        percentage,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        info,
                                                                        buffer_size);
}

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const double*             A,
                                                         rocsparse_int             lda,
                                                         double                    percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const double*             csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
{
    return rocsparse_prune_dense2csr_by_percentage_buffer_size_template(handle,
                                                                        m,
                                                                        n,
                                                                        A,
                                                                        lda,
                                                                        percentage,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        info,
                                                                        buffer_size);
}

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const float*              A,
                                                 rocsparse_int             lda,
                                                 float                     percentage,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_int*            csr_row_ptr,
                                                 rocsparse_int*            nnz_total_dev_host_ptr,
                                                 rocsparse_mat_info        info,
                                                 void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_nnz_by_percentage_template(handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                percentage,
                                                                descr,
                                                                csr_row_ptr,
                                                                nnz_total_dev_host_ptr,
                                                                info,
                                                                temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const double*             A,
                                                 rocsparse_int             lda,
                                                 double                    percentage,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_int*            csr_row_ptr,
                                                 rocsparse_int*            nnz_total_dev_host_ptr,
                                                 rocsparse_mat_info        info,
                                                 void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_nnz_by_percentage_template(handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                percentage,
                                                                descr,
                                                                csr_row_ptr,
                                                                nnz_total_dev_host_ptr,
                                                                info,
                                                                temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const float*              A,
                                             rocsparse_int             lda,
                                             float                     percentage,
                                             const rocsparse_mat_descr descr,
                                             float*                    csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info        info,
                                             void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_by_percentage_template(handle,
                                                            m,
                                                            n,
                                                            A,
                                                            lda,
                                                            percentage,
                                                            descr,
                                                            csr_val,
                                                            csr_row_ptr,
                                                            csr_col_ind,
                                                            info,
                                                            temp_buffer);
}

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const double*             A,
                                             rocsparse_int             lda,
                                             double                    percentage,
                                             const rocsparse_mat_descr descr,
                                             double*                   csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info        info,
                                             void*                     temp_buffer)
{
    return rocsparse_prune_dense2csr_by_percentage_template(handle,
                                                            m,
                                                            n,
                                                            A,
                                                            lda,
                                                            percentage,
                                                            descr,
                                                            csr_val,
                                                            csr_row_ptr,
                                                            csr_col_ind,
                                                            info,
                                                            temp_buffer);
}

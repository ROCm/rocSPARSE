/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H
#define ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function computes the size of the user allocated temporary storage buffer used when converting and pruning by percentage a
*  dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the following steps are performed. First the user
*  calls \p rocsparse_prune_dense2csr_by_percentage_buffer_size which determines the size of the temporary storage buffer. Once
*  determined, this buffer must be allocated by the user. Next the user allocates the csr_row_ptr array to have \p m+1 elements
*  and calls \p rocsparse_prune_dense2csr_nnz_by_percentage. Finally the user finishes the conversion by allocating the csr_col_ind
*  and csr_val arrays (whos size is determined by the value at nnz_total_dev_host_ptr) and calling \p rocsparse_prune_dense2csr_by_percentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense matrix \p A. We then determine a position in this
*  sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in \p rocsparse_prune_dense2csr.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*
*  @param[in]
*  lda         leading dimension of dense array \p A.
*
*  @param[in]
*  percentage  percentage >= 0 and percentage <= 100.
*
*  @param[in]
*  descr      the descriptor of the dense matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  csr_val    array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*
*  @param[in]
*  csr_row_ptr integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*
*  @param[in]
*  csr_col_ind integer array of nnz ( = \p csr_row_ptr[m] - csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[in]
*  info prune information structure
*
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sprune_dense2csr_nnz_by_percentage(), rocsparse_dprune_dense2csr_nnz_by_percentage(),
*              rocsparse_sprune_dense2csr_by_percentage() and rocsparse_dprune_dense2csr_by_percentage().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status
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
                                                         size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status
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
                                                         size_t*                   buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of nonzero elements in a dense matrix
*  when converting and pruning by percentage a dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the following steps are performed. First the user
*  calls \p rocsparse_prune_dense2csr_by_percentage_buffer_size which determines the size of the temporary storage buffer. Once
*  determined, this buffer must be allocated by the user. Next the user allocates the csr_row_ptr array to have \p m+1 elements
*  and calls \p rocsparse_prune_dense2csr_nnz_by_percentage. Finally the user finishes the conversion by allocating the csr_col_ind
*  and csr_val arrays (whos size is determined by the value at nnz_total_dev_host_ptr) and calling \p rocsparse_prune_dense2csr_by_percentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense matrix \p A. We then determine a position in this
*  sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in \p rocsparse_prune_dense2csr.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*
*  @param[in]
*  lda         leading dimension of dense array \p A.
*
*  @param[in]
*  percentage  percentage >= 0 and percentage <= 100.
*
*  @param[in]
*  descr       the descriptor of the dense matrix \p A.
*
*  @param[out]
*  csr_row_ptr integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*
*  @param[out]
*  nnz_total_dev_host_ptr total number of nonzero elements in device or host memory.
*
*  @param[in]
*  info prune information structure
*
*  @param[out]
*  temp_buffer buffer allocated by the user whose size is determined by calling rocsparse_xprune_dense2csr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda or \p percentage is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p descr or \p info or \p csr_row_ptr
*              or \p nnz_total_dev_host_ptr or \p temp_buffer pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                              rocsparse_int             m,
                                                              rocsparse_int             n,
                                                              const float*              A,
                                                              rocsparse_int             lda,
                                                              float                     percentage,
                                                              const rocsparse_mat_descr descr,
                                                              rocsparse_int*            csr_row_ptr,
                                                              rocsparse_int* nnz_total_dev_host_ptr,
                                                              rocsparse_mat_info info,
                                                              void*              temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                              rocsparse_int             m,
                                                              rocsparse_int             n,
                                                              const double*             A,
                                                              rocsparse_int             lda,
                                                              double                    percentage,
                                                              const rocsparse_mat_descr descr,
                                                              rocsparse_int*            csr_row_ptr,
                                                              rocsparse_int* nnz_total_dev_host_ptr,
                                                              rocsparse_mat_info info,
                                                              void*              temp_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix A in dense format into a sparse matrix in CSR format while pruning values
*  based on percentage.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the following steps are performed. First the user
*  calls \p rocsparse_prune_dense2csr_by_percentage_buffer_size which determines the size of the temporary storage buffer. Once
*  determined, this buffer must be allocated by the user. Next the user allocates the csr_row_ptr array to have \p m+1 elements
*  and calls \p rocsparse_prune_dense2csr_nnz_by_percentage. Finally the user finishes the conversion by allocating the csr_col_ind
*  and csr_val arrays (whos size is determined by the value at nnz_total_dev_host_ptr) and calling \p rocsparse_prune_dense2csr_by_percentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense matrix \p A. We then determine a position in this
*  sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in \p rocsparse_prune_dense2csr.
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*
*  @param[in]
*  A           array of dimensions (\p lda, \p n)
*
*  @param[in]
*  lda         leading dimension of dense array \p A.
*
*  @param[in]
*  percentage  percentage >= 0 and percentage <= 100.
*
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[out]
*  csr_val array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*
*  @param[in]
*  csr_row_ptr integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*
*  @param[out]
*  csr_col_ind integer array of nnz ( = \p csr_row_ptr[m] - csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[in]
*  info prune information structure
*
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              rocsparse_xprune_dense2csr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda or \p percentage is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p descr or \p info or \p csr_val
*              or \p csr_row_ptr or \p csr_col_ind or \p temp_buffer pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_by_percentage(rocsparse_handle          handle,
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
                                                          void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_by_percentage(rocsparse_handle          handle,
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
                                                          void*                     temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_PRUNE_DENSE2CSR_BY_PERCENTAGE_H */

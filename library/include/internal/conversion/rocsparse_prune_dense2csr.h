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

#ifndef ROCSPARSE_PRUNE_DENSE2CSR_H
#define ROCSPARSE_PRUNE_DENSE2CSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function computes the the size of the user allocated temporary storage buffer used when converting and pruning
*  a dense matrix to a CSR matrix.
*
*  \details
*  \p rocsparse_prune_dense2csr_buffer_size returns the size of the temporary storage buffer
*  that is required by rocsparse_sprune_dense2csr_nnz(), rocsparse_dprune_dense2csr_nnz(),
*  rocsparse_sprune_dense2csr(), and rocsparse_dprune_dense2csr(). The temporary
*  storage buffer must be allocated by the user.
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
*  threshold   pointer to the pruning non-negative threshold which can exist in either host or device memory.
*
*  @param[in]
*  descr      the descriptor of the dense matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  csr_val
*              array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csr_row_ptr
*              integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csr_col_ind
*              integer array of nnz ( = \p csr_row_ptr[m] - csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sprune_dense2csr_nnz(), rocsparse_dprune_dense2csr_nnz(),
*              rocsparse_sprune_dense2csr() and rocsparse_dprune_dense2csr().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_buffer_size(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const float*              A,
                                                        rocsparse_int             lda,
                                                        const float*              threshold,
                                                        const rocsparse_mat_descr descr,
                                                        const float*              csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_buffer_size(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const double*             A,
                                                        rocsparse_int             lda,
                                                        const double*             threshold,
                                                        const rocsparse_mat_descr descr,
                                                        const double*             csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        size_t*                   buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of nonzero elements in a dense matrix once
*  elements less than the threshold are pruned from the matrix.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
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
*  threshold   pointer to the pruning non-negative threshold which can exist in either host or device memory.
*
*  @param[in]
*  descr      the descriptor of the dense matrix \p A.
*
*  @param[out]
*  csr_row_ptr
*              integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  nnz_total_dev_host_ptr
*              total number of nonzero elements in device or host memory.
*
*  @param[out]
*  temp_buffer
*              buffer allocated by the user whose size is determined by calling rocsparse_xprune_dense2csr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p threshold or \p descr or \p csr_row_ptr
*              or \p nnz_total_dev_host_ptr or \p temp_buffer pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr_nnz(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const float*              A,
                                                rocsparse_int             lda,
                                                const float*              threshold,
                                                const rocsparse_mat_descr descr,
                                                rocsparse_int*            csr_row_ptr,
                                                rocsparse_int*            nnz_total_dev_host_ptr,
                                                void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr_nnz(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const double*             A,
                                                rocsparse_int             lda,
                                                const double*             threshold,
                                                const rocsparse_mat_descr descr,
                                                rocsparse_int*            csr_row_ptr,
                                                rocsparse_int*            nnz_total_dev_host_ptr,
                                                void*                     temp_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix A in dense format into a sparse matrix in CSR format while pruning values
*  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
*
*  \details
*  The user first allocates \p csr_row_ptr to have \p m+1 elements and then calls rocsparse_xprune_dense2csr_nnz()
*  which fills in the \p csr_row_ptr array and stores the number of elements that are larger than the pruning threshold
*  in \p nnz_total_dev_host_ptr. The user then allocates \p csr_col_ind and \p csr_val to have size \p nnz_total_dev_host_ptr
*  and completes the conversion by calling rocsparse_xprune_dense2csr(). A temporary storage buffer is used by both
*  rocsparse_xprune_dense2csr_nnz() and rocsparse_xprune_dense2csr() and must be allocated by the user and whose size is determined
*  by rocsparse_xprune_dense2csr_buffer_size().
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
*  threshold   pointer to the non-negative pruning threshold which can exist in either host or device memory.
*
*  @param[in]
*  descr      the descriptor of the dense matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[out]
*  csr_val
*              array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csr_row_ptr
*              integer array of \p m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csr_col_ind
*              integer array of nnz ( = \p csr_row_ptr[m] - csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              rocsparse_xprune_dense2csr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p lda is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p descr or \p threshold or \p csr_val
*              or \p csr_row_ptr or \p csr_col_ind or \p temp_buffer pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_dense2csr(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const float*              A,
                                            rocsparse_int             lda,
                                            const float*              threshold,
                                            const rocsparse_mat_descr descr,
                                            float*                    csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            rocsparse_int*            csr_col_ind,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_dense2csr(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const double*             A,
                                            rocsparse_int             lda,
                                            const double*             threshold,
                                            const rocsparse_mat_descr descr,
                                            double*                   csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            rocsparse_int*            csr_col_ind,
                                            void*                     temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_PRUNE_DENSE2CSR_H */

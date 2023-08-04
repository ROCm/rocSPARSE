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

#ifndef ROCSPARSE_GEBSR2GEBSR_H
#define ROCSPARSE_GEBSR2GEBSR_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function computes the the size of the user allocated temporary storage buffer used when converting a sparse
*  general BSR matrix to another sparse general BSR matrix.
*
*  \details
*  \p rocsparse_gebsr2gebsr_buffer_size returns the size of the temporary storage buffer
*  that is required by rocsparse_gebsr2gebsr_nnz(), rocsparse_sgebsr2gebsr(), rocsparse_dgebsr2gebsr(),
*  rocsparse_cgebsr2gebsr(), and rocsparse_zgebsr2gebsr(). The temporary
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
*  dir         the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*
*  @param[in]
*  mb           number of block rows of the general BSR sparse matrix \p A.
*
*  @param[in]
*  nb           number of block columns of the general BSR sparse matrix \p A.
*
*  @param[in]
*  nnzb         number of blocks in the general BSR sparse matrix \p A.
*
*  @param[in]
*  descr_A      the descriptor of the general BSR sparse matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  bsr_val_A    array of \p nnzb*row_block_dim_A*col_block_dim_A containing the values of the sparse general BSR matrix \p A.
*
*  @param[in]
*  bsr_row_ptr_A array of \p mb+1 elements that point to the start of every block row of the
*              sparse general BSR matrix \p A.
*  @param[in]
*  bsr_col_ind_A array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \p A.
*
*  @param[in]
*  row_block_dim_A   row size of the blocks in the sparse general BSR matrix \p A.
*
*  @param[in]
*  col_block_dim_A   column size of the blocks in the sparse general BSR matrix \p A.
*
*  @param[in]
*  row_block_dim_C   row size of the blocks in the sparse general BSR matrix \p C.
*
*  @param[in]
*  col_block_dim_C   column size of the blocks in the sparse general BSR matrix \p C.
*
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by rocsparse_gebsr2gebsr_nnz(),
*              rocsparse_sgebsr2gebsr(), rocsparse_dgebsr2gebsr(), rocsparse_cgebsr2gebsr(), and rocsparse_zgebsr2gebsr().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p nnzb or \p row_block_dim_A or
*              \p col_block_dim_A or \p row_block_dim_C or \p col_block_dim_C is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr_A or \p bsr_col_ind_A
*              or \p descr_A or \p buffer_size pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    rocsparse_int             nnzb,
                                                    const rocsparse_mat_descr descr_A,
                                                    const float*              bsr_val_A,
                                                    const rocsparse_int*      bsr_row_ptr_A,
                                                    const rocsparse_int*      bsr_col_ind_A,
                                                    rocsparse_int             row_block_dim_A,
                                                    rocsparse_int             col_block_dim_A,
                                                    rocsparse_int             row_block_dim_C,
                                                    rocsparse_int             col_block_dim_C,
                                                    size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    rocsparse_int             nnzb,
                                                    const rocsparse_mat_descr descr_A,
                                                    const double*             bsr_val_A,
                                                    const rocsparse_int*      bsr_row_ptr_A,
                                                    const rocsparse_int*      bsr_col_ind_A,
                                                    rocsparse_int             row_block_dim_A,
                                                    rocsparse_int             col_block_dim_A,
                                                    rocsparse_int             row_block_dim_C,
                                                    rocsparse_int             col_block_dim_C,
                                                    size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsr_buffer_size(rocsparse_handle               handle,
                                                    rocsparse_direction            dir,
                                                    rocsparse_int                  mb,
                                                    rocsparse_int                  nb,
                                                    rocsparse_int                  nnzb,
                                                    const rocsparse_mat_descr      descr_A,
                                                    const rocsparse_float_complex* bsr_val_A,
                                                    const rocsparse_int*           bsr_row_ptr_A,
                                                    const rocsparse_int*           bsr_col_ind_A,
                                                    rocsparse_int                  row_block_dim_A,
                                                    rocsparse_int                  col_block_dim_A,
                                                    rocsparse_int                  row_block_dim_C,
                                                    rocsparse_int                  col_block_dim_C,
                                                    size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsr_buffer_size(rocsparse_handle                handle,
                                                    rocsparse_direction             dir,
                                                    rocsparse_int                   mb,
                                                    rocsparse_int                   nb,
                                                    rocsparse_int                   nnzb,
                                                    const rocsparse_mat_descr       descr_A,
                                                    const rocsparse_double_complex* bsr_val_A,
                                                    const rocsparse_int*            bsr_row_ptr_A,
                                                    const rocsparse_int*            bsr_col_ind_A,
                                                    rocsparse_int                   row_block_dim_A,
                                                    rocsparse_int                   col_block_dim_A,
                                                    rocsparse_int                   row_block_dim_C,
                                                    rocsparse_int                   col_block_dim_C,
                                                    size_t*                         buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \brief This function is used when converting a general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
*  Specifically, this function determines the number of non-zero blocks that will exist in \p C (stored using either a host
*  or device pointer), and computes the row pointer array for \p C.
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
*  dir         the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*
*  @param[in]
*  mb           number of block rows of the general BSR sparse matrix \p A.
*
*  @param[in]
*  nb           number of block columns of the general BSR sparse matrix \p A.
*
*  @param[in]
*  nnzb         number of blocks in the general BSR sparse matrix \p A.
*
*  @param[in]
*  descr_A      the descriptor of the general BSR sparse matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  bsr_row_ptr_A array of \p mb+1 elements that point to the start of every block row of the
*              sparse general BSR matrix \p A.
*  @param[in]
*  bsr_col_ind_A array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \p A.
*
*  @param[in]
*  row_block_dim_A   row size of the blocks in the sparse general BSR matrix \p A.
*
*  @param[in]
*  col_block_dim_A   column size of the blocks in the sparse general BSR matrix \p A.
*
*  @param[in]
*  descr_C      the descriptor of the general BSR sparse matrix \p C, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  bsr_row_ptr_C array of \p mb_C+1 elements that point to the start of every block row of the
*              sparse general BSR matrix \p C where \p mb_C=(m+row_block_dim_C-1)/row_block_dim_C.
*  @param[in]
*  row_block_dim_C   row size of the blocks in the sparse general BSR matrix \p C.
*
*  @param[in]
*  col_block_dim_C   column size of the blocks in the sparse general BSR matrix \p C.
*
*  @param[out]
*  nnz_total_dev_host_ptr
*              total number of nonzero blocks in general BSR sparse matrix \p C stored using device or host memory.
*
*  @param[out]
*  temp_buffer
*              buffer allocated by the user whose size is determined by calling rocsparse_xgebsr2gebsr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p nnzb or \p row_block_dim_A or
*              \p col_block_dim_A or \p row_block_dim_C or \p col_block_dim_C is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr_A or \p bsr_col_ind_A
*              or \p bsr_row_ptr_C or \p descr_A or \p descr_C or \p temp_buffer pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_gebsr2gebsr_nnz(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_int             mb,
                                           rocsparse_int             nb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr_A,
                                           const rocsparse_int*      bsr_row_ptr_A,
                                           const rocsparse_int*      bsr_col_ind_A,
                                           rocsparse_int             row_block_dim_A,
                                           rocsparse_int             col_block_dim_A,
                                           const rocsparse_mat_descr descr_C,
                                           rocsparse_int*            bsr_row_ptr_C,
                                           rocsparse_int             row_block_dim_C,
                                           rocsparse_int             col_block_dim_C,
                                           rocsparse_int*            nnz_total_dev_host_ptr,
                                           void*                     temp_buffer);

/*! \ingroup conv_module
*  \brief
*  This function converts the general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
*
*  \details
*  The conversion uses three steps. First, the user calls rocsparse_xgebsr2gebsr_buffer_size() to determine the size of
*  the required temporary storage buffer. The user then allocates this buffer. Secondly, the user then allocates \p mb_C+1
*  integers for the row pointer array for \p C where \p mb_C=(m+row_block_dim_C-1)/row_block_dim_C. The user then calls
*  rocsparse_xgebsr2gebsr_nnz() to fill in the row pointer array for \p C ( \p bsr_row_ptr_C ) and determine the number of
*  non-zero blocks that will exist in \p C. Finally, the user allocates space for the colimn indices array of \p C to have
*  \p nnzb_C elements and space for the values array of \p C to have \p nnzb_C*roc_block_dim_C*col_block_dim_C and then calls
*  rocsparse_xgebsr2gebsr() to complete the conversion.
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
*  dir         the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*
*  @param[in]
*  mb           number of block rows of the general BSR sparse matrix \p A.
*
*  @param[in]
*  nb           number of block columns of the general BSR sparse matrix \p A.
*
*  @param[in]
*  nnzb         number of blocks in the general BSR sparse matrix \p A.
*
*  @param[in]
*  descr_A      the descriptor of the general BSR sparse matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  bsr_val_A    array of \p nnzb*row_block_dim_A*col_block_dim_A containing the values of the sparse general BSR matrix \p A.
*
*  @param[in]
*  bsr_row_ptr_A array of \p mb+1 elements that point to the start of every block row of the
*              sparse general BSR matrix \p A.
*  @param[in]
*  bsr_col_ind_A array of \p nnzb elements containing the block column indices of the sparse general BSR matrix \p A.
*
*  @param[in]
*  row_block_dim_A   row size of the blocks in the sparse general BSR matrix \p A.
*
*  @param[in]
*  col_block_dim_A   column size of the blocks in the sparse general BSR matrix \p A.
*
*  @param[in]
*  descr_C      the descriptor of the general BSR sparse matrix \p C, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  bsr_val_C    array of \p nnzb_C*row_block_dim_C*col_block_dim_C containing the values of the sparse general BSR matrix \p C.
*
*  @param[in]
*  bsr_row_ptr_C array of \p mb_C+1 elements that point to the start of every block row of the
*              sparse general BSR matrix \p C.
*  @param[in]
*  bsr_col_ind_C array of \p nnzb_C elements containing the block column indices of the sparse general BSR matrix \p C.
*
*  @param[in]
*  row_block_dim_C   row size of the blocks in the sparse general BSR matrix \p C.
*
*  @param[in]
*  col_block_dim_C   column size of the blocks in the sparse general BSR matrix \p C.
*
*  @param[out]
*  temp_buffer
*              buffer allocated by the user whose size is determined by calling rocsparse_xgebsr2gebsr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p nnzb or \p row_block_dim_A or
*              \p col_block_dim_A or \p row_block_dim_C or \p col_block_dim_C is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr_A or \p bsr_col_ind_A or \p bsr_val_A
*              or \p bsr_row_ptr_C or \p bsr_col_ind_C or \p bsr_val_C or \p descr_A or \p descr_C
*              or \p temp_buffer pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsr(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const rocsparse_mat_descr descr_A,
                                        const float*              bsr_val_A,
                                        const rocsparse_int*      bsr_row_ptr_A,
                                        const rocsparse_int*      bsr_col_ind_A,
                                        rocsparse_int             row_block_dim_A,
                                        rocsparse_int             col_block_dim_A,
                                        const rocsparse_mat_descr descr_C,
                                        float*                    bsr_val_C,
                                        rocsparse_int*            bsr_row_ptr_C,
                                        rocsparse_int*            bsr_col_ind_C,
                                        rocsparse_int             row_block_dim_C,
                                        rocsparse_int             col_block_dim_C,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsr(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const rocsparse_mat_descr descr_A,
                                        const double*             bsr_val_A,
                                        const rocsparse_int*      bsr_row_ptr_A,
                                        const rocsparse_int*      bsr_col_ind_A,
                                        rocsparse_int             row_block_dim_A,
                                        rocsparse_int             col_block_dim_A,
                                        const rocsparse_mat_descr descr_C,
                                        double*                   bsr_val_C,
                                        rocsparse_int*            bsr_row_ptr_C,
                                        rocsparse_int*            bsr_col_ind_C,
                                        rocsparse_int             row_block_dim_C,
                                        rocsparse_int             col_block_dim_C,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsr(rocsparse_handle               handle,
                                        rocsparse_direction            dir,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nb,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_mat_descr      descr_A,
                                        const rocsparse_float_complex* bsr_val_A,
                                        const rocsparse_int*           bsr_row_ptr_A,
                                        const rocsparse_int*           bsr_col_ind_A,
                                        rocsparse_int                  row_block_dim_A,
                                        rocsparse_int                  col_block_dim_A,
                                        const rocsparse_mat_descr      descr_C,
                                        rocsparse_float_complex*       bsr_val_C,
                                        rocsparse_int*                 bsr_row_ptr_C,
                                        rocsparse_int*                 bsr_col_ind_C,
                                        rocsparse_int                  row_block_dim_C,
                                        rocsparse_int                  col_block_dim_C,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsr(rocsparse_handle                handle,
                                        rocsparse_direction             dir,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nb,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_mat_descr       descr_A,
                                        const rocsparse_double_complex* bsr_val_A,
                                        const rocsparse_int*            bsr_row_ptr_A,
                                        const rocsparse_int*            bsr_col_ind_A,
                                        rocsparse_int                   row_block_dim_A,
                                        rocsparse_int                   col_block_dim_A,
                                        const rocsparse_mat_descr       descr_C,
                                        rocsparse_double_complex*       bsr_val_C,
                                        rocsparse_int*                  bsr_row_ptr_C,
                                        rocsparse_int*                  bsr_col_ind_C,
                                        rocsparse_int                   row_block_dim_C,
                                        rocsparse_int                   col_block_dim_C,
                                        void*                           temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GEBSR2GEBSR_H */

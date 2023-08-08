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

#ifndef ROCSPARSE_PRUNE_CSR2CSR_BY_PERCENTAGE_H
#define ROCSPARSE_PRUNE_CSR2CSR_BY_PERCENTAGE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_prune_csr2csr__by_percentage_buffer_size returns the size of the temporary buffer that
 *  is required by \p rocsparse_sprune_csr2csr_nnz_by_percentage, \p rocsparse_dprune_csr2csr_nnz_by_percentage,
 *  \p rocsparse_sprune_csr2csr_by_percentage, and \p rocsparse_dprune_csr2csr_by_percentage. The temporary storage
 *  buffer must be allocated by the user.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  This routine supports execution in a hipGraph context.
 *
 *  @param[in]
 *  handle        handle to the rocsparse library context queue.
 *  @param[in]
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnz_A         number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  csr_descr_A   descriptor of the sparse CSR matrix A. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_A     array of \p nnz_A elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix A.
 *  @param[in]
 *  csr_col_ind_A array of \p nnz_A elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage     percentage >= 0 and percentage <= 100.
 *  @param[in]
 *  csr_descr_C   descriptor of the sparse CSR matrix C. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_C     array of \p nnz_C elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[in]
 *  csr_col_ind_C array of \p nnz_C elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info          prune info structure.
 *  @param[out]
 *  buffer_size   number of bytes of the temporary storage buffer required by \p rocsparse_sprune_csr2csr_nnz_by_percentage,
 *                \p rocsparse_dprune_csr2csr_nnz_by_percentage, \p rocsparse_sprune_csr2csr_by_percentage,
 *                and \p rocsparse_dprune_csr2csr_by_percentage.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_sprune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const float*              csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       float                     percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       const float*              csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       const rocsparse_int*      csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_dprune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const double*             csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       double                    percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       const double*             csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       const rocsparse_int*      csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       size_t*                   buffer_size);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_prune_csr2csr_nnz_by_percentage computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
 *
 *  \note
 *  This routine does not support execution in a hipGraph context.
 *
 *  @param[in]
 *  handle        handle to the rocsparse library context queue.
 *  @param[in]
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnz_A         number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  csr_descr_A   descriptor of the sparse CSR matrix A. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_A     array of \p nnz_A elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix A.
 *  @param[in]
 *  csr_col_ind_A array of \p nnz_A elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage    percentage >= 0 and percentage <= 100.
 *  @param[in]
 *  csr_descr_C   descriptor of the sparse CSR matrix C. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[out]
 *  nnz_total_dev_host_ptr total number of nonzero elements in device or host memory.
 *  @param[in]
 *  info          prune info structure.
 *  @param[out]
 *  temp_buffer   buffer allocated by the user whose size is determined by calling
 *                \p rocsparse_xprune_csr2csr_by_percentage_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz_A or \p percentage is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_descr_A or \p csr_descr_C or \p info or \p csr_val_A
 *              or \p csr_row_ptr_A or \p csr_col_ind_A or \p csr_row_ptr_C or \p nnz_total_dev_host_ptr
 *              or \p temp_buffer pointer is invalid.
 *
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                            rocsparse_int             m,
                                                            rocsparse_int             n,
                                                            rocsparse_int             nnz_A,
                                                            const rocsparse_mat_descr csr_descr_A,
                                                            const float*              csr_val_A,
                                                            const rocsparse_int*      csr_row_ptr_A,
                                                            const rocsparse_int*      csr_col_ind_A,
                                                            float                     percentage,
                                                            const rocsparse_mat_descr csr_descr_C,
                                                            rocsparse_int*            csr_row_ptr_C,
                                                            rocsparse_int* nnz_total_dev_host_ptr,
                                                            rocsparse_mat_info info,
                                                            void*              temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                            rocsparse_int             m,
                                                            rocsparse_int             n,
                                                            rocsparse_int             nnz_A,
                                                            const rocsparse_mat_descr csr_descr_A,
                                                            const double*             csr_val_A,
                                                            const rocsparse_int*      csr_row_ptr_A,
                                                            const rocsparse_int*      csr_col_ind_A,
                                                            double                    percentage,
                                                            const rocsparse_mat_descr csr_descr_C,
                                                            rocsparse_int*            csr_row_ptr_C,
                                                            rocsparse_int* nnz_total_dev_host_ptr,
                                                            rocsparse_mat_info info,
                                                            void*              temp_buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls rocsparse_xprune_csr2csr_buffer_size() to determine the size of the buffer used
 *  by rocsparse_xprune_csr2csr_nnz() and rocsparse_xprune_csr2csr() which the user then allocates. The user then
 *  allocates \p csr_row_ptr_C to have \p m+1 elements and then calls rocsparse_xprune_csr2csr_nnz() which fills
 *  in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
 *  in \p nnz_total_dev_host_ptr. The user then calls rocsparse_xprune_csr2csr() to complete the conversion.
 *
 *  \note
 *  This function is blocking with respect to the host.
 *
 *  \note
 *  This routine does not support execution in a hipGraph context.
 *
 *  @param[in]
 *  handle        handle to the rocsparse library context queue.
 *  @param[in]
 *  m             number of rows in the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns in the sparse CSR matrix.
 *  @param[in]
 *  nnz_A         number of non-zeros in the sparse CSR matrix A.
 *  @param[in]
 *  csr_descr_A   descriptor of the sparse CSR matrix A. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val_A     array of \p nnz_A elements containing the values of the sparse CSR matrix A.
 *  @param[in]
 *  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix A.
 *  @param[in]
 *  csr_col_ind_A array of \p nnz_A elements containing the column indices of the sparse CSR matrix A.
 *  @param[in]
 *  percentage    percentage >= 0 and percentage <= 100.
 *  @param[in]
 *  csr_descr_C   descriptor of the sparse CSR matrix C. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_val_C     array of \p nnz_C elements containing the values of the sparse CSR matrix C.
 *  @param[in]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[out]
 *  csr_col_ind_C array of \p nnz_C elements containing the column indices of the sparse CSR matrix C.
 *  @param[in]
 *  info          prune info structure.
 *  @param[in]
 *  temp_buffer   buffer allocated by the user whose size is determined by calling \p rocsparse_xprune_csr2csr_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz_A or \p percentage is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_descr_A or \p csr_descr_C or \p info or \p csr_val_A
 *              or \p csr_row_ptr_A or \p csr_col_ind_A or \p csr_val_C or \p csr_row_ptr_C or \p csr_col_ind_C
 *              or \p temp_buffer pointer is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_csr2csr_by_percentage(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_mat_descr csr_descr_A,
                                                        const float*              csr_val_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        float                     percentage,
                                                        const rocsparse_mat_descr csr_descr_C,
                                                        float*                    csr_val_C,
                                                        const rocsparse_int*      csr_row_ptr_C,
                                                        rocsparse_int*            csr_col_ind_C,
                                                        rocsparse_mat_info        info,
                                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_csr2csr_by_percentage(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        rocsparse_int             nnz_A,
                                                        const rocsparse_mat_descr csr_descr_A,
                                                        const double*             csr_val_A,
                                                        const rocsparse_int*      csr_row_ptr_A,
                                                        const rocsparse_int*      csr_col_ind_A,
                                                        double                    percentage,
                                                        const rocsparse_mat_descr csr_descr_C,
                                                        double*                   csr_val_C,
                                                        const rocsparse_int*      csr_row_ptr_C,
                                                        rocsparse_int*            csr_col_ind_C,
                                                        rocsparse_mat_info        info,
                                                        void*                     temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_PRUNE_CSR2CSR_BY_PERCENTAGE_H */

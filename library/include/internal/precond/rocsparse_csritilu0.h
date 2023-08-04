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

#ifndef ROCSPARSE_CSRITILU0_H
#define ROCSPARSE_CSRITILU0_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \brief Iterative Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format.
*
*  \details
*  \p rocsparse_csritilu0_buffer_size computes the size in bytes of the buffer that has to be allocated by the user.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[in]
*  option      combination of enumeration values from \ref rocsparse_itilu0_option.
*  @param[in]
*  nmaxiter     maximum number of iterations.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  datatype    Type of numerical values, \ref rocsparse_datatype.
*  @param[out]
*  buffer_size size of the temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_value \p alg, \p base or datatype is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_zero_pivot if nnz is zero.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csritilu0_buffer_size(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int        nmaxiter,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 rocsparse_index_base idx_base,
                                                 rocsparse_datatype   datatype,
                                                 size_t*              buffer_size);

/*! \ingroup precond_module
*  \brief Iterative Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format.
*
*  \details
*  \p rocsparse_csritilu0_preprocess computes the information required to run \ref rocsparse_scsritilu0_compute,
*  \ref rocsparse_dcsritilu0_compute, \ref rocsparse_ccsritilu0_compute, or \ref rocsparse_zcsritilu0_compute,
*  and stores it in the buffer.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[in]
*  option      combination of enumeration values from \ref rocsparse_itilu0_option.
*  @param[in]
*  nmaxiter     maximum number of iterations.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  datatype    Type of numerical values, \ref rocsparse_datatype.
*  @param[in]
*  buffer_size size of the storage buffer allocated by the user.
*  @param[in]
*  buffer      storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_value \p alg, \p base or datatype is invalid.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_zero_pivot if missing diagonal element is detected.
*
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csritilu0_preprocess(rocsparse_handle     handle,
                                                rocsparse_itilu0_alg alg,
                                                rocsparse_int        option,
                                                rocsparse_int        nmaxiter,
                                                rocsparse_int        m,
                                                rocsparse_int        nnz,
                                                const rocsparse_int* csr_row_ptr,
                                                const rocsparse_int* csr_col_ind,
                                                rocsparse_index_base idx_base,
                                                rocsparse_datatype   datatype,
                                                size_t               buffer_size,
                                                void*                buffer);

/*! \ingroup precond_module
*  \brief Iterative Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format.
*
*  \details
*  \p rocsparse_csritilu0_compute computes iteratively the incomplete LU factorization with 0 fill-ins and no
*  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LU
*  \f]
*
*  \p rocsparse_csritilu0 requires a user allocated temporary buffer. Its size is returned
*  by rocsparse_csritilu0_buffer_size(). Furthermore,
*  analysis meta data is required. It can be obtained by rocsparse_csritlu0_preprocess().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[in]
*  option      combination of enumeration values from \ref rocsparse_itilu0_option.
*  @param[inout]
*  nmaxiter     maximum number of iterations.
*  @param[in]
*  tol tolerance to use for stopping criteria.
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start
*              of every row of the sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[inout]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[out]
*  ilu0        incomplete factorization.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  buffer_size size of the storage buffer allocated by the user.
*  @param[in]
*  buffer      storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_value \p alg or \p base is invalid.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr
*              or \p csr_col_ind pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsritilu0_compute(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int        option,
                                              rocsparse_int*       nmaxiter,
                                              float                tol,
                                              rocsparse_int        m,
                                              rocsparse_int        nnz,
                                              const rocsparse_int* csr_row_ptr,
                                              const rocsparse_int* csr_col_ind,
                                              const float*         csr_val,
                                              float*               ilu0,
                                              rocsparse_index_base idx_base,
                                              size_t               buffer_size,
                                              void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsritilu0_compute(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int        option,
                                              rocsparse_int*       nmaxiter,
                                              double               tol,
                                              rocsparse_int        m,
                                              rocsparse_int        nnz,
                                              const rocsparse_int* csr_row_ptr,
                                              const rocsparse_int* csr_col_ind,
                                              const double*        csr_val,
                                              double*              ilu0,
                                              rocsparse_index_base idx_base,
                                              size_t               buffer_size,
                                              void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsritilu0_compute(rocsparse_handle               handle,
                                              rocsparse_itilu0_alg           alg,
                                              rocsparse_int                  option,
                                              rocsparse_int*                 nmaxiter,
                                              float                          tol,
                                              rocsparse_int                  m,
                                              rocsparse_int                  nnz,
                                              const rocsparse_int*           csr_row_ptr,
                                              const rocsparse_int*           csr_col_ind,
                                              const rocsparse_float_complex* csr_val,
                                              rocsparse_float_complex*       ilu0,
                                              rocsparse_index_base           idx_base,
                                              size_t                         buffer_size,
                                              void*                          buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsritilu0_compute(rocsparse_handle                handle,
                                              rocsparse_itilu0_alg            alg,
                                              rocsparse_int                   option,
                                              rocsparse_int*                  nmaxiter,
                                              double                          tol,
                                              rocsparse_int                   m,
                                              rocsparse_int                   nnz,
                                              const rocsparse_int*            csr_row_ptr,
                                              const rocsparse_int*            csr_col_ind,
                                              const rocsparse_double_complex* csr_val,
                                              rocsparse_double_complex*       ilu0,
                                              rocsparse_index_base            idx_base,
                                              size_t                          buffer_size,
                                              void*                           buffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Iterative Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format.
*
*  \details
*  \p rocsparse_csritilu0_history fetches convergence history data.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  rocsparse_csrsort().
*
*  \note
*  This function is blocking with respect to the host.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  alg         algorithm to use, \ref rocsparse_itilu0_alg
*  @param[out]
*  niter       number of performed iterations.
*  @param[out]
*  data        norms.
*  @param[in]
*  buffer_size size of the buffer allocated by the user.
*  @param[in]
*  buffer buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p niter or \p data is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              float*               data,
                                              size_t               buffer_size,
                                              void*                buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              double*              data,
                                              size_t               buffer_size,
                                              void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              float*               data,
                                              size_t               buffer_size,
                                              void*                buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsritilu0_history(rocsparse_handle     handle,
                                              rocsparse_itilu0_alg alg,
                                              rocsparse_int*       niter,
                                              double*              data,
                                              size_t               buffer_size,
                                              void*                buffer);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_CSRITILU0_H */

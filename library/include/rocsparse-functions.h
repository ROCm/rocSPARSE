/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

/*! \file
 *  \brief rocsparse-functions.h provides Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3, using HIP optimized for AMD HCC-based GPU hardware.
 */

#pragma once
#ifndef _ROCSPARSE_FUNCTIONS_H_
#define _ROCSPARSE_FUNCTIONS_H_

#include "rocsparse-export.h"
#include "rocsparse-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */

/*! \ingroup level1_module
 *  \brief Scale a sparse vector and add it to a dense vector.
 *
 *  \details
 *  \p rocsparse_axpyi multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
 *  adds the result to the dense vector \f$y\f$, such that
 *
 *  \f[
 *      y := y + \alpha \cdot x
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          y[x_ind[i]] = y[x_ind[i]] + alpha * x_val[i];
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of vector \f$x\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  x_val       array of \p nnz elements containing the values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_invalid_value \p idx_base is invalid.
 *  \retval rocsparse_status_invalid_size \p nnz is invalid.
 *  \retval rocsparse_status_invalid_pointer \p alpha, \p x_val, \p x_ind or \p y pointer
 *          is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_saxpyi(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  const float*         alpha,
                                  const float*         x_val,
                                  const rocsparse_int* x_ind,
                                  float*               y,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_daxpyi(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  const double*        alpha,
                                  const double*        x_val,
                                  const rocsparse_int* x_ind,
                                  double*              y,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_caxpyi(rocsparse_handle               handle,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_float_complex* x_val,
                                  const rocsparse_int*           x_ind,
                                  rocsparse_float_complex*       y,
                                  rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zaxpyi(rocsparse_handle                handle,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_double_complex* x_val,
                                  const rocsparse_int*            x_ind,
                                  rocsparse_double_complex*       y,
                                  rocsparse_index_base            idx_base);
/**@}*/

/*! \ingroup level1_module
 *  \brief Compute the dot product of a sparse vector with a dense vector.
 *
 *  \details
 *  \p rocsparse_doti computes the dot product of the sparse vector \f$x\f$ with the
 *  dense vector \f$y\f$, such that
 *  \f[
 *    \text{result} := y^T x
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          result += x_val[i] * y[x_ind[i]];
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of vector \f$x\f$.
 *  @param[in]
 *  x_val       array of \p nnz values.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[in]
 *  y           array of values in dense format.
 *  @param[out]
 *  result      pointer to the result, can be host or device memory
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_invalid_value \p idx_base is invalid.
 *  \retval rocsparse_status_invalid_size \p nnz is invalid.
 *  \retval rocsparse_status_invalid_pointer \p x_val, \p x_ind, \p y or \p result
 *          pointer is invalid.
 *  \retval rocsparse_status_memory_error the buffer for the dot product reduction
 *          could not be allocated.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sdoti(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const float*         x_val,
                                 const rocsparse_int* x_ind,
                                 const float*         y,
                                 float*               result,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ddoti(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const double*        x_val,
                                 const rocsparse_int* x_ind,
                                 const double*        y,
                                 double*              result,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdoti(rocsparse_handle               handle,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* x_val,
                                 const rocsparse_int*           x_ind,
                                 const rocsparse_float_complex* y,
                                 rocsparse_float_complex*       result,
                                 rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdoti(rocsparse_handle                handle,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* x_val,
                                 const rocsparse_int*            x_ind,
                                 const rocsparse_double_complex* y,
                                 rocsparse_double_complex*       result,
                                 rocsparse_index_base            idx_base);
/**@}*/

/*! \ingroup level1_module
 *  \brief Compute the dot product of a complex conjugate sparse vector with a dense
 *  vector.
 *
 *  \details
 *  \p rocsparse_dotci computes the dot product of the complex conjugate sparse vector
 *  \f$x\f$ with the dense vector \f$y\f$, such that
 *  \f[
 *    \text{result} := \bar{x}^H y
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          result += conj(x_val[i]) * y[x_ind[i]];
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of vector \f$x\f$.
 *  @param[in]
 *  x_val       array of \p nnz values.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[in]
 *  y           array of values in dense format.
 *  @param[out]
 *  result      pointer to the result, can be host or device memory
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_invalid_value \p idx_base is invalid.
 *  \retval rocsparse_status_invalid_size \p nnz is invalid.
 *  \retval rocsparse_status_invalid_pointer \p x_val, \p x_ind, \p y or \p result
 *          pointer is invalid.
 *  \retval rocsparse_status_memory_error the buffer for the dot product reduction
 *          could not be allocated.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdotci(rocsparse_handle               handle,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* x_val,
                                  const rocsparse_int*           x_ind,
                                  const rocsparse_float_complex* y,
                                  rocsparse_float_complex*       result,
                                  rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdotci(rocsparse_handle                handle,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* x_val,
                                  const rocsparse_int*            x_ind,
                                  const rocsparse_double_complex* y,
                                  rocsparse_double_complex*       result,
                                  rocsparse_index_base            idx_base);
/**@}*/

/*! \ingroup level1_module
 *  \brief Gather elements from a dense vector and store them into a sparse vector.
 *
 *  \details
 *  \p rocsparse_gthr gathers the elements that are listed in \p x_ind from the dense
 *  vector \f$y\f$ and stores them in the sparse vector \f$x\f$.
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          x_val[i] = y[x_ind[i]];
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[in]
 *  y           array of values in dense format.
 *  @param[out]
 *  x_val       array of \p nnz elements containing the values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_value \p idx_base is invalid.
 *  \retval     rocsparse_status_invalid_size \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p y, \p x_val or \p x_ind pointer is
 *              invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgthr(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const float*         y,
                                 float*               x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgthr(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const double*        y,
                                 double*              x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgthr(rocsparse_handle               handle,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* y,
                                 rocsparse_float_complex*       x_val,
                                 const rocsparse_int*           x_ind,
                                 rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgthr(rocsparse_handle                handle,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* y,
                                 rocsparse_double_complex*       x_val,
                                 const rocsparse_int*            x_ind,
                                 rocsparse_index_base            idx_base);
/**@}*/

/*! \ingroup level1_module
 *  \brief Gather and zero out elements from a dense vector and store them into a sparse
 *  vector.
 *
 *  \details
 *  \p rocsparse_gthrz gathers the elements that are listed in \p x_ind from the dense
 *  vector \f$y\f$ and stores them in the sparse vector \f$x\f$. The gathered elements
 *  in \f$y\f$ are replaced by zero.
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          x_val[i]    = y[x_ind[i]];
 *          y[x_ind[i]] = 0;
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[out]
 *  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_value \p idx_base is invalid.
 *  \retval     rocsparse_status_invalid_size \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p y, \p x_val or \p x_ind pointer is
 *              invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgthrz(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  float*               y,
                                  float*               x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgthrz(rocsparse_handle     handle,
                                  rocsparse_int        nnz,
                                  double*              y,
                                  double*              x_val,
                                  const rocsparse_int* x_ind,
                                  rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgthrz(rocsparse_handle         handle,
                                  rocsparse_int            nnz,
                                  rocsparse_float_complex* y,
                                  rocsparse_float_complex* x_val,
                                  const rocsparse_int*     x_ind,
                                  rocsparse_index_base     idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgthrz(rocsparse_handle          handle,
                                  rocsparse_int             nnz,
                                  rocsparse_double_complex* y,
                                  rocsparse_double_complex* x_val,
                                  const rocsparse_int*      x_ind,
                                  rocsparse_index_base      idx_base);
/**@}*/

/*! \ingroup level1_module
 *  \brief Apply Givens rotation to a dense and a sparse vector.
 *
 *  \details
 *  \p rocsparse_roti applies the Givens rotation matrix \f$G\f$ to the sparse vector
 *  \f$x\f$ and the dense vector \f$y\f$, where
 *  \f[
 *    G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          x_tmp = x_val[i];
 *          y_tmp = y[x_ind[i]];
 *
 *          x_val[i]    = c * x_tmp + s * y_tmp;
 *          y[x_ind[i]] = c * y_tmp - s * x_tmp;
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[inout]
 *  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of \f$x\f$.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[in]
 *  c           pointer to the cosine element of \f$G\f$, can be on host or device.
 *  @param[in]
 *  s           pointer to the sine element of \f$G\f$, can be on host or device.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_value \p idx_base is invalid.
 *  \retval     rocsparse_status_invalid_size \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p c, \p s, \p x_val, \p x_ind or \p y
 *              pointer is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sroti(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 float*               x_val,
                                 const rocsparse_int* x_ind,
                                 float*               y,
                                 const float*         c,
                                 const float*         s,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_droti(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 double*              x_val,
                                 const rocsparse_int* x_ind,
                                 double*              y,
                                 const double*        c,
                                 const double*        s,
                                 rocsparse_index_base idx_base);
/**@}*/

/*! \ingroup level1_module
 *  \brief Scatter elements from a dense vector across a sparse vector.
 *
 *  \details
 *  \p rocsparse_sctr scatters the elements that are listed in \p x_ind from the sparse
 *  vector \f$x\f$ into the dense vector \f$y\f$. Indices of \f$y\f$ that are not listed
 *  in \p x_ind remain unchanged.
 *
 *  \code{.c}
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          y[x_ind[i]] = x_val[i];
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *  @param[in]
 *  x_val       array of \p nnz elements containing the non-zero values of \f$x\f$.
 *  @param[in]
 *  x_ind       array of \p nnz elements containing the indices of the non-zero
 *              values of x.
 *  @param[inout]
 *  y           array of values in dense format.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_value \p idx_base is invalid.
 *  \retval     rocsparse_status_invalid_size \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p x_val, \p x_ind or \p y pointer is
 *              invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ssctr(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const float*         x_val,
                                 const rocsparse_int* x_ind,
                                 float*               y,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dsctr(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const double*        x_val,
                                 const rocsparse_int* x_ind,
                                 double*              y,
                                 rocsparse_index_base idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csctr(rocsparse_handle               handle,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* x_val,
                                 const rocsparse_int*           x_ind,
                                 rocsparse_float_complex*       y,
                                 rocsparse_index_base           idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zsctr(rocsparse_handle                handle,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* x_val,
                                 const rocsparse_int*            x_ind,
                                 rocsparse_double_complex*       y,
                                 rocsparse_index_base            idx_base);
/**@}*/

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */

/*! \ingroup level2_module
 *  \brief Sparse matrix vector multiplication using COO storage format
 *
 *  \details
 *  \p rocsparse_coomv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in COO storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == rocsparse\_operation\_none} \\
 *        A^T, & \text{if trans == rocsparse\_operation\_transpose} \\
 *        A^H, & \text{if trans == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  The COO matrix has to be sorted by row indices. This can be achieved by using
 *  rocsparse_coosort_by_row().
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *      }
 *
 *      for(i = 0; i < nnz; ++i)
 *      {
 *          y[coo_row_ind[i]] += alpha * coo_val[i] * x[coo_col_ind[i]];
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse COO matrix.
 *  @param[in]
 *  n           number of columns of the sparse COO matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse COO matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse COO matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  coo_val     array of \p nnz elements of the sparse COO matrix.
 *  @param[in]
 *  coo_row_ind array of \p nnz elements containing the row indices of the sparse COO
 *              matrix.
 *  @param[in]
 *  coo_col_ind array of \p nnz elements containing the column indices of the sparse
 *              COO matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) = A\f$) or \p m elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) = A\f$) or \p n elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p coo_val,
 *              \p coo_row_ind, \p coo_col_ind, \p x, \p beta or \p y pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scoomv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              coo_val,
                                  const rocsparse_int*      coo_row_ind,
                                  const rocsparse_int*      coo_col_ind,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcoomv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             coo_val,
                                  const rocsparse_int*      coo_row_ind,
                                  const rocsparse_int*      coo_col_ind,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccoomv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* coo_val,
                                  const rocsparse_int*           coo_row_ind,
                                  const rocsparse_int*           coo_col_ind,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcoomv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* coo_val,
                                  const rocsparse_int*            coo_row_ind,
                                  const rocsparse_int*            coo_col_ind,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse matrix vector multiplication using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmv_analysis performs the analysis step for rocsparse_scsrmv(),
 *  rocsparse_dcsrmv(), rocsparse_ccsrmv() and rocsparse_zcsrmv(). It is expected that
 *  this function will be executed only once for a given matrix and particular operation
 *  type. The gathered analysis meta data can be cleared by rocsparse_csrmv_clear().
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  info        structure that holds the information collected during the analysis step.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
 *              \p csr_col_ind or \p info pointer is invalid.
 *  \retval     rocsparse_status_memory_error the buffer for the gathered information
 *              could not be allocated.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const float*              csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const double*             csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmv_analysis(rocsparse_handle               handle,
                                           rocsparse_operation            trans,
                                           rocsparse_int                  m,
                                           rocsparse_int                  n,
                                           rocsparse_int                  nnz,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* csr_val,
                                           const rocsparse_int*           csr_row_ptr,
                                           const rocsparse_int*           csr_col_ind,
                                           rocsparse_mat_info             info);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmv_analysis(rocsparse_handle                handle,
                                           rocsparse_operation             trans,
                                           rocsparse_int                   m,
                                           rocsparse_int                   n,
                                           rocsparse_int                   nnz,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* csr_val,
                                           const rocsparse_int*            csr_row_ptr,
                                           const rocsparse_int*            csr_col_ind,
                                           rocsparse_mat_info              info);
/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse matrix vector multiplication using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmv_clear deallocates all memory that was allocated by
 *  rocsparse_scsrmv_analysis(), rocsparse_dcsrmv_analysis(), rocsparse_ccsrmv_analysis()
 *  or rocsparse_zcsrmv_analysis(). This is especially useful, if memory is an issue and
 *  the analysis data is not required anymore for further computation, e.g. when
 *  switching to another sparse matrix format.
 *
 *  \note
 *  Calling \p rocsparse_csrmv_clear is optional. All allocated resources will be
 *  cleared, when the opaque \ref rocsparse_mat_info struct is destroyed using
 *  rocsparse_destroy_mat_info().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[inout]
 *  info        structure that holds the information collected during analysis step.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval     rocsparse_status_memory_error the buffer for the gathered information
 *              could not be deallocated.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 * */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrmv_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup level2_module
 *  \brief Sparse matrix vector multiplication using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == rocsparse\_operation\_none} \\
 *        A^T, & \text{if trans == rocsparse\_operation\_transpose} \\
 *        A^H, & \text{if trans == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  The \p info parameter is optional and contains information collected by
 *  rocsparse_scsrmv_analysis(), rocsparse_dcsrmv_analysis(), rocsparse_ccsrmv_analysis()
 *  or rocsparse_zcsrmv_analysis(). If present, the information will be used to speed up
 *  the \p csrmv computation. If \p info == \p NULL, general \p csrmv routine will be
 *  used instead.
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *
 *          for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
 *          {
 *              y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
 *          }
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  info        information collected by rocsparse_scsrmv_analysis(),
 *              rocsparse_dcsrmv_analysis(), rocsparse_ccsrmv_analysis() or
 *              rocsparse_dcsrmv_analysis(), can be \p NULL if no information is
 *              available.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p x, \p beta or \p y pointer is
 *              invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example performs a sparse matrix vector multiplication in CSR format
 *  using additional meta data to improve performance.
 *  \code{.c}
 *      // Create matrix info structure
 *      rocsparse_mat_info info;
 *      rocsparse_create_mat_info(&info);
 *
 *      // Perform analysis step to obtain meta data
 *      rocsparse_scsrmv_analysis(handle,
 *                                rocsparse_operation_none,
 *                                m,
 *                                n,
 *                                nnz,
 *                                descr,
 *                                csr_val,
 *                                csr_row_ptr,
 *                                csr_col_ind,
 *                                info);
 *
 *      // Compute y = Ax
 *      rocsparse_scsrmv(handle,
 *                       rocsparse_operation_none,
 *                       m,
 *                       n,
 *                       nnz,
 *                       &alpha,
 *                       descr,
 *                       csr_val,
 *                       csr_row_ptr,
 *                       csr_col_ind,
 *                       info,
 *                       x,
 *                       &beta,
 *                       y);
 *
 *      // Do more work
 *      // ...
 *
 *      // Clean up
 *      rocsparse_destroy_mat_info(info);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int*           csr_row_ptr,
                                  const rocsparse_int*           csr_col_ind,
                                  rocsparse_mat_info             info,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int*            csr_row_ptr,
                                  const rocsparse_int*            csr_col_ind,
                                  rocsparse_mat_info              info,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse triangular solve using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrsv_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
 *  structural or numerical zero has been found during rocsparse_scsrsv_solve(),
 *  rocsparse_dcsrsv_solve(), rocsparse_ccsrsv_solve() or rocsparse_zcsrsv_solve()
 *  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
 *  using same index base as the CSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
 *
 *  \note \p rocsparse_csrsv_zero_pivot is a blocking function. It might influence
 *  performance negatively.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[inout]
 *  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
 *              invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_zero_pivot zero pivot has been found.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsv_zero_pivot(rocsparse_handle          handle,
                                            const rocsparse_mat_descr descr,
                                            rocsparse_mat_info        info,
                                            rocsparse_int*            position);

/*! \ingroup level2_module
 *  \brief Sparse triangular solve using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrsv_buffer_size returns the size of the temporary storage buffer that
 *  is required by rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(),
 *  rocsparse_ccsrsv_analysis(), rocsparse_zcsrsv_analysis(), rocsparse_scsrsv_solve(),
 *  rocsparse_dcsrsv_solve(), rocsparse_ccsrsv_solve() and rocsparse_zcsrsv_solve(). The
 *  temporary storage buffer must be allocated by the user. The size of the temporary
 *  storage buffer is identical to the size returned by rocsparse_scsrilu0_buffer_size(),
 *  rocsparse_dcsrilu0_buffer_size(), rocsparse_ccsrilu0_buffer_size() and
 *  rocsparse_zcsrilu0_buffer_size() if the matrix sparsity pattern is identical. The
 *  user allocated buffer can thus be shared between subsequent calls to those functions.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  buffer_size number of bytes of the temporary storage buffer required by
 *              rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(),
 *              rocsparse_ccsrsv_analysis(), rocsparse_zcsrsv_analysis(),
 *              rocsparse_scsrsv_solve(), rocsparse_dcsrsv_solve(),
 *              rocsparse_ccsrsv_solve() and rocsparse_zcsrsv_solve().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
 *              \p csr_col_ind, \p info or \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsv_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsv_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsv_buffer_size(rocsparse_handle               handle,
                                              rocsparse_operation            trans,
                                              rocsparse_int                  m,
                                              rocsparse_int                  nnz,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* csr_val,
                                              const rocsparse_int*           csr_row_ptr,
                                              const rocsparse_int*           csr_col_ind,
                                              rocsparse_mat_info             info,
                                              size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsv_buffer_size(rocsparse_handle                handle,
                                              rocsparse_operation             trans,
                                              rocsparse_int                   m,
                                              rocsparse_int                   nnz,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* csr_val,
                                              const rocsparse_int*            csr_row_ptr,
                                              const rocsparse_int*            csr_col_ind,
                                              rocsparse_mat_info              info,
                                              size_t*                         buffer_size);
/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse triangular solve using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrsv_analysis performs the analysis step for rocsparse_scsrsv_solve(),
 *  rocsparse_dcsrsv_solve(), rocsparse_ccsrsv_solve() and rocsparse_zcsrsv_solve(). It
 *  is expected that this function will be executed only once for a given matrix and
 *  particular operation type. The analysis meta data can be cleared by
 *  rocsparse_csrsv_clear().
 *
 *  \p rocsparse_csrsv_analysis can share its meta data with
 *  rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
 *  rocsparse_ccsrilu0_analysis() and rocsparse_zcsrilu0_analysis(). Selecting
 *  \ref rocsparse_analysis_policy_reuse policy can greatly improve computation
 *  performance of meta data. However, the user need to make sure that the sparsity
 *  pattern remains unchanged. If this cannot be assured,
 *  \ref rocsparse_analysis_policy_force has to be used.
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  info        structure that holds the information collected during
 *              the analysis step.
 *  @param[in]
 *  analysis    \ref rocsparse_analysis_policy_reuse or
 *              \ref rocsparse_analysis_policy_force.
 *  @param[in]
 *  solve       \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_row_ptr,
 *              \p csr_col_ind, \p info or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const float*              csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsv_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const double*             csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsv_analysis(rocsparse_handle               handle,
                                           rocsparse_operation            trans,
                                           rocsparse_int                  m,
                                           rocsparse_int                  nnz,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* csr_val,
                                           const rocsparse_int*           csr_row_ptr,
                                           const rocsparse_int*           csr_col_ind,
                                           rocsparse_mat_info             info,
                                           rocsparse_analysis_policy      analysis,
                                           rocsparse_solve_policy         solve,
                                           void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsv_analysis(rocsparse_handle                handle,
                                           rocsparse_operation             trans,
                                           rocsparse_int                   m,
                                           rocsparse_int                   nnz,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* csr_val,
                                           const rocsparse_int*            csr_row_ptr,
                                           const rocsparse_int*            csr_col_ind,
                                           rocsparse_mat_info              info,
                                           rocsparse_analysis_policy       analysis,
                                           rocsparse_solve_policy          solve,
                                           void*                           temp_buffer);
/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse triangular solve using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrsv_clear deallocates all memory that was allocated by
 *  rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(), rocsparse_ccsrsv_analysis()
 *  or rocsparse_zcsrsv_analysis(). This is especially useful, if memory is an issue and
 *  the analysis data is not required for further computation, e.g. when switching to
 *  another sparse matrix format. Calling \p rocsparse_csrsv_clear is optional. All
 *  allocated resources will be cleared, when the opaque \ref rocsparse_mat_info struct
 *  is destroyed using rocsparse_destroy_mat_info().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[inout]
 *  info        structure that holds the information collected during the analysis step.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval     rocsparse_status_memory_error the buffer holding the meta data could not
 *              be deallocated.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsv_clear(rocsparse_handle          handle,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_mat_info        info);

/*! \ingroup level2_module
 *  \brief Sparse triangular solve using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrsv_solve solves a sparse triangular linear system of a sparse
 *  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
 *  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
 *  \f[
 *    op(A) \cdot y = \alpha \cdot x,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == rocsparse\_operation\_none} \\
 *        A^T, & \text{if trans == rocsparse\_operation\_transpose} \\
 *        A^H, & \text{if trans == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \p rocsparse_csrsv_solve requires a user allocated temporary buffer. Its size is
 *  returned by rocsparse_scsrsv_buffer_size(), rocsparse_dcsrsv_buffer_size(),
 *  rocsparse_ccsrsv_buffer_size() or rocsparse_zcsrsv_buffer_size(). Furthermore,
 *  analysis meta data is required. It can be obtained by rocsparse_scsrsv_analysis(),
 *  rocsparse_dcsrsv_analysis(), rocsparse_ccsrsv_analysis() or
 *  rocsparse_zcsrsv_analysis(). \p rocsparse_csrsv_solve reports the first zero pivot
 *  (either numerical or structural zero). The zero pivot status can be checked calling
 *  rocsparse_csrsv_zero_pivot(). If
 *  \ref rocsparse_diag_type == \ref rocsparse_diag_type_unit, no zero pivot will be
 *  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
 *
 *  \note
 *  The sparse CSR matrix has to be sorted. This can be achieved by calling
 *  rocsparse_csrsort().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  x           array of \p m elements, holding the right-hand side.
 *  @param[out]
 *  y           array of \p m elements, holding the solution.
 *  @param[in]
 *  policy      \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p x or \p y pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  Consider the lower triangular \f$m \times m\f$ matrix \f$L\f$, stored in CSR
 *  storage format with unit diagonal. The following example solves \f$L \cdot y = x\f$.
 *  \code{.c}
 *      // Create rocSPARSE handle
 *      rocsparse_handle handle;
 *      rocsparse_create_handle(&handle);
 *
 *      // Create matrix descriptor
 *      rocsparse_mat_descr descr;
 *      rocsparse_create_mat_descr(&descr);
 *      rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower);
 *      rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_unit);
 *
 *      // Create matrix info structure
 *      rocsparse_mat_info info;
 *      rocsparse_create_mat_info(&info);
 *
 *      // Obtain required buffer size
 *      size_t buffer_size;
 *      rocsparse_dcsrsv_buffer_size(handle,
 *                                   rocsparse_operation_none,
 *                                   m,
 *                                   nnz,
 *                                   descr,
 *                                   csr_val,
 *                                   csr_row_ptr,
 *                                   csr_col_ind,
 *                                   info,
 *                                   &buffer_size);
 *
 *      // Allocate temporary buffer
 *      void* temp_buffer;
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Perform analysis step
 *      rocsparse_dcsrsv_analysis(handle,
 *                                rocsparse_operation_none,
 *                                m,
 *                                nnz,
 *                                descr,
 *                                csr_val,
 *                                csr_row_ptr,
 *                                csr_col_ind,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *
 *      // Solve Ly = x
 *      rocsparse_dcsrsv_solve(handle,
 *                             rocsparse_operation_none,
 *                             m,
 *                             nnz,
 *                             &alpha,
 *                             descr,
 *                             csr_val,
 *                             csr_row_ptr,
 *                             csr_col_ind,
 *                             info,
 *                             x,
 *                             y,
 *                             rocsparse_solve_policy_auto,
 *                             temp_buffer);
 *
 *      // No zero pivot should be found, with L having unit diagonal
 *
 *      // Clean up
 *      hipFree(temp_buffer);
 *      rocsparse_destroy_mat_info(info);
 *      rocsparse_destroy_mat_descr(descr);
 *      rocsparse_destroy_handle(handle);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsv_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        rocsparse_int             m,
                                        rocsparse_int             nnz,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        rocsparse_mat_info        info,
                                        const float*              x,
                                        float*                    y,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsv_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans,
                                        rocsparse_int             m,
                                        rocsparse_int             nnz,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        rocsparse_mat_info        info,
                                        const double*             x,
                                        double*                   y,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsv_solve(rocsparse_handle               handle,
                                        rocsparse_operation            trans,
                                        rocsparse_int                  m,
                                        rocsparse_int                  nnz,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* csr_val,
                                        const rocsparse_int*           csr_row_ptr,
                                        const rocsparse_int*           csr_col_ind,
                                        rocsparse_mat_info             info,
                                        const rocsparse_float_complex* x,
                                        rocsparse_float_complex*       y,
                                        rocsparse_solve_policy         policy,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsv_solve(rocsparse_handle                handle,
                                        rocsparse_operation             trans,
                                        rocsparse_int                   m,
                                        rocsparse_int                   nnz,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* csr_val,
                                        const rocsparse_int*            csr_row_ptr,
                                        const rocsparse_int*            csr_col_ind,
                                        rocsparse_mat_info              info,
                                        const rocsparse_double_complex* x,
                                        rocsparse_double_complex*       y,
                                        rocsparse_solve_policy          policy,
                                        void*                           temp_buffer);
/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse matrix vector multiplication using ELL storage format
 *
 *  \details
 *  \p rocsparse_ellmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in ELL storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == rocsparse\_operation\_none} \\
 *        A^T, & \text{if trans == rocsparse\_operation\_transpose} \\
 *        A^H, & \text{if trans == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *
 *          for(p = 0; p < ell_width; ++p)
 *          {
 *              idx = p * m + i;
 *
 *              if((ell_col_ind[idx] >= 0) && (ell_col_ind[idx] < n))
 *              {
 *                  y[i] = y[i] + alpha * ell_val[idx] * x[ell_col_ind[idx]];
 *              }
 *          }
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  m           number of rows of the sparse ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse ELL matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_val     array that contains the elements of the sparse ELL matrix. Padded
 *              elements should be zero.
 *  @param[in]
 *  ell_col_ind array that contains the column indices of the sparse ELL matrix.
 *              Padded column indices should be -1.
 *  @param[in]
 *  ell_width   number of non-zero elements per row of the sparse ELL matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p ell_val,
 *              \p ell_col_ind, \p x, \p beta or \p y pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sellmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              ell_val,
                                  const rocsparse_int*      ell_col_ind,
                                  rocsparse_int             ell_width,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dellmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             ell_val,
                                  const rocsparse_int*      ell_col_ind,
                                  rocsparse_int             ell_width,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cellmv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* ell_val,
                                  const rocsparse_int*           ell_col_ind,
                                  rocsparse_int                  ell_width,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zellmv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* ell_val,
                                  const rocsparse_int*            ell_col_ind,
                                  rocsparse_int                   ell_width,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse matrix vector multiplication using HYB storage format
 *
 *  \details
 *  \p rocsparse_hybmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in HYB storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == rocsparse\_operation\_none} \\
 *        A^T, & \text{if trans == rocsparse\_operation\_transpose} \\
 *        A^H, & \text{if trans == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref rocsparse_operation_none is supported.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse HYB matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  hyb         matrix in HYB storage format.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p hyb structure was not initialized with
 *              valid matrix sizes.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p hyb, \p x,
 *              \p beta or \p y pointer is invalid.
 *  \retval     rocsparse_status_invalid_value \p hyb structure was not initialized
 *              with a valid partitioning type.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_memory_error the buffer could not be allocated.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_shybmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat   hyb,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dhybmv(rocsparse_handle          handle,
                                  rocsparse_operation       trans,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const rocsparse_hyb_mat   hyb,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_chybmv(rocsparse_handle               handle,
                                  rocsparse_operation            trans,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_hyb_mat        hyb,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zhybmv(rocsparse_handle                handle,
                                  rocsparse_operation             trans,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_hyb_mat         hyb,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
 *  matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
 *  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans\_A == rocsparse\_operation\_none} \\
 *        A^T, & \text{if trans\_A == rocsparse\_operation\_transpose} \\
 *        A^H, & \text{if trans\_A == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans\_B == rocsparse\_operation\_none} \\
 *        B^T, & \text{if trans\_B == rocsparse\_operation\_transpose} \\
 *        B^H, & \text{if trans\_B == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < ldc; ++i)
 *      {
 *          for(j = 0; j < n; ++j)
 *          {
 *              C[i][j] = beta * C[i][j];
 *
 *              for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
 *              {
 *                  C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
 *              }
 *          }
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type.
 *  @param[in]
 *  trans_B     matrix \f$B\f$ operation type.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  n           number of columns of the dense matrix \f$op(B)\f$ and \f$C\f$.
 *  @param[in]
 *  k           number of columns of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix \f$A\f$.
 *  @param[in]
 *  B           array of dimension \f$ldb \times n\f$ (\f$op(B) == B\f$) or
 *              \f$ldb \times k\f$ (\f$op(B) == B^T\f$ or \f$op(B) == B^H\f$).
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$
 *              (\f$op(A) == A\f$) or \f$\max{(1, m)}\f$ (\f$op(A) == A^T\f$ or
 *              \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  C           array of dimension \f$ldc \times n\f$.
 *  @param[in]
 *  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$
 *              (\f$op(A) == A\f$) or \f$\max{(1, k)}\f$ (\f$op(A) == A^T\f$ or
 *              \f$op(A) == A^H\f$).
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz, \p ldb or \p ldc
 *              is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p B, \p beta or \p C pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans_A != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example multiplies a CSR matrix with a dense matrix.
 *  \code{.c}
 *      //     1 2 0 3 0
 *      // A = 0 4 5 0 0
 *      //     6 0 0 7 8
 *
 *      rocsparse_int m   = 3;
 *      rocsparse_int k   = 5;
 *      rocsparse_int nnz = 8;
 *
 *      csr_row_ptr[m+1] = {0, 3, 5, 8};             // device memory
 *      csr_col_ind[nnz] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
 *      csr_val[nnz]     = {1, 2, 3, 4, 5, 6, 7, 8}; // device memory
 *
 *      // Set dimension n of B
 *      rocsparse_int n = 64;
 *
 *      // Allocate and generate dense matrix B
 *      std::vector<float> hB(k * n);
 *      for(rocsparse_int i = 0; i < k * n; ++i)
 *      {
 *          hB[i] = static_cast<float>(rand()) / RAND_MAX;
 *      }
 *
 *      // Copy B to the device
 *      float* B;
 *      hipMalloc((void**)&B, sizeof(float) * k * n);
 *      hipMemcpy(B, hB.data(), sizeof(float) * k * n, hipMemcpyHostToDevice);
 *
 *      // alpha and beta
 *      float alpha = 1.0f;
 *      float beta  = 0.0f;
 *
 *      // Allocate memory for the resulting matrix C
 *      float* C;
 *      hipMalloc((void**)&C, sizeof(float) * m * n);
 *
 *      // Perform the matrix multiplication
 *      rocsparse_scsrmm(handle,
 *                       rocsparse_operation_none,
 *                       rocsparse_operation_none,
 *                       m,
 *                       n,
 *                       k,
 *                       nnz,
 *                       &alpha,
 *                       descr,
 *                       csr_val,
 *                       csr_row_ptr,
 *                       csr_col_ind,
 *                       B,
 *                       k,
 *                       &beta,
 *                       C,
 *                       m);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrmm(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const float*              B,
                                  rocsparse_int             ldb,
                                  const float*              beta,
                                  float*                    C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrmm(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const double*             B,
                                  rocsparse_int             ldb,
                                  const double*             beta,
                                  double*                   C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrmm(rocsparse_handle               handle,
                                  rocsparse_operation            trans_A,
                                  rocsparse_operation            trans_B,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  k,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int*           csr_row_ptr,
                                  const rocsparse_int*           csr_col_ind,
                                  const rocsparse_float_complex* B,
                                  rocsparse_int                  ldb,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       C,
                                  rocsparse_int                  ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrmm(rocsparse_handle                handle,
                                  rocsparse_operation             trans_A,
                                  rocsparse_operation             trans_B,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   k,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int*            csr_row_ptr,
                                  const rocsparse_int*            csr_col_ind,
                                  const rocsparse_double_complex* B,
                                  rocsparse_int                   ldb,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       C,
                                  rocsparse_int                   ldc);
/**@}*/

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */

/*! \ingroup extra_module
 *  \brief Sparse matrix sparse matrix multiplication using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrgemm_buffer_size returns the size of the temporary storage buffer
 *  that is required by rocsparse_csrgemm_nnz(), rocsparse_scsrgemm(),
 *  rocsparse_dcsrgemm(), rocsparse_ccsrgemm() and rocsparse_zcsrgemm(). The temporary
 *  storage buffer must be allocated by the user.
 *
 *  \note
 *  Please note, that for matrix products with more than 4096 non-zero entries per row,
 *  additional temporary storage buffer is allocated by the algorithm.
 *  \note
 *  Please note, that for matrix products with more than 8192 intermediate products per
 *  row, additional temporary storage buffer is allocated by the algorithm.
 *  \note
 *  Currently, only \p trans_A == \p trans_B == \ref rocsparse_operation_none is
 *  supported.
 *  \note
 *  Currently, only \ref rocsparse_matrix_type_general is supported.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  trans_A         matrix \f$A\f$ operation type.
 *  @param[in]
 *  trans_B         matrix \f$B\f$ operation type.
 *  @param[in]
 *  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
 *  @param[in]
 *  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
 *                  \f$C\f$.
 *  @param[in]
 *  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
 *                  rows of the sparse CSR matrix \f$op(B)\f$.
 *  @param[in]
 *  alpha           scalar \f$\alpha\f$.
 *  @param[in]
 *  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
 *                  that point to the start of every row of the sparse CSR matrix
 *                  \f$op(A)\f$.
 *  @param[in]
 *  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
 *                  sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
 *  @param[in]
 *  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
 *                  that point to the start of every row of the sparse CSR matrix
 *                  \f$op(B)\f$.
 *  @param[in]
 *  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
 *                  sparse CSR matrix \f$B\f$.
 *  @param[in]
 *  beta            scalar \f$\beta\f$.
 *  @param[in]
 *  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the sparse
 *                  CSR matrix \f$D\f$.
 *  @param[inout]
 *  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
 *  @param[out]
 *  buffer_size     number of bytes of the temporary storage buffer required by
 *                  rocsparse_csrgemm_nnz(), rocsparse_scsrgemm(), rocsparse_dcsrgemm(),
 *                  rocsparse_ccsrgemm() and rocsparse_zcsrgemm().
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
 *          \p nnz_D is invalid.
 *  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
 *          \p descr_A, \p csr_row_ptr_A, \p csr_col_ind_A, \p descr_B,
 *          \p csr_row_ptr_B or \p csr_col_ind_B are invalid if \p alpha is valid,
 *          \p descr_D, \p csr_row_ptr_D or \p csr_col_ind_D is invalid if \p beta is
 *          valid, \p info_C or \p buffer_size is invalid.
 *  \retval rocsparse_status_not_implemented
 *          \p trans_A != \ref rocsparse_operation_none,
 *          \p trans_B != \ref rocsparse_operation_none, or
 *          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrgemm_buffer_size(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                rocsparse_int             k,
                                                const float*              alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnz_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnz_B,
                                                const rocsparse_int*      csr_row_ptr_B,
                                                const rocsparse_int*      csr_col_ind_B,
                                                const float*              beta,
                                                const rocsparse_mat_descr descr_D,
                                                rocsparse_int             nnz_D,
                                                const rocsparse_int*      csr_row_ptr_D,
                                                const rocsparse_int*      csr_col_ind_D,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrgemm_buffer_size(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                rocsparse_int             k,
                                                const double*             alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnz_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnz_B,
                                                const rocsparse_int*      csr_row_ptr_B,
                                                const rocsparse_int*      csr_col_ind_B,
                                                const double*             beta,
                                                const rocsparse_mat_descr descr_D,
                                                rocsparse_int             nnz_D,
                                                const rocsparse_int*      csr_row_ptr_D,
                                                const rocsparse_int*      csr_col_ind_D,
                                                rocsparse_mat_info        info_C,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrgemm_buffer_size(rocsparse_handle               handle,
                                                rocsparse_operation            trans_A,
                                                rocsparse_operation            trans_B,
                                                rocsparse_int                  m,
                                                rocsparse_int                  n,
                                                rocsparse_int                  k,
                                                const rocsparse_float_complex* alpha,
                                                const rocsparse_mat_descr      descr_A,
                                                rocsparse_int                  nnz_A,
                                                const rocsparse_int*           csr_row_ptr_A,
                                                const rocsparse_int*           csr_col_ind_A,
                                                const rocsparse_mat_descr      descr_B,
                                                rocsparse_int                  nnz_B,
                                                const rocsparse_int*           csr_row_ptr_B,
                                                const rocsparse_int*           csr_col_ind_B,
                                                const rocsparse_float_complex* beta,
                                                const rocsparse_mat_descr      descr_D,
                                                rocsparse_int                  nnz_D,
                                                const rocsparse_int*           csr_row_ptr_D,
                                                const rocsparse_int*           csr_col_ind_D,
                                                rocsparse_mat_info             info_C,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrgemm_buffer_size(rocsparse_handle                handle,
                                                rocsparse_operation             trans_A,
                                                rocsparse_operation             trans_B,
                                                rocsparse_int                   m,
                                                rocsparse_int                   n,
                                                rocsparse_int                   k,
                                                const rocsparse_double_complex* alpha,
                                                const rocsparse_mat_descr       descr_A,
                                                rocsparse_int                   nnz_A,
                                                const rocsparse_int*            csr_row_ptr_A,
                                                const rocsparse_int*            csr_col_ind_A,
                                                const rocsparse_mat_descr       descr_B,
                                                rocsparse_int                   nnz_B,
                                                const rocsparse_int*            csr_row_ptr_B,
                                                const rocsparse_int*            csr_col_ind_B,
                                                const rocsparse_double_complex* beta,
                                                const rocsparse_mat_descr       descr_D,
                                                rocsparse_int                   nnz_D,
                                                const rocsparse_int*            csr_row_ptr_D,
                                                const rocsparse_int*            csr_col_ind_D,
                                                rocsparse_mat_info              info_C,
                                                size_t*                         buffer_size);
/**@}*/

/*! \ingroup extra_module
 *  \brief Sparse matrix sparse matrix multiplication using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrgemm_nnz computes the total CSR non-zero elements and the CSR row
 *  offsets, that point to the start of every row of the sparse CSR matrix, of the
 *  resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
 *  with size \p m + 1.
 *  The required buffer size can be obtained by rocsparse_scsrgemm_buffer_size(),
 *  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() and
 *  rocsparse_zcsrgemm_buffer_size(), respectively.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *  \note
 *  Please note, that for matrix products with more than 8192 intermediate products per
 *  row, additional temporary storage buffer is allocated by the algorithm.
 *  \note
 *  Currently, only \p trans_A == \p trans_B == \ref rocsparse_operation_none is
 *  supported.
 *  \note
 *  Currently, only \ref rocsparse_matrix_type_general is supported.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  trans_A         matrix \f$A\f$ operation type.
 *  @param[in]
 *  trans_B         matrix \f$B\f$ operation type.
 *  @param[in]
 *  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
 *  @param[in]
 *  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
 *                  \f$C\f$.
 *  @param[in]
 *  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
 *                  rows of the sparse CSR matrix \f$op(B)\f$.
 *  @param[in]
 *  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
 *                  that point to the start of every row of the sparse CSR matrix
 *                  \f$op(A)\f$.
 *  @param[in]
 *  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
 *                  sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
 *  @param[in]
 *  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
 *                  that point to the start of every row of the sparse CSR matrix
 *                  \f$op(B)\f$.
 *  @param[in]
 *  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
 *                  sparse CSR matrix \f$B\f$.
 *  @param[in]
 *  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the sparse
 *                  CSR matrix \f$D\f$.
 *  @param[in]
 *  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix \f$C\f$.
 *  @param[out]
 *  nnz_C           pointer to the number of non-zero entries of the sparse CSR
 *                  matrix \f$C\f$.
 *  @param[in]
 *  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned
 *                  by rocsparse_scsrgemm_buffer_size(),
 *                  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() or
 *                  rocsparse_zcsrgemm_buffer_size().
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
 *          \p nnz_D is invalid.
 *  \retval rocsparse_status_invalid_pointer \p descr_A, \p csr_row_ptr_A,
 *          \p csr_col_ind_A, \p descr_B, \p csr_row_ptr_B, \p csr_col_ind_B,
 *          \p descr_D, \p csr_row_ptr_D, \p csr_col_ind_D, \p descr_C,
 *          \p csr_row_ptr_C, \p nnz_C, \p info_C or \p temp_buffer is invalid.
 *  \retval rocsparse_status_memory_error additional buffer for long rows could not be
 *          allocated.
 *  \retval rocsparse_status_not_implemented
 *          \p trans_A != \ref rocsparse_operation_none,
 *          \p trans_B != \ref rocsparse_operation_none, or
 *          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       rocsparse_int             k,
                                       const rocsparse_mat_descr descr_A,
                                       rocsparse_int             nnz_A,
                                       const rocsparse_int*      csr_row_ptr_A,
                                       const rocsparse_int*      csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       rocsparse_int             nnz_B,
                                       const rocsparse_int*      csr_row_ptr_B,
                                       const rocsparse_int*      csr_col_ind_B,
                                       const rocsparse_mat_descr descr_D,
                                       rocsparse_int             nnz_D,
                                       const rocsparse_int*      csr_row_ptr_D,
                                       const rocsparse_int*      csr_col_ind_D,
                                       const rocsparse_mat_descr descr_C,
                                       rocsparse_int*            csr_row_ptr_C,
                                       rocsparse_int*            nnz_C,
                                       const rocsparse_mat_info  info_C,
                                       void*                     temp_buffer);

/*! \ingroup extra_module
 *  \brief Sparse matrix sparse matrix multiplication using CSR storage format
 *
 *  \details
 *  \p rocsparse_csrgemm multiplies the scalar \f$\alpha\f$ with the sparse
 *  \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
 *  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
 *  to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
 *  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
 *  storage format, such
 *  that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot D,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans\_A == rocsparse\_operation\_none} \\
 *        A^T, & \text{if trans\_A == rocsparse\_operation\_transpose} \\
 *        A^H, & \text{if trans\_A == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans\_B == rocsparse\_operation\_none} \\
 *        B^T, & \text{if trans\_B == rocsparse\_operation\_transpose} \\
 *        B^H, & \text{if trans\_B == rocsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
 *  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
 *  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
 *  the sparse CSR matrix C. Both can be obtained by rocsparse_csrgemm_nnz(). The
 *  required buffer size for the computation can be obtained by
 *  rocsparse_scsrgemm_buffer_size(), rocsparse_dcsrgemm_buffer_size(),
 *  rocsparse_ccsrgemm_buffer_size() and rocsparse_zcsrgemm_buffer_size(), respectively.
 *
 *  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
 *  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot op(A) \cdot op(B)\f$ will be computed.
 *  \note \f$\alpha == beta == 0\f$ is invalid.
 *  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
 *  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
 *  \note Currently, only \ref rocsparse_matrix_type_general is supported.
 *  \note This function is non blocking and executed asynchronously with respect to the
 *        host. It may return before the actual computation has finished.
 *  \note Please note, that for matrix products with more than 4096 non-zero entries per
 *  row, additional temporary storage buffer is allocated by the algorithm.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  trans_A         matrix \f$A\f$ operation type.
 *  @param[in]
 *  trans_B         matrix \f$B\f$ operation type.
 *  @param[in]
 *  m               number of rows of the sparse CSR matrix \f$op(A)\f$ and \f$C\f$.
 *  @param[in]
 *  n               number of columns of the sparse CSR matrix \f$op(B)\f$ and
 *                  \f$C\f$.
 *  @param[in]
 *  k               number of columns of the sparse CSR matrix \f$op(A)\f$ and number of
 *                  rows of the sparse CSR matrix \f$op(B)\f$.
 *  @param[in]
 *  alpha           scalar \f$\alpha\f$.
 *  @param[in]
 *  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_val_A       array of \p nnz_A elements of the sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  csr_row_ptr_A   array of \p m+1 elements (\f$op(A) == A\f$, \p k+1 otherwise)
 *                  that point to the start of every row of the sparse CSR matrix
 *                  \f$op(A)\f$.
 *  @param[in]
 *  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
 *                  sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
 *  @param[in]
 *  csr_val_B       array of \p nnz_B elements of the sparse CSR matrix \f$B\f$.
 *  @param[in]
 *  csr_row_ptr_B   array of \p k+1 elements (\f$op(B) == B\f$, \p m+1 otherwise)
 *                  that point to the start of every row of the sparse CSR matrix
 *                  \f$op(B)\f$.
 *  @param[in]
 *  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
 *                  sparse CSR matrix \f$B\f$.
 *  @param[in]
 *  beta            scalar \f$\beta\f$.
 *  @param[in]
 *  descr_D         descriptor of the sparse CSR matrix \f$D\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  nnz_D           number of non-zero entries of the sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  csr_val_D       array of \p nnz_D elements of the sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  csr_row_ptr_D   array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  csr_col_ind_D   array of \p nnz_D elements containing the column indices of the
 *                  sparse CSR matrix \f$D\f$.
 *  @param[in]
 *  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currenty, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_val_C       array of \p nnz_C elements of the sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix \f$C\f$.
 *  @param[out]
 *  csr_col_ind_C   array of \p nnz_C elements containing the column indices of the
 *                  sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  info_C          structure that holds meta data for the sparse CSR matrix \f$C\f$.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned
 *                  by rocsparse_scsrgemm_buffer_size(),
 *                  rocsparse_dcsrgemm_buffer_size(), rocsparse_ccsrgemm_buffer_size() or
 *                  rocsparse_zcsrgemm_buffer_size().
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz_A, \p nnz_B or
 *          \p nnz_D is invalid.
 *  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
 *          \p descr_A, \p csr_val_A, \p csr_row_ptr_A, \p csr_col_ind_A, \p descr_B,
 *          \p csr_val_B, \p csr_row_ptr_B or \p csr_col_ind_B are invalid if \p alpha
 *          is valid, \p descr_D, \p csr_val_D, \p csr_row_ptr_D or \p csr_col_ind_D is
 *          invalid if \p beta is valid, \p csr_val_C, \p csr_row_ptr_C,
 *          \p csr_col_ind_C, \p info_C or \p temp_buffer is invalid.
 *  \retval rocsparse_status_memory_error additional buffer for long rows could not be
 *          allocated.
 *  \retval rocsparse_status_not_implemented
 *          \p trans_A != \ref rocsparse_operation_none,
 *          \p trans_B != \ref rocsparse_operation_none, or
 *          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example multiplies two CSR matrices with a scalar alpha and adds the result to
 *  another CSR matrix.
 *  \code{.c}
 *  // Initialize scalar multipliers
 *  float alpha = 2.0f;
 *  float beta  = 1.0f;
 *
 *  // Create matrix descriptors
 *  rocsparse_mat_descr descr_A;
 *  rocsparse_mat_descr descr_B;
 *  rocsparse_mat_descr descr_C;
 *  rocsparse_mat_descr descr_D;
 *
 *  rocsparse_create_mat_descr(&descr_A);
 *  rocsparse_create_mat_descr(&descr_B);
 *  rocsparse_create_mat_descr(&descr_C);
 *  rocsparse_create_mat_descr(&descr_D);
 *
 *  // Create matrix info structure
 *  rocsparse_mat_info info_C;
 *  rocsparse_create_mat_info(&info_C);
 *
 *  // Set pointer mode
 *  rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
 *
 *  // Query rocsparse for the required buffer size
 *  size_t buffer_size;
 *
 *  rocsparse_scsrgemm_buffer_size(handle,
 *                                 rocsparse_operation_none,
 *                                 rocsparse_operation_none,
 *                                 m,
 *                                 n,
 *                                 k,
 *                                 &alpha,
 *                                 descr_A,
 *                                 nnz_A,
 *                                 csr_row_ptr_A,
 *                                 csr_col_ind_A,
 *                                 descr_B,
 *                                 nnz_B,
 *                                 csr_row_ptr_B,
 *                                 csr_col_ind_B,
 *                                 &beta,
 *                                 descr_D,
 *                                 nnz_D,
 *                                 csr_row_ptr_D,
 *                                 csr_col_ind_D,
 *                                 info_C,
 *                                 &buffer_size);
 *
 *  // Allocate buffer
 *  void* buffer;
 *  hipMalloc(&buffer, buffer_size);
 *
 *  // Obtain number of total non-zero entries in C and row pointers of C
 *  rocsparse_int nnz_C;
 *  hipMalloc((void**)&csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1));
 *
 *  rocsparse_csrgemm_nnz(handle,
 *                        rocsparse_operation_none,
 *                        rocsparse_operation_none,
 *                        m,
 *                        n,
 *                        k,
 *                        descr_A,
 *                        nnz_A,
 *                        csr_row_ptr_A,
 *                        csr_col_ind_A,
 *                        descr_B,
 *                        nnz_B,
 *                        csr_row_ptr_B,
 *                        csr_col_ind_B,
 *                        descr_D,
 *                        nnz_D,
 *                        csr_row_ptr_D,
 *                        csr_col_ind_D,
 *                        descr_C,
 *                        csr_row_ptr_C,
 *                        &nnz_C,
 *                        info_C,
 *                        buffer);
 *
 *  // Compute column indices and values of C
 *  hipMalloc((void**)&csr_col_ind_C, sizeof(rocsparse_int) * nnz_C);
 *  hipMalloc((void**)&csr_val_C, sizeof(float) * nnz_C);
 *
 *  rocsparse_scsrgemm(handle,
 *                     rocsparse_operation_none,
 *                     rocsparse_operation_none,
 *                     m,
 *                     n,
 *                     k,
 *                     &alpha,
 *                     descr_A,
 *                     nnz_A,
 *                     csr_val_A,
 *                     csr_row_ptr_A,
 *                     csr_col_ind_A,
 *                     descr_B,
 *                     nnz_B,
 *                     csr_val_B,
 *                     csr_row_ptr_B,
 *                     csr_col_ind_B,
 *                     &beta,
 *                     descr_D,
 *                     nnz_D,
 *                     csr_val_D,
 *                     csr_row_ptr_D,
 *                     csr_col_ind_D,
 *                     descr_C,
 *                     csr_val_C,
 *                     csr_row_ptr_C,
 *                     csr_col_ind_C,
 *                     info_C,
 *                     buffer);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrgemm(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    rocsparse_int             k,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const float*              csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const float*              csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const float*              beta,
                                    const rocsparse_mat_descr descr_D,
                                    rocsparse_int             nnz_D,
                                    const float*              csr_val_D,
                                    const rocsparse_int*      csr_row_ptr_D,
                                    const rocsparse_int*      csr_col_ind_D,
                                    const rocsparse_mat_descr descr_C,
                                    float*                    csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C,
                                    const rocsparse_mat_info  info_C,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrgemm(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    rocsparse_int             k,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const double*             csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const double*             csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const double*             beta,
                                    const rocsparse_mat_descr descr_D,
                                    rocsparse_int             nnz_D,
                                    const double*             csr_val_D,
                                    const rocsparse_int*      csr_row_ptr_D,
                                    const rocsparse_int*      csr_col_ind_D,
                                    const rocsparse_mat_descr descr_C,
                                    double*                   csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C,
                                    const rocsparse_mat_info  info_C,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrgemm(rocsparse_handle               handle,
                                    rocsparse_operation            trans_A,
                                    rocsparse_operation            trans_B,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    rocsparse_int                  k,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr_A,
                                    rocsparse_int                  nnz_A,
                                    const rocsparse_float_complex* csr_val_A,
                                    const rocsparse_int*           csr_row_ptr_A,
                                    const rocsparse_int*           csr_col_ind_A,
                                    const rocsparse_mat_descr      descr_B,
                                    rocsparse_int                  nnz_B,
                                    const rocsparse_float_complex* csr_val_B,
                                    const rocsparse_int*           csr_row_ptr_B,
                                    const rocsparse_int*           csr_col_ind_B,
                                    const rocsparse_float_complex* beta,
                                    const rocsparse_mat_descr      descr_D,
                                    rocsparse_int                  nnz_D,
                                    const rocsparse_float_complex* csr_val_D,
                                    const rocsparse_int*           csr_row_ptr_D,
                                    const rocsparse_int*           csr_col_ind_D,
                                    const rocsparse_mat_descr      descr_C,
                                    rocsparse_float_complex*       csr_val_C,
                                    const rocsparse_int*           csr_row_ptr_C,
                                    rocsparse_int*                 csr_col_ind_C,
                                    const rocsparse_mat_info       info_C,
                                    void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrgemm(rocsparse_handle                handle,
                                    rocsparse_operation             trans_A,
                                    rocsparse_operation             trans_B,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    rocsparse_int                   k,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr_A,
                                    rocsparse_int                   nnz_A,
                                    const rocsparse_double_complex* csr_val_A,
                                    const rocsparse_int*            csr_row_ptr_A,
                                    const rocsparse_int*            csr_col_ind_A,
                                    const rocsparse_mat_descr       descr_B,
                                    rocsparse_int                   nnz_B,
                                    const rocsparse_double_complex* csr_val_B,
                                    const rocsparse_int*            csr_row_ptr_B,
                                    const rocsparse_int*            csr_col_ind_B,
                                    const rocsparse_double_complex* beta,
                                    const rocsparse_mat_descr       descr_D,
                                    rocsparse_int                   nnz_D,
                                    const rocsparse_double_complex* csr_val_D,
                                    const rocsparse_int*            csr_row_ptr_D,
                                    const rocsparse_int*            csr_col_ind_D,
                                    const rocsparse_mat_descr       descr_C,
                                    rocsparse_double_complex*       csr_val_C,
                                    const rocsparse_int*            csr_row_ptr_C,
                                    rocsparse_int*                  csr_col_ind_C,
                                    const rocsparse_mat_info        info_C,
                                    void*                           temp_buffer);
/**@}*/

/*
 * ===========================================================================
 *    preconditioner SPARSE
 * ===========================================================================
 */

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_csrilu0_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
 *  structural or numerical zero has been found during rocsparse_scsrilu0(),
 *  rocsparse_dcsrilu0(), rocsparse_ccsrilu0() or rocsparse_zcsrilu0() computation. The
 *  first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same index
 *  base as the CSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
 *
 *  \note \p rocsparse_csrilu0_zero_pivot is a blocking function. It might influence
 *  performance negatively.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[inout]
 *  position    pointer to zero pivot \f$j\f$, can be in host or device memory.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info or \p position pointer is
 *              invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_zero_pivot zero pivot has been found.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle   handle,
                                              rocsparse_mat_info info,
                                              rocsparse_int*     position);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_csrilu0_buffer_size returns the size of the temporary storage buffer
 *  that is required by rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
 *  rocsparse_ccsrilu0_analysis(), rocsparse_zcsrilu0_analysis(), rocsparse_scsrilu0(),
 *  rocsparse_dcsrilu0(), rocsparse_ccsrilu0() and rocsparse_zcsrilu0(). The temporary
 *  storage buffer must be allocated by the user. The size of the temporary storage
 *  buffer is identical to the size returned by rocsparse_scsrsv_buffer_size(),
 *  rocsparse_dcsrsv_buffer_size(), rocsparse_ccsrsv_buffer_size() and
 *  rocsparse_zcsrsv_buffer_size() if the matrix sparsity pattern is identical. The user
 *  allocated buffer can thus be shared between subsequent calls to those functions.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  buffer_size number of bytes of the temporary storage buffer required by
 *              rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
 *              rocsparse_ccsrilu0_analysis(), rocsparse_zcsrilu0_analysis(),
 *              rocsparse_scsrilu0(), rocsparse_dcsrilu0(), rocsparse_ccsrilu0() and
 *              rocsparse_zcsrilu0().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
 *              \p csr_col_ind, \p info or \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const float*              csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const double*             csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_buffer_size(rocsparse_handle               handle,
                                                rocsparse_int                  m,
                                                rocsparse_int                  nnz,
                                                const rocsparse_mat_descr      descr,
                                                const rocsparse_float_complex* csr_val,
                                                const rocsparse_int*           csr_row_ptr,
                                                const rocsparse_int*           csr_col_ind,
                                                rocsparse_mat_info             info,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_buffer_size(rocsparse_handle                handle,
                                                rocsparse_int                   m,
                                                rocsparse_int                   nnz,
                                                const rocsparse_mat_descr       descr,
                                                const rocsparse_double_complex* csr_val,
                                                const rocsparse_int*            csr_row_ptr,
                                                const rocsparse_int*            csr_col_ind,
                                                rocsparse_mat_info              info,
                                                size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_csrilu0_analysis performs the analysis step for rocsparse_scsrilu0(),
 *  rocsparse_dcsrilu0(), rocsparse_ccsrilu0() and rocsparse_zcsrilu0(). It is expected
 *  that this function will be executed only once for a given matrix and particular
 *  operation type. The analysis meta data can be cleared by rocsparse_csrilu0_clear().
 *
 *  \p rocsparse_csrilu0_analysis can share its meta data with
 *  rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(), rocsparse_ccsrsv_analysis()
 *  and rocsparse_zcsrsv_analysis(). Selecting \ref rocsparse_analysis_policy_reuse
 *  policy can greatly improve computation performance of meta data. However, the user
 *  need to make sure that the sparsity pattern remains unchanged. If this cannot be
 *  assured, \ref rocsparse_analysis_policy_force has to be used.
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  info        structure that holds the information collected during
 *              the analysis step.
 *  @param[in]
 *  analysis    \ref rocsparse_analysis_policy_reuse or
 *              \ref rocsparse_analysis_policy_force.
 *  @param[in]
 *  solve       \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr,
 *              \p csr_col_ind, \p info or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const float*              csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const double*             csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_analysis(rocsparse_handle               handle,
                                             rocsparse_int                  m,
                                             rocsparse_int                  nnz,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* csr_val,
                                             const rocsparse_int*           csr_row_ptr,
                                             const rocsparse_int*           csr_col_ind,
                                             rocsparse_mat_info             info,
                                             rocsparse_analysis_policy      analysis,
                                             rocsparse_solve_policy         solve,
                                             void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_analysis(rocsparse_handle                handle,
                                             rocsparse_int                   m,
                                             rocsparse_int                   nnz,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* csr_val,
                                             const rocsparse_int*            csr_row_ptr,
                                             const rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info              info,
                                             rocsparse_analysis_policy       analysis,
                                             rocsparse_solve_policy          solve,
                                             void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_csrilu0_clear deallocates all memory that was allocated by
 *  rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
 *  rocsparse_ccsrilu0_analysis() or rocsparse_zcsrilu0_analysis(). This is especially
 *  useful, if memory is an issue and the analysis data is not required for further
 *  computation.
 *
 *  \note
 *  Calling \p rocsparse_csrilu0_clear is optional. All allocated resources will be
 *  cleared, when the opaque \ref rocsparse_mat_info struct is destroyed using
 *  rocsparse_destroy_mat_info().
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[inout]
 *  info        structure that holds the information collected during the analysis step.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval     rocsparse_status_memory_error the buffer holding the meta data could not
 *              be deallocated.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrilu0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_csrilu0 computes the incomplete LU factorization with 0 fill-ins and no
 *  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LU
 *  \f]
 *
 *  \p rocsparse_csrilu0 requires a user allocated temporary buffer. Its size is returned
 *  by rocsparse_scsrilu0_buffer_size(), rocsparse_dcsrilu0_buffer_size(),
 *  rocsparse_ccsrilu0_buffer_size() or rocsparse_zcsrilu0_buffer_size(). Furthermore,
 *  analysis meta data is required. It can be obtained by rocsparse_scsrilu0_analysis(),
 *  rocsparse_dcsrilu0_analysis(), rocsparse_ccsrilu0_analysis() or
 *  rocsparse_zcsrilu0_analysis(). \p rocsparse_csrilu0 reports the first zero pivot
 *  (either numerical or structural zero). The zero pivot status can be obtained by
 *  calling rocsparse_csrilu0_zero_pivot().
 *
 *  \note
 *  The sparse CSR matrix has to be sorted. This can be achieved by calling
 *  rocsparse_csrsort().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[inout]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy      \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_val, \p csr_row_ptr
 *              or \p csr_col_ind pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans != \ref rocsparse_operation_none or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  Consider the sparse \f$m \times m\f$ matrix \f$A\f$, stored in CSR
 *  storage format. The following example computes the incomplete LU factorization
 *  \f$M \approx LU\f$ and solves the preconditioned system \f$My = x\f$.
 *  \code{.c}
 *      // Create rocSPARSE handle
 *      rocsparse_handle handle;
 *      rocsparse_create_handle(&handle);
 *
 *      // Create matrix descriptor for M
 *      rocsparse_mat_descr descr_M;
 *      rocsparse_create_mat_descr(&descr_M);
 *
 *      // Create matrix descriptor for L
 *      rocsparse_mat_descr descr_L;
 *      rocsparse_create_mat_descr(&descr_L);
 *      rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower);
 *      rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit);
 *
 *      // Create matrix descriptor for U
 *      rocsparse_mat_descr descr_U;
 *      rocsparse_create_mat_descr(&descr_U);
 *      rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper);
 *      rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit);
 *
 *      // Create matrix info structure
 *      rocsparse_mat_info info;
 *      rocsparse_create_mat_info(&info);
 *
 *      // Obtain required buffer size
 *      size_t buffer_size_M;
 *      size_t buffer_size_L;
 *      size_t buffer_size_U;
 *      rocsparse_dcsrilu0_buffer_size(handle,
 *                                    m,
 *                                    nnz,
 *                                    descr_M,
 *                                    csr_val,
 *                                    csr_row_ptr,
 *                                    csr_col_ind,
 *                                    info,
 *                                    &buffer_size_M);
 *      rocsparse_dcsrsv_buffer_size(handle,
 *                                   rocsparse_operation_none,
 *                                   m,
 *                                   nnz,
 *                                   descr_L,
 *                                   csr_val,
 *                                   csr_row_ptr,
 *                                   csr_col_ind,
 *                                   info,
 *                                   &buffer_size_L);
 *      rocsparse_dcsrsv_buffer_size(handle,
 *                                   rocsparse_operation_none,
 *                                   m,
 *                                   nnz,
 *                                   descr_U,
 *                                   csr_val,
 *                                   csr_row_ptr,
 *                                   csr_col_ind,
 *                                   info,
 *                                   &buffer_size_U);
 *
 *      size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_U));
 *
 *      // Allocate temporary buffer
 *      void* temp_buffer;
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
 *      // computation performance
 *      rocsparse_dcsrilu0_analysis(handle,
 *                                  m,
 *                                  nnz,
 *                                  descr_M,
 *                                  csr_val,
 *                                  csr_row_ptr,
 *                                  csr_col_ind,
 *                                  info,
 *                                  rocsparse_analysis_policy_reuse,
 *                                  rocsparse_solve_policy_auto,
 *                                  temp_buffer);
 *      rocsparse_dcsrsv_analysis(handle,
 *                                rocsparse_operation_none,
 *                                m,
 *                                nnz,
 *                                descr_L,
 *                                csr_val,
 *                                csr_row_ptr,
 *                                csr_col_ind,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *      rocsparse_dcsrsv_analysis(handle,
 *                                rocsparse_operation_none,
 *                                m,
 *                                nnz,
 *                                descr_U,
 *                                csr_val,
 *                                csr_row_ptr,
 *                                csr_col_ind,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *
 *      // Check for zero pivot
 *      rocsparse_int position;
 *      if(rocsparse_status_zero_pivot == rocsparse_csrilu0_zero_pivot(handle,
 *                                                                     info,
 *                                                                     &position))
 *      {
 *          printf("A has structural zero at A(%d,%d)\n", position, position);
 *      }
 *
 *      // Compute incomplete LU factorization
 *      rocsparse_dcsrilu0(handle,
 *                         m,
 *                         nnz,
 *                         descr_M,
 *                         csr_val,
 *                         csr_row_ptr,
 *                         csr_col_ind,
 *                         info,
 *                         rocsparse_solve_policy_auto,
 *                         temp_buffer);
 *
 *      // Check for zero pivot
 *      if(rocsparse_status_zero_pivot == rocsparse_csrilu0_zero_pivot(handle,
 *                                                                     info,
 *                                                                     &position))
 *      {
 *          printf("U has structural and/or numerical zero at U(%d,%d)\n",
 *                 position,
 *                 position);
 *      }
 *
 *      // Solve Lz = x
 *      rocsparse_dcsrsv_solve(handle,
 *                             rocsparse_operation_none,
 *                             m,
 *                             nnz,
 *                             &alpha,
 *                             descr_L,
 *                             csr_val,
 *                             csr_row_ptr,
 *                             csr_col_ind,
 *                             info,
 *                             x,
 *                             z,
 *                             rocsparse_solve_policy_auto,
 *                             temp_buffer);
 *
 *      // Solve Uy = z
 *      rocsparse_dcsrsv_solve(handle,
 *                             rocsparse_operation_none,
 *                             m,
 *                             nnz,
 *                             &alpha,
 *                             descr_U,
 *                             csr_val,
 *                             csr_row_ptr,
 *                             csr_col_ind,
 *                             info,
 *                             z,
 *                             y,
 *                             rocsparse_solve_policy_auto,
 *                             temp_buffer);
 *
 *      // Clean up
 *      hipFree(temp_buffer);
 *      rocsparse_destroy_mat_info(info);
 *      rocsparse_destroy_mat_descr(descr_M);
 *      rocsparse_destroy_mat_descr(descr_L);
 *      rocsparse_destroy_mat_descr(descr_U);
 *      rocsparse_destroy_handle(handle);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    float*                    csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    double*                   csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_float_complex*  csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_double_complex* csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);
/**@}*/

/*
 * ===========================================================================
 *    Sparse Format Conversions
 * ===========================================================================
 */

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a sparse COO matrix
 *
 *  \details
 *  \p rocsparse_csr2coo converts the CSR array containing the row offsets, that point
 *  to the start of every row, into a COO array of row indices.
 *
 *  \note
 *  It can also be used to convert a CSC array containing the column offsets into a COO
 *  array of column indices.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row
 *              of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[out]
 *  coo_row_ind array of \p nnz elements containing the row indices of the sparse COO
 *              matrix.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr or \p coo_row_ind
 *              pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *
 *  \par Example
 *  This example converts a CSR matrix into a COO matrix.
 *  \code{.c}
 *      //     1 2 0 3 0
 *      // A = 0 4 5 0 0
 *      //     6 0 0 7 8
 *
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 5;
 *      rocsparse_int nnz = 8;
 *
 *      csr_row_ptr[m+1] = {0, 3, 5, 8};             // device memory
 *      csr_col_ind[nnz] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
 *      csr_val[nnz]     = {1, 2, 3, 4, 5, 6, 7, 8}; // device memory
 *
 *      // Allocate COO matrix arrays
 *      rocsparse_int* coo_row_ind;
 *      rocsparse_int* coo_col_ind;
 *      float* coo_val;
 *
 *      hipMalloc((void**)&coo_row_ind, sizeof(rocsparse_int) * nnz);
 *      hipMalloc((void**)&coo_col_ind, sizeof(rocsparse_int) * nnz);
 *      hipMalloc((void**)&coo_val, sizeof(float) * nnz);
 *
 *      // Convert the csr row offsets into coo row indices
 *      rocsparse_csr2coo(handle,
 *                        csr_row_ptr,
 *                        nnz,
 *                        m,
 *                        coo_row_ind,
 *                        rocsparse_index_base_zero);
 *
 *      // Copy the column and value arrays
 *      hipMemcpy(coo_col_ind,
 *                csr_col_ind,
 *                sizeof(rocsparse_int) * nnz,
 *                hipMemcpyDeviceToDevice);
 *
 *      hipMemcpy(coo_val,
 *                csr_val,
 *                sizeof(float) * nnz,
 *                hipMemcpyDeviceToDevice);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2coo(rocsparse_handle     handle,
                                   const rocsparse_int* csr_row_ptr,
                                   rocsparse_int        nnz,
                                   rocsparse_int        m,
                                   rocsparse_int*       coo_row_ind,
                                   rocsparse_index_base idx_base);

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a sparse CSC matrix
 *
 *  \details
 *  \p rocsparse_csr2csc_buffer_size returns the size of the temporary storage buffer
 *  required by rocsparse_scsr2csc(), rocsparse_dcsr2csc(), rocsparse_ccsr2csc() and
 *  rocsparse_zcsr2csc(). The temporary storage buffer must be allocated by the user.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  copy_values \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
 *  @param[out]
 *  buffer_size number of bytes of the temporary storage buffer required by
 *              rocsparse_scsr2csc(), rocsparse_dcsr2csc(), rocsparse_ccsr2csc() and
 *              rocsparse_zcsr2csc().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr, \p csr_col_ind or
 *              \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2csc_buffer_size(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               rocsparse_action     copy_values,
                                               size_t*              buffer_size);

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a sparse CSC matrix
 *
 *  \details
 *  \p rocsparse_csr2csc converts a CSR matrix into a CSC matrix. \p rocsparse_csr2csc
 *  can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
 *  whether \p csc_val is being filled during conversion (\ref rocsparse_action_numeric)
 *  or not (\ref rocsparse_action_symbolic).
 *
 *  \p rocsparse_csr2csc requires extra temporary storage buffer that has to be allocated
 *  by the user. Storage buffer size can be determined by rocsparse_csr2csc_buffer_size().
 *
 *  \note
 *  The resulting matrix can also be seen as the transpose of the input matrix.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  csc_val     array of \p nnz elements of the sparse CSC matrix.
 *  @param[out]
 *  csc_row_ind array of \p nnz elements containing the row indices of the sparse CSC
 *              matrix.
 *  @param[out]
 *  csc_col_ptr array of \p n+1 elements that point to the start of every column of the
 *              sparse CSC matrix.
 *  @param[in]
 *  copy_values \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user, size is returned by
 *              rocsparse_csr2csc_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_val, \p csr_row_ptr,
 *              \p csr_col_ind, \p csc_val, \p csc_row_ind, \p csc_col_ptr or
 *              \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *
 *  \par Example
 *  This example computes the transpose of a CSR matrix.
 *  \code{.c}
 *      //     1 2 0 3 0
 *      // A = 0 4 5 0 0
 *      //     6 0 0 7 8
 *
 *      rocsparse_int m_A   = 3;
 *      rocsparse_int n_A   = 5;
 *      rocsparse_int nnz_A = 8;
 *
 *      csr_row_ptr_A[m+1] = {0, 3, 5, 8};             // device memory
 *      csr_col_ind_A[nnz] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
 *      csr_val_A[nnz]     = {1, 2, 3, 4, 5, 6, 7, 8}; // device memory
 *
 *      // Allocate memory for transposed CSR matrix
 *      rocsparse_int m_T   = n_A;
 *      rocsparse_int n_T   = m_A;
 *      rocsparse_int nnz_T = nnz_A;
 *
 *      rocsparse_int* csr_row_ptr_T;
 *      rocsparse_int* csr_col_ind_T;
 *      float* csr_val_T;
 *
 *      hipMalloc((void**)&csr_row_ptr_T, sizeof(rocsparse_int) * (m_T + 1));
 *      hipMalloc((void**)&csr_col_ind_T, sizeof(rocsparse_int) * nnz_T);
 *      hipMalloc((void**)&csr_val_T, sizeof(float) * nnz_T);
 *
 *      // Obtain the temporary buffer size
 *      size_t buffer_size;
 *      rocsparse_csr2csc_buffer_size(handle,
 *                                    m_A,
 *                                    n_A,
 *                                    nnz_A,
 *                                    csr_row_ptr_A,
 *                                    csr_col_ind_A,
 *                                    rocsparse_action_numeric,
 *                                    &buffer_size);
 *
 *      // Allocate temporary buffer
 *      void* temp_buffer;
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      rocsparse_scsr2csc(handle,
 *                         m_A,
 *                         n_A,
 *                         nnz_A,
 *                         csr_val_A,
 *                         csr_row_ptr_A,
 *                         csr_col_ind_A,
 *                         csr_val_T,
 *                         csr_col_ind_T,
 *                         csr_row_ptr_T,
 *                         rocsparse_action_numeric,
 *                         rocsparse_index_base_zero,
 *                         temp_buffer);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2csc(rocsparse_handle     handle,
                                    rocsparse_int        m,
                                    rocsparse_int        n,
                                    rocsparse_int        nnz,
                                    const float*         csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    float*               csc_val,
                                    rocsparse_int*       csc_row_ind,
                                    rocsparse_int*       csc_col_ptr,
                                    rocsparse_action     copy_values,
                                    rocsparse_index_base idx_base,
                                    void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2csc(rocsparse_handle     handle,
                                    rocsparse_int        m,
                                    rocsparse_int        n,
                                    rocsparse_int        nnz,
                                    const double*        csr_val,
                                    const rocsparse_int* csr_row_ptr,
                                    const rocsparse_int* csr_col_ind,
                                    double*              csc_val,
                                    rocsparse_int*       csc_row_ind,
                                    rocsparse_int*       csc_col_ptr,
                                    rocsparse_action     copy_values,
                                    rocsparse_index_base idx_base,
                                    void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2csc(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    rocsparse_int                  nnz,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    const rocsparse_int*           csr_col_ind,
                                    rocsparse_float_complex*       csc_val,
                                    rocsparse_int*                 csc_row_ind,
                                    rocsparse_int*                 csc_col_ptr,
                                    rocsparse_action               copy_values,
                                    rocsparse_index_base           idx_base,
                                    void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2csc(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    rocsparse_int                   nnz,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    const rocsparse_int*            csr_col_ind,
                                    rocsparse_double_complex*       csc_val,
                                    rocsparse_int*                  csc_row_ind,
                                    rocsparse_int*                  csc_col_ptr,
                                    rocsparse_action                copy_values,
                                    rocsparse_index_base            idx_base,
                                    void*                           temp_buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a sparse ELL matrix
 *
 *  \details
 *  \p rocsparse_csr2ell_width computes the maximum of the per row non-zero elements
 *  over all rows, the ELL \p width, for a given CSR matrix.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  ell_width   pointer to the number of non-zero elements per row in ELL storage
 *              format.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_descr, \p csr_row_ptr, or
 *              \p ell_width pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2ell_width(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         const rocsparse_mat_descr csr_descr,
                                         const rocsparse_int*      csr_row_ptr,
                                         const rocsparse_mat_descr ell_descr,
                                         rocsparse_int*            ell_width);

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a sparse ELL matrix
 *
 *  \details
 *  \p rocsparse_csr2ell converts a CSR matrix into an ELL matrix. It is assumed,
 *  that \p ell_val and \p ell_col_ind are allocated. Allocation size is computed by the
 *  number of rows times the number of ELL non-zero elements per row, such that
 *  \f$\text{nnz}_{\text{ELL}} = m \cdot \text{ell\_width}\f$. The number of ELL
 *  non-zero elements per row is obtained by rocsparse_csr2ell_width().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val     array containing the values of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array containing the column indices of the sparse CSR matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_width   number of non-zero elements per row in ELL storage format.
 *  @param[out]
 *  ell_val     array of \p m times \p ell_width elements of the sparse ELL matrix.
 *  @param[out]
 *  ell_col_ind array of \p m times \p ell_width elements containing the column indices
 *              of the sparse ELL matrix.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p ell_width is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_descr, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p ell_descr, \p ell_val or
 *              \p ell_col_ind pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example converts a CSR matrix into an ELL matrix.
 *  \code{.c}
 *      //     1 2 0 3 0
 *      // A = 0 4 5 0 0
 *      //     6 0 0 7 8
 *
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 5;
 *      rocsparse_int nnz = 8;
 *
 *      csr_row_ptr[m+1] = {0, 3, 5, 8};             // device memory
 *      csr_col_ind[nnz] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
 *      csr_val[nnz]     = {1, 2, 3, 4, 5, 6, 7, 8}; // device memory
 *
 *      // Create ELL matrix descriptor
 *      rocsparse_mat_descr ell_descr;
 *      rocsparse_create_mat_descr(&ell_descr);
 *
 *      // Obtain the ELL width
 *      rocsparse_int ell_width;
 *      rocsparse_csr2ell_width(handle,
 *                              m,
 *                              csr_descr,
 *                              csr_row_ptr,
 *                              ell_descr,
 *                              &ell_width);
 *
 *      // Compute ELL non-zero entries
 *      rocsparse_int ell_nnz = m * ell_width;
 *
 *      // Allocate ELL column and value arrays
 *      rocsparse_int* ell_col_ind;
 *      hipMalloc((void**)&ell_col_ind, sizeof(rocsparse_int) * ell_nnz);
 *
 *      float* ell_val;
 *      hipMalloc((void**)&ell_val, sizeof(float) * ell_nnz);
 *
 *      // Format conversion
 *      rocsparse_scsr2ell(handle,
 *                         m,
 *                         csr_descr,
 *                         csr_val,
 *                         csr_row_ptr,
 *                         csr_col_ind,
 *                         ell_descr,
 *                         ell_width,
 *                         ell_val,
 *                         ell_col_ind);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2ell(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    const rocsparse_mat_descr csr_descr,
                                    const float*              csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    float*                    ell_val,
                                    rocsparse_int*            ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2ell(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    const rocsparse_mat_descr csr_descr,
                                    const double*             csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    double*                   ell_val,
                                    rocsparse_int*            ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2ell(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    const rocsparse_mat_descr      csr_descr,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    const rocsparse_int*           csr_col_ind,
                                    const rocsparse_mat_descr      ell_descr,
                                    rocsparse_int                  ell_width,
                                    rocsparse_float_complex*       ell_val,
                                    rocsparse_int*                 ell_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2ell(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    const rocsparse_mat_descr       csr_descr,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    const rocsparse_int*            csr_col_ind,
                                    const rocsparse_mat_descr       ell_descr,
                                    rocsparse_int                   ell_width,
                                    rocsparse_double_complex*       ell_val,
                                    rocsparse_int*                  ell_col_ind);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a sparse HYB matrix
 *
 *  \details
 *  \p rocsparse_csr2hyb converts a CSR matrix into a HYB matrix. It is assumed
 *  that \p hyb has been initialized with rocsparse_create_hyb_mat().
 *
 *  \note
 *  This function requires a significant amount of storage for the HYB matrix,
 *  depending on the matrix structure.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n               number of columns of the sparse CSR matrix.
 *  @param[in]
 *  descr           descriptor of the sparse CSR matrix. Currently, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_val         array containing the values of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind     array containing the column indices of the sparse CSR matrix.
 *  @param[out]
 *  hyb             sparse matrix in HYB format.
 *  @param[in]
 *  user_ell_width  width of the ELL part of the HYB matrix (only required if
 *                  \p partition_type == \ref rocsparse_hyb_partition_user).
 *  @param[in]
 *  partition_type  \ref rocsparse_hyb_partition_auto (recommended),
 *                  \ref rocsparse_hyb_partition_user or
 *                  \ref rocsparse_hyb_partition_max.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p user_ell_width is invalid.
 *  \retval     rocsparse_status_invalid_value \p partition_type is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p hyb, \p csr_val,
 *              \p csr_row_ptr or \p csr_col_ind pointer is invalid.
 *  \retval     rocsparse_status_memory_error the buffer for the HYB matrix could not be
 *              allocated.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example converts a CSR matrix into a HYB matrix using user defined partitioning.
 *  \code{.c}
 *      // Create HYB matrix structure
 *      rocsparse_hyb_mat hyb;
 *      rocsparse_create_hyb_mat(&hyb);
 *
 *      // User defined ell width
 *      rocsparse_int user_ell_width = 5;
 *
 *      // Perform the conversion
 *      rocsparse_scsr2hyb(handle,
 *                         m,
 *                         n,
 *                         descr,
 *                         csr_val,
 *                         csr_row_ptr,
 *                         csr_col_ind,
 *                         hyb,
 *                         user_ell_width,
 *                         rocsparse_hyb_partition_user);
 *
 *      // Do some work
 *
 *      // Clean up
 *      rocsparse_destroy_hyb_mat(hyb);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2hyb(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr descr,
                                    const float*              csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_hyb_mat         hyb,
                                    rocsparse_int             user_ell_width,
                                    rocsparse_hyb_partition   partition_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr descr,
                                    const double*             csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_hyb_mat         hyb,
                                    rocsparse_int             user_ell_width,
                                    rocsparse_hyb_partition   partition_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2hyb(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    const rocsparse_mat_descr      descr,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    const rocsparse_int*           csr_col_ind,
                                    rocsparse_hyb_mat              hyb,
                                    rocsparse_int                  user_ell_width,
                                    rocsparse_hyb_partition        partition_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2hyb(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    const rocsparse_mat_descr       descr,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    const rocsparse_int*            csr_col_ind,
                                    rocsparse_hyb_mat               hyb,
                                    rocsparse_int                   user_ell_width,
                                    rocsparse_hyb_partition         partition_type);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert a sparse COO matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_coo2csr converts the COO array containing the row indices into a
 *  CSR array of row offsets, that point to the start of every row.
 *  It is assumed that the COO row index array is sorted.
 *
 *  \note It can also be used, to convert a COO array containing the column indices into
 *  a CSC array of column offsets, that point to the start of every column. Then, it is
 *  assumed that the COO column index array is sorted, instead.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  coo_row_ind array of \p nnz elements containing the row indices of the sparse COO
 *              matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[out]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p coo_row_ind or \p csr_row_ptr
 *              pointer is invalid.
 *
 *  \par Example
 *  This example converts a COO matrix into a CSR matrix.
 *  \code{.c}
 *      //     1 2 0 3 0
 *      // A = 0 4 5 0 0
 *      //     6 0 0 7 8
 *
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 5;
 *      rocsparse_int nnz = 8;
 *
 *      coo_row_ind[nnz] = {0, 0, 0, 1, 1, 2, 2, 2}; // device memory
 *      coo_col_ind[nnz] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
 *      coo_val[nnz]     = {1, 2, 3, 4, 5, 6, 7, 8}; // device memory
 *
 *      // Allocate CSR matrix arrays
 *      rocsparse_int* csr_row_ptr;
 *      rocsparse_int* csr_col_ind;
 *      float* csr_val;
 *
 *      hipMalloc((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
 *      hipMalloc((void**)&csr_col_ind, sizeof(rocsparse_int) * nnz);
 *      hipMalloc((void**)&csr_val, sizeof(float) * nnz);
 *
 *      // Convert the coo row indices into csr row offsets
 *      rocsparse_coo2csr(handle,
 *                        coo_row_ind,
 *                        nnz,
 *                        m,
 *                        csr_row_ptr,
 *                        rocsparse_index_base_zero);
 *
 *      // Copy the column and value arrays
 *      hipMemcpy(csr_col_ind,
 *                coo_col_ind,
 *                sizeof(rocsparse_int) * nnz,
 *                hipMemcpyDeviceToDevice);
 *
 *      hipMemcpy(csr_val,
 *                coo_val,
 *                sizeof(float) * nnz,
 *                hipMemcpyDeviceToDevice);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo2csr(rocsparse_handle     handle,
                                   const rocsparse_int* coo_row_ind,
                                   rocsparse_int        nnz,
                                   rocsparse_int        m,
                                   rocsparse_int*       csr_row_ptr,
                                   rocsparse_index_base idx_base);

/*! \ingroup conv_module
 *  \brief Convert a sparse ELL matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_ell2csr_nnz computes the total CSR non-zero elements and the CSR
 *  row offsets, that point to the start of every row of the sparse CSR matrix, for
 *  a given ELL matrix. It is assumed that \p csr_row_ptr has been allocated with
 *  size \p m + 1.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse ELL matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_width   number of non-zero elements per row in ELL storage format.
 *  @param[in]
 *  ell_col_ind array of \p m times \p ell_width elements containing the column indices
 *              of the sparse ELL matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[out]
 *  csr_nnz     pointer to the total number of non-zero elements in CSR storage
 *              format.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p ell_descr, \p ell_col_ind,
 *              \p csr_descr, \p csr_row_ptr or \p csr_nnz pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ell2csr_nnz(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr ell_descr,
                                       rocsparse_int             ell_width,
                                       const rocsparse_int*      ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       rocsparse_int*            csr_row_ptr,
                                       rocsparse_int*            csr_nnz);

/*! \ingroup conv_module
 *  \brief Convert a sparse ELL matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_ell2csr converts an ELL matrix into a CSR matrix. It is assumed
 *  that \p csr_row_ptr has already been filled and that \p csr_val and \p csr_col_ind
 *  are allocated by the user. \p csr_row_ptr and allocation size of \p csr_col_ind and
 *  \p csr_val is defined by the number of CSR non-zero elements. Both can be obtained
 *  by rocsparse_ell2csr_nnz().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  m           number of rows of the sparse ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse ELL matrix.
 *  @param[in]
 *  ell_descr   descriptor of the sparse ELL matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_width   number of non-zero elements per row in ELL storage format.
 *  @param[in]
 *  ell_val     array of \p m times \p ell_width elements of the sparse ELL matrix.
 *  @param[in]
 *  ell_col_ind array of \p m times \p ell_width elements containing the column indices
 *              of the sparse ELL matrix.
 *  @param[in]
 *  csr_descr   descriptor of the sparse CSR matrix. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_val     array containing the values of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
 *              sparse CSR matrix.
 *  @param[out]
 *  csr_col_ind array containing the column indices of the sparse CSR matrix.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_descr, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p ell_descr, \p ell_val or
 *              \p ell_col_ind pointer is invalid.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example converts an ELL matrix into a CSR matrix.
 *  \code{.c}
 *      //     1 2 0 3 0
 *      // A = 0 4 5 0 0
 *      //     6 0 0 7 8
 *
 *      rocsparse_int m         = 3;
 *      rocsparse_int n         = 5;
 *      rocsparse_int nnz       = 9;
 *      rocsparse_int ell_width = 3;
 *
 *      ell_col_ind[nnz] = {0, 1, 0, 1, 2, 3, 3, -1, 4}; // device memory
 *      ell_val[nnz]     = {1, 4, 6, 2, 5, 7, 3, 0, 8};  // device memory
 *
 *      // Create CSR matrix descriptor
 *      rocsparse_mat_descr csr_descr;
 *      rocsparse_create_mat_descr(&csr_descr);
 *
 *      // Allocate csr_row_ptr array for row offsets
 *      rocsparse_int* csr_row_ptr;
 *      hipMalloc((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
 *
 *      // Obtain the number of CSR non-zero entries
 *      // and fill csr_row_ptr array with row offsets
 *      rocsparse_int csr_nnz;
 *      rocsparse_ell2csr_nnz(handle,
 *                            m,
 *                            n,
 *                            ell_descr,
 *                            ell_width,
 *                            ell_col_ind,
 *                            csr_descr,
 *                            csr_row_ptr,
 *                            &csr_nnz);
 *
 *      // Allocate CSR column and value arrays
 *      rocsparse_int* csr_col_ind;
 *      hipMalloc((void**)&csr_col_ind, sizeof(rocsparse_int) * csr_nnz);
 *
 *      float* csr_val;
 *      hipMalloc((void**)&csr_val, sizeof(float) * csr_nnz);
 *
 *      // Format conversion
 *      rocsparse_sell2csr(handle,
 *                         m,
 *                         n,
 *                         ell_descr,
 *                         ell_width,
 *                         ell_val,
 *                         ell_col_ind,
 *                         csr_descr,
 *                         csr_val,
 *                         csr_row_ptr,
 *                         csr_col_ind);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sell2csr(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    const float*              ell_val,
                                    const rocsparse_int*      ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    float*                    csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dell2csr(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr ell_descr,
                                    rocsparse_int             ell_width,
                                    const double*             ell_val,
                                    const rocsparse_int*      ell_col_ind,
                                    const rocsparse_mat_descr csr_descr,
                                    double*                   csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cell2csr(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    const rocsparse_mat_descr      ell_descr,
                                    rocsparse_int                  ell_width,
                                    const rocsparse_float_complex* ell_val,
                                    const rocsparse_int*           ell_col_ind,
                                    const rocsparse_mat_descr      csr_descr,
                                    rocsparse_float_complex*       csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zell2csr(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    const rocsparse_mat_descr       ell_descr,
                                    rocsparse_int                   ell_width,
                                    const rocsparse_double_complex* ell_val,
                                    const rocsparse_int*            ell_col_ind,
                                    const rocsparse_mat_descr       csr_descr,
                                    rocsparse_double_complex*       csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*                  csr_col_ind);
/**@}*/

/*! \ingroup conv_module
 *  \brief Create the identity map
 *
 *  \details
 *  \p rocsparse_create_identity_permutation stores the identity map in \p p, such that
 *  \f$p = 0:1:(n-1)\f$.
 *
 *  \code{.c}
 *      for(i = 0; i < n; ++i)
 *      {
 *          p[i] = i;
 *      }
 *  \endcode
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  n           size of the map \p p.
 *  @param[out]
 *  p           array of \p n integers containing the map.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p n is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p p pointer is invalid.
 *
 *  \par Example
 *  The following example creates an identity permutation.
 *  \code{.c}
 *      rocsparse_int size = 200;
 *
 *      // Allocate memory to hold the identity map
 *      rocsparse_int* perm;
 *      hipMalloc((void**)&perm, sizeof(rocsparse_int) * size);
 *
 *      // Fill perm with the identity permutation
 *      rocsparse_create_identity_permutation(handle, size, perm);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_identity_permutation(rocsparse_handle handle,
                                                       rocsparse_int    n,
                                                       rocsparse_int*   p);

/*! \ingroup conv_module
 *  \brief Sort a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_csrsort_buffer_size returns the size of the temporary storage buffer
 *  required by rocsparse_csrsort(). The temporary storage buffer must be allocated by
 *  the user.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n               number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  CSR matrix.
 *  @param[out]
 *  buffer_size     number of bytes of the temporary storage buffer required by
 *                  rocsparse_csrsort().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr, \p csr_col_ind or
 *              \p buffer_size pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const rocsparse_int* csr_row_ptr,
                                               const rocsparse_int* csr_col_ind,
                                               size_t*              buffer_size);

/*! \ingroup conv_module
 *  \brief Sort a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_csrsort sorts a matrix in CSR format. The sorted permutation vector
 *  \p perm can be used to obtain sorted \p csr_val array. In this case, \p perm must be
 *  initialized as the identity permutation, see rocsparse_create_identity_permutation().
 *
 *  \p rocsparse_csrsort requires extra temporary storage buffer that has to be allocated by
 *  the user. Storage buffer size can be determined by rocsparse_csrsort_buffer_size().
 *
 *  \note
 *  \p perm can be \p NULL if a sorted permutation vector is not required.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n               number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  descr           descriptor of the sparse CSR matrix. Currently, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
 *                  sparse CSR matrix.
 *  @param[inout]
 *  csr_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  CSR matrix.
 *  @param[inout]
 *  perm            array of \p nnz integers containing the unsorted map indices, can be
 *                  \p NULL.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned by
 *                  rocsparse_csrsort_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csr_row_ptr, \p csr_col_ind
 *              or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  The following example sorts a \f$3 \times 3\f$ CSR matrix.
 *  \code{.c}
 *      //     1 2 3
 *      // A = 4 5 6
 *      //     7 8 9
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 3;
 *      rocsparse_int nnz = 9;
 *
 *      csr_row_ptr[m + 1] = {0, 3, 6, 9};                // device memory
 *      csr_col_ind[nnz]   = {2, 0, 1, 0, 1, 2, 0, 2, 1}; // device memory
 *      csr_val[nnz]       = {3, 1, 2, 4, 5, 6, 7, 9, 8}; // device memory
 *
 *      // Create permutation vector perm as the identity map
 *      rocsparse_int* perm;
 *      hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz);
 *      rocsparse_create_identity_permutation(handle, nnz, perm);
 *
 *      // Allocate temporary buffer
 *      size_t buffer_size;
 *      void* temp_buffer;
 *      rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, csr_col_ind, &buffer_size);
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Sort the CSR matrix
 *      rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, temp_buffer);
 *
 *      // Gather sorted csr_val array
 *      float* csr_val_sorted;
 *      hipMalloc((void**)&csr_val_sorted, sizeof(float) * nnz);
 *      rocsparse_sgthr(handle, nnz, csr_val, csr_val_sorted, perm, rocsparse_index_base_zero);
 *
 *      // Clean up
 *      hipFree(temp_buffer);
 *      hipFree(perm);
 *      hipFree(csr_val);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrsort(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_int*      csr_row_ptr,
                                   rocsparse_int*            csr_col_ind,
                                   rocsparse_int*            perm,
                                   void*                     temp_buffer);

/*! \ingroup conv_module
 *  \brief Sort a sparse CSC matrix
 *
 *  \details
 *  \p rocsparse_cscsort_buffer_size returns the size of the temporary storage buffer
 *  required by rocsparse_cscsort(). The temporary storage buffer must be allocated by
 *  the user.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse CSC matrix.
 *  @param[in]
 *  n               number of columns of the sparse CSC matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse CSC matrix.
 *  @param[in]
 *  csc_col_ptr     array of \p n+1 elements that point to the start of every column of
 *                  the sparse CSC matrix.
 *  @param[in]
 *  csc_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  CSC matrix.
 *  @param[out]
 *  buffer_size     number of bytes of the temporary storage buffer required by
 *                  rocsparse_cscsort().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csc_col_ptr, \p csc_row_ind or
 *              \p buffer_size pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cscsort_buffer_size(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const rocsparse_int* csc_col_ptr,
                                               const rocsparse_int* csc_row_ind,
                                               size_t*              buffer_size);

/*! \ingroup conv_module
 *  \brief Sort a sparse CSC matrix
 *
 *  \details
 *  \p rocsparse_cscsort sorts a matrix in CSC format. The sorted permutation vector
 *  \p perm can be used to obtain sorted \p csc_val array. In this case, \p perm must be
 *  initialized as the identity permutation, see rocsparse_create_identity_permutation().
 *
 *  \p rocsparse_cscsort requires extra temporary storage buffer that has to be allocated by
 *  the user. Storage buffer size can be determined by rocsparse_cscsort_buffer_size().
 *
 *  \note
 *  \p perm can be \p NULL if a sorted permutation vector is not required.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse CSC matrix.
 *  @param[in]
 *  n               number of columns of the sparse CSC matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse CSC matrix.
 *  @param[in]
 *  descr           descriptor of the sparse CSC matrix. Currently, only
 *                  \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  csc_col_ptr     array of \p n+1 elements that point to the start of every column of
 *                  the sparse CSC matrix.
 *  @param[inout]
 *  csc_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  CSC matrix.
 *  @param[inout]
 *  perm            array of \p nnz integers containing the unsorted map indices, can be
 *                  \p NULL.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned by
 *                  rocsparse_cscsort_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p csc_col_ptr, \p csc_row_ind
 *              or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  The following example sorts a \f$3 \times 3\f$ CSC matrix.
 *  \code{.c}
 *      //     1 2 3
 *      // A = 4 5 6
 *      //     7 8 9
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 3;
 *      rocsparse_int nnz = 9;
 *
 *      csc_col_ptr[m + 1] = {0, 3, 6, 9};                // device memory
 *      csc_row_ind[nnz]   = {2, 0, 1, 0, 1, 2, 0, 2, 1}; // device memory
 *      csc_val[nnz]       = {7, 1, 4, 2, 5, 8, 3, 9, 6}; // device memory
 *
 *      // Create permutation vector perm as the identity map
 *      rocsparse_int* perm;
 *      hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz);
 *      rocsparse_create_identity_permutation(handle, nnz, perm);
 *
 *      // Allocate temporary buffer
 *      size_t buffer_size;
 *      void* temp_buffer;
 *      rocsparse_cscsort_buffer_size(handle, m, n, nnz, csc_col_ptr, csc_row_ind, &buffer_size);
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Sort the CSC matrix
 *      rocsparse_cscsort(handle, m, n, nnz, descr, csc_col_ptr, csc_row_ind, perm, temp_buffer);
 *
 *      // Gather sorted csc_val array
 *      float* csc_val_sorted;
 *      hipMalloc((void**)&csc_val_sorted, sizeof(float) * nnz);
 *      rocsparse_sgthr(handle, nnz, csc_val, csc_val_sorted, perm, rocsparse_index_base_zero);
 *
 *      // Clean up
 *      hipFree(temp_buffer);
 *      hipFree(perm);
 *      hipFree(csc_val);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cscsort(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_int*      csc_col_ptr,
                                   rocsparse_int*            csc_row_ind,
                                   rocsparse_int*            perm,
                                   void*                     temp_buffer);

/*! \ingroup conv_module
 *  \brief Sort a sparse COO matrix
 *
 *  \details
 *  \p coosort_buffer_size returns the size of the temporary storage buffer that is
 *  required by rocsparse_coosort_by_row() and rocsparse_coosort_by_column(). The
 *  temporary storage buffer has to be allocated by the user.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse COO matrix.
 *  @param[in]
 *  n               number of columns of the sparse COO matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse COO matrix.
 *  @param[in]
 *  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  COO matrix.
 *  @param[in]
 *  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  COO matrix.
 *  @param[out]
 *  buffer_size     number of bytes of the temporary storage buffer required by
 *                  rocsparse_coosort_by_row() and rocsparse_coosort_by_column().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
 *              \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle     handle,
                                               rocsparse_int        m,
                                               rocsparse_int        n,
                                               rocsparse_int        nnz,
                                               const rocsparse_int* coo_row_ind,
                                               const rocsparse_int* coo_col_ind,
                                               size_t*              buffer_size);

/*! \ingroup conv_module
 *  \brief Sort a sparse COO matrix by row
 *
 *  \details
 *  \p rocsparse_coosort_by_row sorts a matrix in COO format by row. The sorted
 *  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
 *  case, \p perm must be initialized as the identity permutation, see
 *  rocsparse_create_identity_permutation().
 *
 *  \p rocsparse_coosort_by_row requires extra temporary storage buffer that has to be
 *  allocated by the user. Storage buffer size can be determined by
 *  rocsparse_coosort_buffer_size().
 *
 *  \note
 *  \p perm can be \p NULL if a sorted permutation vector is not required.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse COO matrix.
 *  @param[in]
 *  n               number of columns of the sparse COO matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse COO matrix.
 *  @param[inout]
 *  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  COO matrix.
 *  @param[inout]
 *  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  COO matrix.
 *  @param[inout]
 *  perm            array of \p nnz integers containing the unsorted map indices, can be
 *                  \p NULL.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned by
 *                  rocsparse_coosort_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
 *              \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *
 *  \par Example
 *  The following example sorts a \f$3 \times 3\f$ COO matrix by row indices.
 *  \code{.c}
 *      //     1 2 3
 *      // A = 4 5 6
 *      //     7 8 9
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 3;
 *      rocsparse_int nnz = 9;
 *
 *      coo_row_ind[nnz] = {0, 1, 2, 0, 1, 2, 0, 1, 2}; // device memory
 *      coo_col_ind[nnz] = {0, 0, 0, 1, 1, 1, 2, 2, 2}; // device memory
 *      coo_val[nnz]     = {1, 4, 7, 2, 5, 8, 3, 6, 9}; // device memory
 *
 *      // Create permutation vector perm as the identity map
 *      rocsparse_int* perm;
 *      hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz);
 *      rocsparse_create_identity_permutation(handle, nnz, perm);
 *
 *      // Allocate temporary buffer
 *      size_t buffer_size;
 *      void* temp_buffer;
 *      rocsparse_coosort_buffer_size(handle,
 *                                    m,
 *                                    n,
 *                                    nnz,
 *                                    coo_row_ind,
 *                                    coo_col_ind,
 *                                    &buffer_size);
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Sort the COO matrix
 *      rocsparse_coosort_by_row(handle,
 *                               m,
 *                               n,
 *                               nnz,
 *                               coo_row_ind,
 *                               coo_col_ind,
 *                               perm,
 *                               temp_buffer);
 *
 *      // Gather sorted coo_val array
 *      float* coo_val_sorted;
 *      hipMalloc((void**)&coo_val_sorted, sizeof(float) * nnz);
 *      rocsparse_sgthr(handle, nnz, coo_val, coo_val_sorted, perm, rocsparse_index_base_zero);
 *
 *      // Clean up
 *      hipFree(temp_buffer);
 *      hipFree(perm);
 *      hipFree(coo_val);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_by_row(rocsparse_handle handle,
                                          rocsparse_int    m,
                                          rocsparse_int    n,
                                          rocsparse_int    nnz,
                                          rocsparse_int*   coo_row_ind,
                                          rocsparse_int*   coo_col_ind,
                                          rocsparse_int*   perm,
                                          void*            temp_buffer);

/*! \ingroup conv_module
 *  \brief Sort a sparse COO matrix by column
 *
 *  \details
 *  \p rocsparse_coosort_by_column sorts a matrix in COO format by column. The sorted
 *  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
 *  case, \p perm must be initialized as the identity permutation, see
 *  rocsparse_create_identity_permutation().
 *
 *  \p rocsparse_coosort_by_column requires extra temporary storage buffer that has to be
 *  allocated by the user. Storage buffer size can be determined by
 *  rocsparse_coosort_buffer_size().
 *
 *  \note
 *  \p perm can be \p NULL if a sorted permutation vector is not required.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  m               number of rows of the sparse COO matrix.
 *  @param[in]
 *  n               number of columns of the sparse COO matrix.
 *  @param[in]
 *  nnz             number of non-zero entries of the sparse COO matrix.
 *  @param[inout]
 *  coo_row_ind     array of \p nnz elements containing the row indices of the sparse
 *                  COO matrix.
 *  @param[inout]
 *  coo_col_ind     array of \p nnz elements containing the column indices of the sparse
 *                  COO matrix.
 *  @param[inout]
 *  perm            array of \p nnz integers containing the unsorted map indices, can be
 *                  \p NULL.
 *  @param[in]
 *  temp_buffer     temporary storage buffer allocated by the user, size is returned by
 *                  rocsparse_coosort_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p coo_row_ind, \p coo_col_ind or
 *              \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *
 *  \par Example
 *  The following example sorts a \f$3 \times 3\f$ COO matrix by column indices.
 *  \code{.c}
 *      //     1 2 3
 *      // A = 4 5 6
 *      //     7 8 9
 *      rocsparse_int m   = 3;
 *      rocsparse_int n   = 3;
 *      rocsparse_int nnz = 9;
 *
 *      coo_row_ind[nnz] = {0, 0, 0, 1, 1, 1, 2, 2, 2}; // device memory
 *      coo_col_ind[nnz] = {0, 1, 2, 0, 1, 2, 0, 1, 2}; // device memory
 *      coo_val[nnz]     = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // device memory
 *
 *      // Create permutation vector perm as the identity map
 *      rocsparse_int* perm;
 *      hipMalloc((void**)&perm, sizeof(rocsparse_int) * nnz);
 *      rocsparse_create_identity_permutation(handle, nnz, perm);
 *
 *      // Allocate temporary buffer
 *      size_t buffer_size;
 *      void* temp_buffer;
 *      rocsparse_coosort_buffer_size(handle,
 *                                    m,
 *                                    n,
 *                                    nnz,
 *                                    coo_row_ind,
 *                                    coo_col_ind,
 *                                    &buffer_size);
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Sort the COO matrix
 *      rocsparse_coosort_by_column(handle,
 *                                  m,
 *                                  n,
 *                                  nnz,
 *                                  coo_row_ind,
 *                                  coo_col_ind,
 *                                  perm,
 *                                  temp_buffer);
 *
 *      // Gather sorted coo_val array
 *      float* coo_val_sorted;
 *      hipMalloc((void**)&coo_val_sorted, sizeof(float) * nnz);
 *      rocsparse_sgthr(handle, nnz, coo_val, coo_val_sorted, perm, rocsparse_index_base_zero);
 *
 *      // Clean up
 *      hipFree(temp_buffer);
 *      hipFree(perm);
 *      hipFree(coo_val);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coosort_by_column(rocsparse_handle handle,
                                             rocsparse_int    m,
                                             rocsparse_int    n,
                                             rocsparse_int    nnz,
                                             rocsparse_int*   coo_row_ind,
                                             rocsparse_int*   coo_col_ind,
                                             rocsparse_int*   perm,
                                             void*            temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSE_FUNCTIONS_H_ */

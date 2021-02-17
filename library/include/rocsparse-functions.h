/* ************************************************************************
* Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
*  of Level 1, 2 and 3, using HIP optimized for AMD GPU hardware.
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
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p rocsparse_bsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{bsr_dim}) \times (nb \cdot \text{bsr_dim})\f$
*  matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
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
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  bsr_dim     block dimension of the sparse BSR matrix.
*  @param[in]
*  x           array of \p nb*bsr_dim elements (\f$op(A) = A\f$) or \p mb*bsr_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*bsr_dim elements (\f$op(A) = A\f$) or \p nb*bsr_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb, \p nnzb or \p bsr_dim is
*              invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrmv(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans,
                                  rocsparse_int             mb,
                                  rocsparse_int             nb,
                                  rocsparse_int             nnzb,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             bsr_dim,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrmv(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans,
                                  rocsparse_int             mb,
                                  rocsparse_int             nb,
                                  rocsparse_int             nnzb,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             bsr_dim,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrmv(rocsparse_handle               handle,
                                  rocsparse_direction            dir,
                                  rocsparse_operation            trans,
                                  rocsparse_int                  mb,
                                  rocsparse_int                  nb,
                                  rocsparse_int                  nnzb,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* bsr_val,
                                  const rocsparse_int*           bsr_row_ptr,
                                  const rocsparse_int*           bsr_col_ind,
                                  rocsparse_int                  bsr_dim,
                                  const rocsparse_float_complex* x,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrmv(rocsparse_handle                handle,
                                  rocsparse_direction             dir,
                                  rocsparse_operation             trans,
                                  rocsparse_int                   mb,
                                  rocsparse_int                   nb,
                                  rocsparse_int                   nnzb,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* bsr_val,
                                  const rocsparse_int*            bsr_row_ptr,
                                  const rocsparse_int*            bsr_col_ind,
                                  rocsparse_int                   bsr_dim,
                                  const rocsparse_double_complex* x,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       y);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsv_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during rocsparse_sbsrsv_solve(),
*  rocsparse_dbsrsv_solve(), rocsparse_cbsrsv_solve() or rocsparse_zbsrsv_solve()
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_bsrsv_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_bsrsv_zero_pivot(rocsparse_handle   handle,
                                            rocsparse_mat_info info,
                                            rocsparse_int*     position);

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsv_buffer_size returns the size of the temporary storage buffer that
*  is required by rocsparse_sbsrsv_analysis(), rocsparse_dbsrsv_analysis(),
*  rocsparse_cbsrsv_analysis(), rocsparse_zbsrsv_analysis(), rocsparse_sbsrsv_solve(),
*  rocsparse_dbsrsv_solve(), rocsparse_cbsrsv_solve() and rocsparse_zbsrsv_solve(). The
*  temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  bsr_dim     block dimension of the sparse BSR matrix.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sbsrsv_analysis(), rocsparse_dbsrsv_analysis(),
*              rocsparse_cbsrsv_analysis(), rocsparse_zbsrsv_analysis(),
*              rocsparse_sbsrsv_solve(), rocsparse_dbsrsv_solve(),
*              rocsparse_cbsrsv_solve() and rocsparse_zbsrsv_solve().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nnzb or \p bsr_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
*              \p bsr_col_ind, \p info or \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsv_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const float*              bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             bsr_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsv_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const double*             bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             bsr_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsv_buffer_size(rocsparse_handle               handle,
                                              rocsparse_direction            dir,
                                              rocsparse_operation            trans,
                                              rocsparse_int                  mb,
                                              rocsparse_int                  nnzb,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* bsr_val,
                                              const rocsparse_int*           bsr_row_ptr,
                                              const rocsparse_int*           bsr_col_ind,
                                              rocsparse_int                  bsr_dim,
                                              rocsparse_mat_info             info,
                                              size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsv_buffer_size(rocsparse_handle                handle,
                                              rocsparse_direction             dir,
                                              rocsparse_operation             trans,
                                              rocsparse_int                   mb,
                                              rocsparse_int                   nnzb,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* bsr_val,
                                              const rocsparse_int*            bsr_row_ptr,
                                              const rocsparse_int*            bsr_col_ind,
                                              rocsparse_int                   bsr_dim,
                                              rocsparse_mat_info              info,
                                              size_t*                         buffer_size);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsv_analysis performs the analysis step for rocsparse_sbsrsv_solve(),
*  rocsparse_dbsrsv_solve(), rocsparse_cbsrsv_solve() and rocsparse_zbsrsv_solve(). It
*  is expected that this function will be executed only once for a given matrix and
*  particular operation type. The analysis meta data can be cleared by
*  rocsparse_bsrsv_clear().
*
*  \p rocsparse_bsrsv_analysis can share its meta data with
*  rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(),
*  rocsparse_cbsrsm_analysis(), rocsparse_zbsrsm_analysis(),
*  rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(),
*  rocsparse_cbsrilu0_analysis(), rocsparse_zbsrilu0_analysis(),
*  rocsparse_sbsric0_analysis(), rocsparse_dbsric0_analysis(),
*  rocsparse_cbsric0_analysis() and rocsparse_zbsric0_analysis(). Selecting
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
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  bsr_dim     block dimension of the sparse BSR matrix.
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
*  \retval     rocsparse_status_invalid_size \p mb, \p nnzb or \p bsr_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_row_ptr,
*              \p bsr_col_ind, \p info or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrsv_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans,
                                           rocsparse_int             mb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const float*              bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             bsr_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsv_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans,
                                           rocsparse_int             mb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const double*             bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             bsr_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsv_analysis(rocsparse_handle               handle,
                                           rocsparse_direction            dir,
                                           rocsparse_operation            trans,
                                           rocsparse_int                  mb,
                                           rocsparse_int                  nnzb,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* bsr_val,
                                           const rocsparse_int*           bsr_row_ptr,
                                           const rocsparse_int*           bsr_col_ind,
                                           rocsparse_int                  bsr_dim,
                                           rocsparse_mat_info             info,
                                           rocsparse_analysis_policy      analysis,
                                           rocsparse_solve_policy         solve,
                                           void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsv_analysis(rocsparse_handle                handle,
                                           rocsparse_direction             dir,
                                           rocsparse_operation             trans,
                                           rocsparse_int                   mb,
                                           rocsparse_int                   nnzb,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* bsr_val,
                                           const rocsparse_int*            bsr_row_ptr,
                                           const rocsparse_int*            bsr_col_ind,
                                           rocsparse_int                   bsr_dim,
                                           rocsparse_mat_info              info,
                                           rocsparse_analysis_policy       analysis,
                                           rocsparse_solve_policy          solve,
                                           void*                           temp_buffer);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsv_clear deallocates all memory that was allocated by
*  rocsparse_sbsrsv_analysis(), rocsparse_dbsrsv_analysis(), rocsparse_cbsrsv_analysis()
*  or rocsparse_zbsrsv_analysis(). This is especially useful, if memory is an issue and
*  the analysis data is not required for further computation, e.g. when switching to
*  another sparse matrix format. Calling \p rocsparse_bsrsv_clear is optional. All
*  allocated resources will be cleared, when the opaque \ref rocsparse_mat_info struct
*  is destroyed using rocsparse_destroy_mat_info().
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
rocsparse_status rocsparse_bsrsv_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p rocsparse_bsrsv_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \p rocsparse_bsrsv_solve requires a user allocated temporary buffer. Its size is
*  returned by rocsparse_sbsrsv_buffer_size(), rocsparse_dbsrsv_buffer_size(),
*  rocsparse_cbsrsv_buffer_size() or rocsparse_zbsrsv_buffer_size(). Furthermore,
*  analysis meta data is required. It can be obtained by rocsparse_sbsrsv_analysis(),
*  rocsparse_dbsrsv_analysis(), rocsparse_cbsrsv_analysis() or
*  rocsparse_zbsrsv_analysis(). \p rocsparse_bsrsv_solve reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be checked calling
*  rocsparse_bsrsv_zero_pivot(). If
*  \ref rocsparse_diag_type == \ref rocsparse_diag_type_unit, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref rocsparse_operation_none and
*  \p trans == \ref rocsparse_operation_transpose is supported.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         matrix storage of BSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  bsr_dim     block dimension of the sparse BSR matrix.
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
*  \retval     rocsparse_status_invalid_size \p mb, \p nnzb or \p bsr_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p x or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  Consider the lower triangular \f$m \times m\f$ matrix \f$L\f$, stored in BSR
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
*      rocsparse_dbsrsv_buffer_size(handle,
*                                   rocsparse_direction_column,
*                                   rocsparse_operation_none,
*                                   mb,
*                                   nnzb,
*                                   descr,
*                                   bsr_val,
*                                   bsr_row_ptr,
*                                   bsr_col_ind,
*                                   bsr_dim,
*                                   info,
*                                   &buffer_size);
*
*      // Allocate temporary buffer
*      void* temp_buffer;
*      hipMalloc(&temp_buffer, buffer_size);
*
*      // Perform analysis step
*      rocsparse_dbsrsv_analysis(handle,
*                                rocsparse_direction_column,
*                                rocsparse_operation_none,
*                                mb,
*                                nnzb,
*                                descr,
*                                bsr_val,
*                                bsr_row_ptr,
*                                bsr_col_ind,
*                                bsr_dim,
*                                info,
*                                rocsparse_analysis_policy_reuse,
*                                rocsparse_solve_policy_auto,
*                                temp_buffer);
*
*      // Solve Ly = x
*      rocsparse_dbsrsv_solve(handle,
*                             rocsparse_direction_column,
*                             rocsparse_operation_none,
*                             mb,
*                             nnzb,
*                             &alpha,
*                             descr,
*                             bsr_val,
*                             bsr_row_ptr,
*                             bsr_col_ind,
*                             bsr_dim,
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
rocsparse_status rocsparse_sbsrsv_solve(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans,
                                        rocsparse_int             mb,
                                        rocsparse_int             nnzb,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             bsr_dim,
                                        rocsparse_mat_info        info,
                                        const float*              x,
                                        float*                    y,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrsv_solve(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans,
                                        rocsparse_int             mb,
                                        rocsparse_int             nnzb,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             bsr_dim,
                                        rocsparse_mat_info        info,
                                        const double*             x,
                                        double*                   y,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrsv_solve(rocsparse_handle               handle,
                                        rocsparse_direction            dir,
                                        rocsparse_operation            trans,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* bsr_val,
                                        const rocsparse_int*           bsr_row_ptr,
                                        const rocsparse_int*           bsr_col_ind,
                                        rocsparse_int                  bsr_dim,
                                        rocsparse_mat_info             info,
                                        const rocsparse_float_complex* x,
                                        rocsparse_float_complex*       y,
                                        rocsparse_solve_policy         policy,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrsv_solve(rocsparse_handle                handle,
                                        rocsparse_direction             dir,
                                        rocsparse_operation             trans,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* bsr_val,
                                        const rocsparse_int*            bsr_row_ptr,
                                        const rocsparse_int*            bsr_col_ind,
                                        rocsparse_int                   bsr_dim,
                                        rocsparse_mat_info              info,
                                        const rocsparse_double_complex* x,
                                        rocsparse_double_complex*       y,
                                        rocsparse_solve_policy          policy,
                                        void*                           temp_buffer);
/**@}*/

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
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
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
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
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
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
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
*  rocsparse_scsrsm_analysis(), rocsparse_dcsrsm_analysis(),
*  rocsparse_ccsrsm_analysis(), rocsparse_zcsrsm_analysis(),
*  rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
*  rocsparse_ccsrilu0_analysis(), rocsparse_zcsrilu0_analysis(),
*  rocsparse_scsric0_analysis(), rocsparse_dcsric0_analysis(),
*  rocsparse_ccsric0_analysis() and rocsparse_zcsric0_analysis(). Selecting
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
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
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
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
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
*  Currently, only \p trans == \ref rocsparse_operation_none and
*  \p trans == \ref rocsparse_operation_transpose is supported.
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
*              \p trans == \ref rocsparse_operation_conjugate_transpose or
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
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
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
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
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

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using GEBSR storage format
*
*  \details
*  \p rocsparse_gebsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{row_block_dim}) \times (nb \cdot \text{col_block_dim})\f$
*  matrix, defined in GEBSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
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
*  dir         matrix storage of GEBSR blocks.
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse GEBSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse GEBSR matrix.
*  @param[in]
*  nnzb        number of non-zero blocks of the sparse GEBSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse GEBSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse GEBSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse GEBSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              GEBSR matrix.
*  @param[in]
*  row_block_dim row block dimension of the sparse GEBSR matrix.
*  @param[in]
*  col_block_dim column block dimension of the sparse GEBSR matrix.
*  @param[in]
*  x           array of \p nb*col_block_dim elements (\f$op(A) = A\f$) or \p mb*row_block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*row_block_dim elements (\f$op(A) = A\f$) or \p nb*col_block_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb, \p nnzb, \p row_block_dim
*              or \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_not_implemented
*              \p trans != \ref rocsparse_operation_none or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsrmv(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             nnzb,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr,
                                    const float*              bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             row_block_dim,
                                    rocsparse_int             col_block_dim,
                                    const float*              x,
                                    const float*              beta,
                                    float*                    y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsrmv(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    rocsparse_int             nnzb,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr,
                                    const double*             bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             row_block_dim,
                                    rocsparse_int             col_block_dim,
                                    const double*             x,
                                    const double*             beta,
                                    double*                   y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsrmv(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_operation            trans,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  nb,
                                    rocsparse_int                  nnzb,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr,
                                    const rocsparse_float_complex* bsr_val,
                                    const rocsparse_int*           bsr_row_ptr,
                                    const rocsparse_int*           bsr_col_ind,
                                    rocsparse_int                  row_block_dim,
                                    rocsparse_int                  col_block_dim,
                                    const rocsparse_float_complex* x,
                                    const rocsparse_float_complex* beta,
                                    rocsparse_float_complex*       y);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsrmv(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_operation             trans,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   nb,
                                    rocsparse_int                   nnzb,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr,
                                    const rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*            bsr_row_ptr,
                                    const rocsparse_int*            bsr_col_ind,
                                    rocsparse_int                   row_block_dim,
                                    rocsparse_int                   col_block_dim,
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
 *  \brief Sparse matrix dense matrix multiplication using BSR storage format
 *
 *  \details
 *  \p rocsparse_bsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$mb \times kb\f$
 *  matrix \f$A\f$, defined in BSR storage format, and the dense \f$k \times n\f$
 *  matrix \f$B\f$ (where \f$k = block\_dim \times kb\f$) and adds the result to the dense
 *  \f$m \times n\f$ matrix \f$C\f$ (where \f$m = block\_dim \times mb\f$) that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans_A == rocsparse_operation_none} \\
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans_B == rocsparse_operation_none} \\
 *        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
 *    \end{array}
 *    \right.
 *  \f]
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
 *  dir         the storage format of the blocks. Can be \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type. Currently, only \ref rocsparse_operation_none is supported.
 *  @param[in]
 *  trans_B     matrix \f$B\f$ operation type. Currently, only \ref rocsparse_operation_none and rocsparse_operation_transpose
 *              are supported.
 *  @param[in]
 *  mb          number of block rows of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  n           number of columns of the dense matrix \f$op(B)\f$ and \f$C\f$.
 *  @param[in]
 *  kb          number of block columns of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  nnzb        number of non-zero blocks of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix \f$A\f$. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  bsr_val     array of \p nnzb*block_dim*block_dim elements of the sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix \f$A\f$.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
 *              BSR matrix \f$A\f$.
 *  @param[in]
 *  block_dim   size of the blocks in the sparse BSR matrix.
 *  @param[in]
 *  B           array of dimension \f$ldb \times n\f$ (\f$op(B) == B\f$) or
 *              \f$ldb \times k\f$ (\f$op(B) == B^T\f$).
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$ where \f$k = block\_dim \times kb\f$.
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  C           array of dimension \f$ldc \times n\f$.
 *  @param[in]
 *  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$ where \f$m = block\_dim \times mb\f$.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p n, \p kb, \p nnzb, \p ldb or \p ldc
 *              is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
 *              \p bsr_row_ptr, \p bsr_col_ind, \p B, \p beta or \p C pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans_A != \ref rocsparse_operation_none or
 *              \p trans_B == \ref rocsparse_operation_conjugate_transpose or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example multiplies a BSR matrix with a dense matrix.
 *  \code{.c}
 *      //     1 2 0 3 0 0
 *      // A = 0 4 5 0 0 0
 *      //     0 0 0 7 8 0
 *      //     0 0 1 2 4 1
 *
 *      rocsparse_int block_dim = 2;
 *      rocsparse_int mb   = 2;
 *      rocsparse_int kb   = 3;
 *      rocsparse_int nnzb = 4;
 *      rocsparse_direction dir = rocsparse_direction_row;
 *
 *      bsr_row_ptr[mb+1]                 = {0, 2, 4};                                        // device memory
 *      bsr_col_ind[nnzb]                 = {0, 1, 1, 2};                                     // device memory
 *      bsr_val[nnzb*block_dim*block_dim] = {1, 2, 0, 4, 0, 3, 5, 0, 0, 7, 1, 2, 8, 0, 4, 1}; // device memory
 *
 *      // Set dimension n of B
 *      rocsparse_int n = 64;
 *      rocsparse_int m = mb * block_dim;
 *      rocsparse_int k = kb * block_dim;
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
 *      rocsparse_sbsrmm(handle,
 *                       dir,
 *                       rocsparse_operation_none,
 *                       rocsparse_operation_none,
 *                       mb,
 *                       n,
 *                       kb,
 *                       nnzb,
 *                       &alpha,
 *                       descr,
 *                       bsr_val,
 *                       bsr_row_ptr,
 *                       bsr_col_ind,
 *                       block_dim,
 *                       B,
 *                       k,
 *                       &beta,
 *                       C,
 *                       m);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrmm(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             mb,
                                  rocsparse_int             n,
                                  rocsparse_int             kb,
                                  rocsparse_int             nnzb,
                                  const float*              alpha,
                                  const rocsparse_mat_descr descr,
                                  const float*              bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  const float*              B,
                                  rocsparse_int             ldb,
                                  const float*              beta,
                                  float*                    C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrmm(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             mb,
                                  rocsparse_int             n,
                                  rocsparse_int             kb,
                                  rocsparse_int             nnzb,
                                  const double*             alpha,
                                  const rocsparse_mat_descr descr,
                                  const double*             bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  const double*             B,
                                  rocsparse_int             ldb,
                                  const double*             beta,
                                  double*                   C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrmm(rocsparse_handle               handle,
                                  rocsparse_direction            dir,
                                  rocsparse_operation            trans_A,
                                  rocsparse_operation            trans_B,
                                  rocsparse_int                  mb,
                                  rocsparse_int                  n,
                                  rocsparse_int                  kb,
                                  rocsparse_int                  nnzb,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* bsr_val,
                                  const rocsparse_int*           bsr_row_ptr,
                                  const rocsparse_int*           bsr_col_ind,
                                  rocsparse_int                  block_dim,
                                  const rocsparse_float_complex* B,
                                  rocsparse_int                  ldb,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       C,
                                  rocsparse_int                  ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrmm(rocsparse_handle                handle,
                                  rocsparse_direction             dir,
                                  rocsparse_operation             trans_A,
                                  rocsparse_operation             trans_B,
                                  rocsparse_int                   mb,
                                  rocsparse_int                   n,
                                  rocsparse_int                   kb,
                                  rocsparse_int                   nnzb,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* bsr_val,
                                  const rocsparse_int*            bsr_row_ptr,
                                  const rocsparse_int*            bsr_col_ind,
                                  rocsparse_int                   block_dim,
                                  const rocsparse_double_complex* B,
                                  rocsparse_int                   ldb,
                                  const rocsparse_double_complex* beta,
                                  rocsparse_double_complex*       C,
                                  rocsparse_int                   ldc);
/**@}*/

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using GEneral BSR storage format
 *
 *  \details
 *  \p rocsparse_gebsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$mb \times kb\f$
 *  matrix \f$A\f$, defined in GEneral BSR storage format, and the dense \f$k \times n\f$
 *  matrix \f$B\f$ (where \f$k = col_block\_dim \times kb\f$) and adds the result to the dense
 *  \f$m \times n\f$ matrix \f$C\f$ (where \f$m = row_block\_dim \times mb\f$) that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans_A == rocsparse_operation_none} \\
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans_B == rocsparse_operation_none} \\
 *        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
 *    \end{array}
 *    \right.
 *  \f]
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
 *  dir         the storage format of the blocks. Can be \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type. Currently, only \ref rocsparse_operation_none is supported.
 *  @param[in]
 *  trans_B     matrix \f$B\f$ operation type. Currently, only \ref rocsparse_operation_none and rocsparse_operation_transpose
 *              are supported.
 *  @param[in]
 *  mb          number of block rows of the sparse GEneral BSR matrix \f$A\f$.
 *  @param[in]
 *  n           number of columns of the dense matrix \f$op(B)\f$ and \f$C\f$.
 *  @param[in]
 *  kb          number of block columns of the sparse GEneral BSR matrix \f$A\f$.
 *  @param[in]
 *  nnzb        number of non-zero blocks of the sparse GEneral BSR matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  descr       descriptor of the sparse GEneral BSR matrix \f$A\f$. Currently, only
 *              \ref rocsparse_matrix_type_general is supported.
 *  @param[in]
 *  bsr_val     array of \p nnzb*row_block_dim*col_block_dim elements of the sparse GEneral BSR matrix \f$A\f$.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse GEneral BSR matrix \f$A\f$.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse
 *              GEneral BSR matrix \f$A\f$.
 *  @param[in]
 *  row_block_dim   row size of the blocks in the sparse GEneral BSR matrix.
 *  @param[in]
 *  col_block_dim   column size of the blocks in the sparse GEneral BSR matrix.
 *  @param[in]
 *  B           array of dimension \f$ldb \times n\f$ (\f$op(B) == B\f$) or
 *              \f$ldb \times k\f$ (\f$op(B) == B^T\f$).
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$ where \f$k = col_block\_dim \times kb\f$.
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  C           array of dimension \f$ldc \times n\f$.
 *  @param[in]
 *  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$ where \f$m = row_block\_dim \times mb\f$.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p n, \p kb, \p nnzb, \p ldb, \p ldc, \p row_block_dim
 *              or \p col_block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
 *              \p bsr_row_ptr, \p bsr_col_ind, \p B, \p beta or \p C pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_not_implemented
 *              \p trans_A != \ref rocsparse_operation_none or
 *              \p trans_B == \ref rocsparse_operation_conjugate_transpose or
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  This example multiplies a GEneral BSR matrix with a dense matrix.
 *  \code{.c}
 *      //     1 2 0 3 0 0
 *      // A = 0 4 5 0 0 0
 *      //     0 0 0 7 8 0
 *      //     0 0 1 2 4 1
 *
 *      rocsparse_int row_block_dim = 2;
 *      rocsparse_int col_block_dim = 3;
 *      rocsparse_int mb   = 2;
 *      rocsparse_int kb   = 2;
 *      rocsparse_int nnzb = 4;
 *      rocsparse_direction dir = rocsparse_direction_row;
 *
 *      bsr_row_ptr[mb+1]                 = {0, 2, 4};                                        // device memory
 *      bsr_col_ind[nnzb]                 = {0, 1, 0, 1};                                     // device memory
 *      bsr_val[nnzb*row_block_dim*col_block_dim] = {1, 2, 0, 0, 4, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 8, 0, 2, 4, 1}; // device memory
 *
 *      // Set dimension n of B
 *      rocsparse_int n = 64;
 *      rocsparse_int m = mb * row_block_dim;
 *      rocsparse_int k = kb * col_block_dim;
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
 *      rocsparse_sgebsrmm(handle,
 *                         dir,
 *                         rocsparse_operation_none,
 *                         rocsparse_operation_none,
 *                         mb,
 *                         n,
 *                         kb,
 *                         nnzb,
 *                         &alpha,
 *                         descr,
 *                         bsr_val,
 *                         bsr_row_ptr,
 *                         bsr_col_ind,
 *                         row_block_dim,
 *                         col_block_dim,
 *                         B,
 *                         k,
 *                         &beta,
 *                         C,
 *                         m);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsrmm(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             mb,
                                    rocsparse_int             n,
                                    rocsparse_int             kb,
                                    rocsparse_int             nnzb,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr,
                                    const float*              bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             row_block_dim,
                                    rocsparse_int             col_block_dim,
                                    const float*              B,
                                    rocsparse_int             ldb,
                                    const float*              beta,
                                    float*                    C,
                                    rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsrmm(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_int             mb,
                                    rocsparse_int             n,
                                    rocsparse_int             kb,
                                    rocsparse_int             nnzb,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr,
                                    const double*             bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             row_block_dim,
                                    rocsparse_int             col_block_dim,
                                    const double*             B,
                                    rocsparse_int             ldb,
                                    const double*             beta,
                                    double*                   C,
                                    rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsrmm(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_operation            trans_A,
                                    rocsparse_operation            trans_B,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  n,
                                    rocsparse_int                  kb,
                                    rocsparse_int                  nnzb,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr,
                                    const rocsparse_float_complex* bsr_val,
                                    const rocsparse_int*           bsr_row_ptr,
                                    const rocsparse_int*           bsr_col_ind,
                                    rocsparse_int                  row_block_dim,
                                    rocsparse_int                  col_block_dim,
                                    const rocsparse_float_complex* B,
                                    rocsparse_int                  ldb,
                                    const rocsparse_float_complex* beta,
                                    rocsparse_float_complex*       C,
                                    rocsparse_int                  ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsrmm(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_operation             trans_A,
                                    rocsparse_operation             trans_B,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   n,
                                    rocsparse_int                   kb,
                                    rocsparse_int                   nnzb,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr,
                                    const rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*            bsr_row_ptr,
                                    const rocsparse_int*            bsr_col_ind,
                                    rocsparse_int                   row_block_dim,
                                    rocsparse_int                   colblock_dim,
                                    const rocsparse_double_complex* B,
                                    rocsparse_int                   ldb,
                                    const rocsparse_double_complex* beta,
                                    rocsparse_double_complex*       C,
                                    rocsparse_int                   ldc);

/**@}*/

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
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
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

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p rocsparse_csrsm_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during rocsparse_scsrsm_solve(),
*  rocsparse_dcsrsm_solve(), rocsparse_ccsrsm_solve() or rocsparse_zcsrsm_solve()
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_csrsm_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_csrsm_zero_pivot(rocsparse_handle   handle,
                                            rocsparse_mat_info info,
                                            rocsparse_int*     position);

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p rocsparse_csrsm_buffer_size returns the size of the temporary storage buffer that
*  is required by rocsparse_scsrsm_analysis(), rocsparse_dcsrsm_analysis(),
*  rocsparse_ccsrsm_analysis(), rocsparse_zcsrsm_analysis(), rocsparse_scsrsm_solve(),
*  rocsparse_dcsrsm_solve(), rocsparse_ccsrsm_solve() and rocsparse_zcsrsm_solve(). The
*  temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_B     matrix B operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix A.
*  @param[in]
*  nrhs        number of columns of the dense matrix op(B).
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix A.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix A.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix A.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix A.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix A.
*  @param[in]
*  B           array of \p m \f$\times\f$ \p nrhs elements of the rhs matrix B.
*  @param[in]
*  ldb         leading dimension of rhs matrix B.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref rocsparse_solve_policy_auto.
*  @param[out]
*  buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_scsrsm_analysis(), rocsparse_dcsrsm_analysis(),
*              rocsparse_ccsrsm_analysis(), rocsparse_zcsrsm_analysis(),
*              rocsparse_scsrsm_solve(), rocsparse_dcsrsm_solve(),
*              rocsparse_ccsrsm_solve() and rocsparse_zcsrsm_solve().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p nrhs or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p alpha, \p descr, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p B, \p info or \p buffer_size pointer
*              is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_B == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_int             m,
                                              rocsparse_int             nrhs,
                                              rocsparse_int             nnz,
                                              const float*              alpha,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              const float*              B,
                                              rocsparse_int             ldb,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsm_buffer_size(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_int             m,
                                              rocsparse_int             nrhs,
                                              rocsparse_int             nnz,
                                              const double*             alpha,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              const double*             B,
                                              rocsparse_int             ldb,
                                              rocsparse_mat_info        info,
                                              rocsparse_solve_policy    policy,
                                              size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsm_buffer_size(rocsparse_handle               handle,
                                              rocsparse_operation            trans_A,
                                              rocsparse_operation            trans_B,
                                              rocsparse_int                  m,
                                              rocsparse_int                  nrhs,
                                              rocsparse_int                  nnz,
                                              const rocsparse_float_complex* alpha,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* csr_val,
                                              const rocsparse_int*           csr_row_ptr,
                                              const rocsparse_int*           csr_col_ind,
                                              const rocsparse_float_complex* B,
                                              rocsparse_int                  ldb,
                                              rocsparse_mat_info             info,
                                              rocsparse_solve_policy         policy,
                                              size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsm_buffer_size(rocsparse_handle                handle,
                                              rocsparse_operation             trans_A,
                                              rocsparse_operation             trans_B,
                                              rocsparse_int                   m,
                                              rocsparse_int                   nrhs,
                                              rocsparse_int                   nnz,
                                              const rocsparse_double_complex* alpha,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* csr_val,
                                              const rocsparse_int*            csr_row_ptr,
                                              const rocsparse_int*            csr_col_ind,
                                              const rocsparse_double_complex* B,
                                              rocsparse_int                   ldb,
                                              rocsparse_mat_info              info,
                                              rocsparse_solve_policy          policy,
                                              size_t*                         buffer_size);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p rocsparse_csrsm_analysis performs the analysis step for rocsparse_scsrsm_solve(),
*  rocsparse_dcsrsm_solve(), rocsparse_ccsrsm_solve() and rocsparse_zcsrsm_solve(). It
*  is expected that this function will be executed only once for a given matrix and
*  particular operation type. The analysis meta data can be cleared by
*  rocsparse_csrsm_clear().
*
*  \p rocsparse_csrsm_analysis can share its meta data with
*  rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
*  rocsparse_ccsrilu0_analysis(), rocsparse_zcsrilu0_analysis(),
*  rocsparse_scsric0_analysis(), rocsparse_dcsric0_analysis(),
*  rocsparse_ccsric0_analysis(), rocsparse_zcsric0_analysis(),
*  rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(),
*  rocsparse_ccsrsv_analysis() and rocsparse_zcsrsv_analysis(). Selecting
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
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_B     matrix B operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix A.
*  @param[in]
*  nrhs        number of columns of the dense matrix op(B).
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix A.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix A.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix A.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix A.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix A.
*  @param[in]
*  B           array of \p m \f$\times\f$ \p nrhs elements of the rhs matrix B.
*  @param[in]
*  ldb         leading dimension of rhs matrix B.
*  @param[out]
*  info        structure that holds the information collected during the analysis step.
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
*  \retval     rocsparse_status_invalid_size \p m, \p nrhs or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p alpha, \p descr, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p B, \p info or \p temp_buffer pointer
*              is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_B == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrsm_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_int             m,
                                           rocsparse_int             nrhs,
                                           rocsparse_int             nnz,
                                           const float*              alpha,
                                           const rocsparse_mat_descr descr,
                                           const float*              csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           const float*              B,
                                           rocsparse_int             ldb,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsm_analysis(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_int             m,
                                           rocsparse_int             nrhs,
                                           rocsparse_int             nnz,
                                           const double*             alpha,
                                           const rocsparse_mat_descr descr,
                                           const double*             csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           const double*             B,
                                           rocsparse_int             ldb,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsm_analysis(rocsparse_handle               handle,
                                           rocsparse_operation            trans_A,
                                           rocsparse_operation            trans_B,
                                           rocsparse_int                  m,
                                           rocsparse_int                  nrhs,
                                           rocsparse_int                  nnz,
                                           const rocsparse_float_complex* alpha,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* csr_val,
                                           const rocsparse_int*           csr_row_ptr,
                                           const rocsparse_int*           csr_col_ind,
                                           const rocsparse_float_complex* B,
                                           rocsparse_int                  ldb,
                                           rocsparse_mat_info             info,
                                           rocsparse_analysis_policy      analysis,
                                           rocsparse_solve_policy         solve,
                                           void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsm_analysis(rocsparse_handle                handle,
                                           rocsparse_operation             trans_A,
                                           rocsparse_operation             trans_B,
                                           rocsparse_int                   m,
                                           rocsparse_int                   nrhs,
                                           rocsparse_int                   nnz,
                                           const rocsparse_double_complex* alpha,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* csr_val,
                                           const rocsparse_int*            csr_row_ptr,
                                           const rocsparse_int*            csr_col_ind,
                                           const rocsparse_double_complex* B,
                                           rocsparse_int                   ldb,
                                           rocsparse_mat_info              info,
                                           rocsparse_analysis_policy       analysis,
                                           rocsparse_solve_policy          solve,
                                           void*                           temp_buffer);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p rocsparse_csrsm_clear deallocates all memory that was allocated by
*  rocsparse_scsrsm_analysis(), rocsparse_dcsrsm_analysis(), rocsparse_ccsrsm_analysis()
*  or rocsparse_zcsrsm_analysis(). This is especially useful, if memory is an issue and
*  the analysis data is not required for further computation, e.g. when switching to
*  another sparse matrix format. Calling \p rocsparse_csrsm_clear is optional. All
*  allocated resources will be cleared, when the opaque \ref rocsparse_mat_info struct
*  is destroyed using rocsparse_destroy_mat_info().
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
rocsparse_status rocsparse_csrsm_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p rocsparse_csrsm_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution matrix
*  \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot op(X) = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  ,
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if trans_B == rocsparse_operation_none} \\
*        X^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        X^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \p rocsparse_csrsm_solve requires a user allocated temporary buffer. Its size is
*  returned by rocsparse_scsrsm_buffer_size(), rocsparse_dcsrsm_buffer_size(),
*  rocsparse_ccsrsm_buffer_size() or rocsparse_zcsrsm_buffer_size(). Furthermore,
*  analysis meta data is required. It can be obtained by rocsparse_scsrsm_analysis(),
*  rocsparse_dcsrsm_analysis(), rocsparse_ccsrsm_analysis() or
*  rocsparse_zcsrsm_analysis(). \p rocsparse_csrsm_solve reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be checked calling
*  rocsparse_csrsm_zero_pivot(). If
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
*  Currently, only \p trans_A != \ref rocsparse_operation_conjugate_transpose and
*  \p trans_B != \ref rocsparse_operation_conjugate_transpose is supported.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  trans_A     matrix A operation type.
*  @param[in]
*  trans_B     matrix B operation type.
*  @param[in]
*  m           number of rows of the sparse CSR matrix A.
*  @param[in]
*  nrhs        number of columns of the dense matrix op(B).
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix A.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse CSR matrix A.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix A.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix A.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix A.
*  @param[inout]
*  B           array of \p m \f$\times\f$ \p nrhs elements of the rhs matrix B.
*  @param[in]
*  ldb         leading dimension of rhs matrix B.
*  @param[in]
*  info        structure that holds the information collected during the analysis step.
*  @param[in]
*  policy      \ref rocsparse_solve_policy_auto.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p nrhs or \p nnz is invalid.
*  \retval     rocsparse_status_invalid_pointer \p alpha, \p descr, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p B, \p info or \p temp_buffer pointer
*              is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \p trans_A == \ref rocsparse_operation_conjugate_transpose,
*              \p trans_B == \ref rocsparse_operation_conjugate_transpose or
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  Consider the lower triangular \f$m \times m\f$ matrix \f$L\f$, stored in CSR
*  storage format with unit diagonal. The following example solves \f$L \cdot X = B\f$.
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
*      rocsparse_dcsrsm_buffer_size(handle,
*                                   rocsparse_operation_none,
*                                   rocsparse_operation_none,
*                                   m,
*                                   nrhs,
*                                   nnz,
*                                   &alpha,
*                                   descr,
*                                   csr_val,
*                                   csr_row_ptr,
*                                   csr_col_ind,
*                                   B,
*                                   ldb,
*                                   info,
*                                   rocsparse_solve_policy_auto,
*                                   &buffer_size);
*
*      // Allocate temporary buffer
*      void* temp_buffer;
*      hipMalloc(&temp_buffer, buffer_size);
*
*      // Perform analysis step
*      rocsparse_dcsrsm_analysis(handle,
*                                rocsparse_operation_none,
*                                rocsparse_operation_none,
*                                m,
*                                nrhs,
*                                nnz,
*                                &alpha,
*                                descr,
*                                csr_val,
*                                csr_row_ptr,
*                                csr_col_ind,
*                                B,
*                                ldb,
*                                info,
*                                rocsparse_analysis_policy_reuse,
*                                rocsparse_solve_policy_auto,
*                                temp_buffer);
*
*      // Solve LX = B
*      rocsparse_dcsrsm_solve(handle,
*                             rocsparse_operation_none,
*                             rocsparse_operation_none,
*                             m,
*                             nrhs,
*                             nnz,
*                             &alpha,
*                             descr,
*                             csr_val,
*                             csr_row_ptr,
*                             csr_col_ind,
*                             B,
*                             ldb,
*                             info,
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
rocsparse_status rocsparse_scsrsm_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             m,
                                        rocsparse_int             nrhs,
                                        rocsparse_int             nnz,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        float*                    B,
                                        rocsparse_int             ldb,
                                        rocsparse_mat_info        info,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrsm_solve(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             m,
                                        rocsparse_int             nrhs,
                                        rocsparse_int             nnz,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        double*                   B,
                                        rocsparse_int             ldb,
                                        rocsparse_mat_info        info,
                                        rocsparse_solve_policy    policy,
                                        void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrsm_solve(rocsparse_handle               handle,
                                        rocsparse_operation            trans_A,
                                        rocsparse_operation            trans_B,
                                        rocsparse_int                  m,
                                        rocsparse_int                  nrhs,
                                        rocsparse_int                  nnz,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* csr_val,
                                        const rocsparse_int*           csr_row_ptr,
                                        const rocsparse_int*           csr_col_ind,
                                        rocsparse_float_complex*       B,
                                        rocsparse_int                  ldb,
                                        rocsparse_mat_info             info,
                                        rocsparse_solve_policy         policy,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrsm_solve(rocsparse_handle                handle,
                                        rocsparse_operation             trans_A,
                                        rocsparse_operation             trans_B,
                                        rocsparse_int                   m,
                                        rocsparse_int                   nrhs,
                                        rocsparse_int                   nnz,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* csr_val,
                                        const rocsparse_int*            csr_row_ptr,
                                        const rocsparse_int*            csr_col_ind,
                                        rocsparse_double_complex*       B,
                                        rocsparse_int                   ldb,
                                        rocsparse_mat_info              info,
                                        rocsparse_solve_policy          policy,
                                        void*                           temp_buffer);
/**@}*/

/*! \ingroup level3_module
*  \brief Dense matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p rocsparse_gemmi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times k\f$
*  matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  trans_A     matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B     matrix \f$B\f$ operation type.
*  @param[in]
*  m           number of rows of the dense matrix \f$A\f$.
*  @param[in]
*  n           number of columns of the sparse CSR matrix \f$op(B)\f$ and \f$C\f$.
*  @param[in]
*  k           number of columns of the dense matrix \f$A\f$.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  A           array of dimension \f$lda \times k\f$ (\f$op(A) == A\f$) or
*              \f$lda \times m\f$ (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
*  @param[in]
*  lda         leading dimension of \f$A\f$, must be at least \f$m\f$
*              (\f$op(A) == A\f$) or \f$k\f$ (\f$op(A) == A^T\f$ or
*              \f$op(A) == A^H\f$).
*  @param[in]
*  descr       descriptor of the sparse CSR matrix \f$B\f$. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse CSR
*              matrix \f$B\f$.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  C           array of dimension \f$ldc \times n\f$ that holds the values of \f$C\f$.
*  @param[in]
*  ldc         leading dimension of \f$C\f$, must be at least \f$m\f$.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p n, \p k, \p nnz, \p lda or \p ldc
*              is invalid.
*  \retval     rocsparse_status_invalid_pointer \p alpha, \p A, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind, \p beta or \p C pointer is invalid.
*
*  \par Example
*  This example multiplies a dense matrix with a CSC matrix.
*  \code{.c}
*      rocsparse_int m   = 2;
*      rocsparse_int n   = 5;
*      rocsparse_int k   = 3;
*      rocsparse_int nnz = 8;
*      rocsparse_int lda = m;
*      rocsparse_int ldc = m;
*
*      // Matrix A (m x k)
*      // (  9.0  10.0  11.0 )
*      // ( 12.0  13.0  14.0 )
*
*      // Matrix B (k x n)
*      // ( 1.0  2.0  0.0  3.0  0.0 )
*      // ( 0.0  4.0  5.0  0.0  0.0 )
*      // ( 6.0  0.0  0.0  7.0  8.0 )
*
*      // Matrix C (m x n)
*      // ( 15.0  16.0  17.0  18.0  19.0 )
*      // ( 20.0  21.0  22.0  23.0  24.0 )
*
*      A[lda * k]           = {9.0, 12.0, 10.0, 13.0, 11.0, 14.0};      // device memory
*      csc_col_ptr_B[n + 1] = {0, 2, 4, 5, 7, 8};                       // device memory
*      csc_row_ind_B[nnz]   = {0, 0, 1, 1, 2, 3, 3, 4};                 // device memory
*      csc_val_B[nnz]       = {1.0, 6.0, 2.0, 4.0, 5.0, 3.0, 7.0, 8.0}; // device memory
*      C[ldc * n]           = {15.0, 20.0, 16.0, 21.0, 17.0, 22.0,      // device memory
*                              18.0, 23.0, 19.0, 24.0};
*
*      // alpha and beta
*      float alpha = 1.0f;
*      float beta  = 0.0f;
*
*      // Perform the matrix multiplication
*      rocsparse_sgemmi(handle,
*                       rocsparse_operation_none,
*                       rocsparse_operation_transpose,
*                       m,
*                       n,
*                       k,
*                       nnz,
*                       &alpha,
*                       A,
*                       lda,
*                       descr_B,
*                       csc_val_B,
*                       csc_col_ptr_B,
*                       csc_row_ind_B,
*                       &beta,
*                       C,
*                       ldc);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgemmi(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const float*              alpha,
                                  const float*              A,
                                  rocsparse_int             lda,
                                  const rocsparse_mat_descr descr,
                                  const float*              csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const float*              beta,
                                  float*                    C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgemmi(rocsparse_handle          handle,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             m,
                                  rocsparse_int             n,
                                  rocsparse_int             k,
                                  rocsparse_int             nnz,
                                  const double*             alpha,
                                  const double*             A,
                                  rocsparse_int             lda,
                                  const rocsparse_mat_descr descr,
                                  const double*             csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  const double*             beta,
                                  double*                   C,
                                  rocsparse_int             ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgemmi(rocsparse_handle               handle,
                                  rocsparse_operation            trans_A,
                                  rocsparse_operation            trans_B,
                                  rocsparse_int                  m,
                                  rocsparse_int                  n,
                                  rocsparse_int                  k,
                                  rocsparse_int                  nnz,
                                  const rocsparse_float_complex* alpha,
                                  const rocsparse_float_complex* A,
                                  rocsparse_int                  lda,
                                  const rocsparse_mat_descr      descr,
                                  const rocsparse_float_complex* csr_val,
                                  const rocsparse_int*           csr_row_ptr,
                                  const rocsparse_int*           csr_col_ind,
                                  const rocsparse_float_complex* beta,
                                  rocsparse_float_complex*       C,
                                  rocsparse_int                  ldc);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgemmi(rocsparse_handle                handle,
                                  rocsparse_operation             trans_A,
                                  rocsparse_operation             trans_B,
                                  rocsparse_int                   m,
                                  rocsparse_int                   n,
                                  rocsparse_int                   k,
                                  rocsparse_int                   nnz,
                                  const rocsparse_double_complex* alpha,
                                  const rocsparse_double_complex* A,
                                  rocsparse_int                   lda,
                                  const rocsparse_mat_descr       descr,
                                  const rocsparse_double_complex* csr_val,
                                  const rocsparse_int*            csr_row_ptr,
                                  const rocsparse_int*            csr_col_ind,
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
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p rocsparse_csrgeam_nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
*  size \p m + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*  \note
*  Currently, only \ref rocsparse_matrix_type_general is supported.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  descr_A         descriptor of the sparse CSR matrix \f$A\f$. Currenty, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_A           number of non-zero entries of the sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_row_ptr_A   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currenty, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currenty, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  nnz_C           pointer to the number of non-zero entries of the sparse CSR
*                  matrix \f$C\f$. \p nnz_C can be a host or device pointer.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p nnz_A or \p nnz_B is invalid.
*  \retval rocsparse_status_invalid_pointer \p descr_A, \p csr_row_ptr_A,
*          \p csr_col_ind_A, \p descr_B, \p csr_row_ptr_B, \p csr_col_ind_B,
*          \p descr_C, \p csr_row_ptr_C or \p nnz_C is invalid.
*  \retval rocsparse_status_not_implemented
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr descr_A,
                                       rocsparse_int             nnz_A,
                                       const rocsparse_int*      csr_row_ptr_A,
                                       const rocsparse_int*      csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       rocsparse_int             nnz_B,
                                       const rocsparse_int*      csr_row_ptr_B,
                                       const rocsparse_int*      csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       rocsparse_int*            csr_row_ptr_C,
                                       rocsparse_int*            nnz_C);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p rocsparse_csrgeam multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by rocsparse_csrgeam_nnz().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref rocsparse_matrix_type_general is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  m               number of rows of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
*  @param[in]
*  n               number of columns of the sparse CSR matrix \f$A\f$, \f$B\f$ and \f$C\f$.
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
*  csr_row_ptr_A   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  csr_col_ind_A   array of \p nnz_A elements containing the column indices of the
*                  sparse CSR matrix \f$A\f$.
*  @param[in]
*  beta            scalar \f$\beta\f$.
*  @param[in]
*  descr_B         descriptor of the sparse CSR matrix \f$B\f$. Currenty, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  nnz_B           number of non-zero entries of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_val_B       array of \p nnz_B elements of the sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_row_ptr_B   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  csr_col_ind_B   array of \p nnz_B elements containing the column indices of the
*                  sparse CSR matrix \f$B\f$.
*  @param[in]
*  descr_C         descriptor of the sparse CSR matrix \f$C\f$. Currenty, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val_C       array of elements of the sparse CSR matrix \f$C\f$.
*  @param[in]
*  csr_row_ptr_C   array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix \f$C\f$.
*  @param[out]
*  csr_col_ind_C   array of elements containing the column indices of the
*                  sparse CSR matrix \f$C\f$.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_size \p m, \p n, \p nnz_A or \p nnz_B is invalid.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p descr_A, \p csr_val_A,
*          \p csr_row_ptr_A, \p csr_col_ind_A, \p beta, \p descr_B, \p csr_val_B,
*          \p csr_row_ptr_B, \p csr_col_ind_B, \p descr_C, \p csr_val_C,
*          \p csr_row_ptr_C or \p csr_col_ind_C is invalid.
*  \retval rocsparse_status_not_implemented
*          \p rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example adds two CSR matrices.
*  \code{.c}
*  // Initialize scalar multipliers
*  float alpha = 1.0f;
*  float beta  = 1.0f;
*
*  // Create matrix descriptors
*  rocsparse_mat_descr descr_A;
*  rocsparse_mat_descr descr_B;
*  rocsparse_mat_descr descr_C;
*
*  rocsparse_create_mat_descr(&descr_A);
*  rocsparse_create_mat_descr(&descr_B);
*  rocsparse_create_mat_descr(&descr_C);
*
*  // Set pointer mode
*  rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
*
*  // Obtain number of total non-zero entries in C and row pointers of C
*  rocsparse_int nnz_C;
*  hipMalloc((void**)&csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1));
*
*  rocsparse_csrgeam_nnz(handle,
*                        m,
*                        n,
*                        descr_A,
*                        nnz_A,
*                        csr_row_ptr_A,
*                        csr_col_ind_A,
*                        descr_B,
*                        nnz_B,
*                        csr_row_ptr_B,
*                        csr_col_ind_B,
*                        descr_C,
*                        csr_row_ptr_C,
*                        &nnz_C);
*
*  // Compute column indices and values of C
*  hipMalloc((void**)&csr_col_ind_C, sizeof(rocsparse_int) * nnz_C);
*  hipMalloc((void**)&csr_val_C, sizeof(float) * nnz_C);
*
*  rocsparse_scsrgeam(handle,
*                     m,
*                     n,
*                     &alpha,
*                     descr_A,
*                     nnz_A,
*                     csr_val_A,
*                     csr_row_ptr_A,
*                     csr_col_ind_A,
*                     &beta,
*                     descr_B,
*                     nnz_B,
*                     csr_val_B,
*                     csr_row_ptr_B,
*                     csr_col_ind_B,
*                     descr_C,
*                     csr_val_C,
*                     csr_row_ptr_C,
*                     csr_col_ind_C);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrgeam(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const float*              alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const float*              csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const float*              beta,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const float*              csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const rocsparse_mat_descr descr_C,
                                    float*                    csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrgeam(rocsparse_handle          handle,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const double*             alpha,
                                    const rocsparse_mat_descr descr_A,
                                    rocsparse_int             nnz_A,
                                    const double*             csr_val_A,
                                    const rocsparse_int*      csr_row_ptr_A,
                                    const rocsparse_int*      csr_col_ind_A,
                                    const double*             beta,
                                    const rocsparse_mat_descr descr_B,
                                    rocsparse_int             nnz_B,
                                    const double*             csr_val_B,
                                    const rocsparse_int*      csr_row_ptr_B,
                                    const rocsparse_int*      csr_col_ind_B,
                                    const rocsparse_mat_descr descr_C,
                                    double*                   csr_val_C,
                                    const rocsparse_int*      csr_row_ptr_C,
                                    rocsparse_int*            csr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrgeam(rocsparse_handle               handle,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    const rocsparse_float_complex* alpha,
                                    const rocsparse_mat_descr      descr_A,
                                    rocsparse_int                  nnz_A,
                                    const rocsparse_float_complex* csr_val_A,
                                    const rocsparse_int*           csr_row_ptr_A,
                                    const rocsparse_int*           csr_col_ind_A,
                                    const rocsparse_float_complex* beta,
                                    const rocsparse_mat_descr      descr_B,
                                    rocsparse_int                  nnz_B,
                                    const rocsparse_float_complex* csr_val_B,
                                    const rocsparse_int*           csr_row_ptr_B,
                                    const rocsparse_int*           csr_col_ind_B,
                                    const rocsparse_mat_descr      descr_C,
                                    rocsparse_float_complex*       csr_val_C,
                                    const rocsparse_int*           csr_row_ptr_C,
                                    rocsparse_int*                 csr_col_ind_C);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrgeam(rocsparse_handle                handle,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    const rocsparse_double_complex* alpha,
                                    const rocsparse_mat_descr       descr_A,
                                    rocsparse_int                   nnz_A,
                                    const rocsparse_double_complex* csr_val_A,
                                    const rocsparse_int*            csr_row_ptr_A,
                                    const rocsparse_int*            csr_col_ind_A,
                                    const rocsparse_double_complex* beta,
                                    const rocsparse_mat_descr       descr_B,
                                    rocsparse_int                   nnz_B,
                                    const rocsparse_double_complex* csr_val_B,
                                    const rocsparse_int*            csr_row_ptr_B,
                                    const rocsparse_int*            csr_col_ind_B,
                                    const rocsparse_mat_descr       descr_C,
                                    rocsparse_double_complex*       csr_val_C,
                                    const rocsparse_int*            csr_row_ptr_C,
                                    rocsparse_int*                  csr_col_ind_C);
/**@}*/

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
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
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
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_bsric0_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
 *  structural or numerical zero has been found during rocsparse_sbsric0(),
 *  rocsparse_dbsric0(), rocsparse_cbsric0() or rocsparse_zbsric0() computation.
 *  The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same
 *  index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position=j means that either the diagonal block \p A(j,j)
 *  is missing (structural zero) or the diagonal block \p A(j,j) is not positive definite
 *  (numerical zero).
 *
 *  \note \p rocsparse_bsric0_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_bsric0_zero_pivot(rocsparse_handle   handle,
                                             rocsparse_mat_info info,
                                             rocsparse_int*     position);

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_bsric0_buffer_size returns the size of the temporary storage buffer
 *  that is required by rocsparse_sbsric0_analysis(), rocsparse_dbsric0_analysis(),
 *  rocsparse_cbsric0_analysis(), rocsparse_zbsric0_analysis(), rocsparse_sbsric0(),
 *  rocsparse_dbsric0(), rocsparse_sbsric0() and rocsparse_dbsric0(). The temporary
 *  storage buffer must be allocated by the user. The size of the temporary storage
 *  buffer is identical to the size returned by rocsparse_sbsrsv_buffer_size(),
 *  rocsparse_dbsrsv_buffer_size(), rocsparse_cbsrsv_buffer_size(), rocsparse_zbsrsv_buffer_size(),
 *  rocsparse_sbsrilu0_buffer_size(), rocsparse_dbsrilu0_buffer_size(), rocsparse_cbsrilu0_buffer_size()
 *  and rocsparse_zbsrilu0_buffer_size() if the matrix sparsity pattern is identical. The user
 *  allocated buffer can thus be shared between subsequent calls to those functions.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  dir 	    direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
 *              \ref rocsparse_direction_row.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
 *  @param[out]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  buffer_size number of bytes of the temporary storage buffer required by
 *              rocsparse_sbsric0_analysis(), rocsparse_dbsric0_analysis(),
 *              rocsparse_cbsric0_analysis(), rocsparse_zbsric0_analysis(),
 *              rocsparse_sbsric0(), rocsparse_dbsric0(), rocsparse_cbsric0()
 *              and rocsparse_zbsric0().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
 *              \p bsr_col_ind, \p info or \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsric0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               const float*              bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsric0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               const double*             bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsric0_buffer_size(rocsparse_handle               handle,
                                               rocsparse_direction            dir,
                                               rocsparse_int                  mb,
                                               rocsparse_int                  nnzb,
                                               const rocsparse_mat_descr      descr,
                                               const rocsparse_float_complex* bsr_val,
                                               const rocsparse_int*           bsr_row_ptr,
                                               const rocsparse_int*           bsr_col_ind,
                                               rocsparse_int                  block_dim,
                                               rocsparse_mat_info             info,
                                               size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsric0_buffer_size(rocsparse_handle                handle,
                                               rocsparse_direction             dir,
                                               rocsparse_int                   mb,
                                               rocsparse_int                   nnzb,
                                               const rocsparse_mat_descr       descr,
                                               const rocsparse_double_complex* bsr_val,
                                               const rocsparse_int*            bsr_row_ptr,
                                               const rocsparse_int*            bsr_col_ind,
                                               rocsparse_int                   block_dim,
                                               rocsparse_mat_info              info,
                                               size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_bsric0_analysis performs the analysis step for rocsparse_sbsric0()
 *  rocsparse_dbsric0(), rocsparse_cbsric0(), and rocsparse_zbsric0(). It is expected
 *  that this function will be executed only once for a given matrix and particular
 *  operation type. The analysis meta data can be cleared by rocsparse_bsric0_clear().
 *
 *  \p rocsparse_bsric0_analysis can share its meta data with
 *  rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(),
 *  rocsparse_cbsrilu0_analysis(), rocsparse_zbsrilu0_analysis(),
 *  rocsparse_sbsrsv_analysis(), rocsparse_dbsrsv_analysis(),
 *  rocsparse_cbsrsv_analysis(), rocsparse_zbsrsv_analysis(),
 *  rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(),
 *  rocsparse_cbsrsm_analysis() and rocsparse_zbsrsm_analysis(). Selecting
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
 *  dir 	    direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
 *              \ref rocsparse_direction_row.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
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
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
 *              \p bsr_col_ind, \p info or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsric0_analysis(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nnzb,
                                            const rocsparse_mat_descr descr,
                                            const float*              bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsric0_analysis(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nnzb,
                                            const rocsparse_mat_descr descr,
                                            const double*             bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsric0_analysis(rocsparse_handle               handle,
                                            rocsparse_direction            dir,
                                            rocsparse_int                  mb,
                                            rocsparse_int                  nnzb,
                                            const rocsparse_mat_descr      descr,
                                            const rocsparse_float_complex* bsr_val,
                                            const rocsparse_int*           bsr_row_ptr,
                                            const rocsparse_int*           bsr_col_ind,
                                            rocsparse_int                  block_dim,
                                            rocsparse_mat_info             info,
                                            rocsparse_analysis_policy      analysis,
                                            rocsparse_solve_policy         solve,
                                            void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsric0_analysis(rocsparse_handle                handle,
                                            rocsparse_direction             dir,
                                            rocsparse_int                   mb,
                                            rocsparse_int                   nnzb,
                                            const rocsparse_mat_descr       descr,
                                            const rocsparse_double_complex* bsr_val,
                                            const rocsparse_int*            bsr_row_ptr,
                                            const rocsparse_int*            bsr_col_ind,
                                            rocsparse_int                   block_dim,
                                            rocsparse_mat_info              info,
                                            rocsparse_analysis_policy       analysis,
                                            rocsparse_solve_policy          solve,
                                            void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_bsric0_clear deallocates all memory that was allocated by
 *  rocsparse_sbsric0_analysis(), rocsparse_dbsric0_analysis(), rocsparse_cbsric0_analysis()
 *  or rocsparse_zbsric0_analysis(). This is especially useful, if memory is an issue and
 *  the analysis data is not required for further computation.
 *
 *  \note
 *  Calling \p rocsparse_bsric0_clear is optional. All allocated resources will be
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
rocsparse_status rocsparse_bsric0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p rocsparse_bsric0 computes the incomplete Cholesky factorization with 0 fill-ins
 *  and no pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LL^T
 *  \f]
 *
 *  \p rocsparse_bsric0 requires a user allocated temporary buffer. Its size is returned
 *  by rocsparse_sbsric0_buffer_size(), rocsparse_dbsric0_buffer_size(),
 *  rocsparse_cbsric0_buffer_size() or rocsparse_zbsric0_buffer_size(). Furthermore,
 *  analysis meta data is required. It can be obtained by rocsparse_sbsric0_analysis(),
 *  rocsparse_dbsric0_analysis(), rocsparse_cbsric0_analysis() or rocsparse_zbsric0_analysis().
 *  \p rocsparse_bsric0 reports the first zero pivot (either numerical or structural zero).
 *  The zero pivot status can be obtained by calling rocsparse_bsric0_zero_pivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  dir 	    direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
 *              \ref rocsparse_direction_row.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[inout]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy      \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr
 *              or \p bsr_col_ind pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  Consider the sparse \f$m \times m\f$ matrix \f$A\f$, stored in BSR
 *  storage format. The following example computes the incomplete Cholesky factorization
 *  \f$M \approx LL^T\f$ and solves the preconditioned system \f$My = x\f$.
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
 *      // Create matrix descriptor for L'
 *      rocsparse_mat_descr descr_Lt;
 *      rocsparse_create_mat_descr(&descr_Lt);
 *      rocsparse_set_mat_fill_mode(descr_Lt, rocsparse_fill_mode_upper);
 *      rocsparse_set_mat_diag_type(descr_Lt, rocsparse_diag_type_non_unit);
 *
 *      // Create matrix info structure
 *      rocsparse_mat_info info;
 *      rocsparse_create_mat_info(&info);
 *
 *      // Obtain required buffer size
 *      size_t buffer_size_M;
 *      size_t buffer_size_L;
 *      size_t buffer_size_Lt;
 *      rocsparse_dbsric0_buffer_size(handle,
 *                                     rocsparse_direction_row,
 *                                     mb,
 *                                     nnzb,
 *                                     descr_M,
 *                                     bsr_val,
 *                                     bsr_row_ptr,
 *                                     bsr_col_ind,
 *                                     block_dim,
 *                                     info,
 *                                     &buffer_size_M);
 *      rocsparse_dbsrsv_buffer_size(handle,
 *                                   rocsparse_direction_row,
 *                                   rocsparse_operation_none,
 *                                   mb,
 *                                   nnzb,
 *                                   descr_L,
 *                                   bsr_val,
 *                                   bsr_row_ptr,
 *                                   bsr_col_ind,
 *                                   block_dim,
 *                                   info,
 *                                   &buffer_size_L);
 *      rocsparse_dbsrsv_buffer_size(handle,
 *                                   rocsparse_direction_row,
 *                                   rocsparse_operation_transpose,
 *                                   mb,
 *                                   nnzb,
 *                                   descr_Lt,
 *                                   bsr_val,
 *                                   bsr_row_ptr,
 *                                   bsr_col_ind,
 *                                   block_dim,
 *                                   info,
 *                                   &buffer_size_Lt);
 *
 *      size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_Lt));
 *
 *      // Allocate temporary buffer
 *      void* temp_buffer;
 *      hipMalloc(&temp_buffer, buffer_size);
 *
 *      // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
 *      // computation performance
 *      rocsparse_dbsric0_analysis(handle,
 *                                  rocsparse_direction_row,
 *                                  mb,
 *                                  nnzb,
 *                                  descr_M,
 *                                  bsr_val,
 *                                  bsr_row_ptr,
 *                                  bsr_col_ind,
 *                                  block_dim,
 *                                  info,
 *                                  rocsparse_analysis_policy_reuse,
 *                                  rocsparse_solve_policy_auto,
 *                                  temp_buffer);
 *      rocsparse_dbsrsv_analysis(handle,
 *                                rocsparse_direction_row,
 *                                rocsparse_operation_none,
 *                                mb,
 *                                nnzb,
 *                                descr_L,
 *                                bsr_val,
 *                                bsr_row_ptr,
 *                                bsr_col_ind,
 *                                block_dim,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *      rocsparse_dbsrsv_analysis(handle,
 *                                rocsparse_direction_row,
 *                                rocsparse_operation_transpose,
 *                                mb,
 *                                nnzb,
 *                                descr_Lt,
 *                                bsr_val,
 *                                bsr_row_ptr,
 *                                bsr_col_ind,
 *                                block_dim,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *
 *      // Check for zero pivot
 *      rocsparse_int position;
 *      if(rocsparse_status_zero_pivot == rocsparse_bsric0_zero_pivot(handle,
 *                                                                    info,
 *                                                                    &position))
 *      {
 *          printf("A has structural zero at A(%d,%d)\n", position, position);
 *      }
 *
 *      // Compute incomplete Cholesky factorization M = LL'
 *      rocsparse_dbsric0(handle,
 *                         rocsparse_direction_row,
 *                         mb,
 *                         nnzb,
 *                         descr_M,
 *                         bsr_val,
 *                         bsr_row_ptr,
 *                         bsr_col_ind,
 *                         block_dim,
 *                         info,
 *                         rocsparse_solve_policy_auto,
 *                         temp_buffer);
 *
 *      // Check for zero pivot
 *      if(rocsparse_status_zero_pivot == rocsparse_bsric0_zero_pivot(handle,
 *                                                                     info,
 *                                                                     &position))
 *      {
 *          printf("L has structural and/or numerical zero at L(%d,%d)\n",
 *                 position,
 *                 position);
 *      }
 *
 *      // Solve Lz = x
 *      rocsparse_dbsrsv_solve(handle,
 *                             rocsparse_direction_row,
 *                             rocsparse_operation_none,
 *                             mb,
 *                             nnzb,
 *                             &alpha,
 *                             descr_L,
 *                             bsr_val,
 *                             bsr_row_ptr,
 *                             bsr_col_ind,
 *                             block_dim,
 *                             info,
 *                             x,
 *                             z,
 *                             rocsparse_solve_policy_auto,
 *                             temp_buffer);
 *
 *      // Solve L'y = z
 *      rocsparse_dbsrsv_solve(handle,
 *                             rocsparse_direction_row,
 *                             rocsparse_operation_transpose,
 *                             mb,
 *                             nnzb,
 *                             &alpha,
 *                             descr_Lt,
 *                             bsr_val,
 *                             bsr_row_ptr,
 *                             bsr_col_ind,
 *                             block_dim,
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
 *      rocsparse_destroy_mat_descr(descr_Lt);
 *      rocsparse_destroy_handle(handle);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsric0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   float*                    bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsric0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   double*                   bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsric0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_float_complex*  bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsric0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_double_complex* bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
 *  structural or numerical zero has been found during rocsparse_sbsrilu0(),
 *  rocsparse_dbsrilu0(), rocsparse_cbsrilu0() or rocsparse_zbsrilu0() computation.
 *  The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same
 *  index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position \f$=j\f$ means that either the diagonal block
 *  \f$A_{j,j}\f$ is missing (structural zero) or the diagonal block \f$A_{j,j}\f$ is not
 *  invertible (numerical zero).
 *
 *  \note \p rocsparse_bsrilu0_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_bsrilu0_zero_pivot(rocsparse_handle   handle,
                                              rocsparse_mat_info info,
                                              rocsparse_int*     position);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_numeric_boost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  info            structure that holds the information collected during the analysis step.
 *  @param[in]
 *  enable_boost    enable/disable numeric boost.
 *  @param[in]
 *  boost_tol       tolerance to determine whether a numerical value is replaced or not.
 *  @param[in]
 *  boost_val       boost value to replace a numerical value.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info, \p tol or \p boost_val pointer
 *              is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const float*       boost_tol,
                                                  const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const double*      boost_tol,
                                                  const double*      boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0_numeric_boost(rocsparse_handle               handle,
                                                  rocsparse_mat_info             info,
                                                  int                            enable_boost,
                                                  const float*                   boost_tol,
                                                  const rocsparse_float_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0_numeric_boost(rocsparse_handle                handle,
                                                  rocsparse_mat_info              info,
                                                  int                             enable_boost,
                                                  const double*                   boost_tol,
                                                  const rocsparse_double_complex* boost_val);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_buffer_size returns the size of the temporary storage buffer
 *  that is required by rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(),
 *  rocsparse_cbsrilu0_analysis(), rocsparse_zbsrilu0_analysis(), rocsparse_sbsrilu0(),
 *  rocsparse_dbsrilu0(), rocsparse_sbsrilu0() and rocsparse_dbsrilu0(). The temporary
 *  storage buffer must be allocated by the user. The size of the temporary storage
 *  buffer is identical to the size returned by rocsparse_sbsrsv_buffer_size(),
 *  rocsparse_dbsrsv_buffer_size(), rocsparse_cbsrsv_buffer_size(), rocsparse_zbsrsv_buffer_size(),
 *  rocsparse_sbsric0_buffer_size(), rocsparse_dbsric0_buffer_size(), rocsparse_cbsric0_buffer_size()
 *  and rocsparse_zbsric0_buffer_size() if the matrix sparsity pattern is identical. The user
 *  allocated buffer can thus be shared between subsequent calls to those functions.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  dir         direction that specifies whether to count nonzero elements by \ref rocsparse_direction_row or by
 *              \ref rocsparse_direction_row.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
 *  @param[out]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  buffer_size number of bytes of the temporary storage buffer required by
 *              rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(),
 *              rocsparse_cbsrilu0_analysis(), rocsparse_zbsrilu0_analysis(),
 *              rocsparse_sbsrilu0(), rocsparse_dbsrilu0(), rocsparse_cbsrilu0()
 *              and rocsparse_zbsrilu0().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
 *              \p bsr_col_ind, \p info or \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                const rocsparse_mat_descr descr,
                                                const float*              bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0_buffer_size(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_int             mb,
                                                rocsparse_int             nnzb,
                                                const rocsparse_mat_descr descr,
                                                const double*             bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0_buffer_size(rocsparse_handle               handle,
                                                rocsparse_direction            dir,
                                                rocsparse_int                  mb,
                                                rocsparse_int                  nnzb,
                                                const rocsparse_mat_descr      descr,
                                                const rocsparse_float_complex* bsr_val,
                                                const rocsparse_int*           bsr_row_ptr,
                                                const rocsparse_int*           bsr_col_ind,
                                                rocsparse_int                  block_dim,
                                                rocsparse_mat_info             info,
                                                size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0_buffer_size(rocsparse_handle                handle,
                                                rocsparse_direction             dir,
                                                rocsparse_int                   mb,
                                                rocsparse_int                   nnzb,
                                                const rocsparse_mat_descr       descr,
                                                const rocsparse_double_complex* bsr_val,
                                                const rocsparse_int*            bsr_row_ptr,
                                                const rocsparse_int*            bsr_col_ind,
                                                rocsparse_int                   block_dim,
                                                rocsparse_mat_info              info,
                                                size_t*                         buffer_size);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_analysis performs the analysis step for rocsparse_sbsrilu0()
 *  rocsparse_dbsrilu0(), rocsparse_cbsrilu0(), and rocsparse_zbsrilu0(). It is expected
 *  that this function will be executed only once for a given matrix. The analysis meta
 *  data can be cleared by rocsparse_bsrilu0_clear().
 *
 *  \p rocsparse_bsrilu0_analysis can share its meta data with
 *  rocsparse_sbsric0_analysis(), rocsparse_dbsric0_analysis(),
 *  rocsparse_cbsric0_analysis(), rocsparse_zbsric0_analysis(),
 *  rocsparse_sbsrsv_analysis(), rocsparse_dbsrsv_analysis(),
 *  rocsparse_cbsrsv_analysis(), rocsparse_zbsrsv_analysis(),
 *  rocsparse_sbsrsm_analysis(), rocsparse_dbsrsm_analysis(),
 *  rocsparse_cbsrsm_analysis() and rocsparse_zbsrsm_analysis(). Selecting
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
 *  dir         direction that specified whether to count nonzero elements by
 *              \ref rocsparse_direction_row or by \ref rocsparse_direction_row.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[in]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
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
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr,
 *              \p bsr_col_ind, \p info or \p temp_buffer pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_int             mb,
                                             rocsparse_int             nnzb,
                                             const rocsparse_mat_descr descr,
                                             const float*              bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_int             mb,
                                             rocsparse_int             nnzb,
                                             const rocsparse_mat_descr descr,
                                             const double*             bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0_analysis(rocsparse_handle               handle,
                                             rocsparse_direction            dir,
                                             rocsparse_int                  mb,
                                             rocsparse_int                  nnzb,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* bsr_val,
                                             const rocsparse_int*           bsr_row_ptr,
                                             const rocsparse_int*           bsr_col_ind,
                                             rocsparse_int                  block_dim,
                                             rocsparse_mat_info             info,
                                             rocsparse_analysis_policy      analysis,
                                             rocsparse_solve_policy         solve,
                                             void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0_analysis(rocsparse_handle                handle,
                                             rocsparse_direction             dir,
                                             rocsparse_int                   mb,
                                             rocsparse_int                   nnzb,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* bsr_val,
                                             const rocsparse_int*            bsr_row_ptr,
                                             const rocsparse_int*            bsr_col_ind,
                                             rocsparse_int                   block_dim,
                                             rocsparse_mat_info              info,
                                             rocsparse_analysis_policy       analysis,
                                             rocsparse_solve_policy          solve,
                                             void*                           temp_buffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0_clear deallocates all memory that was allocated by
 *  rocsparse_sbsrilu0_analysis(), rocsparse_dbsrilu0_analysis(), rocsparse_cbsrilu0_analysis()
 *  or rocsparse_zbsrilu0_analysis(). This is especially useful, if memory is an issue and
 *  the analysis data is not required for further computation.
 *
 *  \note
 *  Calling \p rocsparse_bsrilu0_clear is optional. All allocated resources will be
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
rocsparse_status rocsparse_bsrilu0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_bsrilu0 computes the incomplete LU factorization with 0 fill-ins and no
 *  pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LU
 *  \f]
 *
 *  \p rocsparse_bsrilu0 requires a user allocated temporary buffer. Its size is returned
 *  by rocsparse_sbsrilu0_buffer_size(), rocsparse_dbsrilu0_buffer_size(),
 *  rocsparse_cbsrilu0_buffer_size() or rocsparse_zbsrilu0_buffer_size(). Furthermore,
 *  analysis meta data is required. It can be obtained by rocsparse_sbsrilu0_analysis(),
 *  rocsparse_dbsrilu0_analysis(), rocsparse_cbsrilu0_analysis() or
 *  rocsparse_zbsrilu0_analysis(). \p rocsparse_bsrilu0 reports the first zero pivot
 *  (either numerical or structural zero). The zero pivot status can be obtained by
 *  calling rocsparse_bsrilu0_zero_pivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  @param[in]
 *  handle      handle to the rocsparse library context queue.
 *  @param[in]
 *  dir         direction that specified whether to count nonzero elements by
 *              \ref rocsparse_direction_row or by \ref rocsparse_direction_row.
 *  @param[in]
 *  mb          number of block rows in the sparse BSR matrix.
 *  @param[in]
 *  nnzb        number of non-zero block entries of the sparse BSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse BSR matrix.
 *  @param[inout]
 *  bsr_val     array of length \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
 *  @param[in]
 *  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
 *              sparse BSR matrix.
 *  @param[in]
 *  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
 *  @param[in]
 *  block_dim   the block dimension of the BSR matrix. Between 1 and m where \p m=mb*block_dim.
 *  @param[in]
 *  info        structure that holds the information collected during the analysis step.
 *  @param[in]
 *  policy      \ref rocsparse_solve_policy_auto.
 *  @param[in]
 *  temp_buffer temporary storage buffer allocated by the user.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p mb, \p nnzb, or \p block_dim is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p descr, \p bsr_val, \p bsr_row_ptr
 *              or \p bsr_col_ind pointer is invalid.
 *  \retval     rocsparse_status_arch_mismatch the device is not supported.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *  \retval     rocsparse_status_not_implemented
 *              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
 *
 *  \par Example
 *  Consider the sparse \f$m \times m\f$ matrix \f$A\f$, stored in BSR
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
 *      rocsparse_dbsrilu0_buffer_size(handle,
 *                                     rocsparse_direction_row,
 *                                     mb,
 *                                     nnzb,
 *                                     descr_M,
 *                                     bsr_val,
 *                                     bsr_row_ptr,
 *                                     bsr_col_ind,
 *                                     block_dim,
 *                                     info,
 *                                     &buffer_size_M);
 *      rocsparse_dbsrsv_buffer_size(handle,
 *                                   rocsparse_direction_row,
 *                                   rocsparse_operation_none,
 *                                   mb,
 *                                   nnzb,
 *                                   descr_L,
 *                                   bsr_val,
 *                                   bsr_row_ptr,
 *                                   bsr_col_ind,
 *                                   block_dim,
 *                                   info,
 *                                   &buffer_size_L);
 *      rocsparse_dbsrsv_buffer_size(handle,
 *                                   rocsparse_direction_row,
 *                                   rocsparse_operation_transpose,
 *                                   mb,
 *                                   nnzb,
 *                                   descr_U,
 *                                   bsr_val,
 *                                   bsr_row_ptr,
 *                                   bsr_col_ind,
 *                                   block_dim,
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
 *      rocsparse_dbsrilu0_analysis(handle,
 *                                  rocsparse_direction_row,
 *                                  mb,
 *                                  nnzb,
 *                                  descr_M,
 *                                  bsr_val,
 *                                  bsr_row_ptr,
 *                                  bsr_col_ind,
 *                                  block_dim,
 *                                  info,
 *                                  rocsparse_analysis_policy_reuse,
 *                                  rocsparse_solve_policy_auto,
 *                                  temp_buffer);
 *      rocsparse_dbsrsv_analysis(handle,
 *                                rocsparse_direction_row,
 *                                rocsparse_operation_none,
 *                                mb,
 *                                nnzb,
 *                                descr_L,
 *                                bsr_val,
 *                                bsr_row_ptr,
 *                                bsr_col_ind,
 *                                block_dim,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *      rocsparse_dbsrsv_analysis(handle,
 *                                rocsparse_direction_row,
 *                                rocsparse_operation_transpose,
 *                                mb,
 *                                nnzb,
 *                                descr_U,
 *                                bsr_val,
 *                                bsr_row_ptr,
 *                                bsr_col_ind,
 *                                block_dim,
 *                                info,
 *                                rocsparse_analysis_policy_reuse,
 *                                rocsparse_solve_policy_auto,
 *                                temp_buffer);
 *
 *      // Check for zero pivot
 *      rocsparse_int position;
 *      if(rocsparse_status_zero_pivot == rocsparse_bsrilu0_zero_pivot(handle,
 *                                                                    info,
 *                                                                    &position))
 *      {
 *          printf("A has structural zero at A(%d,%d)\n", position, position);
 *      }
 *
 *      // Compute incomplete LU factorization M = LU
 *      rocsparse_dbsrilu0(handle,
 *                         rocsparse_direction_row,
 *                         mb,
 *                         nnzb,
 *                         descr_M,
 *                         bsr_val,
 *                         bsr_row_ptr,
 *                         bsr_col_ind,
 *                         block_dim,
 *                         info,
 *                         rocsparse_solve_policy_auto,
 *                         temp_buffer);
 *
 *      // Check for zero pivot
 *      if(rocsparse_status_zero_pivot == rocsparse_bsrilu0_zero_pivot(handle,
 *                                                                     info,
 *                                                                     &position))
 *      {
 *          printf("L has structural and/or numerical zero at L(%d,%d)\n",
 *                 position,
 *                 position);
 *      }
 *
 *      // Solve Lz = x
 *      rocsparse_dbsrsv_solve(handle,
 *                             rocsparse_direction_row,
 *                             rocsparse_operation_none,
 *                             mb,
 *                             nnzb,
 *                             &alpha,
 *                             descr_L,
 *                             bsr_val,
 *                             bsr_row_ptr,
 *                             bsr_col_ind,
 *                             block_dim,
 *                             info,
 *                             x,
 *                             z,
 *                             rocsparse_solve_policy_auto,
 *                             temp_buffer);
 *
 *      // Solve Uy = z
 *      rocsparse_dbsrsv_solve(handle,
 *                             rocsparse_direction_row,
 *                             rocsparse_operation_transpose,
 *                             mb,
 *                             nnzb,
 *                             &alpha,
 *                             descr_U,
 *                             bsr_val,
 *                             bsr_row_ptr,
 *                             bsr_col_ind,
 *                             block_dim,
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
rocsparse_status rocsparse_sbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    float*                    bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    double*                   bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_float_complex*  bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsrilu0(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nnzb,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    rocsparse_mat_info        info,
                                    rocsparse_solve_policy    policy,
                                    void*                     temp_buffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csric_zero_pivot returns \ref rocsparse_status_zero_pivot, if either a
*  structural or numerical zero has been found during rocsparse_scsric0() or
*  rocsparse_dcsric0() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
*  is stored in \p position, using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref rocsparse_status_success is returned instead.
*
*  \note \p rocsparse_csric0_zero_pivot is a blocking function. It might influence
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
rocsparse_status rocsparse_csric0_zero_pivot(rocsparse_handle   handle,
                                             rocsparse_mat_info info,
                                             rocsparse_int*     position);

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csric0_buffer_size returns the size of the temporary storage buffer
*  that is required by rocsparse_scsric0_analysis(), rocsparse_dcsric0_analysis(),
*  rocsparse_scsric0() and rocsparse_dcsric0(). The temporary storage buffer must
*  be allocated by the user. The size of the temporary storage buffer is identical to
*  the size returned by rocsparse_scsrsv_buffer_size(), rocsparse_dcsrsv_buffer_size(),
*  rocsparse_scsrilu0_buffer_size() and rocsparse_dcsrilu0_buffer_size() if the matrix
*  sparsity pattern is identical. The user allocated buffer can thus be shared between
*  subsequent calls to those functions.
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
*              rocsparse_scsric0_analysis(), rocsparse_dcsric0_analysis(),
*              rocsparse_scsric0() and rocsparse_dcsric0().
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
rocsparse_status rocsparse_scsric0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const float*              csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsric0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const double*             csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsric0_buffer_size(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  nnz,
                                               const rocsparse_mat_descr      descr,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               rocsparse_mat_info             info,
                                               size_t*                        buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsric0_buffer_size(rocsparse_handle                handle,
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
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csric0_analysis performs the analysis step for rocsparse_scsric0()
*  and rocsparse_dcsric0(). It is expected that this function will be executed only
*  once for a given matrix and particular operation type. The analysis meta data can be
*  cleared by rocsparse_csric0_clear().
*
*  \p rocsparse_csric0_analysis can share its meta data with
*  rocsparse_scsrilu0_analysis(), rocsparse_dcsrilu0_analysis(),
*  rocsparse_ccsrilu0_analysis(), rocsparse_zcsrilu0_analysis(),
*  rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(),
*  rocsparse_ccsrsv_analysis(), rocsparse_zcsrsv_analysis(),
*  rocsparse_scsrsm_analysis(), rocsparse_dcsrsm_analysis(),
*  rocsparse_scsrsm_analysis() and rocsparse_dcsrsm_analysis(). Selecting
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
rocsparse_status rocsparse_scsric0_analysis(rocsparse_handle          handle,
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
rocsparse_status rocsparse_dcsric0_analysis(rocsparse_handle          handle,
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
rocsparse_status rocsparse_ccsric0_analysis(rocsparse_handle               handle,
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
rocsparse_status rocsparse_zcsric0_analysis(rocsparse_handle                handle,
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
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csric0_clear deallocates all memory that was allocated by
*  rocsparse_scsric0_analysis() or rocsparse_dcsric0_analysis(). This is especially
*  useful, if memory is an issue and the analysis data is not required for further
*  computation.
*
*  \note
*  Calling \p rocsparse_csric0_clear is optional. All allocated resources will be
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
rocsparse_status rocsparse_csric0_clear(rocsparse_handle handle, rocsparse_mat_info info);

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p rocsparse_csric0 computes the incomplete Cholesky factorization with 0 fill-ins
*  and no pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LL^T
*  \f]
*
*  \p rocsparse_csric0 requires a user allocated temporary buffer. Its size is returned
*  by rocsparse_scsric0_buffer_size() or rocsparse_dcsric0_buffer_size(). Furthermore,
*  analysis meta data is required. It can be obtained by rocsparse_scsric0_analysis()
*  or rocsparse_dcsric0_analysis(). \p rocsparse_csric0 reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be obtained by
*  calling rocsparse_csric0_zero_pivot().
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
*  storage format. The following example computes the incomplete Cholesky factorization
*  \f$M \approx LL^T\f$ and solves the preconditioned system \f$My = x\f$.
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
*      // Create matrix descriptor for L'
*      rocsparse_mat_descr descr_Lt;
*      rocsparse_create_mat_descr(&descr_Lt);
*      rocsparse_set_mat_fill_mode(descr_Lt, rocsparse_fill_mode_upper);
*      rocsparse_set_mat_diag_type(descr_Lt, rocsparse_diag_type_non_unit);
*
*      // Create matrix info structure
*      rocsparse_mat_info info;
*      rocsparse_create_mat_info(&info);
*
*      // Obtain required buffer size
*      size_t buffer_size_M;
*      size_t buffer_size_L;
*      size_t buffer_size_Lt;
*      rocsparse_dcsric0_buffer_size(handle,
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
*                                   rocsparse_operation_transpose,
*                                   m,
*                                   nnz,
*                                   descr_Lt,
*                                   csr_val,
*                                   csr_row_ptr,
*                                   csr_col_ind,
*                                   info,
*                                   &buffer_size_Lt);
*
*      size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_Lt));
*
*      // Allocate temporary buffer
*      void* temp_buffer;
*      hipMalloc(&temp_buffer, buffer_size);
*
*      // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
*      // computation performance
*      rocsparse_dcsric0_analysis(handle,
*                                 m,
*                                 nnz,
*                                 descr_M,
*                                 csr_val,
*                                 csr_row_ptr,
*                                 csr_col_ind,
*                                 info,
*                                 rocsparse_analysis_policy_reuse,
*                                 rocsparse_solve_policy_auto,
*                                 temp_buffer);
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
*                                rocsparse_operation_transpose,
*                                m,
*                                nnz,
*                                descr_Lt,
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
*      if(rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle,
*                                                                    info,
*                                                                    &position))
*      {
*          printf("A has structural zero at A(%d,%d)\n", position, position);
*      }
*
*      // Compute incomplete Cholesky factorization M = LL'
*      rocsparse_dcsric0(handle,
*                        m,
*                        nnz,
*                        descr_M,
*                        csr_val,
*                        csr_row_ptr,
*                        csr_col_ind,
*                        info,
*                        rocsparse_solve_policy_auto,
*                        temp_buffer);
*
*      // Check for zero pivot
*      if(rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle,
*                                                                    info,
*                                                                    &position))
*      {
*          printf("L has structural and/or numerical zero at L(%d,%d)\n",
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
*      // Solve L'y = z
*      rocsparse_dcsrsv_solve(handle,
*                             rocsparse_operation_transpose,
*                             m,
*                             nnz,
*                             &alpha,
*                             descr_Lt,
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
*      rocsparse_destroy_mat_descr(descr_Lt);
*      rocsparse_destroy_handle(handle);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsric0(rocsparse_handle          handle,
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
rocsparse_status rocsparse_dcsric0(rocsparse_handle          handle,
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
rocsparse_status rocsparse_ccsric0(rocsparse_handle          handle,
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
rocsparse_status rocsparse_zcsric0(rocsparse_handle          handle,
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
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage
 *  format
 *
 *  \details
 *  \p rocsparse_csrilu0_numeric_boost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 *
 *  @param[in]
 *  handle          handle to the rocsparse library context queue.
 *  @param[in]
 *  info            structure that holds the information collected during the analysis step.
 *  @param[in]
 *  enable_boost    enable/disable numeric boost.
 *  @param[in]
 *  boost_tol       tolerance to determine whether a numerical value is replaced or not.
 *  @param[in]
 *  boost_val       boost value to replace a numerical value.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p info, \p tol or \p boost_val pointer
 *              is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const float*       boost_tol,
                                                  const float*       boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsrilu0_numeric_boost(rocsparse_handle   handle,
                                                  rocsparse_mat_info info,
                                                  int                enable_boost,
                                                  const double*      boost_tol,
                                                  const double*      boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsrilu0_numeric_boost(rocsparse_handle               handle,
                                                  rocsparse_mat_info             info,
                                                  int                            enable_boost,
                                                  const float*                   boost_tol,
                                                  const rocsparse_float_complex* boost_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsrilu0_numeric_boost(rocsparse_handle                handle,
                                                  rocsparse_mat_info              info,
                                                  int                             enable_boost,
                                                  const double*                   boost_tol,
                                                  const rocsparse_double_complex* boost_val);
/**@}*/

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
*  rocsparse_scsric0_analysis(), rocsparse_dcsric0_analysis(),
*  rocsparse_ccsric0_analysis(), rocsparse_zcsric0_analysis(),
*  rocsparse_scsrsv_analysis(), rocsparse_dcsrsv_analysis(),
*  rocsparse_ccsrsv_analysis(), rocsparse_zcsrsv_analysis(),
*  rocsparse_scsrsm_analysis(), rocsparse_dcsrsm_analysis(),
*  rocsparse_scsrsm_analysis() and rocsparse_dcsrsm_analysis(). Selecting
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
*  \brief
*  This function computes the number of nonzero elements per row or column and the total number of nonzero elements in a dense matrix.
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  dir        direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by \ref rocsparse_direction_row.
*
*  @param[in]
*  m           number of rows of the dense matrix \p A.
*
*  @param[in]
*  n           number of columns of the dense matrix \p A.
*
*  @param[in]
*  descr      the descriptor of the dense matrix \p A.
*
*  @param[in]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[in]
*  ld         leading dimension of dense array \p A.
*
*  @param[out]
*  nnz_per_row_columns
*              array of size \p m or \p n containing the number of nonzero elements per row or column, respectively.
*  @param[out]
*  nnz_total_dev_host_ptr
*              total number of nonzero elements in device or host memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p nnz_per_row_columns or \p nnz_total_dev_host_ptr
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_snnz(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                const rocsparse_mat_descr descr,
                                const float*              A,
                                rocsparse_int             ld,
                                rocsparse_int*            nnz_per_row_columns,
                                rocsparse_int*            nnz_total_dev_host_ptr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnnz(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                const rocsparse_mat_descr descr,
                                const double*             A,
                                rocsparse_int             ld,
                                rocsparse_int*            nnz_per_row_columns,
                                rocsparse_int*            nnz_total_dev_host_ptr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cnnz(rocsparse_handle               handle,
                                rocsparse_direction            dir,
                                rocsparse_int                  m,
                                rocsparse_int                  n,
                                const rocsparse_mat_descr      descr,
                                const rocsparse_float_complex* A,
                                rocsparse_int                  ld,
                                rocsparse_int*                 nnz_per_row_columns,
                                rocsparse_int*                 nnz_total_dev_host_ptr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_znnz(rocsparse_handle                handle,
                                rocsparse_direction             dir,
                                rocsparse_int                   m,
                                rocsparse_int                   n,
                                const rocsparse_mat_descr       descr,
                                const rocsparse_double_complex* A,
                                rocsparse_int                   ld,
                                rocsparse_int*                  nnz_per_row_columns,
                                rocsparse_int*                  nnz_total_dev_host_ptr);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix A in dense format into a sparse matrix in CSR format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_row, which can be pre-computed with rocsparse_xnnz().
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*  \details
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
*  descr      the descriptor of the dense matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[in]
*  ld         leading dimension of dense array \p A.
*
*  @param[in]
*  nnz_per_rows   array of size \p n containing the number of non-zero elements per row.
*
*  @param[out]
*  csr_val
*              array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[out]
*  csr_row_ptr
*              integer array of m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[out]
*  csr_col_ind
*              integer array of nnz ( = \p csr_row_ptr[m] - csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p nnz_per_rows or \p csr_val \p csr_row_ptr or \p csr_col_ind
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sdense2csr(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      float*                    csr_val,
                                      rocsparse_int*            csr_row_ptr,
                                      rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ddense2csr(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      double*                   csr_val,
                                      rocsparse_int*            csr_row_ptr,
                                      rocsparse_int*            csr_col_ind);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdense2csr(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* A,
                                      rocsparse_int                  ld,
                                      const rocsparse_int*           nnz_per_rows,
                                      rocsparse_float_complex*       csr_val,
                                      rocsparse_int*                 csr_row_ptr,
                                      rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdense2csr(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* A,
                                      rocsparse_int                   ld,
                                      const rocsparse_int*            nnz_per_rows,
                                      rocsparse_double_complex*       csr_val,
                                      rocsparse_int*                  csr_row_ptr,
                                      rocsparse_int*                  csr_col_ind);
/**@}*/

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
*  by rocsparse_xprune_dense2csr_buffer_size(). The routine rocsparse_xprune_dense2csr() is executed asynchronously with
*  respect to the host and may return control to the application on the host before the entire result is ready.
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
*  Once we have this threshold we prune values in the dense matrix \p A as in \p rocsparse_prune_dense2csr. It is executed
*  asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
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
*  Once we have this threshold we prune values in the dense matrix \p A as in \p rocsparse_prune_dense2csr. The routine does
*  support asynchronous execution if the pointer mode is set to device.
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
*  Once we have this threshold we prune values in the dense matrix \p A as in \p rocsparse_prune_dense2csr. It is executed
*  asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
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

/*! \ingroup conv_module
*  \brief
*
*  This function converts the matrix A in dense format into a sparse matrix in CSC format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_columns, which can be pre-computed with rocsparse_xnnz().
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*
*  \details
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
*  descr      the descriptor of the dense matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[in]
*  ld         leading dimension of dense array \p A.
*
*  @param[in]
*  nnz_per_columns   array of size \p n containing the number of non-zero elements per column.
*
*  @param[out]
*  csc_val
*              array of nnz ( = \p csc_col_ptr[m] - \p csc_col_ptr[0] ) nonzero elements of matrix \p A.
*  @param[out]
*  csc_col_ptr
*              integer array of m+1 elements that contains the start of every column and the end of the last column plus one.
*  @param[out]
*  csc_row_ind
*              integer array of nnz ( = \p csc_col_ptr[m] - csc_col_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p nnz_per_columns or \p csc_val \p csc_col_ptr or \p csc_row_ind
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sdense2csc(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_columns,
                                      float*                    csc_val,
                                      rocsparse_int*            csc_col_ptr,
                                      rocsparse_int*            csc_row_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ddense2csc(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_columns,
                                      double*                   csc_val,
                                      rocsparse_int*            csc_col_ptr,
                                      rocsparse_int*            csc_row_ind);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdense2csc(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* A,
                                      rocsparse_int                  ld,
                                      const rocsparse_int*           nnz_per_columns,
                                      rocsparse_float_complex*       csc_val,
                                      rocsparse_int*                 csc_col_ptr,
                                      rocsparse_int*                 csc_row_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdense2csc(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* A,
                                      rocsparse_int                   ld,
                                      const rocsparse_int*            nnz_per_columns,
                                      rocsparse_double_complex*       csc_val,
                                      rocsparse_int*                  csc_col_ptr,
                                      rocsparse_int*                  csc_row_ind);
/**@}*/

/*! \ingroup conv_module
*  \brief
*
*  This function converts the matrix A in dense format into a sparse matrix in COO format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_rows, which can be pre-computed with rocsparse_xnnz().
*
*  \details
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
*  descr      the descriptor of the dense matrix \p A, the supported matrix type is rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[in]
*  ld         leading dimension of dense array \p A.
*
*  @param[in]
*  nnz_per_rows   array of size \p n containing the number of non-zero elements per row.
*
*  @param[out]
*  coo_val
*              array of nnz nonzero elements of matrix \p A.
*  @param[out]
*  coo_row_ind
*              integer array of nnz row indices of the non-zero elements of matrix \p A.
*  @param[out]
*  coo_col_ind integer array of nnz column indices of the non-zero elements of matrix \p A.
*
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p nnz_per_rows or \p coo_val \p coo_col_ind or \p coo_row_ind
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sdense2coo(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      float*                    coo_val,
                                      rocsparse_int*            coo_row_ind,
                                      rocsparse_int*            coo_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ddense2coo(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             A,
                                      rocsparse_int             ld,
                                      const rocsparse_int*      nnz_per_rows,
                                      double*                   coo_val,
                                      rocsparse_int*            coo_row_ind,
                                      rocsparse_int*            coo_col_ind);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cdense2coo(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* A,
                                      rocsparse_int                  ld,
                                      const rocsparse_int*           nnz_per_rows,
                                      rocsparse_float_complex*       coo_val,
                                      rocsparse_int*                 coo_row_ind,
                                      rocsparse_int*                 coo_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zdense2coo(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* A,
                                      rocsparse_int                   ld,
                                      const rocsparse_int*            nnz_per_rows,
                                      rocsparse_double_complex*       coo_val,
                                      rocsparse_int*                  coo_row_ind,
                                      rocsparse_int*                  coo_col_ind);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in CSR format into a dense matrix.
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*  \details
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
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  csr_val     array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csr_row_ptr integer array of m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csr_col_ind integer array of nnz ( = \p csr_row_ptr[m] - csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[out]
*  ld          leading dimension of dense array \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p csr_val \p csr_row_ptr or \p csr_col_ind
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              csr_val,
                                      const rocsparse_int*      csr_row_ptr,
                                      const rocsparse_int*      csr_col_ind,
                                      float*                    A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             csr_val,
                                      const rocsparse_int*      csr_row_ptr,
                                      const rocsparse_int*      csr_col_ind,
                                      double*                   A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2dense(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* csr_val,
                                      const rocsparse_int*           csr_row_ptr,
                                      const rocsparse_int*           csr_col_ind,
                                      rocsparse_float_complex*       A,
                                      rocsparse_int                  ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2dense(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* csr_val,
                                      const rocsparse_int*            csr_row_ptr,
                                      const rocsparse_int*            csr_col_ind,
                                      rocsparse_double_complex*       A,
                                      rocsparse_int                   ld);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in CSC format into a dense matrix.
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*  \details
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
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  csc_val     array of nnz ( = \p csc_col_ptr[m] - \p csc_col_ptr[0] ) nonzero elements of matrix \p A.
*  @param[in]
*  csc_col_ptr integer array of m+1 elements that contains the start of every row and the end of the last row plus one.
*  @param[in]
*  csc_row_ind integer array of nnz ( = \p csc_col_ptr[m] - csc_col_ptr[0] ) column indices of the non-zero elements of matrix \p A.
*
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[out]
*  ld          leading dimension of dense array \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p csc_val \p csc_col_ptr or \p csc_row_ind
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsc2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const float*              csc_val,
                                      const rocsparse_int*      csc_col_ptr,
                                      const rocsparse_int*      csc_row_ind,
                                      float*                    A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsc2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr descr,
                                      const double*             csc_val,
                                      const rocsparse_int*      csc_col_ptr,
                                      const rocsparse_int*      csc_row_ind,
                                      double*                   A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsc2dense(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* csc_val,
                                      const rocsparse_int*           csc_col_ptr,
                                      const rocsparse_int*           csc_row_ind,
                                      rocsparse_float_complex*       A,
                                      rocsparse_int                  ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsc2dense(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* csc_val,
                                      const rocsparse_int*            csc_col_ptr,
                                      const rocsparse_int*            csc_row_ind,
                                      rocsparse_double_complex*       A,
                                      rocsparse_int                   ld);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in COO format into a dense matrix.
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*  \details
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
*  nnz         number of non-zero entries of the sparse COO matrix.
*  @param[in]
*  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref rocsparse_matrix_type_general and also any valid value of the \ref rocsparse_index_base.
*
*  @param[in]
*  coo_val     array of nnz nonzero elements of matrix \p A.
*  @param[in]
*  coo_row_ind integer array of nnz row indices of the non-zero elements of matrix \p A.
*
*  @param[in]
*  coo_col_ind integer array of nnz column indices of the non-zero elements of matrix \p A.
*  @param[out]
*  A           array of dimensions (\p ld, \p n)
*
*  @param[out]
*  ld          leading dimension of dense array \p A.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz or \p ld is invalid.
*  \retval     rocsparse_status_invalid_pointer \p A or \p coo_val \p coo_col_ind or \p coo_row_ind
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scoo2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      rocsparse_int             nnz,
                                      const rocsparse_mat_descr descr,
                                      const float*              coo_val,
                                      const rocsparse_int*      coo_row_ind,
                                      const rocsparse_int*      coo_col_ind,
                                      float*                    A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcoo2dense(rocsparse_handle          handle,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      rocsparse_int             nnz,
                                      const rocsparse_mat_descr descr,
                                      const double*             coo_val,
                                      const rocsparse_int*      coo_row_ind,
                                      const rocsparse_int*      coo_col_ind,
                                      double*                   A,
                                      rocsparse_int             ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccoo2dense(rocsparse_handle               handle,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      rocsparse_int                  nnz,
                                      const rocsparse_mat_descr      descr,
                                      const rocsparse_float_complex* coo_val,
                                      const rocsparse_int*           coo_row_ind,
                                      const rocsparse_int*           coo_col_ind,
                                      rocsparse_float_complex*       A,
                                      rocsparse_int                  ld);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcoo2dense(rocsparse_handle                handle,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      rocsparse_int                   nnz,
                                      const rocsparse_mat_descr       descr,
                                      const rocsparse_double_complex* coo_val,
                                      const rocsparse_int*            coo_row_ind,
                                      const rocsparse_int*            coo_col_ind,
                                      rocsparse_double_complex*       A,
                                      rocsparse_int                   ld);
/**@}*/

/*! \ingroup conv_module
*  Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
*  in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
*  of remaining elements in the matrix.
*
*  @param[in]
*  handle        handle to the rocsparse library context queue.
*
*  @param[in]
*  m             number of rows of the sparse CSR matrix.
*
*  @param[in]
*  descr_A       the descriptor of the sparse CSR matrix.
*
*  @param[in]
*  csr_val_A     array of \p nnz_A elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
*                uncompressed sparse CSR matrix.
*  @param[out]
*  nnz_per_row   array of length \p m containing the number of entries that will be kept per row in
*                the final compressed CSR matrix.
*  @param[out]
*  nnz_C         number of elements in the column indices and values arrays of the compressed
*                sparse CSR matrix. Can be either host or device pointer.
*  @param[in]
*  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
*                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
*                are removed in output compressed CSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n is invalid.
*  \retval     rocsparse_status_invalid_value \p tol is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_val_A or \p csr_row_ptr_A or \p nnz_per_row or \p nnz_C
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_snnz_compress(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         const rocsparse_mat_descr descr_A,
                                         const float*              csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         rocsparse_int*            nnz_per_row,
                                         rocsparse_int*            nnz_C,
                                         float                     tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnnz_compress(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         const rocsparse_mat_descr descr_A,
                                         const double*             csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         rocsparse_int*            nnz_per_row,
                                         rocsparse_int*            nnz_C,
                                         double                    tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cnnz_compress(rocsparse_handle               handle,
                                         rocsparse_int                  m,
                                         const rocsparse_mat_descr      descr_A,
                                         const rocsparse_float_complex* csr_val_A,
                                         const rocsparse_int*           csr_row_ptr_A,
                                         rocsparse_int*                 nnz_per_row,
                                         rocsparse_int*                 nnz_C,
                                         rocsparse_float_complex        tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_znnz_compress(rocsparse_handle                handle,
                                         rocsparse_int                   m,
                                         const rocsparse_mat_descr       descr_A,
                                         const rocsparse_double_complex* csr_val_A,
                                         const rocsparse_int*            csr_row_ptr_A,
                                         rocsparse_int*                  nnz_per_row,
                                         rocsparse_int*                  nnz_C,
                                         rocsparse_double_complex        tol);
/**@}*/

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
*  \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
*
*  \details
*  \p rocsparse_gebsr2gebsc_buffer_size returns the size of the temporary storage buffer
*  required by rocsparse_sgebsr2gebsc(), rocsparse_dgebsr2gebsc(), rocsparse_cgebsr2gebsc() and
*  rocsparse_zgebsr2gebsc(). The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  mb           number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb           number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb         number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb*row_block_dim*col_block_dim containing the values of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every row of the
*              sparse GEneral BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the column indices of the sparse
*              GEneral BSR matrix.
*  @param[in]
*  row_block_dim   row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  col_block_dim   col size of the blocks in the sparse general BSR matrix.

*  @param[out]
*  p_buffer_size number of bytes of the temporary storage buffer required by
*              rocsparse_sgebsr2gebsc(), rocsparse_dgebsr2gebsc(), rocsparse_cgebsr2gebsc() and
*              rocsparse_zgebsr2gebsc().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb or \p nnzb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_row_ptr, \p bsr_col_ind or
*              \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsc_buffer_size(rocsparse_handle     handle,
                                                    rocsparse_int        mb,
                                                    rocsparse_int        nb,
                                                    rocsparse_int        nnzb,
                                                    const float*         bsr_val,
                                                    const rocsparse_int* bsr_row_ptr,
                                                    const rocsparse_int* bsr_col_ind,
                                                    rocsparse_int        row_block_dim,
                                                    rocsparse_int        col_block_dim,
                                                    size_t*              p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsc_buffer_size(rocsparse_handle     handle,
                                                    rocsparse_int        mb,
                                                    rocsparse_int        nb,
                                                    rocsparse_int        nnzb,
                                                    const double*        bsr_val,
                                                    const rocsparse_int* bsr_row_ptr,
                                                    const rocsparse_int* bsr_col_ind,
                                                    rocsparse_int        row_block_dim,
                                                    rocsparse_int        col_block_dim,
                                                    size_t*              p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsc_buffer_size(rocsparse_handle               handle,
                                                    rocsparse_int                  mb,
                                                    rocsparse_int                  nb,
                                                    rocsparse_int                  nnzb,
                                                    const rocsparse_float_complex* bsr_val,
                                                    const rocsparse_int*           bsr_row_ptr,
                                                    const rocsparse_int*           bsr_col_ind,
                                                    rocsparse_int                  row_block_dim,
                                                    rocsparse_int                  col_block_dim,
                                                    size_t*                        p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsc_buffer_size(rocsparse_handle                handle,
                                                    rocsparse_int                   mb,
                                                    rocsparse_int                   nb,
                                                    rocsparse_int                   nnzb,
                                                    const rocsparse_double_complex* bsr_val,
                                                    const rocsparse_int*            bsr_row_ptr,
                                                    const rocsparse_int*            bsr_col_ind,
                                                    rocsparse_int                   row_block_dim,
                                                    rocsparse_int                   col_block_dim,
                                                    size_t*                         p_buffer_size);

/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
*
*  \details
*  \p rocsparse_gebsr2gebsc converts a GEneral BSR matrix into a GEneral BSC matrix. \p rocsparse_gebsr2gebsc
*  can also be used to convert a GEneral BSC matrix into a GEneral BSR matrix. \p copy_values decides
*  whether \p bsc_val is being filled during conversion (\ref rocsparse_action_numeric)
*  or not (\ref rocsparse_action_symbolic).
*
*  \p rocsparse_gebsr2gebsc requires extra temporary storage buffer that has to be allocated
*  by the user. Storage buffer size can be determined by rocsparse_gebsr2gebsc_buffer_size().
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
*  mb          number of rows of the sparse GEneral BSR matrix.
*  @param[in]
*  nb          number of columns of the sparse GEneral BSR matrix.
*  @param[in]
*  nnzb        number of non-zero entries of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_val     array of \p nnzb * \p row_block_dim * \p col_block_dim  elements of the sparse GEneral BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse GEneral BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz elements containing the column indices of the sparse
*              GEneral BSR matrix.
*  @param[in]
*  row_block_dim   row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  col_block_dim   col size of the blocks in the sparse general BSR matrix.
*  @param[out]
*  bsc_val     array of \p nnz elements of the sparse BSC matrix.
*  @param[out]
*  bsc_row_ind array of \p nnz elements containing the row indices of the sparse BSC
*              matrix.
*  @param[out]
*  bsc_col_ptr array of \p n+1 elements that point to the start of every column of the
*              sparse BSC matrix.
*  @param[in]
*  copy_values \ref rocsparse_action_symbolic or \ref rocsparse_action_numeric.
*  @param[in]
*  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*  @param[in]
*  temp_buffer temporary storage buffer allocated by the user, size is returned by
*              rocsparse_gebsr2gebsc_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb, \p nb or \p nnzb is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val, \p bsr_row_ptr,
*              \p bsr_col_ind, \p bsc_val, \p bsc_row_ind, \p bsc_col_ptr or
*              \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_arch_mismatch the device is not supported.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*
*  \par Example
*  This example computes the transpose of a GEneral BSR matrix.
*  \code{.c}
*      //     1 2 0 3
*      // A = 0 4 5 0
*      //     6 0 0 7
*      //     1 2 3 4
*
*      rocsparse_int mb_A   = 2;
*      rocsparse_int row_block_dim = 2;
*      rocsparse_int col_block_dim = 2;
*      rocsparse_int nb_A   = 2;
*      rocsparse_int nnzb_A = 4;
*
*      bsr_row_ptr_A[mb_A+1] = {0, 2, 4};               // device memory
*      bsr_col_ind_A[nnzb_A] = {0, 1, 0, 1}; // device memory
*      bsr_val_A[nnzb_A]     = {1, 0, 2, 4, 0, 5, 3, 0, 6, 1, 0, 2, 0, 3, 7, 4}; // device memory
*
*      // Allocate memory for transposed BSR matrix
*      rocsparse_int mb_T   = nb_A;
*      rocsparse_int nb_T   = mb_A;
*      rocsparse_int nnzb_T = nnzb_A;
*
*      rocsparse_int* bsr_row_ptr_T;
*      rocsparse_int* bsr_col_ind_T;
*      float* bsr_val_T;
*
*      hipMalloc((void**)&bsr_row_ptr_T, sizeof(rocsparse_int) * (mb_T + 1));
*      hipMalloc((void**)&bsr_col_ind_T, sizeof(rocsparse_int) * nnzb_T);
*      hipMalloc((void**)&bsr_val_T, sizeof(float) * nnzb_T);
*
*      // Obtain the temporary buffer size
*      size_t buffer_size;
*      rocsparse_gebsr2gebsc_buffer_size(handle,
*                                    mb_A,
*                                    nb_A,
*                                    nnzb_A,
*                                    bsr_row_ptr_A,
*                                    bsr_col_ind_A,
*                                    rocsparse_action_numeric,
*                                    &buffer_size);
*
*      // Allocate temporary buffer
*      void* temp_buffer;
*      hipMalloc(&temp_buffer, buffer_size);
*
*      rocsparse_sgebsr2gebsc(handle,
*                         mb_A,
*                         nb_A,
*                         nnzb_A,
*                         bsr_val_A,
*                         bsr_row_ptr_A,
*                         bsr_col_ind_A,
*                         row_block_dim,
*                         col_block_dim,
*                         bsr_val_T,
*                         bsr_col_ind_T,
*                         bsr_row_ptr_T,
*                         rocsparse_action_numeric,
*                         rocsparse_index_base_zero,
*                         temp_buffer);
*  \endcode
*/
/**@{*/

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2gebsc(rocsparse_handle     handle,
                                        rocsparse_int        mb,
                                        rocsparse_int        nb,
                                        rocsparse_int        nnzb,
                                        const float*         bsr_val,
                                        const rocsparse_int* bsr_row_ptr,
                                        const rocsparse_int* bsr_col_ind,
                                        rocsparse_int        row_block_dim,
                                        rocsparse_int        col_block_dim,
                                        float*               bsc_val,
                                        rocsparse_int*       bsc_row_ind,
                                        rocsparse_int*       bsc_col_ptr,
                                        rocsparse_action     copy_values,
                                        rocsparse_index_base idx_base,
                                        void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2gebsc(rocsparse_handle     handle,
                                        rocsparse_int        mb,
                                        rocsparse_int        nb,
                                        rocsparse_int        nnzb,
                                        const double*        bsr_val,
                                        const rocsparse_int* bsr_row_ptr,
                                        const rocsparse_int* bsr_col_ind,
                                        rocsparse_int        row_block_dim,
                                        rocsparse_int        col_block_dim,
                                        double*              bsc_val,
                                        rocsparse_int*       bsc_row_ind,
                                        rocsparse_int*       bsc_col_ptr,
                                        rocsparse_action     copy_values,
                                        rocsparse_index_base idx_base,
                                        void*                temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2gebsc(rocsparse_handle               handle,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nb,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_float_complex* bsr_val,
                                        const rocsparse_int*           bsr_row_ptr,
                                        const rocsparse_int*           bsr_col_ind,
                                        rocsparse_int                  row_block_dim,
                                        rocsparse_int                  col_block_dim,
                                        rocsparse_float_complex*       bsc_val,
                                        rocsparse_int*                 bsc_row_ind,
                                        rocsparse_int*                 bsc_col_ptr,
                                        rocsparse_action               copy_values,
                                        rocsparse_index_base           idx_base,
                                        void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2gebsc(rocsparse_handle                handle,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nb,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_double_complex* bsr_val,
                                        const rocsparse_int*            bsr_row_ptr,
                                        const rocsparse_int*            bsr_col_ind,
                                        rocsparse_int                   row_block_dim,
                                        rocsparse_int                   col_block_dim,
                                        rocsparse_double_complex*       bsc_val,
                                        rocsparse_int*                  bsc_row_ind,
                                        rocsparse_int*                  bsc_col_ptr,
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
*  \f$\text{nnz}_{\text{ELL}} = m \cdot \text{ell_width}\f$. The number of ELL
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
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  BSR matrix given a sparse CSR matrix as input.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  dir         direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
*              \ref rocsparse_direction_row.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_row_ptr integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*
*  @param[in]
*  csr_col_ind integer array of the column indices for each non-zero element in the CSR matrix
*
*  @param[in]
*  block_dim   the block dimension of the BSR matrix. Between 1 and min(m, n)
*
*  @param[in]
*  bsr_descr    descriptor of the sparse BSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_row_ptr integer array containing \p mb+1 elements that point to the start of each block row of the BSR matrix
*
*  @param[out]
*  bsr_nnz     total number of nonzero elements in device or host memory.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr or \p csr_col_ind or \p bsr_row_ptr or \p bsr_nnz
*              pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2bsr_nnz(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_int             m,
                                       rocsparse_int             n,
                                       const rocsparse_mat_descr csr_descr,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_int             block_dim,
                                       const rocsparse_mat_descr bsr_descr,
                                       rocsparse_int*            bsr_row_ptr,
                                       rocsparse_int*            bsr_nnz);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse BSR matrix
*
*  \details
*  \p rocsparse_csr2bsr converts a CSR matrix into a BSR matrix. It is assumed,
*  that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
*  for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
*  the BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
*  \p csr2bsr_nnz() which also fills in \p bsr_row_ptr.
*
*  \p rocsparse_csr2bsr requires extra temporary storage that is allocated internally if \p block_dim>16
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  dir          the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val      array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr  array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csr_col_ind  array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  block_dim    size of the blocks in the sparse BSR matrix.
*  @param[in]
*  bsr_descr    descriptor of the sparse BSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_val      array of \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsr_row_ptr  array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsr_col_ind  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a CSR matrix into an BSR matrix.
*  \code{.c}
*      //     1 4 0 0 0 0
*      // A = 0 2 3 0 0 0
*      //     5 0 0 7 8 0
*      //     0 0 9 0 6 0
*
*      rocsparse_int m   = 4;
*      rocsparse_int n   = 6;
*      rocsparse_int block_dim = 2;
*      rocsparse_int nnz = 9;
*      rocsparse_int mb = (m + block_dim - 1) / block_dim;
*      rocsparse_int nb = (n + block_dim - 1) / block_dim;
*
*      csr_row_ptr[m+1]  = {0, 2, 4, 7, 9};             // device memory
*      csr_col_ind[nnz]  = {0, 1, 1, 2, 0, 3, 4, 2, 4}; // device memory
*      csr_val[nnz]      = {1, 4, 2, 3, 5, 7, 8, 9, 6}; // device memory
*
*      hipMalloc(&bsr_row_ptr, sizeof(rocsparse_int) *(mb + 1));
*      rocsparse_int nnzb;
*      rocsparse_int* nnzTotalHostPtr = &nnzb;
*      csr2bsr_nnz(handle,
*                  rocsparse_direction_row,
*                  m,
*                  n,
*                  csr_descr,
*                  csr_row_ptr,
*                  csr_col_ind,
*                  block_dim,
*                  bsr_descr,
*                  bsr_row_ptr,
*                  nnzTotalHostPtr);
*      nnzb = *nnzTotalDevHostPtr;
*      hipMalloc(&bsr_col_ind, sizeof(int)*nnzb);
*      hipMalloc(&bsr_val, sizeof(float)*(block_dim * block_dim) * nnzb);
*      scsr2bsr(handle,
*               rocsparse_direction_row,
*               m,
*               n,
*               csr_descr,
*               csr_val,
*               csr_row_ptr,
*               csr_col_ind,
*               block_dim,
*               bsr_descr,
*               bsr_val,
*               bsr_row_ptr,
*               bsr_col_ind);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2bsr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr csr_descr,
                                    const float*              csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr bsr_descr,
                                    float*                    bsr_val,
                                    rocsparse_int*            bsr_row_ptr,
                                    rocsparse_int*            bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2bsr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             m,
                                    rocsparse_int             n,
                                    const rocsparse_mat_descr csr_descr,
                                    const double*             csr_val,
                                    const rocsparse_int*      csr_row_ptr,
                                    const rocsparse_int*      csr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr bsr_descr,
                                    double*                   bsr_val,
                                    rocsparse_int*            bsr_row_ptr,
                                    rocsparse_int*            bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2bsr(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_int                  m,
                                    rocsparse_int                  n,
                                    const rocsparse_mat_descr      csr_descr,
                                    const rocsparse_float_complex* csr_val,
                                    const rocsparse_int*           csr_row_ptr,
                                    const rocsparse_int*           csr_col_ind,
                                    rocsparse_int                  block_dim,
                                    const rocsparse_mat_descr      bsr_descr,
                                    rocsparse_float_complex*       bsr_val,
                                    rocsparse_int*                 bsr_row_ptr,
                                    rocsparse_int*                 bsr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2bsr(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_int                   m,
                                    rocsparse_int                   n,
                                    const rocsparse_mat_descr       csr_descr,
                                    const rocsparse_double_complex* csr_val,
                                    const rocsparse_int*            csr_row_ptr,
                                    const rocsparse_int*            csr_col_ind,
                                    rocsparse_int                   block_dim,
                                    const rocsparse_mat_descr       bsr_descr,
                                    rocsparse_double_complex*       bsr_val,
                                    rocsparse_int*                  bsr_row_ptr,
                                    rocsparse_int*                  bsr_col_ind);
/**@}*/

/*! \ingroup conv_module
*  \brief
 *  \details
 *  \p rocsparse_csr2gebsr_buffer_size returns the size of the temporary buffer that
 *  is required by \p rocsparse_csr2gebcsr_nnz,
 *   \p rocsparse_scsr2gebcsr, \p rocsparse_dcsr2gebsr, \p rocsparse_ccsr2gebsr and \p rocsparse_zcsr2gebsr.
 *  The temporary storage
 *  buffer must be allocated by the user.
 *
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  GEneral BSR matrix given a sparse CSR matrix as input.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  dir         direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
*              \ref rocsparse_direction_row.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*
*  @param[in]
*  csr_val      array of \p nnz elements containing the values of the sparse CSR matrix.
*
*  @param[in]
*  csr_row_ptr  integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*
*  @param[in]
*  csr_col_ind  integer array of the column indices for each non-zero element in the CSR matrix
*
*  @param[in]
*  row_block_dim   the row block dimension of the GEneral BSR matrix. Between 1 and \p m
*
*  @param[in]
*  col_block_dim   the col block dimension of the GEneral BSR matrix. Between 1 and \p n
*
*  @param[out]
*  p_buffer_size  (host/device) number of bytes of the temporary storage buffer required by \p rocsparse_csr2gebsr_nnz and \p rocsparse_scsr2gebsr.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p row_block_dim  \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_val or \p csr_row_ptr or \p csr_col_ind or \p bsr_row_ptr or \p p_buffer_size
*              pointer is invalid.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr csr_descr,
                                                  const float*              csr_val,
                                                  const rocsparse_int*      csr_row_ptr,
                                                  const rocsparse_int*      csr_col_ind,
                                                  rocsparse_int             row_block_dim,
                                                  rocsparse_int             col_block_dim,
                                                  size_t*                   p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr csr_descr,
                                                  const double*             csr_val,
                                                  const rocsparse_int*      csr_row_ptr,
                                                  const rocsparse_int*      csr_col_ind,
                                                  rocsparse_int             row_block_dim,
                                                  rocsparse_int             col_block_dim,
                                                  size_t*                   p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2gebsr_buffer_size(rocsparse_handle               handle,
                                                  rocsparse_direction            dir,
                                                  rocsparse_int                  m,
                                                  rocsparse_int                  n,
                                                  const rocsparse_mat_descr      csr_descr,
                                                  const rocsparse_float_complex* csr_val,
                                                  const rocsparse_int*           csr_row_ptr,
                                                  const rocsparse_int*           csr_col_ind,
                                                  rocsparse_int                  row_block_dim,
                                                  rocsparse_int                  col_block_dim,
                                                  size_t*                        p_buffer_size);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2gebsr_buffer_size(rocsparse_handle                handle,
                                                  rocsparse_direction             dir,
                                                  rocsparse_int                   m,
                                                  rocsparse_int                   n,
                                                  const rocsparse_mat_descr       csr_descr,
                                                  const rocsparse_double_complex* csr_val,
                                                  const rocsparse_int*            csr_row_ptr,
                                                  const rocsparse_int*            csr_col_ind,
                                                  rocsparse_int                   row_block_dim,
                                                  rocsparse_int                   col_block_dim,
                                                  size_t*                         p_buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  GEneral BSR matrix given a sparse CSR matrix as input.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  dir         direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by
*              \ref rocsparse_direction_row.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_row_ptr integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*
*  @param[in]
*  csr_col_ind integer array of the column indices for each non-zero element in the CSR matrix
*
*  @param[in]
*  bsr_descr    descriptor of the sparse GEneral BSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_row_ptr integer array containing \p mb+1 elements that point to the start of each block row of the General BSR matrix
*
*  @param[in]
*  row_block_dim   the row block dimension of the GEneral BSR matrix. Between 1 and min(m, n)
*
*  @param[in]
*  col_block_dim   the col block dimension of the GEneral BSR matrix. Between 1 and min(m, n)
*
*  @param[out]
*  bsr_nnz_devhost  total number of nonzero elements in device or host memory.
*
*  @param[in]
*  p_buffer    buffer allocated by the user whose size is determined by calling \p rocsparse_xcsr2gebsr_buffer_size.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p row_block_dim \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p csr_row_ptr or \p csr_col_ind or \p bsr_row_ptr or \p bsr_nnz_devhost
*              pointer is invalid.
*/
/**@{*/

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr2gebsr_nnz(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         const rocsparse_mat_descr csr_descr,
                                         const rocsparse_int*      csr_row_ptr,
                                         const rocsparse_int*      csr_col_ind,
                                         const rocsparse_mat_descr bsr_descr,
                                         rocsparse_int*            bsr_row_ptr,
                                         rocsparse_int             row_block_dim,
                                         rocsparse_int             col_block_dim,
                                         rocsparse_int*            bsr_nnz_devhost,
                                         void*                     p_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse GEneral BSR matrix
*
*  \details
*  \p rocsparse_csr2gebsr converts a CSR matrix into a GEneral BSR matrix. It is assumed,
*  that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
*  for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
*  the GEneral BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
*  \p csr2gebsr_nnz() which also fills in \p bsr_row_ptr.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  dir          the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  csr_descr    descriptor of the sparse CSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  csr_val      array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr  array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csr_col_ind  array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  bsr_descr    descriptor of the sparse BSR matrix. Currently, only
*               \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  bsr_val      array of \p nnzb* \p row_block_dim*\p col_block_dim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsr_row_ptr  array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsr_col_ind  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  row_block_dim    row size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  col_block_dim    col size of the blocks in the sparse GEneral BSR matrix.
*  @param[in]
*  p_buffer    buffer allocated by the user whose size is determined by calling \p rocsparse_xcsr2gebsr_buffer_size.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m or \p n or \p row_block_dim or \p col_block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a CSR matrix into an BSR matrix.
*  \code{.c}
*      //     1 4 0 0 0 0
*      // A = 0 2 3 0 0 0
*      //     5 0 0 7 8 0
*      //     0 0 9 0 6 0
*
*      rocsparse_int m   = 4;
*      rocsparse_int n   = 6;
*      rocsparse_int row_block_dim = 2;
*      rocsparse_int col_block_dim = 3;
*      rocsparse_int nnz = 9;
*      rocsparse_int mb = (m + row_block_dim - 1) / row_block_dim;
*      rocsparse_int nb = (n + col_block_dim - 1) / col_block_dim;
*
*      csr_row_ptr[m+1]  = {0, 2, 4, 7, 9};             // device memory
*      csr_col_ind[nnz]  = {0, 1, 1, 2, 0, 3, 4, 2, 4}; // device memory
*      csr_val[nnz]      = {1, 4, 2, 3, 5, 7, 8, 9, 6}; // device memory
*
*      hipMalloc(&bsr_row_ptr, sizeof(rocsparse_int) *(mb + 1));
*      rocsparse_int nnzb;
*      rocsparse_int* nnzTotalHostPtr = &nnzb;
*      csr2gebsr_nnz(handle,
*                  rocsparse_direction_row,
*                  m,
*                  n,
*                  csr_descr,
*                  csr_row_ptr,
*                  csr_col_ind,
*                  row_block_dim,
*                  col_block_dim,
*                  bsr_descr,
*                  bsr_row_ptr,
*                  nnzTotalHostPtr);
*      nnzb = *nnzTotalDevHostPtr;
*      hipMalloc(&bsr_col_ind, sizeof(int)*nnzb);
*      hipMalloc(&bsr_val, sizeof(float)*(row_block_dim * col_block_dim) * nnzb);
*      scsr2gebsr(handle,
*               rocsparse_direction_row,
*               m,
*               n,
*               csr_descr,
*               csr_val,
*               csr_row_ptr,
*               csr_col_ind,
*               row_block_dim,
*               col_block_dim,
*               bsr_descr,
*               bsr_val,
*               bsr_row_ptr,
*               bsr_col_ind);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2gebsr(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr csr_descr,
                                      const float*              csr_val,
                                      const rocsparse_int*      csr_row_ptr,
                                      const rocsparse_int*      csr_col_ind,
                                      const rocsparse_mat_descr bsr_descr,
                                      float*                    bsr_val,
                                      rocsparse_int*            bsr_row_ptr,
                                      rocsparse_int*            bsr_col_ind,
                                      rocsparse_int             row_block_dim,
                                      rocsparse_int             col_block_dim,
                                      void*                     p_buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2gebsr(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_int             m,
                                      rocsparse_int             n,
                                      const rocsparse_mat_descr csr_descr,
                                      const double*             csr_val,
                                      const rocsparse_int*      csr_row_ptr,
                                      const rocsparse_int*      csr_col_ind,
                                      const rocsparse_mat_descr bsr_descr,
                                      double*                   bsr_val,
                                      rocsparse_int*            bsr_row_ptr,
                                      rocsparse_int*            bsr_col_ind,
                                      rocsparse_int             row_block_dim,
                                      rocsparse_int             col_block_dim,
                                      void*                     p_buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2gebsr(rocsparse_handle               handle,
                                      rocsparse_direction            dir,
                                      rocsparse_int                  m,
                                      rocsparse_int                  n,
                                      const rocsparse_mat_descr      csr_descr,
                                      const rocsparse_float_complex* csr_val,
                                      const rocsparse_int*           csr_row_ptr,
                                      const rocsparse_int*           csr_col_ind,
                                      const rocsparse_mat_descr      bsr_descr,
                                      rocsparse_float_complex*       bsr_val,
                                      rocsparse_int*                 bsr_row_ptr,
                                      rocsparse_int*                 bsr_col_ind,
                                      rocsparse_int                  row_block_dim,
                                      rocsparse_int                  col_block_dim,
                                      void*                          p_buffer);
ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2gebsr(rocsparse_handle                handle,
                                      rocsparse_direction             dir,
                                      rocsparse_int                   m,
                                      rocsparse_int                   n,
                                      const rocsparse_mat_descr       csr_descr,
                                      const rocsparse_double_complex* csr_val,
                                      const rocsparse_int*            csr_row_ptr,
                                      const rocsparse_int*            csr_col_ind,
                                      const rocsparse_mat_descr       bsr_descr,
                                      rocsparse_double_complex*       bsr_val,
                                      rocsparse_int*                  bsr_row_ptr,
                                      rocsparse_int*                  bsr_col_ind,
                                      rocsparse_int                   row_block_dim,
                                      rocsparse_int                   col_block_dim,
                                      void*                           p_buffer);

/**@}*/

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a compressed sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_csr2csr_compress converts a CSR matrix into a compressed CSR matrix by
 *  removing entries in the input CSR matrix that are below a non-negative threshold \p tol
 *
 *  \note
 *  In the case of complex matrices only the magnitude of the real part of \p tol is used.
 *
 *  @param[in]
 *  handle        handle to the rocsparse library context queue.
 *  @param[in]
 *  m             number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n             number of columns of the sparse CSR matrix.
 *  @param[in]
 *  descr_A       matrix descriptor for the CSR matrix
 *  @param[in]
 *  csr_val_A     array of \p nnz_A elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr_A array of \p m+1 elements that point to the start of every row of the
 *                uncompressed sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind_A array of \p nnz_A elements containing the column indices of the uncompressed
 *                sparse CSR matrix.
 *  @param[in]
 *  nnz_A         number of elements in the column indices and values arrays of the uncompressed
 *                sparse CSR matrix.
 *  @param[in]
 *  nnz_per_row   array of length \p m containing the number of entries that will be kept per row in
 *                the final compressed CSR matrix.
 *  @param[out]
 *  csr_val_C     array of \p nnz_C elements of the compressed sparse CSC matrix.
 *  @param[out]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every column of the compressed
 *                sparse CSR matrix.
 *  @param[out]
 *  csr_col_ind_C array of \p nnz_C elements containing the row indices of the compressed
 *                sparse CSR matrix.
 *  @param[in]
 *  tol           the non-negative tolerance used for compression. If \p tol is complex then only the magnitude
 *                of the real part is used. Entries in the input uncompressed CSR array that are below the tolerance
 *                are removed in output compressed CSR matrix.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m, \p n or \p nnz_A is invalid.
 *  \retval     rocsparse_status_invalid_value \p tol is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p csr_val_A, \p csr_row_ptr_A,
 *              \p csr_col_ind_A, \p csr_val_C, \p csr_row_ptr_C, \p csr_col_ind_C or
 *              \p nnz_per_row pointer is invalid.
 *
 *  \par Example
 *  This example demonstrates how to compress a CSR matrix. Compressing a CSR matrix involves two steps. First we use
 *  nnz_compress() to determine how many entries will be in the final compressed CSR matrix. Then we call csr2csr_compress()
 *  to finish the compression and fill in the column indices and values arrays of the compressed CSR matrix.
 *  \code{.c}
 *      //     1 2 0 3 0
 *      // A = 0 4 5 0 0
 *      //     6 0 0 7 8
 *
 *      float tol = 0.0f;
 *
 *      rocsparse_int m     = 3;
 *      rocsparse_int n     = 5;
 *      rocsparse_int nnz_A = 8;
 *
 *      csr_row_ptr_A[m+1]   = {0, 3, 5, 8};             // device memory
 *      csr_col_ind_A[nnz_A] = {0, 1, 3, 1, 2, 0, 3, 4}; // device memory
 *      csr_val_A[nnz_A]     = {1, 0, 3, 4, 0, 6, 7, 0}; // device memory
 *
 *      // Allocate memory for the row pointer array of the compressed CSR matrix
 *      rocsparse_int* csr_row_ptr_C;
 *      hipMalloc(csr_row_ptr_C, sizeof(rocsparse_int) * (m + 1));
 *
 *      // Allocate memory for the nnz_per_row array
 *      rocsparse_int* nnz_per_row;
 *      hipMalloc(nnz_per_row, sizeof(rocsparse_int) * m);
 *
 *      // Call nnz_compress() which fills in nnz_per_row array and finds the number
 *      // of entries that will be in the compressed CSR matrix
 *      rocsparse_int nnz_C;
 *      nnz_compress(handle,
 *                   m,
 *                   descr_A,
 *                   csr_val_A,
 *                   csr_row_ptr_A,
 *                   nnz_per_row,
 *                   &nnz_C,
 *                   tol);
 *
 *      // Allocate column indices and values array for the compressed CSR matrix
 *      rocsparse_int* csr_col_ind_C;
 *      rocsparse_int* csr_val_C;
 *      hipMalloc(csr_col_ind_C, sizeof(rocsparse_int) * nnz_C;
 *      hipMalloc(csr_val_C, sizeof(rocsparse_int) * nnz_C;
 *
 *      // Finish compression by calling csr2csr_compress()
 *      csr2csr_compress(handle,
 *                       m,
 *                       n,
 *                       descr_A,
 *                       csr_val_A,
 *                       csr_row_ptr_A,
 *                       csr_col_ind_A,
 *                       nnz_A,
 *                       nnz_per_row,
 *                       csr_val_C,
 *                       csr_row_ptr_C,
 *                       csr_col_ind_C,
 *                       tol);
 *  \endcode
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scsr2csr_compress(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const rocsparse_mat_descr descr_A,
                                             const float*              csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_int*      nnz_per_row,
                                             float*                    csr_val_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            csr_col_ind_C,
                                             float                     tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dcsr2csr_compress(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const rocsparse_mat_descr descr_A,
                                             const double*             csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_int*      nnz_per_row,
                                             double*                   csr_val_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            csr_col_ind_C,
                                             double                    tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ccsr2csr_compress(rocsparse_handle               handle,
                                             rocsparse_int                  m,
                                             rocsparse_int                  n,
                                             const rocsparse_mat_descr      descr_A,
                                             const rocsparse_float_complex* csr_val_A,
                                             const rocsparse_int*           csr_row_ptr_A,
                                             const rocsparse_int*           csr_col_ind_A,
                                             rocsparse_int                  nnz_A,
                                             const rocsparse_int*           nnz_per_row,
                                             rocsparse_float_complex*       csr_val_C,
                                             rocsparse_int*                 csr_row_ptr_C,
                                             rocsparse_int*                 csr_col_ind_C,
                                             rocsparse_float_complex        tol);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zcsr2csr_compress(rocsparse_handle                handle,
                                             rocsparse_int                   m,
                                             rocsparse_int                   n,
                                             const rocsparse_mat_descr       descr_A,
                                             const rocsparse_double_complex* csr_val_A,
                                             const rocsparse_int*            csr_row_ptr_A,
                                             const rocsparse_int*            csr_col_ind_A,
                                             rocsparse_int                   nnz_A,
                                             const rocsparse_int*            nnz_per_row,
                                             rocsparse_double_complex*       csr_val_C,
                                             rocsparse_int*                  csr_row_ptr_C,
                                             rocsparse_int*                  csr_col_ind_C,
                                             rocsparse_double_complex        tol);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_prune_csr2csr_buffer_size returns the size of the temporary buffer that
 *  is required by \p rocsparse_sprune_csr2csr_nnz, \p rocsparse_dprune_csr2csr_nnz,
 *  \p rocsparse_sprune_csr2csr, and \p rocsparse_dprune_csr2csr. The temporary storage
 *  buffer must be allocated by the user.
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
 *  threshold     pointer to the non-negative pruning threshold which can exist in either host or device memory.
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
 *  @param[out]
 *  buffer_size   number of bytes of the temporary storage buffer required by \p rocsparse_sprune_csr2csr_nnz,
 *                \p rocsparse_dprune_csr2csr_nnz, \p rocsparse_sprune_csr2csr, and \p rocsparse_dprune_csr2csr.
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_pointer \p buffer_size pointer is invalid.
 *  \retval     rocsparse_status_internal_error an internal error occurred.
 *
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_csr2csr_buffer_size(rocsparse_handle          handle,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz_A,
                                                      const rocsparse_mat_descr csr_descr_A,
                                                      const float*              csr_val_A,
                                                      const rocsparse_int*      csr_row_ptr_A,
                                                      const rocsparse_int*      csr_col_ind_A,
                                                      const float*              threshold,
                                                      const rocsparse_mat_descr csr_descr_C,
                                                      const float*              csr_val_C,
                                                      const rocsparse_int*      csr_row_ptr_C,
                                                      const rocsparse_int*      csr_col_ind_C,
                                                      size_t*                   buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_csr2csr_buffer_size(rocsparse_handle          handle,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz_A,
                                                      const rocsparse_mat_descr csr_descr_A,
                                                      const double*             csr_val_A,
                                                      const rocsparse_int*      csr_row_ptr_A,
                                                      const rocsparse_int*      csr_col_ind_A,
                                                      const double*             threshold,
                                                      const rocsparse_mat_descr csr_descr_C,
                                                      const double*             csr_val_C,
                                                      const rocsparse_int*      csr_row_ptr_C,
                                                      const rocsparse_int*      csr_col_ind_C,
                                                      size_t*                   buffer_size);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_prune_csr2csr_nnz computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
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
 *  threshold     pointer to the non-negative pruning threshold which can exist in either host or device memory.
 *  @param[in]
 *  csr_descr_C   descriptor of the sparse CSR matrix C. Currently, only
 *                \ref rocsparse_matrix_type_general is supported.
 *  @param[out]
 *  csr_row_ptr_C array of \p m+1 elements that point to the start of every row of the
 *                sparse CSR matrix C.
 *  @param[out]
 *  nnz_total_dev_host_ptr total number of nonzero elements in device or host memory.
 *  @param[out]
 *  temp_buffer   buffer allocated by the user whose size is determined by calling \p rocsparse_xprune_csr2csr_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz_A is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p threshold or \p csr_descr_A or \p csr_descr_C or \p csr_val_A
 *              or \p csr_row_ptr_A or \p csr_col_ind_A or \p csr_row_ptr_C or \p nnz_total_dev_host_ptr
 *              or \p temp_buffer pointer is invalid.
 *
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_csr2csr_nnz(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz_A,
                                              const rocsparse_mat_descr csr_descr_A,
                                              const float*              csr_val_A,
                                              const rocsparse_int*      csr_row_ptr_A,
                                              const rocsparse_int*      csr_col_ind_A,
                                              const float*              threshold,
                                              const rocsparse_mat_descr csr_descr_C,
                                              rocsparse_int*            csr_row_ptr_C,
                                              rocsparse_int*            nnz_total_dev_host_ptr,
                                              void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_csr2csr_nnz(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz_A,
                                              const rocsparse_mat_descr csr_descr_A,
                                              const double*             csr_val_A,
                                              const rocsparse_int*      csr_row_ptr_A,
                                              const rocsparse_int*      csr_col_ind_A,
                                              const double*             threshold,
                                              const rocsparse_mat_descr csr_descr_C,
                                              rocsparse_int*            csr_row_ptr_C,
                                              rocsparse_int*            nnz_total_dev_host_ptr,
                                              void*                     temp_buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls rocsparse_xprune_csr2csr_buffer_size() to determine the size of the buffer used
 *  by rocsparse_xprune_csr2csr_nnz() and rocsparse_xprune_csr2csr() which the user then allocates. The user then
 *  allocates \p csr_row_ptr_C to have \p m+1 elements and then calls rocsparse_xprune_csr2csr_nnz() which fills
 *  in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
 *  in \p nnz_total_dev_host_ptr. The user then calls rocsparse_xprune_csr2csr() to complete the conversion. It
 *  is executed asynchronously with respect to the host and may return control to the application on the host
 *  before the entire result is ready.
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
 *  threshold     pointer to the non-negative pruning threshold which can exist in either host or device memory.
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
 *  temp_buffer   buffer allocated by the user whose size is determined by calling \p rocsparse_xprune_csr2csr_buffer_size().
 *
 *  \retval     rocsparse_status_success the operation completed successfully.
 *  \retval     rocsparse_status_invalid_handle the library context was not initialized.
 *  \retval     rocsparse_status_invalid_size \p m or \p n or \p nnz_A is invalid.
 *  \retval     rocsparse_status_invalid_pointer \p threshold or \p csr_descr_A or \p csr_descr_C or \p csr_val_A
 *              or \p csr_row_ptr_A or \p csr_col_ind_A or \p csr_val_C or \p csr_row_ptr_C or \p csr_col_ind_C
 *              or \p temp_buffer pointer is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sprune_csr2csr(rocsparse_handle          handle,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             nnz_A,
                                          const rocsparse_mat_descr csr_descr_A,
                                          const float*              csr_val_A,
                                          const rocsparse_int*      csr_row_ptr_A,
                                          const rocsparse_int*      csr_col_ind_A,
                                          const float*              threshold,
                                          const rocsparse_mat_descr csr_descr_C,
                                          float*                    csr_val_C,
                                          const rocsparse_int*      csr_row_ptr_C,
                                          rocsparse_int*            csr_col_ind_C,
                                          void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dprune_csr2csr(rocsparse_handle          handle,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             nnz_A,
                                          const rocsparse_mat_descr csr_descr_A,
                                          const double*             csr_val_A,
                                          const rocsparse_int*      csr_row_ptr_A,
                                          const rocsparse_int*      csr_col_ind_A,
                                          const double*             threshold,
                                          const rocsparse_mat_descr csr_descr_C,
                                          double*                   csr_val_C,
                                          const rocsparse_int*      csr_row_ptr_C,
                                          rocsparse_int*            csr_col_ind_C,
                                          void*                     temp_buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p rocsparse_prune_csr2csr__by_percentage_buffer_size returns the size of the temporary buffer that
 *  is required by \p rocsparse_sprune_csr2csr_nnz_by_percentage, \p rocsparse_dprune_csr2csr_nnz_by_percentage,
 *  \p rocsparse_sprune_csr2csr_by_percentage, and \p rocsparse_dprune_csr2csr_by_percentage. The temporary storage
 *  buffer must be allocated by the user.
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
 *  in \p nnz_total_dev_host_ptr. The user then calls rocsparse_xprune_csr2csr() to complete the conversion. It
 *  is executed asynchronously with respect to the host and may return control to the application on the host
 *  before the entire result is ready.
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
*  \brief Convert a sparse HYB matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_hyb2csr_buffer_size returns the size of the temporary storage buffer
*  required by rocsparse_shyb2csr(), rocsparse_dhyb2csr(), rocsparse_chyb2csr() and
*  rocsparse_dhyb2csr(). The temporary storage buffer must be allocated by the user.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  descr           descriptor of the sparse HYB matrix. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  hyb             sparse matrix in HYB format.
*  @param[in]
*  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[out]
*  buffer_size     number of bytes of the temporary storage buffer required by
*                  rocsparse_shyb2csr(), rocsparse_dhyb2csr(), rocsparse_chyb2csr() and
*                  rocsparse_zhyb2csr().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p hyb, \p csr_row_ptr or
*              \p buffer_size pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_hyb2csr_buffer_size(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               const rocsparse_int*      csr_row_ptr,
                                               size_t*                   buffer_size);

/*! \ingroup conv_module
*  \brief Convert a sparse HYB matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_hyb2csr converts a HYB matrix into a CSR matrix.
*
*  \p rocsparse_hyb2csr requires extra temporary storage buffer that has to be allocated
*  by the user. Storage buffer size can be determined by
*  rocsparse_hyb2csr_buffer_size().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle          handle to the rocsparse library context queue.
*  @param[in]
*  descr           descriptor of the sparse HYB matrix. Currently, only
*                  \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  hyb             sparse matrix in HYB format.
*  @param[out]
*  csr_val         array containing the values of the sparse CSR matrix.
*  @param[out]
*  csr_row_ptr     array of \p m+1 elements that point to the start of every row of the
*                  sparse CSR matrix.
*  @param[out]
*  csr_col_ind     array containing the column indices of the sparse CSR matrix.
*  @param[in]
*  temp_buffer     temporary storage buffer allocated by the user, size is returned by
*                  rocsparse_hyb2csr_buffer_size().
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p descr, \p hyb, \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*  \retval     rocsparse_status_not_implemented
*              \ref rocsparse_matrix_type != \ref rocsparse_matrix_type_general.
*
*  \par Example
*  This example converts a HYB matrix into a CSR matrix.
*  \code{.c}
*      // Create CSR matrix arrays
*      rocsparse_int* csr_row_ptr;
*      rocsparse_int* csr_col_ind;
*      float* csr_val;
*
*      hipMalloc((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*      hipMalloc((void**)&csr_col_ind, sizeof(rocsparse_int) * nnz);
*      hipMalloc((void**)&csr_val, sizeof(float) * nnz);
*
*      // Get required size of temporary buffer
*      size_t size;
*      rocsparse_hyb2csr_buffer_size(handle,
*                                    descr,
*                                    hyb,
*                                    csr_row_ptr,
*                                    &size);
*
*      // Allocate temporary buffer
*      void* buffer;
*      hipMalloc(&buffer, size);
*
*      // Perform the conversion
*      rocsparse_shyb2csr(handle,
*                         descr,
*                         hyb,
*                         csr_val,
*                         csr_row_ptr,
*                         csr_col_ind,
*                         buffer);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_shyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    float*                    csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dhyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    double*                   csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_chyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    rocsparse_float_complex*  csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zhyb2csr(rocsparse_handle          handle,
                                    const rocsparse_mat_descr descr,
                                    const rocsparse_hyb_mat   hyb,
                                    rocsparse_double_complex* csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind,
                                    void*                     temp_buffer);
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

/*! \ingroup conv_module
*  \brief Convert a sparse BSR matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_bsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
*  for \p csr_row_ptr is computed by the number of block rows multiplied by the block
*  dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
*  the number of blocks in the BSR matrix multiplied by the block dimension squared.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  mb          number of block rows in the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse BSR matrix.
*  @param[in]
*  bsr_descr   descriptor of the sparse BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  block_dim   size of the blocks in the sparse BSR matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val     array of \p nnzb*block_dim*block_dim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csr_row_ptr array of \p m+1 where \p m=mb*block_dim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csr_col_ind array of \p nnzb*block_dim*block_dim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a BSR matrix into an CSR matrix.
*  \code{.c}
*      //     1 4 0 0 0 0
*      // A = 0 2 3 0 0 0
*      //     5 0 0 7 8 0
*      //     0 0 9 0 6 0
*
*      rocsparse_int mb   = 2;
*      rocsparse_int nb   = 3;
*      rocsparse_int block_dim = 2;
*      rocsparse_int m = Mb * block_dim;
*      rocsparse_int n = Nb * block_dim;
*
*      bsr_row_ptr[mb+1]                 = {0, 2, 5};                                                    // device memory
*      bsr_col_ind[nnzb]                 = {0, 1, 0, 1, 2};                                              // device memory
*      bsr_val[nnzb*block_dim*block_dim] = {1, 0, 4, 2, 0, 3, 0, 0, 5, 0, 0, 0, 0, 9, 7, 0, 8, 6, 0, 0}; // device memory
*
*      rocsparse_int nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];
*
*      // Create CSR arrays on device
*      rocsparse_int* csr_row_ptr;
*      rocsparse_int* csr_col_ind;
*      float* csr_val;
*      hipMalloc((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*      hipMalloc((void**)&csr_col_ind, sizeof(rocsparse_int) * nnzb * block_dim * block_dim);
*      hipMalloc((void**)&csr_val, sizeof(float) * nnzb * block_dim * block_dim);
*
*      // Create rocsparse handle
*      rocsparse_local_handle handle;
*
*      rocsparse_mat_descr bsr_descr = nullptr;
*      rocsparse_create_mat_descr(&bsr_descr);
*
*      rocsparse_mat_descr csr_descr = nullptr;
*      rocsparse_create_mat_descr(&csr_descr);
*
*      rocsparse_set_mat_index_base(bsr_descr, rocsparse_index_base_zero);
*      rocsparse_set_mat_index_base(csr_descr, rocsparse_index_base_zero);
*
*      // Format conversion
*      rocsparse_sbsr2csr(handle,
*                         rocsparse_direction_column,
*                         mb,
*                         nb,
*                         bsr_descr,
*                         bsr_val,
*                         bsr_row_ptr,
*                         bsr_col_ind,
*                         block_dim,
*                         csr_descr,
*                         csr_val,
*                         csr_row_ptr,
*                         csr_col_ind);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sbsr2csr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    const rocsparse_mat_descr bsr_descr,
                                    const float*              bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr csr_descr,
                                    float*                    csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dbsr2csr(rocsparse_handle          handle,
                                    rocsparse_direction       dir,
                                    rocsparse_int             mb,
                                    rocsparse_int             nb,
                                    const rocsparse_mat_descr bsr_descr,
                                    const double*             bsr_val,
                                    const rocsparse_int*      bsr_row_ptr,
                                    const rocsparse_int*      bsr_col_ind,
                                    rocsparse_int             block_dim,
                                    const rocsparse_mat_descr csr_descr,
                                    double*                   csr_val,
                                    rocsparse_int*            csr_row_ptr,
                                    rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cbsr2csr(rocsparse_handle               handle,
                                    rocsparse_direction            dir,
                                    rocsparse_int                  mb,
                                    rocsparse_int                  nb,
                                    const rocsparse_mat_descr      bsr_descr,
                                    const rocsparse_float_complex* bsr_val,
                                    const rocsparse_int*           bsr_row_ptr,
                                    const rocsparse_int*           bsr_col_ind,
                                    rocsparse_int                  block_dim,
                                    const rocsparse_mat_descr      csr_descr,
                                    rocsparse_float_complex*       csr_val,
                                    rocsparse_int*                 csr_row_ptr,
                                    rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zbsr2csr(rocsparse_handle                handle,
                                    rocsparse_direction             dir,
                                    rocsparse_int                   mb,
                                    rocsparse_int                   nb,
                                    const rocsparse_mat_descr       bsr_descr,
                                    const rocsparse_double_complex* bsr_val,
                                    const rocsparse_int*            bsr_row_ptr,
                                    const rocsparse_int*            bsr_col_ind,
                                    rocsparse_int                   block_dim,
                                    const rocsparse_mat_descr       csr_descr,
                                    rocsparse_double_complex*       csr_val,
                                    rocsparse_int*                  csr_row_ptr,
                                    rocsparse_int*                  csr_col_ind);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse general BSR matrix into a sparse CSR matrix
*
*  \details
*  \p rocsparse_gebsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
*  for \p csr_row_ptr is computed by the number of block rows multiplied by the block
*  dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
*  the number of blocks in the BSR matrix multiplied by the product of the block dimensions.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  dir         the storage format of the blocks, \ref rocsparse_direction_row or \ref rocsparse_direction_column
*  @param[in]
*  mb          number of block rows in the sparse general BSR matrix.
*  @param[in]
*  nb          number of block columns in the sparse general BSR matrix.
*  @param[in]
*  bsr_descr   descriptor of the sparse general BSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb*row_block_dim*col_block_dim containing the values of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of the
*              sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*  @param[in]
*  row_block_dim   row size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  col_block_dim   column size of the blocks in the sparse general BSR matrix.
*  @param[in]
*  csr_descr   descriptor of the sparse CSR matrix. Currently, only
*              \ref rocsparse_matrix_type_general is supported.
*  @param[out]
*  csr_val     array of \p nnzb*row_block_dim*col_block_dim elements containing the values of the sparse CSR matrix.
*  @param[out]
*  csr_row_ptr array of \p m+1 where \p m=mb*row_block_dim elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  csr_col_ind array of \p nnzb*block_dim*block_dim elements containing the column indices of the sparse CSR matrix.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p mb or \p nb or \p block_dim is invalid.
*  \retval     rocsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*  \par Example
*  This example converts a general BSR matrix into an CSR matrix.
*  \code{.c}
*      //     1 4 0 0 0 0
*      // A = 0 2 3 0 0 0
*      //     5 0 0 7 8 0
*      //     0 0 9 0 6 0
*
*      rocsparse_int mb   = 2;
*      rocsparse_int nb   = 2;
*      rocsparse_int row_block_dim = 2;
*      rocsparse_int col_block_dim = 3;
*      rocsparse_int m = Mb * row_block_dim;
*      rocsparse_int n = Nb * col_block_dim;
*
*      bsr_row_ptr[mb+1]                 = {0, 1, 3};                                              // device memory
*      bsr_col_ind[nnzb]                 = {0, 0, 1};                                              // device memory
*      bsr_val[nnzb*block_dim*block_dim] = {1, 0, 4, 2, 0, 3, 5, 0, 0, 0, 0, 9, 7, 0, 8, 6, 0, 0}; // device memory
*
*      rocsparse_int nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];
*
*      // Create CSR arrays on device
*      rocsparse_int* csr_row_ptr;
*      rocsparse_int* csr_col_ind;
*      float* csr_val;
*      hipMalloc((void**)&csr_row_ptr, sizeof(rocsparse_int) * (m + 1));
*      hipMalloc((void**)&csr_col_ind, sizeof(rocsparse_int) * nnzb * row_block_dim * col_block_dim);
*      hipMalloc((void**)&csr_val, sizeof(float) * nnzb * row_block_dim * col_block_dim);
*
*      // Create rocsparse handle
*      rocsparse_local_handle handle;
*
*      rocsparse_mat_descr bsr_descr = nullptr;
*      rocsparse_create_mat_descr(&bsr_descr);
*
*      rocsparse_mat_descr csr_descr = nullptr;
*      rocsparse_create_mat_descr(&csr_descr);
*
*      rocsparse_set_mat_index_base(bsr_descr, rocsparse_index_base_zero);
*      rocsparse_set_mat_index_base(csr_descr, rocsparse_index_base_zero);
*
*      // Format conversion
*      rocsparse_sgebsr2csr(handle,
*                         rocsparse_direction_column,
*                         mb,
*                         nb,
*                         bsr_descr,
*                         bsr_val,
*                         bsr_row_ptr,
*                         bsr_col_ind,
*                         row_block_dim,
*                         col_block_dim,
*                         csr_descr,
*                         csr_val,
*                         csr_row_ptr,
*                         csr_col_ind);
*  \endcode
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgebsr2csr(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_int             mb,
                                      rocsparse_int             nb,
                                      const rocsparse_mat_descr bsr_descr,
                                      const float*              bsr_val,
                                      const rocsparse_int*      bsr_row_ptr,
                                      const rocsparse_int*      bsr_col_ind,
                                      rocsparse_int             row_block_dim,
                                      rocsparse_int             col_block_dim,
                                      const rocsparse_mat_descr csr_descr,
                                      float*                    csr_val,
                                      rocsparse_int*            csr_row_ptr,
                                      rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgebsr2csr(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_int             mb,
                                      rocsparse_int             nb,
                                      const rocsparse_mat_descr bsr_descr,
                                      const double*             bsr_val,
                                      const rocsparse_int*      bsr_row_ptr,
                                      const rocsparse_int*      bsr_col_ind,
                                      rocsparse_int             row_block_dim,
                                      rocsparse_int             col_block_dim,
                                      const rocsparse_mat_descr csr_descr,
                                      double*                   csr_val,
                                      rocsparse_int*            csr_row_ptr,
                                      rocsparse_int*            csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgebsr2csr(rocsparse_handle               handle,
                                      rocsparse_direction            dir,
                                      rocsparse_int                  mb,
                                      rocsparse_int                  nb,
                                      const rocsparse_mat_descr      bsr_descr,
                                      const rocsparse_float_complex* bsr_val,
                                      const rocsparse_int*           bsr_row_ptr,
                                      const rocsparse_int*           bsr_col_ind,
                                      rocsparse_int                  row_block_dim,
                                      rocsparse_int                  col_block_dim,
                                      const rocsparse_mat_descr      csr_descr,
                                      rocsparse_float_complex*       csr_val,
                                      rocsparse_int*                 csr_row_ptr,
                                      rocsparse_int*                 csr_col_ind);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgebsr2csr(rocsparse_handle                handle,
                                      rocsparse_direction             dir,
                                      rocsparse_int                   mb,
                                      rocsparse_int                   nb,
                                      const rocsparse_mat_descr       bsr_descr,
                                      const rocsparse_double_complex* bsr_val,
                                      const rocsparse_int*            bsr_row_ptr,
                                      const rocsparse_int*            bsr_col_ind,
                                      rocsparse_int                   row_block_dim,
                                      rocsparse_int                   col_block_dim,
                                      const rocsparse_mat_descr       csr_descr,
                                      rocsparse_double_complex*       csr_val,
                                      rocsparse_int*                  csr_row_ptr,
                                      rocsparse_int*                  csr_col_ind);
/**@}*/

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
*  \details
*  The routine does support asynchronous execution.
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

/*
* ===========================================================================
*    generic SPARSE
* ===========================================================================
*/

/*! \ingroup generic_module
*  \brief Scale a sparse vector and add it to a scaled dense vector.
*
*  \details
*  \ref rocsparse_axpby multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
*  adds the result to the dense vector \f$y\f$ that is multiplied with scalar
*  \f$\beta\f$, such that
*
*  \f[
*      y := \alpha \cdot x + \beta \cdot y
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] = alpha * x_val[i] + beta * y[x_ind[i]]
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
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  x           sparse matrix descriptor.
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           dense matrix descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p x, \p beta or \p y pointer is
*          invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_axpby(rocsparse_handle            handle,
                                 const void*                 alpha,
                                 const rocsparse_spvec_descr x,
                                 const void*                 beta,
                                 rocsparse_dnvec_descr       y);

/*! \ingroup generic_module
*  \brief Gather elements from a dense vector and store them into a sparse vector.
*
*  \details
*  \ref rocsparse_gather gathers the elements from the dense vector \f$y\f$ and stores
*  them in the sparse vector \f$x\f$.
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
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  y            dense vector \f$y\f$.
*  @param[out]
*  x            sparse vector \f$x\f$.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p x or \p y pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_gather(rocsparse_handle            handle,
                                  const rocsparse_dnvec_descr y,
                                  rocsparse_spvec_descr       x);

/*! \ingroup generic_module
*  \brief Scatter elements from a sparse vector into a dense vector.
*
*  \details
*  \ref rocsparse_scatter scatters the elements from the sparse vector \f$x\f$ in the dense
*  vector \f$y\f$.
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
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  x            sparse vector \f$x\f$.
*  @param[out]
*  y            dense vector \f$y\f$.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p x or \p y pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_scatter(rocsparse_handle            handle,
                                   const rocsparse_spvec_descr x,
                                   rocsparse_dnvec_descr       y);

/*! \ingroup generic_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \ref rocsparse_rot applies the Givens rotation matrix \f$G\f$ to the sparse vector
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
*  c           pointer to the cosine element of \f$G\f$, can be on host or device.
*  @param[in]
*  s           pointer to the sine element of \f$G\f$, can be on host or device.
*  @param[inout]
*  x           sparse vector \f$x\f$.
*  @param[inout]
*  y           dense vector \f$y\f$.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_pointer \p c, \p s, \p x or \p y pointer is
*              invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_rot(rocsparse_handle      handle,
                               const void*           c,
                               const void*           s,
                               rocsparse_spvec_descr x,
                               rocsparse_dnvec_descr y);

/*! \ingroup generic_module
*  \brief Sparse matrix to dense matrix conversion
*
*  \details
*  \p rocsparse_sparse_to_dense
*  \p rocsparse_sparse_to_dense performs the conversion of a sparse matrix in CSR, CSC, or COO format to
*     a dense matrix
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the sparse to dense operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  mat_A        sparse matrix descriptor.
*  @param[in]
*  mat_B        dense matrix descriptor.
*  @param[in]
*  alg          algorithm for the sparse to dense computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the sparse to dense operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p mat_A, \p mat_B, or \p buffer_size
*               pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle              handle,
                                           const rocsparse_spmat_descr   mat_A,
                                           rocsparse_dnmat_descr         mat_B,
                                           rocsparse_sparse_to_dense_alg alg,
                                           size_t*                       buffer_size,
                                           void*                         temp_buffer);

/*! \ingroup generic_module
*  \brief Dense matrix to sparse matrix conversion
*
*  \details
*  \p rocsparse_dense_to_sparse
*  \p rocsparse_dense_to_sparse performs the conversion of a dense matrix to a sparse matrix in CSR, CSC, or COO format.
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the dense to sparse operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  mat_A        dense matrix descriptor.
*  @param[in]
*  mat_B        sparse matrix descriptor.
*  @param[in]
*  alg          algorithm for the sparse to dense computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the dense to sparse operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p mat_A, \p mat_B, or \p buffer_size
*               pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle              handle,
                                           const rocsparse_dnmat_descr   mat_A,
                                           rocsparse_spmat_descr         mat_B,
                                           rocsparse_dense_to_sparse_alg alg,
                                           size_t*                       buffer_size,
                                           void*                         temp_buffer);

/*! \ingroup generic_module
*  \brief Sparse vector inner dot product
*
*  \details
*  \ref rocsparse_spvv computes the inner dot product of the sparse vecotr \f$x\f$ with the
*  dense vector \f$y\f$, such that
*  \f[
*    \text{result} := x^{'} \cdot y,
*  \f]
*  with
*  \f[
*    op(x) = \left\{
*    \begin{array}{ll}
*        x,   & \text{if trans == rocsparse_operation_none} \\
*        \bar{x}, & \text{if trans == rocsparse_operation_conjugate_transpose} \\
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      result = 0;
*      for(i = 0; i < nnz; ++i)
*      {
*          result += x_val[i] * y[x_ind[i]];
*      }
*  \endcode
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the SpVV operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans        sparse vector operation type.
*  @param[in]
*  x            sparse vector descriptor.
*  @param[in]
*  y            dense vector descriptor.
*  @param[out]
*  result       pointer to the result, can be host or device memory
*  @param[in]
*  compute_type floating point precision for the SpVV computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpVV operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p x, \p y, \p result or \p buffer_size
*               pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p compute_type is currently not
*               supported.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvv(rocsparse_handle            handle,
                                rocsparse_operation         trans,
                                const rocsparse_spvec_descr x,
                                const rocsparse_dnvec_descr y,
                                void*                       result,
                                rocsparse_datatype          compute_type,
                                size_t*                     buffer_size,
                                void*                       temp_buffer);

/*! \ingroup generic_module
*  \brief Sparse matrix vector multiplication
*
*  \details
*  \ref rocsparse_spmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix and the dense vector \f$x\f$ and adds the result to the dense vector \f$y\f$
*  that is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == rocsparse_operation_none} \\
*        A^T, & \text{if trans == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the SpMV operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref rocsparse_operation_none is supported.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans        matrix operation type.
*  @paran[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat          matrix descriptor.
*  @param[in]
*  x            vector descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  y            vector descriptor.
*  @param[in]
*  compute_type floating point precision for the SpMV computation.
*  @param[in]
*  alg          SpMV algorithm for the SpMV computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpMV operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat, \p x, \p beta, \p y or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p trans, \p compute_type or \p alg is
*               currently not supported.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmv(rocsparse_handle            handle,
                                rocsparse_operation         trans,
                                const void*                 alpha,
                                const rocsparse_spmat_descr mat,
                                const rocsparse_dnvec_descr x,
                                const void*                 beta,
                                const rocsparse_dnvec_descr y,
                                rocsparse_datatype          compute_type,
                                rocsparse_spmv_alg          alg,
                                size_t*                     buffer_size,
                                void*                       temp_buffer);

/*! \ingroup generic_module
*  \brief Sparse matrix dense matrix multiplication
*
*  \details
*  \p rocsparse_spmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
*  matrix \f$A\f$, defined in CSR or COO storage format, and the dense \f$k \times n\f$
*  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*
*  \note
*  Currently, only CSR and COO sparse formats are supported.
*
*  \note
*  Different algorithms are available which can provide better performance for different matrices.
*  Currently, the available algorithms are rocsparse_spmm_alg_csr for CSR matrices and
*  rocsparse_spmm_alg_coo_segmented or rocsparse_spmm_alg_coo_atomic for COO matrices. Additionally,
*  one can specify the algorithm to be rocsparse_spmm_alg_default. In the case of CSR matrices this will
*  set the algorithm to be rocsparse_spmm_alg_csr and for COO matrices it will set the algorithm to be
*  rocsparse_spmm_alg_coo_atomic.
*
*  \note
*  This function writes the required allocation size (in bytes) to \p buffer_size and
*  returns without performing the SpMM operation, when a nullptr is passed for
*  \p temp_buffer.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans_A      matrix operation type.
*  @param[in]
*  trans_B      matrix operation type.
*  @paran[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        matrix descriptor.
*  @param[in]
*  mat_B        matrix descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[in]
*  mat_C        matrix descriptor.
*  @param[in]
*  compute_type floating point precision for the SpMM computation.
*  @param[in]
*  alg          SpMM algorithm for the SpMM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpMM operation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p alpha, \p mat_A, \p mat_B, \p mat_C, \p beta, or
*               \p buffer_size pointer is invalid.
*  \retval      rocsparse_status_not_implemented \p trans_A, \p trans_B, \p compute_type or \p alg is
*               currently not supported.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                rocsparse_operation         trans_A,
                                rocsparse_operation         trans_B,
                                const void*                 alpha,
                                const rocsparse_spmat_descr mat_A,
                                const rocsparse_dnmat_descr mat_B,
                                const void*                 beta,
                                const rocsparse_dnmat_descr mat_C,
                                rocsparse_datatype          compute_type,
                                rocsparse_spmm_alg          alg,
                                size_t*                     buffer_size,
                                void*                       temp_buffer);

/*! \ingroup generic_module
*  \brief Sparse matrix sparse matrix multiplication
*
*  \details
*  \ref rocsparse_spgemm multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$ and
*  adds the result to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot D,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none} \\
*        A^T, & \text{if trans_A == rocsparse_operation_transpose} \\
*        A^H, & \text{if trans_A == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none} \\
*        B^T, & \text{if trans_B == rocsparse_operation_transpose} \\
*        B^H, & \text{if trans_B == rocsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note SpGEMM requires three stages to complete. The first stage
*  \ref rocsparse_spgemm_stage_buffer_size will return the size of the temporary storage buffer
*  that is required for subsequent calls to \ref rocsparse_spgemm. The second stage
*  \ref rocsparse_spgemm_stage_nnz will determine the number of non-zero elements of the
*  resulting \f$C\f$ matrix. If the sparsity pattern of \f$C\f$ is already known, this
*  stage can be skipped. In the final stage \ref rocsparse_spgemm_stage_compute, the actual
*  computation is performed.
*  \note If \ref rocsparse_spgemm_stage_auto is selected, rocSPARSE will automatically detect
*  which stage is required based on the following indicators:
*  If \p temp_buffer is equal to \p nullptr, the required buffer size will be returned.
*  Else, if the number of non-zeros of \f$C\f$ is zero, the number of non-zero entries will be
*  computed.
*  Else, the SpGEMM algorithm will be executed.
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot op(A) \cdot op(B)\f$ will be
*  computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note It is allowed to pass the same sparse matrix for \f$C\f$ and \f$D\f$, if both
*  matrices have the same sparsity pattern.
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for rare matrix products with more than 4096 non-zero entries
*  per row, additional temporary storage buffer is allocated by the algorithm.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  trans_A      sparse matrix \f$A\f$ operation type.
*  @param[in]
*  trans_B      sparse matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  A            sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  B            sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[in]
*  D            sparse matrix \f$D\f$ descriptor.
*  @param[out]
*  C            sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SpGEMM computation.
*  @param[in]
*  alg          SpGEMM algorithm for the SpGEMM computation.
*  @param[in]
*  stage        SpGEMM stage for the SpGEMM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. buffer_size is set when
*               \p temp_buffer is nullptr.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user. When a nullptr is passed,
*               the required allocation size (in bytes) is written to \p buffer_size and
*               function returns without performing the SpGEMM operation.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p A, \p B, \p D, \p C or \p buffer_size pointer is invalid.
*  \retval rocsparse_status_memory_error additional buffer for long rows could not be
*          allocated.
*  \retval rocsparse_status_not_implemented
*          \p trans_A != \ref rocsparse_operation_none or
*          \p trans_B != \ref rocsparse_operation_none.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spgemm(rocsparse_handle            handle,
                                  rocsparse_operation         trans_A,
                                  rocsparse_operation         trans_B,
                                  const void*                 alpha,
                                  const rocsparse_spmat_descr A,
                                  const rocsparse_spmat_descr B,
                                  const void*                 beta,
                                  const rocsparse_spmat_descr D,
                                  rocsparse_spmat_descr       C,
                                  rocsparse_datatype          compute_type,
                                  rocsparse_spgemm_alg        alg,
                                  rocsparse_spgemm_stage      stage,
                                  size_t*                     buffer_size,
                                  void*                       temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSE_FUNCTIONS_H_ */

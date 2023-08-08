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

#ifndef ROCSPARSE_GPSV_H
#define ROCSPARSE_GPSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup precond_module
*  \brief Batched Pentadiagonal solver
*
*  \details
*  \p rocsparse_gpsv_interleaved_batch_buffer_size calculates the required buffer size
*  for rocsparse_gpsv_interleaved_batch(). It is the users responsibility to allocate
*  this buffer.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  alg          algorithm to solve the linear system.
*  @param[in]
*  m            size of the pentadiagonal linear system.
*  @param[in]
*  ds           lower diagonal (distance 2) of pentadiagonal system. First two entries
*               must be zero.
*  @param[in]
*  dl           lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[in]
*  d            main diagonal of pentadiagonal system.
*  @param[in]
*  du           upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[in]
*  dw           upper diagonal (distance 2) of pentadiagonal system. Last two entries
*               must be zero.
*  @param[in]
*  x            Dense array of right-hand-sides with dimension \p batch_stride by \p m.
*  @param[in]
*  batch_count  The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate consecutive elements in a system.
*               Must satisfy \p batch_stride >= batch_count.
*  @param[out]
*  buffer_size  Number of bytes of the temporary storage buffer required.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p alg, \p batch_count or
*              \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p ds, \p dl, \p d, \p du, \p dw, \p x
*              or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const float*                   ds,
                                                               const float*                   dl,
                                                               const float*                   d,
                                                               const float*                   du,
                                                               const float*                   dw,
                                                               const float*                   x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const double*                  ds,
                                                               const double*                  dl,
                                                               const double*                  d,
                                                               const double*                  du,
                                                               const double*                  dw,
                                                               const double*                  x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg alg,
                                                               rocsparse_int                  m,
                                                               const rocsparse_float_complex* ds,
                                                               const rocsparse_float_complex* dl,
                                                               const rocsparse_float_complex* d,
                                                               const rocsparse_float_complex* du,
                                                               const rocsparse_float_complex* dw,
                                                               const rocsparse_float_complex* x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgpsv_interleaved_batch_buffer_size(rocsparse_handle handle,
                                                               rocsparse_gpsv_interleaved_alg  alg,
                                                               rocsparse_int                   m,
                                                               const rocsparse_double_complex* ds,
                                                               const rocsparse_double_complex* dl,
                                                               const rocsparse_double_complex* d,
                                                               const rocsparse_double_complex* du,
                                                               const rocsparse_double_complex* dw,
                                                               const rocsparse_double_complex* x,
                                                               rocsparse_int batch_count,
                                                               rocsparse_int batch_stride,
                                                               size_t*       buffer_size);
/**@}*/

/*! \ingroup precond_module
*  \brief Batched Pentadiagonal solver
*
*  \details
*  \p rocsparse_gpsv_interleaved_batch  solves a batch of pentadiagonal linear systems.
*  The coefficient matrix of each pentadiagonal linear system is defined by five vectors
*  for the lower part (ds, dl), main diagonal (d) and upper part (du, dw).
*
*  The function requires a temporary buffer. The size of the required buffer is returned
*  by rocsparse_gpsv_interleaved_batch_buffer_size().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  The routine is numerically stable because it uses QR to solve the linear systems.
*
*  \note
*  m need to be at least 3, to be a valid pentadiagonal matrix.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  alg          algorithm to solve the linear system.
*  @param[in]
*  m            size of the pentadiagonal linear system.
*  @param[inout]
*  ds           lower diagonal (distance 2) of pentadiagonal system. First two entries
*               must be zero.
*  @param[inout]
*  dl           lower diagonal of pentadiagonal system. First entry must be zero.
*  @param[inout]
*  d            main diagonal of pentadiagonal system.
*  @param[inout]
*  du           upper diagonal of pentadiagonal system. Last entry must be zero.
*  @param[inout]
*  dw           upper diagonal (distance 2) of pentadiagonal system. Last two entries
*               must be zero.
*  @param[inout]
*  x            Dense array of right-hand-sides with dimension \p batch_stride by \p m.
*  @param[in]
*  batch_count  The number of systems to solve.
*  @param[in]
*  batch_stride The number of elements that separate consecutive elements in a system.
*               Must satisfy \p batch_stride >= batch_count.
*  @param[in]
*  temp_buffer  Temporary storage buffer allocated by the user.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p m, \p alg, \p batch_count or
*              \p batch_stride is invalid.
*  \retval     rocsparse_status_invalid_pointer \p ds, \p dl, \p d, \p du, \p dw, \p x
*              or \p temp_buffer pointer is invalid.
*  \retval     rocsparse_status_internal_error an internal error occurred.
*/
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   float*                         ds,
                                                   float*                         dl,
                                                   float*                         d,
                                                   float*                         du,
                                                   float*                         dw,
                                                   float*                         x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   double*                        ds,
                                                   double*                        dl,
                                                   double*                        d,
                                                   double*                        du,
                                                   double*                        dw,
                                                   double*                        x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_cgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   rocsparse_float_complex*       ds,
                                                   rocsparse_float_complex*       dl,
                                                   rocsparse_float_complex*       d,
                                                   rocsparse_float_complex*       du,
                                                   rocsparse_float_complex*       dw,
                                                   rocsparse_float_complex*       x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_zgpsv_interleaved_batch(rocsparse_handle               handle,
                                                   rocsparse_gpsv_interleaved_alg alg,
                                                   rocsparse_int                  m,
                                                   rocsparse_double_complex*      ds,
                                                   rocsparse_double_complex*      dl,
                                                   rocsparse_double_complex*      d,
                                                   rocsparse_double_complex*      du,
                                                   rocsparse_double_complex*      dw,
                                                   rocsparse_double_complex*      x,
                                                   rocsparse_int                  batch_count,
                                                   rocsparse_int                  batch_stride,
                                                   void*                          temp_buffer);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_GPSV_H */

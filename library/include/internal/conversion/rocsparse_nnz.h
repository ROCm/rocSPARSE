/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_NNZ_H
#define ROCSPARSE_NNZ_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row or column and the total number of nonzero elements in a dense matrix.
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*
*  @param[in]
*  dir        direction that specified whether to count nonzero elements by \ref rocsparse_direction_row or by \ref rocsparse_direction_column.
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
*
*  \par Example
*  \code{.c}
*      // rocSPARSE handle
*      rocsparse_handle handle;
*      rocsparse_create_handle(&handle);
*
*      // Matrix descriptor
*      rocsparse_mat_descr descr;
*      rocsparse_create_mat_descr(&descr);
*
*      // Dense matrix in column order
*      //     1 2 0 3 0
*      // A = 0 4 5 0 0
*      //     6 0 0 7 8
*      float hdense_A[15] = {1.0f, 0.0f, 6.0f, 2.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 0.0f, 8.0f};
*
*      rocsparse_int m         = 3;
*      rocsparse_int n         = 5;
*      rocsparse_direction dir = rocsparse_direction_row;
*
*      float* ddense_A = nullptr;
*      hipMalloc((void**)&ddense_A, sizeof(float) * m * n);
*      hipMemcpy(ddense_A, hdense_A, sizeof(float) * m * n, hipMemcpyHostToDevice);
*
*      // Allocate memory for the nnz_per_row_columns array
*      rocsparse_int* dnnz_per_row;
*      hipMalloc((void**)&dnnz_per_row, sizeof(rocsparse_int) * m);
*
*      rocsparse_int nnz_A;
*      rocsparse_snnz(handle, dir, m, n, descr, ddense_A, m, dnnz_per_row, &nnz_A);
*
*      // Copy result back to host
*      rocsparse_int hnnz_per_row[3];
*      hipMemcpy(hnnz_per_row, dnnz_per_row, sizeof(rocsparse_int) * m, hipMemcpyDeviceToHost);
*
*      hipFree(ddense_A);
*      hipFree(dnnz_per_row);
*
*      rocsparse_destroy_mat_descr(descr);
*      rocsparse_destroy_handle(handle);
*  \endcode
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
#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_NNZ_H */

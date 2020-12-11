/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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
 *  \brief rocsparse-auxiliary.h provides auxilary functions in rocsparse
 */

#pragma once
#ifndef _ROCSPARSE_AUXILIARY_H_
#define _ROCSPARSE_AUXILIARY_H_

#include <stdint.h>

#include "rocsparse-export.h"
#include "rocsparse-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a rocsparse handle
 *
 *  \details
 *  \p rocsparse_create_handle creates the rocSPARSE library context. It must be
 *  initialized before any other rocSPARSE API function is invoked and must be passed to
 *  all subsequent library function calls. The handle should be destroyed at the end
 *  using rocsparse_destroy_handle().
 *
 *  @param[out]
 *  handle  the pointer to the handle to the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the initialization succeeded.
 *  \retval rocsparse_status_invalid_handle \p handle pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_handle(rocsparse_handle* handle);

/*! \ingroup aux_module
 *  \brief Destroy a rocsparse handle
 *
 *  \details
 *  \p rocsparse_destroy_handle destroys the rocSPARSE library context and releases all
 *  resources used by the rocSPARSE library.
 *
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle);

/*! \ingroup aux_module
 *  \brief Specify user defined HIP stream
 *
 *  \details
 *  \p rocsparse_set_stream specifies the stream to be used by the rocSPARSE library
 *  context and all subsequent function calls.
 *
 *  @param[inout]
 *  handle  the handle to the rocSPARSE library context.
 *  @param[in]
 *  stream  the stream to be used by the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 *
 *  \par Example
 *  This example illustrates, how a user defined stream can be used in rocSPARSE.
 *  \code{.c}
 *      // Create rocSPARSE handle
 *      rocsparse_handle handle;
 *      rocsparse_create_handle(&handle);
 *
 *      // Create stream
 *      hipStream_t stream;
 *      hipStreamCreate(&stream);
 *
 *      // Set stream to rocSPARSE handle
 *      rocsparse_set_stream(handle, stream);
 *
 *      // Do some work
 *      // ...
 *
 *      // Clean up
 *      rocsparse_destroy_handle(handle);
 *      hipStreamDestroy(stream);
 *  \endcode
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_stream(rocsparse_handle handle, hipStream_t stream);

/*! \ingroup aux_module
 *  \brief Get current stream from library context
 *
 *  \details
 *  \p rocsparse_get_stream gets the rocSPARSE library context stream which is currently
 *  used for all subsequent function calls.
 *
 *  @param[in]
 *  handle the handle to the rocSPARSE library context.
 *  @param[out]
 *  stream the stream currently used by the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t* stream);

/*! \ingroup aux_module
 *  \brief Specify pointer mode
 *
 *  \details
 *  \p rocsparse_set_pointer_mode specifies the pointer mode to be used by the rocSPARSE
 *  library context and all subsequent function calls. By default, all values are passed
 *  by reference on the host. Valid pointer modes are \ref rocsparse_pointer_mode_host
 *  or \p rocsparse_pointer_mode_device.
 *
 *  @param[in]
 *  handle          the handle to the rocSPARSE library context.
 *  @param[in]
 *  pointer_mode    the pointer mode to be used by the rocSPARSE library context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_pointer_mode(rocsparse_handle       handle,
                                            rocsparse_pointer_mode pointer_mode);

/*! \ingroup aux_module
 *  \brief Get current pointer mode from library context
 *
 *  \details
 *  \p rocsparse_get_pointer_mode gets the rocSPARSE library context pointer mode which
 *  is currently used for all subsequent function calls.
 *
 *  @param[in]
 *  handle          the handle to the rocSPARSE library context.
 *  @param[out]
 *  pointer_mode    the pointer mode that is currently used by the rocSPARSE library
 *                  context.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_pointer_mode(rocsparse_handle        handle,
                                            rocsparse_pointer_mode* pointer_mode);

/*! \ingroup aux_module
 *  \brief Get rocSPARSE version
 *
 *  \details
 *  \p rocsparse_get_version gets the rocSPARSE library version number.
 *  - patch = version % 100
 *  - minor = version / 100 % 1000
 *  - major = version / 100000
 *
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
 *  @param[out]
 *  version the version number of the rocSPARSE library.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_version(rocsparse_handle handle, int* version);

/*! \ingroup aux_module
 *  \brief Get rocSPARSE git revision
 *
 *  \details
 *  \p rocsparse_get_git_rev gets the rocSPARSE library git commit revision (SHA-1).
 *
 *  @param[in]
 *  handle  the handle to the rocSPARSE library context.
 *  @param[out]
 *  rev     the git commit revision (SHA-1).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_handle \p handle is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_git_rev(rocsparse_handle handle, char* rev);

/*! \ingroup aux_module
 *  \brief Create a matrix descriptor
 *  \details
 *  \p rocsparse_create_mat_descr creates a matrix descriptor. It initializes
 *  \ref rocsparse_matrix_type to \ref rocsparse_matrix_type_general and
 *  \ref rocsparse_index_base to \ref rocsparse_index_base_zero. It should be destroyed
 *  at the end using rocsparse_destroy_mat_descr().
 *
 *  @param[out]
 *  descr   the pointer to the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_mat_descr(rocsparse_mat_descr* descr);

/*! \ingroup aux_module
 *  \brief Copy a matrix descriptor
 *  \details
 *  \p rocsparse_copy_mat_descr copies a matrix descriptor. Both, source and destination
 *  matrix descriptors must be initialized prior to calling \p rocsparse_copy_mat_descr.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix descriptor.
 *  @param[in]
 *  src     the pointer to the source matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_mat_descr(rocsparse_mat_descr dest, const rocsparse_mat_descr src);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p rocsparse_destroy_mat_descr destroys a matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_mat_descr(rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the index base of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_index_base sets the index base of a matrix descriptor. Valid
 *  options are \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p base is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_index_base(rocsparse_mat_descr descr, rocsparse_index_base base);

/*! \ingroup aux_module
 *  \brief Get the index base of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_index_base returns the index base of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 */
ROCSPARSE_EXPORT
rocsparse_index_base rocsparse_get_mat_index_base(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_type sets the matrix type of a matrix descriptor. Valid
 *  matrix types are \ref rocsparse_matrix_type_general,
 *  \ref rocsparse_matrix_type_symmetric, \ref rocsparse_matrix_type_hermitian or
 *  \ref rocsparse_matrix_type_triangular.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  type    \ref rocsparse_matrix_type_general, \ref rocsparse_matrix_type_symmetric,
 *          \ref rocsparse_matrix_type_hermitian or
 *          \ref rocsparse_matrix_type_triangular.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_type(rocsparse_mat_descr descr, rocsparse_matrix_type type);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_type returns the matrix type of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocsparse_matrix_type_general, \ref rocsparse_matrix_type_symmetric,
 *              \ref rocsparse_matrix_type_hermitian or
 *              \ref rocsparse_matrix_type_triangular.
 */
ROCSPARSE_EXPORT
rocsparse_matrix_type rocsparse_get_mat_type(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_fill_mode sets the matrix fill mode of a matrix descriptor.
 *  Valid fill modes are \ref rocsparse_fill_mode_lower or
 *  \ref rocsparse_fill_mode_upper.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  fill_mode   \ref rocsparse_fill_mode_lower or \ref rocsparse_fill_mode_upper.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p fill_mode is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_fill_mode(rocsparse_mat_descr descr,
                                             rocsparse_fill_mode fill_mode);

/*! \ingroup aux_module
 *  \brief Get the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_fill_mode returns the matrix fill mode of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocsparse_fill_mode_lower or \ref rocsparse_fill_mode_upper.
 */
ROCSPARSE_EXPORT
rocsparse_fill_mode rocsparse_get_mat_fill_mode(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_diag_type sets the matrix diagonal type of a matrix
 *  descriptor. Valid diagonal types are \ref rocsparse_diag_type_unit or
 *  \ref rocsparse_diag_type_non_unit.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  diag_type   \ref rocsparse_diag_type_unit or \ref rocsparse_diag_type_non_unit.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p diag_type is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_diag_type(rocsparse_mat_descr descr,
                                             rocsparse_diag_type diag_type);

/*! \ingroup aux_module
 *  \brief Get the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_diag_type returns the matrix diagonal type of a matrix
 *  descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref rocsparse_diag_type_unit or \ref rocsparse_diag_type_non_unit.
 */
ROCSPARSE_EXPORT
rocsparse_diag_type rocsparse_get_mat_diag_type(const rocsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Create a \p HYB matrix structure
 *
 *  \details
 *  \p rocsparse_create_hyb_mat creates a structure that holds the matrix in \p HYB
 *  storage format. It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *
 *  @param[inout]
 *  hyb the pointer to the hybrid matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p hyb pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb);

/*! \ingroup aux_module
 *  \brief Destroy a \p HYB matrix structure
 *
 *  \details
 *  \p rocsparse_destroy_hyb_mat destroys a \p HYB structure.
 *
 *  @param[in]
 *  hyb the hybrid matrix structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p hyb pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb);

/*! \ingroup aux_module
 *  \brief Create a matrix info structure
 *
 *  \details
 *  \p rocsparse_create_mat_info creates a structure that holds the matrix info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using rocsparse_destroy_mat_info().
 *
 *  @param[inout]
 *  info    the pointer to the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);

/*! \ingroup aux_module
 *  \brief Destroy a matrix info structure
 *
 *  \details
 *  \p rocsparse_destroy_mat_info destroys a matrix info structure.
 *
 *  @param[in]
 *  info    the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);

// Generic API

// SpVec
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_spvec_descr(rocsparse_spvec_descr* descr,
                                              int64_t                size,
                                              int64_t                nnz,
                                              void*                  indices,
                                              void*                  values,
                                              rocsparse_indextype    idx_type,
                                              rocsparse_index_base   idx_base,
                                              rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_spvec_descr descr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get(const rocsparse_spvec_descr descr,
                                     int64_t*                    size,
                                     int64_t*                    nnz,
                                     void**                      indices,
                                     void**                      values,
                                     rocsparse_indextype*        idx_type,
                                     rocsparse_index_base*       idx_base,
                                     rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get_index_base(const rocsparse_spvec_descr descr,
                                                rocsparse_index_base*       idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get_values(const rocsparse_spvec_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_set_values(rocsparse_spvec_descr descr, void* values);

// SpMat
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_coo_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  coo_row_ind,
                                            void*                  coo_col_ind,
                                            void*                  coo_val,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_coo_aos_descr(rocsparse_spmat_descr* descr,
                                                int64_t                rows,
                                                int64_t                cols,
                                                int64_t                nnz,
                                                void*                  coo_ind,
                                                void*                  coo_val,
                                                rocsparse_indextype    idx_type,
                                                rocsparse_index_base   idx_base,
                                                rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csr_row_ptr,
                                            void*                  csr_col_ind,
                                            void*                  csr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csc_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csc_col_ptr,
                                            void*                  csc_row_ind,
                                            void*                  csc_val,
                                            rocsparse_indextype    col_ptr_type,
                                            rocsparse_indextype    row_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_ell_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            void*                  ell_col_ind,
                                            void*                  ell_val,
                                            int64_t                ell_width,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_spmat_descr descr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_aos_get(const rocsparse_spmat_descr descr,
                                       int64_t*                    rows,
                                       int64_t*                    cols,
                                       int64_t*                    nnz,
                                       void**                      coo_ind,
                                       void**                      coo_val,
                                       rocsparse_indextype*        idx_type,
                                       rocsparse_index_base*       idx_base,
                                       rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      coo_row_ind,
                                   void**                      coo_col_ind,
                                   void**                      coo_val,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csr_row_ptr,
                                   void**                      csr_col_ind,
                                   void**                      csr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_ell_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   void**                      ell_col_ind,
                                   void**                      ell_val,
                                   int64_t*                    ell_width,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 coo_row_ind,
                                            void*                 coo_col_ind,
                                            void*                 coo_val);

ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_coo_aos_set_pointers(rocsparse_spmat_descr descr, void* coo_ind, void* coo_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csr_row_ptr,
                                            void*                 csr_col_ind,
                                            void*                 csr_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csc_col_ptr,
                                            void*                 csc_row_ind,
                                            void*                 csc_val);

ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_ell_set_pointers(rocsparse_spmat_descr descr, void* ell_col_ind, void* ell_val);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_size(rocsparse_spmat_descr descr,
                                          int64_t*              rows,
                                          int64_t*              cols,
                                          int64_t*              nnz);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_format(const rocsparse_spmat_descr descr,
                                            rocsparse_format*           format);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_index_base(const rocsparse_spmat_descr descr,
                                                rocsparse_index_base*       idx_base);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values);

// Dense vector
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr* descr,
                                              int64_t                size,
                                              void*                  values,
                                              rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_dnvec_descr descr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_get(const rocsparse_dnvec_descr descr,
                                     int64_t*                    size,
                                     void**                      values,
                                     rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_get_values(const rocsparse_dnvec_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_set_values(rocsparse_dnvec_descr descr, void* values);

// Dense matrix
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_dnmat_descr(rocsparse_dnmat_descr* descr,
                                              int64_t                rows,
                                              int64_t                cols,
                                              int64_t                ld,
                                              void*                  values,
                                              rocsparse_datatype     data_type,
                                              rocsparse_order        order);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_dnmat_descr(rocsparse_dnmat_descr descr);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get(const rocsparse_dnmat_descr descr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    ld,
                                     void**                      values,
                                     rocsparse_datatype*         data_type,
                                     rocsparse_order*            order);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get_values(const rocsparse_dnmat_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_set_values(rocsparse_dnmat_descr descr, void* values);

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSE_AUXILIARY_H_ */

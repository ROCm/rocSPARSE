/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_AUXILIARY_H
#define ROCSPARSE_AUXILIARY_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

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
 *  \brief Return the string representation of a rocSPARSE status code enum name
 *
 *  \details
 *  \p rocsparse_get_status_name takes a rocSPARSE status as input and returns the string representation of this status.
 *  If the status is not recognized, the function returns "Unrecognized status code"
 *
 *  @param[in]
 *  status  a rocSPARSE status
 *
 *  \retval pointer to null terminated string
 */
ROCSPARSE_EXPORT
const char* rocsparse_get_status_name(rocsparse_status status);

/*! \ingroup aux_module
 *  \brief Return the rocSPARSE status code description as a string
 *
 *  \details
 *  \p rocsparse_get_status_description takes a rocSPARSE status as input and returns the status description as a string.
 *  If the status is not recognized, the function returns "Unrecognized status code"
 *
 *  @param[in]
 *  status  a rocSPARSE status
 *
 *  \retval pointer to null terminated string
 */
ROCSPARSE_EXPORT
const char* rocsparse_get_status_description(rocsparse_status status);

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
 *  \brief Specify the matrix storage mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_set_mat_storage_mode sets the matrix storage mode of a matrix descriptor.
 *  Valid fill modes are \ref rocsparse_storage_mode_sorted or
 *  \ref rocsparse_storage_mode_unsorted.
 *
 *  @param[inout]
 *  descr           the matrix descriptor.
 *  @param[in]
 *  storage_mode    \ref rocsparse_storage_mode_sorted or
 *                  \ref rocsparse_storage_mode_unsorted.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocsparse_status_invalid_value \p storage_mode is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_storage_mode(rocsparse_mat_descr    descr,
                                                rocsparse_storage_mode storage_mode);

/*! \ingroup aux_module
 *  \brief Get the matrix storage mode of a matrix descriptor
 *
 *  \details
 *  \p rocsparse_get_mat_storage_mode returns the matrix storage mode of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref rocsparse_storage_mode_sorted or \ref rocsparse_storage_mode_unsorted.
 */
ROCSPARSE_EXPORT
rocsparse_storage_mode rocsparse_get_mat_storage_mode(const rocsparse_mat_descr descr);

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
 *  \brief Copy a \p HYB matrix structure
 *
 *  \details
 *  \p rocsparse_copy_hyb_mat copies a matrix info structure. Both, source and destination
 *  matrix info structure must be initialized prior to calling \p rocsparse_copy_hyb_mat.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix info structure.
 *  @param[in]
 *  src     the pointer to the source matrix info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p hyb pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_hyb_mat(rocsparse_hyb_mat dest, const rocsparse_hyb_mat src);

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
 *  \brief Copy a matrix info structure
 *  \details
 *  \p rocsparse_copy_mat_info copies a matrix info structure. Both, source and destination
 *  matrix info structure must be initialized prior to calling \p rocsparse_copy_mat_info.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix info structure.
 *  @param[in]
 *  src     the pointer to the source matrix info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_mat_info(rocsparse_mat_info dest, const rocsparse_mat_info src);

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

/*! \ingroup aux_module
 *  \brief Create a color info structure
 *
 *  \details
 *  \p rocsparse_create_color_info creates a structure that holds the color info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using rocsparse_destroy_color_info().
 *
 *  @param[inout]
 *  info    the pointer to the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_color_info(rocsparse_color_info* info);

/*! \ingroup aux_module
 *  \brief Copy a color info structure
 *  \details
 *  \p rocsparse_copy_color_info copies a color info structure. Both, source and destination
 *  color info structure must be initialized prior to calling \p rocsparse_copy_color_info.
 *
 *  @param[out]
 *  dest    the pointer to the destination color info structure.
 *  @param[in]
 *  src     the pointer to the source color info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_copy_color_info(rocsparse_color_info       dest,
                                           const rocsparse_color_info src);

/*! \ingroup aux_module
 *  \brief Destroy a color info structure
 *
 *  \details
 *  \p rocsparse_destroy_color_info destroys a color info structure.
 *
 *  @param[in]
 *  info    the info structure.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p info pointer is invalid.
 *  \retval rocsparse_status_internal_error an internal error occurred.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_color_info(rocsparse_color_info info);

// Generic API

/*! \ingroup aux_module
 *  \brief Create a sparse vector descriptor
 *  \details
 *  \p rocsparse_create_spvec_descr creates a sparse vector descriptor. It should be
 *  destroyed at the end using rocsparse_destroy_mat_descr().
 *
 *  @param[out]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  size   size of the sparse vector.
 *  @param[in]
 *  nnz   number of non-zeros in sparse vector.
 *  @param[in]
 *  indices   indices of the sparse vector where non-zeros occur (must be array of length \p nnz ).
 *  @param[in]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type   \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base   \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p indices or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_create_const_spvec_descr(rocsparse_const_spvec_descr* descr,
                                                    int64_t                      size,
                                                    int64_t                      nnz,
                                                    const void*                  indices,
                                                    const void*                  values,
                                                    rocsparse_indextype          idx_type,
                                                    rocsparse_index_base         idx_base,
                                                    rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a sparse vector descriptor
 *
 *  \details
 *  \p rocsparse_destroy_spvec_descr destroys a sparse vector descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_const_spvec_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse vector descriptor
 *  \details
 *  \p rocsparse_spvec_get gets the fields of the sparse vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[out]
 *  size   size of the sparse vector.
 *  @param[out]
 *  nnz   number of non-zeros in sparse vector.
 *  @param[out]
 *  indices   indices of the sparse vector where non-zeros occur (must be array of length \p nnz ).
 *  @param[out]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type   \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base   \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p indices or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_const_spvec_get(rocsparse_const_spvec_descr descr,
                                           int64_t*                    size,
                                           int64_t*                    nnz,
                                           const void**                indices,
                                           const void**                values,
                                           rocsparse_indextype*        idx_type,
                                           rocsparse_index_base*       idx_base,
                                           rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the index base stored in the sparse vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[out]
 *  idx_base   \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_base is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get_index_base(rocsparse_const_spvec_descr descr,
                                                rocsparse_index_base*       idx_base);

/*! \ingroup aux_module
 *  \brief Get the values array stored in the sparse vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[out]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_get_values(const rocsparse_spvec_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_spvec_get_values(rocsparse_const_spvec_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in the sparse vector descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  values   non-zero values in the sparse vector (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spvec_set_values(rocsparse_spvec_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Create a sparse COO matrix descriptor
 *  \details
 *  \p rocsparse_create_coo_descr creates a sparse COO matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse COO matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO matrix.
 *  @param[in]
 *  cols        number of columns in the COO matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO matrix.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_create_const_coo_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  coo_row_ind,
                                                  const void*                  coo_col_ind,
                                                  const void*                  coo_val,
                                                  rocsparse_indextype          idx_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse COO AoS matrix descriptor
 *  \details
 *  \p rocsparse_create_coo_aos_descr creates a sparse COO AoS matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse COO AoS matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the COO AoS matrix.
 *  @param[in]
 *  cols        number of columns in the COO AoS matrix
 *  @param[in]
 *  nnz         number of non-zeros in the COO AoS matrix.
 *  @param[in]
 *  coo_ind     <row, column> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
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

/*! \ingroup aux_module
 *  \brief Create a sparse BSR matrix descriptor
 *  \details
 *  \p rocsparse_create_bsr_descr creates a sparse BSR matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr        the pointer to the sparse BSR matrix descriptor.
 *  @param[in]
 *  mb           number of rows in the BSR matrix.
 *  @param[in]
 *  nb           number of columns in the BSR matrix
 *  @param[in]
 *  nnzb         number of non-zeros in the BSR matrix.
 *  @param[in]
 *  block_dir    direction of the internal block storage.
 *  @param[in]
 *  block_dim    dimension of the blocks.
 *  @param[in]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p mb+1 ).
 *  @param[in]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p nnzb ).
 *  @param[in]
 *  bsr_val      values of the BSR matrix (must be array of length \p nnzb * \p block_dim * \p block_dim ).
 *  @param[in]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p bsr_row_ptr or \p bsr_col_ind or \p bsr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p mb or \p nb or \p nnzb \p block_dim is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type or \p block_dir is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_bsr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                mb,
                                            int64_t                nb,
                                            int64_t                nnzb,
                                            rocsparse_direction    block_dir,
                                            int64_t                block_dim,
                                            void*                  bsr_row_ptr,
                                            void*                  bsr_col_ind,
                                            void*                  bsr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type);

/*! \ingroup aux_module
 *  \brief Create a sparse CSR matrix descriptor
 *  \details
 *  \p rocsparse_create_csr_descr creates a sparse CSR matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr        the pointer to the sparse CSR matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSR matrix.
 *  @param[in]
 *  cols         number of columns in the CSR matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_create_const_csr_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csr_row_ptr,
                                                  const void*                  csr_col_ind,
                                                  const void*                  csr_val,
                                                  rocsparse_indextype          row_ptr_type,
                                                  rocsparse_indextype          col_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse CSC matrix descriptor
 *  \details
 *  \p rocsparse_create_csc_descr creates a sparse CSC matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse CSC matrix descriptor.
 *  @param[in]
 *  rows         number of rows in the CSC matrix.
 *  @param[in]
 *  cols         number of columns in the CSC matrix
 *  @param[in]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[in]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  col_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  row_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_create_const_csc_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csc_col_ptr,
                                                  const void*                  csc_row_ind,
                                                  const void*                  csc_val,
                                                  rocsparse_indextype          col_ptr_type,
                                                  rocsparse_indextype          row_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Create a sparse ELL matrix descriptor
 *  \details
 *  \p rocsparse_create_ell_descr creates a sparse ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr       the pointer to the sparse ELL matrix descriptor.
 *  @param[in]
 *  rows        number of rows in the ELL matrix.
 *  @param[in]
 *  cols        number of columns in the ELL matrix
 *  @param[in]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_width   width of the ELL matrix.
 *  @param[in]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_width is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
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

/*! \ingroup aux_module
 *  \brief Create a sparse blocked ELL matrix descriptor
 *  \details
 *  \p rocsparse_create_bell_descr creates a sparse blocked ELL matrix descriptor. It should be
 *  destroyed at the end using \p rocsparse_destroy_spmat_descr.
 *
 *  @param[out]
 *  descr         the pointer to the sparse blocked ELL matrix descriptor.
 *  @param[in]
 *  rows          number of rows in the blocked ELL matrix.
 *  @param[in]
 *  cols          number of columns in the blocked ELL matrix
 *  @param[in]
 *  ell_block_dir \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[in]
 *  ell_block_dim block dimension of the sparse blocked ELL matrix.
 *  @param[in]
 *  ell_cols      column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_col_ind   column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val       values of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  idx_type      \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[in]
 *  idx_base      \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[in]
 *  data_type     \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_cols or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_bell_descr(rocsparse_spmat_descr* descr,
                                             int64_t                rows,
                                             int64_t                cols,
                                             rocsparse_direction    ell_block_dir,
                                             int64_t                ell_block_dim,
                                             int64_t                ell_cols,
                                             void*                  ell_col_ind,
                                             void*                  ell_val,
                                             rocsparse_indextype    idx_type,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_bell_descr(rocsparse_const_spmat_descr* descr,
                                                   int64_t                      rows,
                                                   int64_t                      cols,
                                                   rocsparse_direction          ell_block_dir,
                                                   int64_t                      ell_block_dim,
                                                   int64_t                      ell_cols,
                                                   const void*                  ell_col_ind,
                                                   const void*                  ell_val,
                                                   rocsparse_indextype          idx_type,
                                                   rocsparse_index_base         idx_base,
                                                   rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a sparse matrix descriptor
 *
 *  \details
 *  \p rocsparse_destroy_spmat_descr destroys a sparse matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_const_spmat_descr descr);

/*! \ingroup aux_module
*  \brief Sparse matrix to sparse matrix conversion.
*
*  \details
*  \p rocsparse_create_sparse_to_sparse_descr creates the descriptor of the sparse_to_sparse algorithm.

*  @param[out]
*  descr        pointer to the descriptor of the sparse_to_sparse algorithm.
*  @param[in]
*  source       source sparse matrix descriptor.
*  @param[in]
*  target       target sparse matrix descriptor.
*  @param[in]
*  alg          algorithm for the sparse_to_sparse computation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_value if any required enumeration is invalid.
*  \retval      rocsparse_status_invalid_pointer \p descr, \p source, or \p target
*               pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_sparse_to_sparse_descr(rocsparse_sparse_to_sparse_descr* descr,
                                                         rocsparse_const_spmat_descr       source,
                                                         rocsparse_spmat_descr             target,
                                                         rocsparse_sparse_to_sparse_alg    alg);

/*! \ingroup aux_module
*  \brief Sparse matrix to sparse matrix conversion.
*
*  \details
*  \p rocsparse_sparse_to_sparse_permissive allows the routine to allocate an intermediate sparse matrix
*  in order to perform the conversion. By default, the routine is not permissive.
*  @param[in]
*  descr        descriptor of the sparse_to_sparse algorithm.
*  \retval      rocsparse_status_success the operation completed successfully.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sparse_to_sparse_permissive(rocsparse_sparse_to_sparse_descr descr);

/*! \ingroup aux_module
*  \brief Sparse matrix to sparse matrix conversion.
*
*  \details
*  \p rocsparse_destroy_sparse_to_sparse_descr destroys the descriptor of the sparse_to_sparse algorithm.
*
*  @param[in]
*  descr        descriptor of the sparse_to_sparse algorithm.
*  \retval      rocsparse_status_success the operation completed successfully.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_sparse_to_sparse_descr(rocsparse_sparse_to_sparse_descr descr);

/*! \ingroup aux_module
*  \brief Sparse matrix extraction.
*
*  \details
*  \p rocsparse_create_extract_descr creates the descriptor of the extract algorithm.

*  @param[out]
*  descr        pointer to the descriptor of the extract algorithm.
*  @param[in]
*  source       source sparse matrix descriptor.
*  @param[in]
*  target       target sparse matrix descriptor.
*  @param[in]
*  alg          algorithm for the extract computation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_value if any required enumeration is invalid.
*  \retval      rocsparse_status_invalid_pointer \p descr, \p source, or \p target
*               pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_extract_descr(rocsparse_extract_descr*    descr,
                                                rocsparse_const_spmat_descr source,
                                                rocsparse_spmat_descr       target,
                                                rocsparse_extract_alg       alg);

/*! \ingroup aux_module
*  \brief Sparse matrix extraction.
*
*  \details
*  \p rocsparse_destroy_extract_descr destroys the descriptor of the extract algorithm.
*
*  @param[in]
*  descr        descriptor of the extract algorithm.
*  \retval      rocsparse_status_success the operation completed successfully.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_extract_descr(rocsparse_extract_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse COO matrix descriptor
 *  \details
 *  \p rocsparse_coo_get gets the fields of the sparse COO matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse COO matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse COO matrix.
 *  @param[out]
 *  cols        number of columns in the sparse COO matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse COO matrix.
 *  @param[out]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_const_coo_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                coo_row_ind,
                                         const void**                coo_col_ind,
                                         const void**                coo_val,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse COO AoS matrix descriptor
 *  \details
 *  \p rocsparse_coo_aos_get gets the fields of the sparse COO AoS matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse COO AoS matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse COO AoS matrix.
 *  @param[out]
 *  cols        number of columns in the sparse COO AoS matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse COO AoS matrix.
 *  @param[out]
 *  coo_ind     <row, columns> indices of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  coo_val     values of the COO AoS matrix (must be array of length \p nnz ).
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_const_coo_aos_get(rocsparse_const_spmat_descr descr,
                                             int64_t*                    rows,
                                             int64_t*                    cols,
                                             int64_t*                    nnz,
                                             const void**                coo_ind,
                                             const void**                coo_val,
                                             rocsparse_indextype*        idx_type,
                                             rocsparse_index_base*       idx_base,
                                             rocsparse_datatype*         data_type);

/**@}*/
/*! \ingroup aux_module
 *  \brief Get the fields of the sparse CSR matrix descriptor
 *  \details
 *  \p rocsparse_csr_get gets the fields of the sparse CSR matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse CSR matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSR matrix.
 *  @param[out]
 *  cols         number of columns in the CSR matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSR matrix.
 *  @param[out]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[out]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *  @param[out]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_const_csr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csr_row_ptr,
                                         const void**                csr_col_ind,
                                         const void**                csr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse CSC matrix descriptor
 *  \details
 *  \p rocsparse_csc_get gets the fields of the sparse CSC matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse CSC matrix descriptor.
 *  @param[out]
 *  rows         number of rows in the CSC matrix.
 *  @param[out]
 *  cols         number of columns in the CSC matrix
 *  @param[out]
 *  nnz          number of non-zeros in the CSC matrix.
 *  @param[out]
 *  csc_col_ptr  column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[out]
 *  csc_row_ind  row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  csc_val      values of the CSC matrix (must be array of length \p nnz ).
 *  @param[out]
 *  col_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  row_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p col_ptr_type or \p row_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csc_col_ptr,
                                   void**                      csc_row_ind,
                                   void**                      csc_val,
                                   rocsparse_indextype*        col_ptr_type,
                                   rocsparse_indextype*        row_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_csc_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csc_col_ptr,
                                         const void**                csc_row_ind,
                                         const void**                csc_val,
                                         rocsparse_indextype*        col_ptr_type,
                                         rocsparse_indextype*        row_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse ELL matrix descriptor
 *  \details
 *  \p rocsparse_ell_get gets the fields of the sparse ELL matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse ELL matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the ELL matrix.
 *  @param[out]
 *  cols        number of columns in the ELL matrix
 *  @param[out]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_width   width of the ELL matrix.
 *  @param[out]
 *  idx_type    \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_width is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
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
rocsparse_status rocsparse_const_ell_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         const void**                ell_col_ind,
                                         const void**                ell_val,
                                         int64_t*                    ell_width,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse blocked ELL matrix descriptor
 *  \details
 *  \p rocsparse_bell_get gets the fields of the sparse blocked ELL matrix descriptor
 *
 *  @param[in]
 *  descr         the pointer to the sparse blocked ELL matrix descriptor.
 *  @param[out]
 *  rows          number of rows in the blocked ELL matrix.
 *  @param[out]
 *  cols          number of columns in the blocked ELL matrix
 *  @param[out]
 *  ell_block_dir \ref rocsparse_direction_row or \ref rocsparse_direction_column.
 *  @param[out]
 *  ell_block_dim block dimension of the sparse blocked ELL matrix.
 *  @param[out]
 *  ell_cols      column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_col_ind   column indices of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  ell_val       values of the blocked ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[out]
 *  idx_type      \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base      \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type     \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *                \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_cols or \p ell_col_ind or \p ell_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ell_block_dim is invalid.
 *  \retval rocsparse_status_invalid_value if \p ell_block_dir or \p idx_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bell_get(const rocsparse_spmat_descr descr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    rocsparse_direction*        ell_block_dir,
                                    int64_t*                    ell_block_dim,
                                    int64_t*                    ell_cols,
                                    void**                      ell_col_ind,
                                    void**                      ell_val,
                                    rocsparse_indextype*        idx_type,
                                    rocsparse_index_base*       idx_base,
                                    rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_bell_get(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          rocsparse_direction*        ell_block_dir,
                                          int64_t*                    ell_block_dim,
                                          int64_t*                    ell_cols,
                                          const void**                ell_col_ind,
                                          const void**                ell_val,
                                          rocsparse_indextype*        idx_type,
                                          rocsparse_index_base*       idx_base,
                                          rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the fields of the sparse BSR matrix descriptor
 *  \details
 *  \p rocsparse_bsr_get gets the fields of the sparse BSR matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the sparse BSR matrix descriptor.
 *  @param[out]
 *  brows         number of rows in the BSR matrix.
 *  @param[out]
 *  bcols         number of columns in the BSR matrix
 *  @param[out]
 *  bnnz          number of non-zeros in the BSR matrix.
 *  @param[out]
 *  bdir          storage layout of the dense block matrices.
 *  @param[out]
 *  bdim          block dimension.
 *  @param[out]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p brows+1 ).
 *  @param[out]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p bnnz ).
 *  @param[out]
 *  bsr_val      values of the BSR matrix (must be array of length \p bnnz * \p bdim * \p bdim ).
 *  @param[out]
 *  row_ptr_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  col_ind_type \ref rocsparse_indextype_i32 or \ref rocsparse_indextype_i64.
 *  @param[out]
 *  idx_base     \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
 *  @param[out]
 *  data_type    \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *               \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csr_row_ptr or \p csr_col_ind or \p csr_val is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 *  \retval rocsparse_status_invalid_value if \p row_ptr_type or \p col_ind_type or \p idx_base or \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    brows,
                                   int64_t*                    bcols,
                                   int64_t*                    bnnz,
                                   rocsparse_direction*        bdir,
                                   int64_t*                    bdim,
                                   void**                      bsr_row_ptr,
                                   void**                      bsr_col_ind,
                                   void**                      bsr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_bsr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    brows,
                                         int64_t*                    bcols,
                                         int64_t*                    bnnz,
                                         rocsparse_direction*        bdir,
                                         int64_t*                    bdim,
                                         const void**                bsr_row_ptr,
                                         const void**                bsr_col_ind,
                                         const void**                bsr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the row indices, column indices and values array in the sparse COO matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  coo_row_ind row indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_col_ind column indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val     values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_row_ind or \p coo_col_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 coo_row_ind,
                                            void*                 coo_col_ind,
                                            void*                 coo_val);

/*! \ingroup aux_module
 *  \brief Set the <row, column> indices and values array in the sparse COO AoS matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  coo_ind <row, column> indices of the COO matrix (must be array of length \p nnz ).
 *  @param[in]
 *  coo_val values of the COO matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_coo_aos_set_pointers(rocsparse_spmat_descr descr, void* coo_ind, void* coo_val);

/*! \ingroup aux_module
 *  \brief Set the row offsets, column indices and values array in the sparse CSR matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  csr_row_ptr  row offsets of the CSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  csr_col_ind  column indices of the CSR matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csr_val      values of the CSR matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p coo_ind or \p coo_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csr_row_ptr,
                                            void*                 csr_col_ind,
                                            void*                 csr_val);

/*! \ingroup aux_module
 *  \brief Set the column offsets, row indices and values array in the sparse CSC matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse vector descriptor.
 *  @param[in]
 *  csc_col_ptr column offsets of the CSC matrix (must be array of length \p cols+1 ).
 *  @param[in]
 *  csc_row_ind row indices of the CSC matrix (must be array of length \p nnz ).
 *  @param[in]
 *  csc_val     values of the CSC matrix (must be array of length \p nnz ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p csc_col_ptr or \p csc_row_ind or \p csc_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csc_col_ptr,
                                            void*                 csc_row_ind,
                                            void*                 csc_val);

/*! \ingroup aux_module
 *  \brief Set the column indices and values array in the sparse ELL matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse vector descriptor.
 *  @param[in]
 *  ell_col_ind column indices of the ELL matrix (must be array of length \p rows*ell_width ).
 *  @param[in]
 *  ell_val     values of the ELL matrix (must be array of length \p rows*ell_width ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p ell_col_ind or \p ell_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_ell_set_pointers(rocsparse_spmat_descr descr, void* ell_col_ind, void* ell_val);

/*! \ingroup aux_module
 *  \brief Set the row offsets, column indices and values array in the sparse BSR matrix descriptor
 *
 *  @param[inout]
 *  descr   the pointer to the sparse vector descriptor.
 *  @param[in]
 *  bsr_row_ptr  row offsets of the BSR matrix (must be array of length \p rows+1 ).
 *  @param[in]
 *  bsr_col_ind  column indices of the BSR matrix (must be array of length \p nnzb ).
 *  @param[in]
 *  bsr_val      values of the BSR matrix (must be array of length \p nnzb*block_dim*block_dim ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p bsr_row_ptr or \p bsr_col_ind or \p bsr_val is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_bsr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 bsr_row_ptr,
                                            void*                 bsr_col_ind,
                                            void*                 bsr_val);

/*! \ingroup aux_module
 *  \brief Get the number of rows, columns and non-zeros from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  rows        number of rows in the sparse matrix.
 *  @param[out]
 *  cols        number of columns in the sparse matrix.
 *  @param[out]
 *  nnz         number of non-zeros in sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p nnz is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_size(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          int64_t*                    nnz);

/*! \ingroup aux_module
 *  \brief Get the sparse matrix format from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  format      \ref rocsparse_format_coo or \ref rocsparse_format_coo_aos or
 *              \ref rocsparse_format_csr or \ref rocsparse_format_csc or
 *              \ref rocsparse_format_ell
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p format is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_format(rocsparse_const_spmat_descr descr,
                                            rocsparse_format*           format);

/*! \ingroup aux_module
 *  \brief Get the sparse matrix index base from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  idx_base    \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_value if \p idx_base is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_index_base(rocsparse_const_spmat_descr descr,
                                                rocsparse_index_base*       idx_base);

/*! \ingroup aux_module
 *  \brief Get the values array from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr     the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  values    values array of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_spmat_get_values(rocsparse_const_spmat_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in the sparse matrix descriptor
 *
 *  @param[inout]
 *  descr     the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  values    values array of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Get the strided batch count from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[out]
 *  batch_count batch_count of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_strided_batch(rocsparse_const_spmat_descr descr,
                                                   rocsparse_int*              batch_count);

/*! \ingroup aux_module
 *  \brief Set the strided batch count in the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  batch_count batch_count of the sparse matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_strided_batch(rocsparse_spmat_descr descr,
                                                   rocsparse_int         batch_count);

/*! \ingroup aux_module
 *  \brief Set the batch count and batch stride in the sparse COO matrix descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the sparse COO matrix descriptor.
 *  @param[in]
 *  batch_count  batch_count of the sparse COO matrix.
 *  @param[in]
 *  batch_stride batch stride of the sparse COO matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_coo_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, row offset batch stride and the column indices batch stride in the sparse CSR matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the sparse CSR matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the sparse CSR matrix.
 *  @param[in]
 *  offsets_batch_stride        row offset batch stride of the sparse CSR matrix.
 *  @param[in]
 *  columns_values_batch_stride column indices batch stride of the sparse CSR matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p columns_values_batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csr_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               columns_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count, column offset batch stride and the row indices batch stride in the sparse CSC matrix descriptor
 *
 *  @param[inout]
 *  descr                       the pointer to the sparse CSC matrix descriptor.
 *  @param[in]
 *  batch_count                 batch_count of the sparse CSC matrix.
 *  @param[in]
 *  offsets_batch_stride        column offset batch stride of the sparse CSC matrix.
 *  @param[in]
 *  rows_values_batch_stride    row indices batch stride of the sparse CSC matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p offsets_batch_stride or \p rows_values_batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_csc_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               rows_values_batch_stride);

/*! \ingroup aux_module
 *  \brief Get the requested attribute data from the sparse matrix descriptor
 *
 *  @param[in]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  attribute \ref rocsparse_spmat_fill_mode or \ref rocsparse_spmat_diag_type or
 *            \ref rocsparse_spmat_matrix_type or \ref rocsparse_spmat_storage_mode
 *  @param[out]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p attribute is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_attribute(rocsparse_const_spmat_descr descr,
                                               rocsparse_spmat_attribute   attribute,
                                               void*                       data,
                                               size_t                      data_size);

/*! \ingroup aux_module
 *  \brief Set the requested attribute data in the sparse matrix descriptor
 *
 *  @param[inout]
 *  descr       the pointer to the sparse matrix descriptor.
 *  @param[in]
 *  attribute \ref rocsparse_spmat_fill_mode or \ref rocsparse_spmat_diag_type or
 *            \ref rocsparse_spmat_matrix_type or \ref rocsparse_spmat_storage_mode
 *  @param[in]
 *  data      attribute data
 *  @param[in]
 *  data_size attribute data size.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p data is invalid.
 *  \retval rocsparse_status_invalid_value if \p attribute is invalid.
 *  \retval rocsparse_status_invalid_size if \p data_size is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_attribute(rocsparse_spmat_descr     descr,
                                               rocsparse_spmat_attribute attribute,
                                               const void*               data,
                                               size_t                    data_size);

/*! \ingroup aux_module
 *  \brief Create a dense vector descriptor
 *  \details
 *  \p rocsparse_create_dnvec_descr creates a dense vector descriptor. It should be
 *  destroyed at the end using rocsparse_destroy_dnvec_descr().
 *
 *  @param[out]
 *  descr   the pointer to the dense vector descriptor.
 *  @param[in]
 *  size   size of the dense vector.
 *  @param[in]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *  @param[in]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr* descr,
                                              int64_t                size,
                                              void*                  values,
                                              rocsparse_datatype     data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_dnvec_descr(rocsparse_const_dnvec_descr* descr,
                                                    int64_t                      size,
                                                    const void*                  values,
                                                    rocsparse_datatype           data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a dense vector descriptor
 *
 *  \details
 *  \p rocsparse_destroy_dnvec_descr destroys a dense vector descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_const_dnvec_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the dense vector descriptor
 *  \details
 *  \p rocsparse_dnvec_get gets the fields of the dense vector descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense vector descriptor.
 *  @param[out]
 *  size   size of the dense vector.
 *  @param[out]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p size is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_get(const rocsparse_dnvec_descr descr,
                                     int64_t*                    size,
                                     void**                      values,
                                     rocsparse_datatype*         data_type);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnvec_get(rocsparse_const_dnvec_descr descr,
                                           int64_t*                    size,
                                           const void**                values,
                                           rocsparse_datatype*         data_type);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the values array from a dense vector descriptor
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *  @param[out]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_get_values(const rocsparse_dnvec_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnvec_get_values(rocsparse_const_dnvec_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in a dense vector descriptor
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  values   non-zero values in the dense vector (must be array of length \p size ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnvec_set_values(rocsparse_dnvec_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Create a dense matrix descriptor
 *  \details
 *  \p rocsparse_create_dnmat_descr creates a dense matrix descriptor. It should be
 *  destroyed at the end using rocsparse_destroy_dnmat_descr().
 *
 *  @param[out]
 *  descr     the pointer to the dense matrix descriptor.
 *  @param[in]
 *  rows      number of rows in the dense matrix.
 *  @param[in]
 *  cols      number of columns in the dense matrix.
 *  @param[in]
 *  ld        leading dimension of the dense matrix.
 *  @param[in]
 *  values    non-zero values in the dense vector (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *  @param[in]
 *  data_type \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *            \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *  @param[in]
 *  order     \ref rocsparse_order_row or \ref rocsparse_order_column.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ld is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type or \p order is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_dnmat_descr(rocsparse_dnmat_descr* descr,
                                              int64_t                rows,
                                              int64_t                cols,
                                              int64_t                ld,
                                              void*                  values,
                                              rocsparse_datatype     data_type,
                                              rocsparse_order        order);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_const_dnmat_descr(rocsparse_const_dnmat_descr* descr,
                                                    int64_t                      rows,
                                                    int64_t                      cols,
                                                    int64_t                      ld,
                                                    const void*                  values,
                                                    rocsparse_datatype           data_type,
                                                    rocsparse_order              order);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a dense matrix descriptor
 *
 *  \details
 *  \p rocsparse_destroy_dnmat_descr destroys a dense matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_dnmat_descr(rocsparse_const_dnmat_descr descr);

/*! \ingroup aux_module
 *  \brief Get the fields of the dense matrix descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense matrix descriptor.
 *  @param[out]
 *  rows   number of rows in the dense matrix.
 *  @param[out]
 *  cols   number of columns in the dense matrix.
 *  @param[out]
 *  ld        leading dimension of the dense matrix.
 *  @param[out]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *  @param[out]
 *  data_type   \ref rocsparse_datatype_f32_r, \ref rocsparse_datatype_f64_r,
 *              \ref rocsparse_datatype_f32_c or \ref rocsparse_datatype_f64_c.
 *  @param[out]
 *  order     \ref rocsparse_order_row or \ref rocsparse_order_column.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 *  \retval rocsparse_status_invalid_size if \p rows or \p cols or \p ld is invalid.
 *  \retval rocsparse_status_invalid_value if \p data_type or \p order is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get(const rocsparse_dnmat_descr descr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    ld,
                                     void**                      values,
                                     rocsparse_datatype*         data_type,
                                     rocsparse_order*            order);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnmat_get(rocsparse_const_dnmat_descr descr,
                                           int64_t*                    rows,
                                           int64_t*                    cols,
                                           int64_t*                    ld,
                                           const void**                values,
                                           rocsparse_datatype*         data_type,
                                           rocsparse_order*            order);
/**@}*/

/*! \ingroup aux_module
 *  \brief Get the values array from the dense matrix descriptor
 *
 *  @param[in]
 *  descr   the pointer to the dense matrix descriptor.
 *  @param[out]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr or \p values is invalid.
 */
/**@{*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get_values(const rocsparse_dnmat_descr descr, void** values);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_const_dnmat_get_values(rocsparse_const_dnmat_descr descr,
                                                  const void**                values);
/**@}*/

/*! \ingroup aux_module
 *  \brief Set the values array in a dense matrix descriptor
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  values    non-zero values in the dense matrix (must be array of length
 *            \p ld*rows if \p order=rocsparse_order_column or \p ld*cols if \p order=rocsparse_order_row ).
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer \p descr or \p values is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_set_values(rocsparse_dnmat_descr descr, void* values);

/*! \ingroup aux_module
 *  \brief Get the batch count and batch stride from the dense matrix descriptor
 *
 *  @param[in]
 *  descr        the pointer to the dense matrix descriptor.
 *  @param[out]
 *  batch_count  the batch count in the dense matrix.
 *  @param[out]
 *  batch_stride the batch stride in the dense matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_get_strided_batch(rocsparse_const_dnmat_descr descr,
                                                   rocsparse_int*              batch_count,
                                                   int64_t*                    batch_stride);

/*! \ingroup aux_module
 *  \brief Set the batch count and batch stride in the dense matrix descriptor
 *
 *  @param[inout]
 *  descr        the pointer to the dense matrix descriptor.
 *  @param[in]
 *  batch_count  the batch count in the dense matrix.
 *  @param[in]
 *  batch_stride the batch stride in the dense matrix.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p descr is invalid.
 *  \retval rocsparse_status_invalid_size if \p batch_count or \p batch_stride is invalid.
 */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_dnmat_set_strided_batch(rocsparse_dnmat_descr descr,
                                                   rocsparse_int         batch_count,
                                                   int64_t               batch_stride);

/*! \ingroup aux_module
   *  \brief Enable debug kernel launch.
   * \details If the debug kernel launch is enabled then hip errors are checked before and after every kernel launch.
   * \note This routine ignores the environment variable ROCSPARSE_DEBUG_KERNEL_LAUNCH.
   */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_kernel_launch();

/*! \ingroup aux_module
   *  \brief Disable debug kernel launch.
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_KERNEL_LAUNCH.
   */
ROCSPARSE_EXPORT void rocsparse_disable_debug_kernel_launch();

/*! \ingroup aux_module
   * \return 1 if enabled, 0 otherwise.
   */
ROCSPARSE_EXPORT int rocsparse_state_debug_kernel_launch();

/*! \ingroup aux_module
   *  \brief Enable debug arguments.
   * \details If the debug arguments is enabled then argument descriptors are internally available when an argument checking occurs. It provide information to the user depending of the setup of the verbosity
   * \ref rocsparse_enable_debug_arguments_verbose, \ref rocsparse_disable_debug_arguments_verbose and \ref rocsparse_state_debug_arguments_verbose.
   * \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS.
   * \note This routine enables debug arguments verbose with \ref rocsparse_enable_debug_arguments_verbose.
   */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_arguments();

/*! \ingroup aux_module
   *  \brief Disable debug arguments.
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS.
   *  \note This routines disables debug arguments verbose.
   */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_arguments();

/*! \ingroup aux_module
   * \return 1 if enabled, 0 otherwise.
   */
ROCSPARSE_EXPORT
int rocsparse_state_debug_arguments();

/*! \ingroup aux_module
   *  \brief Enable debug arguments verbose.
   *  \details The debug argument verbose displays information related to argument descriptors created from argument checking failures.
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE)
   */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_arguments_verbose();

/*! \ingroup aux_module
   *  \brief Disable debug arguments verbose.
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE)
   */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_arguments_verbose();

/*! \ingroup aux_module
 * \brief Get state of debug arguments verbose.
 * \return 1 if enabled, 0 otherwise.
 */
ROCSPARSE_EXPORT
int rocsparse_state_debug_arguments_verbose();

/*! \ingroup aux_module
   *  \brief Enable debug.
   * \details If the debug is enabled then code traces are generated when unsuccessful status returns occur. It provides information to the user depending of the set of the verbosity
   * (\ref rocsparse_enable_debug_verbose, \ref rocsparse_disable_debug_verbose and \ref rocsparse_state_debug_verbose).
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG.
   * \note \ref rocsparse_enable_debug_verbose and \ref rocsparse_enable_debug_arguments are called.
   */
ROCSPARSE_EXPORT
void rocsparse_enable_debug();

/*! \ingroup aux_module
   *  \brief Disable debug.
   *  \note This routine also disables debug arguments with \ref rocsparse_disable_debug_arguments.
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG.
   */
ROCSPARSE_EXPORT
void rocsparse_disable_debug();
/*! \ingroup aux_module
   * \brief Get state of  debug.
   * \return 1 if enabled, 0 otherwise.
   */
ROCSPARSE_EXPORT
int rocsparse_state_debug();

/*! \ingroup aux_module
   *  \brief Enable debug verbose.
   *  \details The debug verbose displays a stack of code traces showing where the code is handling a unsuccessful status.
   *  \note This routine enables debug arguments verbose with \ref rocsparse_enable_debug_arguments_verbose.
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_VERBOSE.
   */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_verbose();

/*! \ingroup aux_module
   *  \brief Disable debug verbose.
   *  \note This routine disables debug arguments verbose with  \ref rocsparse_disable_debug_arguments.
   *  \note This routine ignores the environment variable ROCSPARSE_DEBUG_VERBOSE.
   */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_verbose();
/*! \ingroup aux_module
   * \brief Get state of  debug verbose.
   * \return 1 if enabled, 0 otherwise.
   */
ROCSPARSE_EXPORT
int rocsparse_state_debug_verbose();

/*! \ingroup aux_module
   *  \brief Enable debug force host assert.
   *  \details The debug force host assert forces the evaluation of assert on host when the compiler directive NDEBUG is used.
   */
ROCSPARSE_EXPORT
void rocsparse_enable_debug_force_host_assert();

/*! \ingroup aux_module
   *  \brief Disable debug force host assert.
   */
ROCSPARSE_EXPORT
void rocsparse_disable_debug_force_host_assert();

/*! \ingroup aux_module
   * \brief Get state of  debug force host assert.
   * \return 1 if enabled, 0 otherwise.
   */
ROCSPARSE_EXPORT
int rocsparse_state_debug_force_host_assert();

//
// If ROCSPARSE_WITH_MEMSTAT is defined
// then a set of extra routines is offered
// to manage memory with a recording of some traces.
//
#ifdef ROCSPARSE_WITH_MEMSTAT
/*! \ingroup aux_module
   *  \brief Set the memory report filename.
   *
   *  \details
   *  \p rocsparse_memstat_report set the filename to use for the memory report.
   *  This routine is optional, but it must be called before any hip memory operation.
   *  Note that the default memory report filename is 'rocsparse_memstat.json'.
   *  Also note that if any operation occurs before calling this routine, the default filename rocsparse_memstat.json
   *  will be used but renamed after this call.
   *  The content of the memory report summarizes memory operations from the use of the routines
   *  \ref rocsparse_hip_malloc,
   *  \ref rocsparse_hip_free,
   *  \ref rocsparse_hip_host_malloc,
   *  \ref rocsparse_hip_host_free,
   *  \ref rocsparse_hip_host_managed,
   *  \ref rocsparse_hip_free_managed.
   *
   *  @param[in]
   *  filename  the memory report filename.
   *
   *  \retval rocsparse_status_success the operation succeeded.
   *  \retval rocsparse_status_invalid_pointer \p handle filename is an invalid pointer.
   *  \retval rocsparse_status_internal_error an internal error occurred.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_memstat_report(const char* filename);

/*! \ingroup aux_module
   *  \brief Wrap hipFree.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_free(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMalloc.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_malloc(void** mem, size_t nbytes, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipFreeAsync.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  stream  the stream to be used by the asynchronous operation
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_free_async(void* mem, hipStream_t stream, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMallocAsync.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  stream  the stream to be used by the asynchronous operation
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t
    rocsparse_hip_malloc_async(void** mem, size_t nbytes, hipStream_t stream, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipHostFree.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_host_free(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipHostMalloc.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_host_malloc(void** mem, size_t nbytes, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipFreeManaged.
   *
   *  @param[in]
   *  mem  memory pointer
   *  @param[in]
   *  tag  tag to attach to the operation.
   *
   *  \retval error from the related hip operation.
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_free_managed(void* mem, const char* tag);

/*! \ingroup aux_module
   *  \brief Wrap hipMallocManaged.
   *
   *  @param[in]
   *  mem  pointer of memory pointer
   *  @param[in]
   *  nbytes  number of bytes
   *  @param[in]
   *  tag  tag to attach to the operation
   *
   *  \retval error from the related hip operation
   */
ROCSPARSE_EXPORT
hipError_t rocsparse_hip_malloc_managed(void** mem, size_t nbytes, const char* tag);

#endif

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_AUXILIARY_H */

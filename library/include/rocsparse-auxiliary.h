/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/*!\file
 * \brief rocsparse-auxiliary.h provides auxilary functions in rocsparse
*/

#pragma once
#ifndef _ROCSPARSE_AUXILIARY_H_
#define _ROCSPARSE_AUXILIARY_H_

#include "rocsparse-types.h"
#include "rocsparse-export.h"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocsparse_handle is a structure holding the rocsparse library context.
 * It must be initialized using rocsparse_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparse_destroy_handle().
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_handle(rocsparse_handle *handle);

/********************************************************************************
 * \brief Destroy handle.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle);

/********************************************************************************
 * \brief Remove any streams from handle, and add one.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_stream(rocsparse_handle handle, hipStream_t stream);

/********************************************************************************
 * \brief Get stream [0] from handle.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t *stream);

/********************************************************************************
 * \brief Set rocsparse_pointer_mode.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_pointer_mode(rocsparse_handle handle,
                                            rocsparse_pointer_mode pointer_mode);

/********************************************************************************
 * \brief Get rocsparse_pointer_mode.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_pointer_mode(rocsparse_handle handle,
                                            rocsparse_pointer_mode *pointer_mode);

/********************************************************************************
 * \brief Get rocSPARSE version
 * version % 100        = patch level
 * version / 100 % 1000 = minor version
 * version / 100000     = major version
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_get_version(rocsparse_handle handle, int *version);

/********************************************************************************
 * \brief rocsparse_create_mat_descr_t is a structure holding the rocsparse matrix
 * descriptor. It must be initialized using rocsparse_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocsparse_destroy_mat_descr().
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_mat_descr(rocsparse_mat_descr *descrA);

/********************************************************************************
 * \brief Destroy the matrix descriptor.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_mat_descr(rocsparse_mat_descr descrA);

/********************************************************************************
 * \brief Set the index base of the matrix descriptor.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_index_base(rocsparse_mat_descr descrA,
                                              rocsparse_index_base base);

/********************************************************************************
 * \brief Returns the index base of the matrix descriptor.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_index_base rocsparse_get_mat_index_base(const rocsparse_mat_descr descrA);

/********************************************************************************
 * \brief Set the matrix type of the matrix descriptor.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_mat_type(rocsparse_mat_descr descrA,
                                        rocsparse_matrix_type type);

/********************************************************************************
 * \brief Returns the matrix type of the matrix descriptor.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_matrix_type rocsparse_get_mat_type(const rocsparse_mat_descr descrA);

/********************************************************************************
 * \brief rocsparse_create_hyb_mat is a structure holding the rocsparse HYB
 * matrix. It must be initialized using rocsparse_create_hyb_mat()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the HYB matrix.
 * It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat *hyb);

/********************************************************************************
 * \brief Destroy HYB matrix.
 *******************************************************************************/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb);

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSE_AUXILIARY_H_ */

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

#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime_api.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

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
rocsparse_status rocsparse_create_handle(rocsparse_handle* handle)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else
    {
        // Allocate
        try
        {
            *handle = new _rocsparse_handle();
            log_trace(*handle, "rocsparse_create_handle");
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle)
{
    log_trace(handle, "rocsparse_destroy_handle");
    // Destruct
    try
    {
        delete handle;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Indicates whether the scalar value pointers are on the host or device.
 * Set pointer mode, can be host or device
 *******************************************************************************/
rocsparse_status rocsparse_set_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode mode)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    handle->pointer_mode = mode;
    log_trace(handle, "rocsparse_set_pointer_mode", mode);
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Get pointer mode, can be host or device.
 *******************************************************************************/
rocsparse_status rocsparse_get_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode* mode)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    *mode = handle->pointer_mode;
    log_trace(handle, "rocsparse_get_pointer_mode", *mode);
    return rocsparse_status_success;
}

/********************************************************************************
 *! \brief Set rocsparse stream used for all subsequent library function calls.
 * If not set, all hip kernels will take the default NULL stream.
 *******************************************************************************/
rocsparse_status rocsparse_set_stream(rocsparse_handle handle, hipStream_t stream_id)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    log_trace(handle, "rocsparse_set_stream", stream_id);
    return handle->set_stream(stream_id);
}

/********************************************************************************
 *! \brief Get rocsparse stream used for all subsequent library function calls.
 *******************************************************************************/
rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t* stream_id)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    log_trace(handle, "rocsparse_get_stream", *stream_id);
    return handle->get_stream(stream_id);
}

/********************************************************************************
 * \brief Get rocSPARSE version
 * version % 100        = patch level
 * version / 100 % 1000 = minor version
 * version / 100000     = major version
 *******************************************************************************/
rocsparse_status rocsparse_get_version(rocsparse_handle handle, int* version)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    *version = ROCSPARSE_VERSION_MAJOR * 100000 + ROCSPARSE_VERSION_MINOR * 100
               + ROCSPARSE_VERSION_PATCH;

    log_trace(handle, "rocsparse_get_version", *version);

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Get rocSPARSE git revision
 *******************************************************************************/
rocsparse_status rocsparse_get_git_rev(rocsparse_handle handle, char* rev)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(rev == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    static constexpr char v[] = TO_STR(ROCSPARSE_VERSION_TWEAK);

    memcpy(rev, v, sizeof(v));

    log_trace(handle, "rocsparse_get_git_rev", rev);

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_mat_descr_t is a structure holding the rocsparse matrix
 * descriptor. It must be initialized using rocsparse_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocsparse_destroy_mat_descr().
 *******************************************************************************/
rocsparse_status rocsparse_create_mat_descr(rocsparse_mat_descr* descr)
{
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *descr = new _rocsparse_mat_descr;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief copy matrix descriptor
 *******************************************************************************/
rocsparse_status rocsparse_copy_mat_descr(rocsparse_mat_descr dest, const rocsparse_mat_descr src)
{
    if(dest == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(src == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    dest->type      = src->type;
    dest->fill_mode = src->fill_mode;
    dest->diag_type = src->diag_type;
    dest->base      = src->base;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocsparse_status rocsparse_destroy_mat_descr(rocsparse_mat_descr descr)
{
    // Destruct
    try
    {
        delete descr;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Set the index base of the matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_set_mat_index_base(rocsparse_mat_descr descr, rocsparse_index_base base)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(base != rocsparse_index_base_zero && base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    descr->base = base;
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Returns the index base of the matrix descriptor.
 *******************************************************************************/
rocsparse_index_base rocsparse_get_mat_index_base(const rocsparse_mat_descr descr)
{
    // If descriptor is invalid, default index base is returned
    if(descr == nullptr)
    {
        return rocsparse_index_base_zero;
    }
    return descr->base;
}

/********************************************************************************
 * \brief Set the matrix type of the matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_set_mat_type(rocsparse_mat_descr descr, rocsparse_matrix_type type)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(type != rocsparse_matrix_type_general && type != rocsparse_matrix_type_symmetric
       && type != rocsparse_matrix_type_hermitian)
    {
        return rocsparse_status_invalid_value;
    }
    descr->type = type;
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Returns the matrix type of the matrix descriptor.
 *******************************************************************************/
rocsparse_matrix_type rocsparse_get_mat_type(const rocsparse_mat_descr descr)
{
    // If descriptor is invalid, default matrix type is returned
    if(descr == nullptr)
    {
        return rocsparse_matrix_type_general;
    }
    return descr->type;
}

rocsparse_status rocsparse_set_mat_fill_mode(rocsparse_mat_descr descr,
                                             rocsparse_fill_mode fill_mode)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(fill_mode != rocsparse_fill_mode_lower && fill_mode != rocsparse_fill_mode_upper)
    {
        return rocsparse_status_invalid_value;
    }
    descr->fill_mode = fill_mode;
    return rocsparse_status_success;
}

rocsparse_fill_mode rocsparse_get_mat_fill_mode(const rocsparse_mat_descr descr)
{
    // If descriptor is invalid, default fill mode is returned
    if(descr == nullptr)
    {
        return rocsparse_fill_mode_lower;
    }
    return descr->fill_mode;
}

rocsparse_status rocsparse_set_mat_diag_type(rocsparse_mat_descr descr,
                                             rocsparse_diag_type diag_type)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(diag_type != rocsparse_diag_type_unit && diag_type != rocsparse_diag_type_non_unit)
    {
        return rocsparse_status_invalid_value;
    }
    descr->diag_type = diag_type;
    return rocsparse_status_success;
}

rocsparse_diag_type rocsparse_get_mat_diag_type(const rocsparse_mat_descr descr)
{
    // If descriptor is invalid, default diagonal type is returned
    if(descr == nullptr)
    {
        return rocsparse_diag_type_non_unit;
    }
    return descr->diag_type;
}

/********************************************************************************
 * \brief rocsparse_create_hyb_mat is a structure holding the rocsparse HYB
 * matrix. It must be initialized using rocsparse_create_hyb_mat()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the HYB matrix.
 * It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *******************************************************************************/
rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb)
{
    if(hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *hyb = new _rocsparse_hyb_mat;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Destroy HYB matrix.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb)
{
    // Destruct
    try
    {
        // Clean up ELL part
        if(hyb->ell_col_ind != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipFree(hyb->ell_col_ind));
        }
        if(hyb->ell_val != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipFree(hyb->ell_val));
        }

        // Clean up COO part
        if(hyb->coo_row_ind != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipFree(hyb->coo_row_ind));
        }
        if(hyb->coo_col_ind != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipFree(hyb->coo_col_ind));
        }
        if(hyb->coo_val != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipFree(hyb->coo_val));
        }

        delete hyb;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_mat_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info)
{
    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *info = new _rocsparse_mat_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Destroy mat info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info)
{
    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    // Uncouple shared meta data
    // TODO add more crossover data here
    if(info->csrsv_lower_info == info->csrilu0_info)
    {
        info->csrsv_lower_info = nullptr;
    }

    // Clear csrmv info struct
    if(info->csrmv_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrmv_info(info->csrmv_info));
    }

    // Clear csrilu0 info struct
    if(info->csrilu0_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrilu0_info));
    }

    // Clear csrsv upper info struct
    if(info->csrsv_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsv_upper_info));
    }

    // Clear csrsv lower info struct
    if(info->csrsv_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrtr_info(info->csrsv_lower_info));
    }

    // Clear csrgemm info struct
    if(info->csrgemm_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrgemm_info(info->csrgemm_info));
    }

    // Destruct
    try
    {
        delete info;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}

#ifdef __cplusplus
}
#endif

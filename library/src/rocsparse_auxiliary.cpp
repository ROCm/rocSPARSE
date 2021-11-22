/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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
#include "rocsparse/rocsparse.h"
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
        *handle = nullptr;
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
        *descr = nullptr;
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
    if(dest == nullptr || src == nullptr || dest == src)
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

    if(rocsparse_enum_utils::is_invalid(base))
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

    if(rocsparse_enum_utils::is_invalid(type))
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

    if(rocsparse_enum_utils::is_invalid(fill_mode))
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

    if(rocsparse_enum_utils::is_invalid(diag_type))
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
        *hyb = nullptr;
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
 * \brief Copy HYB matrix.
 *******************************************************************************/
rocsparse_status rocsparse_copy_hyb_mat(rocsparse_hyb_mat dest, const rocsparse_hyb_mat src)
{
    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }

    // check if destination already contains data. If it does, verify its allocated arrays are the same size as source
    bool previously_created = false;
    previously_created |= (dest->m != 0);
    previously_created |= (dest->n != 0);
    previously_created |= (dest->partition != rocsparse_hyb_partition_auto);
    previously_created |= (dest->ell_nnz != 0);
    previously_created |= (dest->ell_width != 0);
    previously_created |= (dest->ell_col_ind != nullptr);
    previously_created |= (dest->ell_val != nullptr);
    previously_created |= (dest->coo_nnz != 0);
    previously_created |= (dest->coo_row_ind != nullptr);
    previously_created |= (dest->coo_col_ind != nullptr);
    previously_created |= (dest->coo_val != nullptr);
    previously_created |= (dest->data_type_T != rocsparse_datatype_f32_r);

    if(previously_created)
    {
        // Sparsity pattern of dest and src must match
        bool invalid = false;
        invalid |= (dest->m != src->m);
        invalid |= (dest->n != src->n);
        invalid |= (dest->partition != src->partition);
        invalid |= (dest->ell_width != src->ell_width);
        invalid |= (dest->ell_nnz != src->ell_nnz);
        invalid |= (dest->coo_nnz != src->coo_nnz);
        invalid |= (dest->data_type_T != src->data_type_T);

        if(invalid)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    size_t T_size = sizeof(float);
    switch(src->data_type_T)
    {
    case rocsparse_datatype_f32_r:
    {
        T_size = sizeof(float);
        break;
    }
    case rocsparse_datatype_f64_r:
    {
        T_size = sizeof(double);
        break;
    }
    case rocsparse_datatype_f32_c:
    {
        T_size = sizeof(rocsparse_float_complex);
        break;
    }
    case rocsparse_datatype_f64_c:
    {
        T_size = sizeof(rocsparse_double_complex);
        break;
    }
    }

    if(src->ell_col_ind != nullptr)
    {
        if(dest->ell_col_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                hipMalloc((void**)&(dest->ell_col_ind), sizeof(rocsparse_int) * src->ell_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->ell_col_ind,
                                      src->ell_col_ind,
                                      sizeof(rocsparse_int) * src->ell_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->ell_val != nullptr)
    {
        if(dest->ell_val == nullptr)
        {
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&(dest->ell_val), T_size * src->ell_nnz));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->ell_val, src->ell_val, T_size * src->ell_nnz, hipMemcpyDeviceToDevice));
    }

    if(src->coo_row_ind != nullptr)
    {
        if(dest->coo_row_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                hipMalloc((void**)&(dest->coo_row_ind), sizeof(rocsparse_int) * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->coo_row_ind,
                                      src->coo_row_ind,
                                      sizeof(rocsparse_int) * src->coo_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->coo_col_ind != nullptr)
    {
        if(dest->coo_col_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                hipMalloc((void**)&(dest->coo_col_ind), sizeof(rocsparse_int) * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->coo_col_ind,
                                      src->coo_col_ind,
                                      sizeof(rocsparse_int) * src->coo_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->coo_val != nullptr)
    {
        if(dest->coo_val == nullptr)
        {
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&(dest->coo_val), T_size * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->coo_val, src->coo_val, T_size * src->coo_nnz, hipMemcpyDeviceToDevice));
    }

    dest->m           = src->m;
    dest->n           = src->n;
    dest->partition   = src->partition;
    dest->ell_width   = src->ell_width;
    dest->ell_nnz     = src->ell_nnz;
    dest->coo_nnz     = src->coo_nnz;
    dest->data_type_T = src->data_type_T;

    return rocsparse_status_success;
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
        *info = nullptr;
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
 * \brief Copy mat info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_mat_info(rocsparse_mat_info dest, const rocsparse_mat_info src)
{
    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_indextype index_type_J = rocsparse_indextype_u16;

    if(src->bsrsv_upper_info != nullptr)
    {
        index_type_J = src->bsrsv_upper_info->index_type_J;

        if(dest->bsrsv_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsv_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsv_upper_info, src->bsrsv_upper_info));
    }

    if(src->bsrsv_lower_info != nullptr)
    {
        index_type_J = src->bsrsv_lower_info->index_type_J;

        if(dest->bsrsv_lower_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsv_lower_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsv_lower_info, src->bsrsv_lower_info));
    }

    if(src->bsrsvt_upper_info != nullptr)
    {
        index_type_J = src->bsrsvt_upper_info->index_type_J;

        if(dest->bsrsvt_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsvt_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsvt_upper_info, src->bsrsvt_upper_info));
    }

    if(src->bsrsvt_lower_info != nullptr)
    {
        index_type_J = src->bsrsvt_lower_info->index_type_J;

        if(dest->bsrsvt_lower_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsvt_lower_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsvt_lower_info, src->bsrsvt_lower_info));
    }

    if(src->bsric0_info != nullptr)
    {
        index_type_J = src->bsric0_info->index_type_J;

        if(dest->bsric0_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsric0_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_copy_trm_info(dest->bsric0_info, src->bsric0_info));
    }

    if(src->bsrilu0_info != nullptr)
    {
        index_type_J = src->bsrilu0_info->index_type_J;

        if(dest->bsrilu0_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrilu0_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_copy_trm_info(dest->bsrilu0_info, src->bsrilu0_info));
    }

    if(src->bsrsm_upper_info != nullptr)
    {
        index_type_J = src->bsrsm_upper_info->index_type_J;

        if(dest->bsrsm_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsm_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsm_upper_info, src->bsrsm_upper_info));
    }

    if(src->bsrsm_lower_info != nullptr)
    {
        index_type_J = src->bsrsm_lower_info->index_type_J;

        if(dest->bsrsm_lower_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsm_lower_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsm_lower_info, src->bsrsm_lower_info));
    }

    if(src->bsrsmt_upper_info != nullptr)
    {
        index_type_J = src->bsrsmt_upper_info->index_type_J;

        if(dest->bsrsmt_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsmt_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsmt_upper_info, src->bsrsmt_upper_info));
    }

    if(src->bsrsmt_lower_info != nullptr)
    {
        index_type_J = src->bsrsmt_lower_info->index_type_J;

        if(dest->bsrsmt_lower_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->bsrsmt_lower_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->bsrsmt_lower_info, src->bsrsmt_lower_info));
    }

    if(src->csrmv_info != nullptr)
    {
        index_type_J = src->csrmv_info->index_type_J;

        if(dest->csrmv_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrmv_info(&dest->csrmv_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_copy_csrmv_info(dest->csrmv_info, src->csrmv_info));
    }

    if(src->csric0_info != nullptr)
    {
        index_type_J = src->csric0_info->index_type_J;

        if(dest->csric0_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csric0_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_copy_trm_info(dest->csric0_info, src->csric0_info));
    }

    if(src->csrilu0_info != nullptr)
    {
        index_type_J = src->csrilu0_info->index_type_J;

        if(dest->csrilu0_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrilu0_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_copy_trm_info(dest->csrilu0_info, src->csrilu0_info));
    }

    if(src->csrsv_upper_info != nullptr)
    {
        index_type_J = src->csrsv_upper_info->index_type_J;

        if(dest->csrsv_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrsv_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->csrsv_upper_info, src->csrsv_upper_info));
    }

    if(src->csrsv_lower_info != nullptr)
    {
        index_type_J = src->csrsv_lower_info->index_type_J;

        if(dest->csrsv_lower_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrsv_lower_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->csrsv_lower_info, src->csrsv_lower_info));
    }

    if(src->csrsvt_upper_info != nullptr)
    {
        index_type_J = src->csrsvt_upper_info->index_type_J;

        if(dest->csrsvt_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrsvt_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->csrsvt_upper_info, src->csrsvt_upper_info));
    }

    if(src->csrsm_upper_info != nullptr)
    {
        index_type_J = src->csrsm_upper_info->index_type_J;

        if(dest->csrsm_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrsm_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->csrsm_upper_info, src->csrsm_upper_info));
    }

    if(src->csrsm_lower_info != nullptr)
    {
        index_type_J = src->csrsm_lower_info->index_type_J;

        if(dest->csrsm_lower_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrsm_lower_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->csrsm_lower_info, src->csrsm_lower_info));
    }

    if(src->csrsmt_upper_info != nullptr)
    {
        index_type_J = src->csrsmt_upper_info->index_type_J;

        if(dest->csrsmt_upper_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrsmt_upper_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->csrsmt_upper_info, src->csrsmt_upper_info));
    }

    if(src->csrsmt_lower_info != nullptr)
    {
        index_type_J = src->csrsmt_lower_info->index_type_J;

        if(dest->csrsmt_lower_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_trm_info(&dest->csrsmt_lower_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_trm_info(dest->csrsmt_lower_info, src->csrsmt_lower_info));
    }

    if(src->csrgemm_info != nullptr)
    {
        if(dest->csrgemm_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csrgemm_info(&dest->csrgemm_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_copy_csrgemm_info(dest->csrgemm_info, src->csrgemm_info));
    }

    if(src->zero_pivot != nullptr)
    {
        // zero pivot for csrsv, csrsm, csrilu0, csric0
        size_t J_size = sizeof(uint16_t);
        switch(index_type_J)
        {
        case rocsparse_indextype_u16:
        {
            J_size = sizeof(uint16_t);
            break;
        }
        case rocsparse_indextype_i32:
        {
            J_size = sizeof(int32_t);
            break;
        }
        case rocsparse_indextype_i64:
        {
            J_size = sizeof(int64_t);
            break;
        }
        }

        if(dest->zero_pivot == nullptr)
        {
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&dest->zero_pivot, J_size));
        }

        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->zero_pivot, src->zero_pivot, J_size, hipMemcpyDeviceToDevice));
    }

    dest->boost_enable        = src->boost_enable;
    dest->use_double_prec_tol = src->use_double_prec_tol;
    dest->boost_tol           = src->boost_tol;
    dest->boost_val           = src->boost_val;

    return rocsparse_status_success;
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
    if(info->bsrsv_lower_info == info->bsrilu0_info || info->bsrsv_lower_info == info->bsric0_info
       || info->bsrsv_lower_info == info->bsrsm_lower_info)
    {
        info->bsrsv_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->bsrsm_lower_info == info->bsrilu0_info || info->bsrsm_lower_info == info->bsric0_info)
    {
        info->bsrsm_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->bsrilu0_info == info->bsric0_info)
    {
        info->bsrilu0_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsv_lower_info == info->csrilu0_info || info->csrsv_lower_info == info->csric0_info
       || info->csrsv_lower_info == info->csrsm_lower_info)
    {
        info->csrsv_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsm_lower_info == info->csrilu0_info || info->csrsm_lower_info == info->csric0_info)
    {
        info->csrsm_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrilu0_info == info->csric0_info)
    {
        info->csrilu0_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsv_upper_info == info->csrsm_upper_info)
    {
        info->csrsv_upper_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->bsrsv_upper_info == info->bsrsm_upper_info)
    {
        info->bsrsv_upper_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsvt_lower_info == info->csrsmt_lower_info)
    {
        info->csrsvt_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->bsrsvt_lower_info == info->bsrsmt_lower_info)
    {
        info->bsrsvt_lower_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->csrsvt_upper_info == info->csrsmt_upper_info)
    {
        info->csrsvt_upper_info = nullptr;
    }

    // Uncouple shared meta data
    if(info->bsrsvt_upper_info == info->bsrsmt_upper_info)
    {
        info->bsrsvt_upper_info = nullptr;
    }

    // Clear csrmv info struct
    if(info->csrmv_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrmv_info(info->csrmv_info));
    }

    // Clear bsrsvt upper info struct
    if(info->bsrsvt_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsvt_upper_info));
    }

    // Clear bsrsvt lower info struct
    if(info->bsrsvt_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsvt_lower_info));
    }

    // Clear bsric0 info struct
    if(info->bsric0_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsric0_info));
    }

    // Clear bsrilu0 info struct
    if(info->bsrilu0_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrilu0_info));
    }

    // Clear csrsvt upper info struct
    if(info->csrsvt_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsvt_upper_info));
    }

    // Clear csrsvt lower info struct
    if(info->csrsvt_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsvt_lower_info));
    }

    // Clear csrsmt upper info struct
    if(info->csrsmt_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsmt_upper_info));
    }

    // Clear csrsmt lower info struct
    if(info->csrsmt_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsmt_lower_info));
    }

    // Clear bsrsmt upper info struct
    if(info->bsrsmt_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsmt_upper_info));
    }

    // Clear bsrsmt lower info struct
    if(info->bsrsmt_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsmt_lower_info));
    }

    // Clear csric0 info struct
    if(info->csric0_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csric0_info));
    }

    // Clear csrilu0 info struct
    if(info->csrilu0_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrilu0_info));
    }

    // Clear bsrsv upper info struct
    if(info->bsrsv_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsv_upper_info));
    }

    // Clear bsrsv lower info struct
    if(info->bsrsv_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsv_lower_info));
    }

    // Clear csrsv upper info struct
    if(info->csrsv_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsv_upper_info));
    }

    // Clear csrsv lower info struct
    if(info->csrsv_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsv_lower_info));
    }

    // Clear csrsm upper info struct
    if(info->csrsm_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsm_upper_info));
    }

    // Clear csrsm lower info struct
    if(info->csrsm_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->csrsm_lower_info));
    }

    // Clear bsrsm upper info struct
    if(info->bsrsm_upper_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsm_upper_info));
    }

    // Clear bsrsm lower info struct
    if(info->bsrsm_lower_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_trm_info(info->bsrsm_lower_info));
    }

    // Clear csrgemm info struct
    if(info->csrgemm_info != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrgemm_info(info->csrgemm_info));
    }

    // Clear zero pivot
    if(info->zero_pivot != nullptr)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->zero_pivot));
        info->zero_pivot = nullptr;
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

/********************************************************************************
 * \brief rocsparse_color_info is a structure holding the color info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_color_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_color_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_color_info(rocsparse_color_info* info)
{
    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        *info = nullptr;
        // Allocate
        try
        {
            *info = new _rocsparse_color_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Copy color info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_color_info(rocsparse_color_info       dest,
                                           const rocsparse_color_info src)
{
    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Destroy color info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_color_info(rocsparse_color_info info)
{
    if(info == nullptr)
    {
        return rocsparse_status_success;
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

/********************************************************************************
 * \brief rocsparse_create_spvec_descr creates a descriptor holding the sparse
 * vector data, sizes and properties. It must be called prior to all subsequent
 * library function calls that involve sparse vectors. It should be destroyed at
 * the end using rocsparse_destroy_spvec_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_spvec_descr(rocsparse_spvec_descr* descr,
                                              int64_t                size,
                                              int64_t                nnz,
                                              void*                  indices,
                                              void*                  values,
                                              rocsparse_indextype    idx_type,
                                              rocsparse_index_base   idx_base,
                                              rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(idx_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid sizes
    if(size < 0 || nnz < 0 || size < nnz)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointers
    if(nnz != 0 && (indices == nullptr || values == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_spvec_descr;

        (*descr)->init = true;

        (*descr)->size = size;
        (*descr)->nnz  = nnz;

        (*descr)->idx_data = indices;
        (*descr)->val_data = values;

        (*descr)->idx_type  = idx_type;
        (*descr)->data_type = data_type;

        (*descr)->idx_base = idx_base;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_destroy_spvec_descr destroys a sparse vector descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_spvec_descr descr)
{
    // Check for valid handle
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        // Do nothing
        return rocsparse_status_success;
    }

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
 * \brief rocsparse_spvec_get returns the sparse vector matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get(const rocsparse_spvec_descr descr,
                                     int64_t*                    size,
                                     int64_t*                    nnz,
                                     void**                      indices,
                                     void**                      values,
                                     rocsparse_indextype*        idx_type,
                                     rocsparse_index_base*       idx_base,
                                     rocsparse_datatype*         data_type)
{
    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(size == nullptr || nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(indices == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(idx_type == nullptr || idx_base == nullptr || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *size = descr->size;
    *nnz  = descr->nnz;

    *indices = descr->idx_data;
    *values  = descr->val_data;

    *idx_type  = descr->idx_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spvec_get_index_base returns the sparse vector index base.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get_index_base(const rocsparse_spvec_descr descr,
                                                rocsparse_index_base*       idx_base)
{
    // Check for valid pointers
    if(descr == nullptr || idx_base == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *idx_base = descr->idx_base;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spvec_get_values returns the sparse vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get_values(const rocsparse_spvec_descr descr, void** values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *values = descr->val_data;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spvec_set_values sets the sparse vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_set_values(rocsparse_spvec_descr descr, void* values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    descr->val_data = values;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_coo_descr creates a descriptor holding the COO matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_coo_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  coo_row_ind,
                                            void*                  coo_col_ind,
                                            void*                  coo_val,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(idx_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid sizes
    if(rows < 0 || cols < 0 || nnz < 0 || nnz > rows * cols)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointers
    if(rows > 0 && cols > 0 && nnz > 0
       && (coo_row_ind == nullptr || coo_col_ind == nullptr || coo_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_spmat_descr;

        (*descr)->init = true;

        (*descr)->rows = rows;
        (*descr)->cols = cols;
        (*descr)->nnz  = nnz;

        (*descr)->row_data = coo_row_ind;
        (*descr)->col_data = coo_col_ind;
        (*descr)->val_data = coo_val;

        (*descr)->row_type  = idx_type;
        (*descr)->col_type  = idx_type;
        (*descr)->data_type = data_type;

        (*descr)->idx_base = idx_base;
        (*descr)->format   = rocsparse_format_coo;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

        // Initialize descriptor
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_coo_aos_descr creates a descriptor holding the COO matrix
 * data, sizes and properties where the row pointer and column indices are stored
 * using array of structure (AoS) format. It must be called prior to all subsequent
 * library function calls that involve sparse matrices. It should be destroyed at
 * the end using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_coo_aos_descr(rocsparse_spmat_descr* descr,
                                                int64_t                rows,
                                                int64_t                cols,
                                                int64_t                nnz,
                                                void*                  coo_ind,
                                                void*                  coo_val,
                                                rocsparse_indextype    idx_type,
                                                rocsparse_index_base   idx_base,
                                                rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(idx_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid sizes
    if(rows < 0 || cols < 0 || nnz < 0 || nnz > rows * cols)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointers
    if(rows > 0 && cols > 0 && nnz > 0 && (coo_ind == nullptr || coo_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_spmat_descr;

        (*descr)->init = true;

        (*descr)->rows = rows;
        (*descr)->cols = cols;
        (*descr)->nnz  = nnz;

        (*descr)->ind_data = coo_ind;
        (*descr)->val_data = coo_val;

        (*descr)->row_type  = idx_type;
        (*descr)->col_type  = idx_type;
        (*descr)->data_type = data_type;

        (*descr)->idx_base = idx_base;
        (*descr)->format   = rocsparse_format_coo_aos;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

        // Initialize descriptor
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_csr_descr creates a descriptor holding the CSR matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
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
                                            rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(row_ptr_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(col_ind_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid sizes
    if(rows < 0 || cols < 0 || nnz < 0 || nnz > rows * cols)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointers
    if(rows > 0 && csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csr_col_ind == nullptr || csr_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_spmat_descr;

        (*descr)->init = true;

        (*descr)->rows = rows;
        (*descr)->cols = cols;
        (*descr)->nnz  = nnz;

        (*descr)->row_data = csr_row_ptr;
        (*descr)->col_data = csr_col_ind;
        (*descr)->val_data = csr_val;

        (*descr)->row_type  = row_ptr_type;
        (*descr)->col_type  = col_ind_type;
        (*descr)->data_type = data_type;

        (*descr)->idx_base = idx_base;
        (*descr)->format   = rocsparse_format_csr;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

        // Initialize descriptor
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_csc_descr creates a descriptor holding the CSC matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
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
                                            rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(col_ptr_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(row_ind_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid sizes
    if(rows < 0 || cols < 0 || nnz < 0 || nnz > rows * cols)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointers
    if(cols > 0 && csc_col_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz != 0 && (csc_row_ind == nullptr || csc_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_spmat_descr;

        (*descr)->init = true;

        (*descr)->rows = rows;
        (*descr)->cols = cols;
        (*descr)->nnz  = nnz;

        (*descr)->row_data = csc_row_ind;
        (*descr)->col_data = csc_col_ptr;
        (*descr)->val_data = csc_val;

        (*descr)->row_type  = row_ind_type;
        (*descr)->col_type  = col_ptr_type;
        (*descr)->data_type = data_type;

        (*descr)->idx_base = idx_base;
        (*descr)->format   = rocsparse_format_csc;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

        // Initialize descriptor
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_ell_descr creates a descriptor holding the ELL matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_ell_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            void*                  ell_col_ind,
                                            void*                  ell_val,
                                            int64_t                ell_width,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(idx_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid sizes
    if(rows < 0 || cols < 0 || ell_width < 0 || ell_width > cols)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointers
    if(rows > 0 && cols > 0 && ell_width > 0 && (ell_col_ind == nullptr || ell_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_spmat_descr;

        (*descr)->init = true;

        (*descr)->rows      = rows;
        (*descr)->cols      = cols;
        (*descr)->ell_width = ell_width;

        (*descr)->col_data = ell_col_ind;
        (*descr)->val_data = ell_val;

        (*descr)->row_type  = idx_type;
        (*descr)->col_type  = idx_type;
        (*descr)->data_type = data_type;

        (*descr)->idx_base = idx_base;
        (*descr)->format   = rocsparse_format_ell;

        //
        // This is not really the number of non-zeros.
        // TODO: refactor the descriptors and having a proper design (get_nnz and different implementation for different format).
        // ell_width = nnz / rows.
        //
        (*descr)->nnz = ell_width * rows;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

        // Initialize descriptor
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_bell_descr creates a descriptor holding the
 * BLOCKED ELL matrix data, sizes and properties. It must be called prior to all
 * subsequent library function calls that involve sparse matrices.
 * It should be destroyed at the end using rocsparse_destroy_spmat_descr().
 * All data pointers remain valid.
 *******************************************************************************/
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
                                             rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(ell_block_dir))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid sizes
    if(rows < 0 || cols < 0 || ell_cols < 0 || ell_cols > cols || ell_block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointers
    if(rows > 0 && cols > 0 && ell_cols > 0 && (ell_col_ind == nullptr || ell_val == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_spmat_descr;

        (*descr)->init = true;

        (*descr)->rows = rows;
        (*descr)->cols = cols;

        (*descr)->ell_cols  = ell_cols;
        (*descr)->block_dir = ell_block_dir;
        (*descr)->block_dim = ell_block_dim;

        (*descr)->col_data = ell_col_ind;
        (*descr)->val_data = ell_val;

        (*descr)->row_type  = idx_type;
        (*descr)->col_type  = idx_type;
        (*descr)->data_type = data_type;

        (*descr)->idx_base = idx_base;
        (*descr)->format   = rocsparse_format_bell;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

        // Initialize descriptor
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_destroy_spmat_descr destroys a sparse matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_spmat_descr descr)
{
    // Check for valid handle
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        // Do nothing
        return rocsparse_status_success;
    }

    // Destruct
    try
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr->descr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(descr->info));

        delete descr;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_coo_get returns the sparse COO matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      coo_row_ind,
                                   void**                      coo_col_ind,
                                   void**                      coo_val,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
{
    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(rows == nullptr || cols == nullptr || nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(coo_row_ind == nullptr || coo_col_ind == nullptr || coo_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(idx_type == nullptr || idx_base == nullptr || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_row_ind = descr->row_data;
    *coo_col_ind = descr->col_data;
    *coo_val     = descr->val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_coo_aos_get returns the sparse COO (AoS) matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_coo_aos_get(const rocsparse_spmat_descr descr,
                                       int64_t*                    rows,
                                       int64_t*                    cols,
                                       int64_t*                    nnz,
                                       void**                      coo_ind,
                                       void**                      coo_val,
                                       rocsparse_indextype*        idx_type,
                                       rocsparse_index_base*       idx_base,
                                       rocsparse_datatype*         data_type)
{
    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(rows == nullptr || cols == nullptr || nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(coo_ind == nullptr || coo_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(idx_type == nullptr || idx_base == nullptr || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_ind = descr->ind_data;
    *coo_val = descr->val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_csr_get returns the sparse CSR matrix data, sizes and
 * properties.
 *******************************************************************************/
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
                                   rocsparse_datatype*         data_type)
{
    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(rows == nullptr || cols == nullptr || nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(csr_row_ptr == nullptr || csr_col_ind == nullptr || csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(row_ptr_type == nullptr || col_ind_type == nullptr || idx_base == nullptr
       || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csr_row_ptr = descr->row_data;
    *csr_col_ind = descr->col_data;
    *csr_val     = descr->val_data;

    *row_ptr_type = descr->row_type;
    *col_ind_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_ell_get returns the sparse ELL matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_ell_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   void**                      ell_col_ind,
                                   void**                      ell_val,
                                   int64_t*                    ell_width,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
{
    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(rows == nullptr || cols == nullptr || ell_width == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(ell_col_ind == nullptr || ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(idx_type == nullptr || idx_base == nullptr || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *rows = descr->rows;
    *cols = descr->cols;

    *ell_col_ind = descr->col_data;
    *ell_val     = descr->val_data;
    *ell_width   = descr->ell_width;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_bell_get returns the sparse BLOCKED ELL matrix data,
 * sizes and properties.
 *******************************************************************************/
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
                                    rocsparse_datatype*         data_type)
{
    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(rows == nullptr || cols == nullptr || ell_cols == nullptr || ell_block_dim == nullptr
       || ell_block_dir == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(ell_col_ind == nullptr || ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(idx_type == nullptr || idx_base == nullptr || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *rows = descr->rows;
    *cols = descr->cols;

    *ell_col_ind   = descr->col_data;
    *ell_val       = descr->val_data;
    *ell_cols      = descr->ell_cols;
    *ell_block_dir = descr->block_dir;
    *ell_block_dim = descr->block_dim;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_coo_set_pointers sets the sparse COO matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 coo_row_ind,
                                            void*                 coo_col_ind,
                                            void*                 coo_val)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid pointers
    if(coo_row_ind == nullptr || coo_col_ind == nullptr || coo_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    descr->row_data = coo_row_ind;
    descr->col_data = coo_col_ind;
    descr->val_data = coo_val;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_coo_aos_set_pointers sets the sparse COO (AoS) matrix data pointers.
 *******************************************************************************/
rocsparse_status
    rocsparse_coo_aos_set_pointers(rocsparse_spmat_descr descr, void* coo_ind, void* coo_val)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid pointers
    if(coo_ind == nullptr || coo_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    descr->ind_data = coo_ind;
    descr->val_data = coo_val;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_csr_set_pointers sets the sparse CSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csr_row_ptr,
                                            void*                 csr_col_ind,
                                            void*                 csr_val)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid pointers
    if(csr_row_ptr == nullptr || csr_col_ind == nullptr || csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->analysed = false;

    descr->row_data = csr_row_ptr;
    descr->col_data = csr_col_ind;
    descr->val_data = csr_val;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_csc_set_pointers sets the sparse CSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csc_col_ptr,
                                            void*                 csc_row_ind,
                                            void*                 csc_val)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid pointers
    if(csc_col_ptr == nullptr || csc_row_ind == nullptr || csc_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->analysed = false;

    descr->row_data = csc_row_ind;
    descr->col_data = csc_col_ptr;
    descr->val_data = csc_val;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_ell_set_pointers sets the sparse ELL matrix data pointers.
 *******************************************************************************/
rocsparse_status
    rocsparse_ell_set_pointers(rocsparse_spmat_descr descr, void* ell_col_ind, void* ell_val)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid pointers
    if(ell_col_ind == nullptr || ell_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    descr->col_data = ell_col_ind;
    descr->val_data = ell_val;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spmat_get_size returns the sparse matrix sizes.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_size(rocsparse_spmat_descr descr,
                                          int64_t*              rows,
                                          int64_t*              cols,
                                          int64_t*              nnz)
{
    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(rows == nullptr || cols == nullptr || nnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spmat_get_format returns the sparse matrix format.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_format(const rocsparse_spmat_descr descr,
                                            rocsparse_format*           format)
{
    // Check for valid pointers
    if(descr == nullptr || format == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *format = descr->format;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spmat_get_index_base returns the sparse matrix index base.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_index_base(const rocsparse_spmat_descr descr,
                                                rocsparse_index_base*       idx_base)
{
    // Check for valid pointers
    if(descr == nullptr || idx_base == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *idx_base = descr->idx_base;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spmat_get_values returns the sparse matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *values = descr->val_data;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spmat_set_values sets the sparse matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    descr->val_data = values;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_spmat_get_attribute gets the sparse matrix attribute.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_attribute(rocsparse_spmat_descr     descr,
                                               rocsparse_spmat_attribute attribute,
                                               void*                     data,
                                               size_t                    data_size)
{
    if(descr == nullptr || data == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    switch(attribute)
    {
    case rocsparse_spmat_fill_mode:
    {
        if(data_size != sizeof(rocsparse_spmat_fill_mode))
        {
            return rocsparse_status_invalid_size;
        }
        rocsparse_fill_mode* uplo = reinterpret_cast<rocsparse_fill_mode*>(data);
        *uplo                     = rocsparse_get_mat_fill_mode(descr->descr);
        return rocsparse_status_success;
    }
    case rocsparse_spmat_diag_type:
    {
        if(data_size != sizeof(rocsparse_spmat_diag_type))
        {
            return rocsparse_status_invalid_size;
        }
        rocsparse_diag_type* uplo = reinterpret_cast<rocsparse_diag_type*>(data);
        *uplo                     = rocsparse_get_mat_diag_type(descr->descr);
        return rocsparse_status_success;
    }
    case rocsparse_spmat_matrix_type:
    {
        if(data_size != sizeof(rocsparse_spmat_matrix_type))
        {
            return rocsparse_status_invalid_size;
        }
        rocsparse_matrix_type* matrix = reinterpret_cast<rocsparse_matrix_type*>(data);
        *matrix                       = rocsparse_get_mat_type(descr->descr);
        return rocsparse_status_success;
    }
    }

    return rocsparse_status_invalid_value;
}

/********************************************************************************
 * \brief rocsparse_spmat_get_attribute sets the sparse matrix attribute.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_attribute(rocsparse_spmat_descr     descr,
                                               rocsparse_spmat_attribute attribute,
                                               const void*               data,
                                               size_t                    data_size)
{
    if(descr == nullptr || data == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    switch(attribute)
    {
    case rocsparse_spmat_fill_mode:
    {
        if(data_size != sizeof(rocsparse_spmat_fill_mode))
        {
            return rocsparse_status_invalid_size;
        }
        rocsparse_fill_mode uplo = *reinterpret_cast<const rocsparse_fill_mode*>(data);
        return rocsparse_set_mat_fill_mode(descr->descr, uplo);
    }
    case rocsparse_spmat_diag_type:
    {
        if(data_size != sizeof(rocsparse_spmat_diag_type))
        {
            return rocsparse_status_invalid_size;
        }
        rocsparse_diag_type diag = *reinterpret_cast<const rocsparse_diag_type*>(data);
        return rocsparse_set_mat_diag_type(descr->descr, diag);
    }

    case rocsparse_spmat_matrix_type:
    {
        if(data_size != sizeof(rocsparse_spmat_matrix_type))
        {
            return rocsparse_status_invalid_size;
        }
        rocsparse_matrix_type matrix = *reinterpret_cast<const rocsparse_matrix_type*>(data);
        return rocsparse_set_mat_type(descr->descr, matrix);
    }
    }

    return rocsparse_status_invalid_value;
}

/********************************************************************************
 * \brief rocsparse_create_dnvec_descr creates a descriptor holding the dense
 * vector data, size and properties. It must be called prior to all subsequent
 * library function calls that involve the dense vector. It should be destroyed
 * at the end using rocsparse_destroy_dnvec_descr(). The data pointer remains
 * valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr* descr,
                                              int64_t                size,
                                              void*                  values,
                                              rocsparse_datatype     data_type)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid size
    if(size < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointer
    if(size > 0 && values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_dnvec_descr;

        (*descr)->init = true;

        (*descr)->size      = size;
        (*descr)->values    = values;
        (*descr)->data_type = data_type;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_destroy_dnvec_descr destroys a dense vector descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_dnvec_descr descr)
{
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

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
 * \brief rocsparse_dnvec_get returns the dense vector data, size and properties.
 *******************************************************************************/
rocsparse_status rocsparse_dnvec_get(const rocsparse_dnvec_descr descr,
                                     int64_t*                    size,
                                     void**                      values,
                                     rocsparse_datatype*         data_type)
{
    // Check for valid pointers
    if(descr == nullptr || size == nullptr || values == nullptr || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *size      = descr->size;
    *values    = descr->values;
    *data_type = descr->data_type;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_dnvec_get_values returns the dense vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnvec_get_values(const rocsparse_dnvec_descr descr, void** values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *values = descr->values;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_dnvec_set_values sets the dense vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnvec_set_values(rocsparse_dnvec_descr descr, void* values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    descr->values = values;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_create_dnmat_descr creates a descriptor holding the dense
 * matrix data, size and properties. It must be called prior to all subsequent
 * library function calls that involve the dense matrix. It should be destroyed
 * at the end using rocsparse_destroy_dnmat_descr(). The data pointer remains
 * valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_dnmat_descr(rocsparse_dnmat_descr* descr,
                                              int64_t                rows,
                                              int64_t                cols,
                                              int64_t                ld,
                                              void*                  values,
                                              rocsparse_datatype     data_type,
                                              rocsparse_order        order)
{
    // Check for valid descriptor
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(data_type))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid size
    if(rows < 0 || cols < 0 || ld < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid pointer
    if(rows > 0 && cols > 0 && ld > 0 && values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    *descr = nullptr;
    // Allocate
    try
    {
        *descr = new _rocsparse_dnmat_descr;

        (*descr)->init = true;

        (*descr)->rows      = rows;
        (*descr)->cols      = cols;
        (*descr)->ld        = ld;
        (*descr)->values    = values;
        (*descr)->data_type = data_type;
        (*descr)->order     = order;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_destroy_dnmat_descr destroys a dense matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_dnmat_descr(rocsparse_dnmat_descr descr)
{
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

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
 * \brief rocsparse_dnmat_get returns the dense matrix data, size and properties.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_get(const rocsparse_dnmat_descr descr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    ld,
                                     void**                      values,
                                     rocsparse_datatype*         data_type,
                                     rocsparse_order*            order)
{
    // Check for valid pointers
    if(descr == nullptr || rows == nullptr || cols == nullptr || ld == nullptr || values == nullptr
       || data_type == nullptr || order == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *rows      = descr->rows;
    *cols      = descr->cols;
    *ld        = descr->ld;
    *values    = descr->values;
    *data_type = descr->data_type;
    *order     = descr->order;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_dnmat_get_values returns the dense matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_get_values(const rocsparse_dnmat_descr descr, void** values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *values = descr->values;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_dnmat_set_values sets the dense matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_set_values(rocsparse_dnmat_descr descr, void* values)
{
    // Check for valid pointers
    if(descr == nullptr || values == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    descr->values = values;

    return rocsparse_status_success;
}

#ifdef __cplusplus
}
#endif

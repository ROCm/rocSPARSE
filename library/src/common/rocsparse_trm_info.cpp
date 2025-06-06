/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_trm_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_memstat.hpp"

/********************************************************************************
 * \brief rocsparse_trm_info is a structure holding the rocsparse bsrsv, csrsv,
 * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
 * csrilu0_analysis and csric0_analysis. It must be initialized using the
 * create_trm_info() routine. It should be destroyed at the end
 * using destroy_trm_info().
 *******************************************************************************/
rocsparse_status rocsparse::create_trm_info(rocsparse_trm_info* info)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *info = new _rocsparse_trm_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Copy trm info.
 *******************************************************************************/
rocsparse_status rocsparse::copy_trm_info(rocsparse_trm_info dest, const rocsparse_trm_info src)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }

    // check if destination already contains data. If it does, verify its allocated arrays are the same size as source
    bool previously_created = false;
    previously_created |= (dest->max_nnz != 0);

    previously_created |= (dest->row_map != nullptr);
    previously_created |= (dest->trm_diag_ind != nullptr);
    previously_created |= (dest->trmt_perm != nullptr);
    previously_created |= (dest->trmt_row_ptr != nullptr);
    previously_created |= (dest->trmt_col_ind != nullptr);

    previously_created |= (dest->m != 0);
    previously_created |= (dest->nnz != 0);
    previously_created |= (dest->descr != nullptr);
    previously_created |= (dest->trm_row_ptr != nullptr);
    previously_created |= (dest->trm_col_ind != nullptr);
    previously_created |= (dest->index_type_I != rocsparse_indextype_u16);
    previously_created |= (dest->index_type_J != rocsparse_indextype_u16);

    if(previously_created)
    {
        // Sparsity pattern of dest and src must match
        bool invalid = false;
        invalid |= (dest->max_nnz != src->max_nnz);
        invalid |= (dest->m != src->m);
        invalid |= (dest->nnz != src->nnz);
        invalid |= (dest->index_type_I != src->index_type_I);
        invalid |= (dest->index_type_J != src->index_type_J);

        if(invalid)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    size_t I_size = sizeof(uint16_t);
    switch(src->index_type_I)
    {
    case rocsparse_indextype_u16:
    {
        I_size = sizeof(uint16_t);
        break;
    }
    case rocsparse_indextype_i32:
    {
        I_size = sizeof(int32_t);
        break;
    }
    case rocsparse_indextype_i64:
    {
        I_size = sizeof(int64_t);
        break;
    }
    }

    size_t J_size = sizeof(uint16_t);
    switch(src->index_type_J)
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

    if(src->row_map != nullptr)
    {
        if(dest->row_map == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc((void**)&(dest->row_map), J_size * src->m));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->row_map, src->row_map, J_size * src->m, hipMemcpyDeviceToDevice));
    }

    if(src->trm_diag_ind != nullptr)
    {
        if(dest->trm_diag_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc((void**)&(dest->trm_diag_ind), I_size * src->m));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->trm_diag_ind, src->trm_diag_ind, I_size * src->m, hipMemcpyDeviceToDevice));
    }

    if(src->trmt_perm != nullptr)
    {
        if(dest->trmt_perm == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc((void**)&(dest->trmt_perm), I_size * src->nnz));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->trmt_perm, src->trmt_perm, I_size * src->nnz, hipMemcpyDeviceToDevice));
    }

    if(src->trmt_row_ptr != nullptr)
    {
        if(dest->trmt_row_ptr == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc((void**)&(dest->trmt_row_ptr), I_size * (src->m + 1)));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->trmt_row_ptr, src->trmt_row_ptr, I_size * (src->m + 1), hipMemcpyDeviceToDevice));
    }

    if(src->trmt_col_ind != nullptr)
    {
        if(dest->trmt_col_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc((void**)&(dest->trmt_col_ind), J_size * src->nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->trmt_col_ind, src->trmt_col_ind, J_size * src->nnz, hipMemcpyDeviceToDevice));
    }

    dest->max_nnz      = src->max_nnz;
    dest->m            = src->m;
    dest->nnz          = src->nnz;
    dest->index_type_I = src->index_type_I;
    dest->index_type_J = src->index_type_J;

    // Not owned by the info struct. Just pointers to externally allocated memory
    dest->descr       = src->descr;
    dest->trm_row_ptr = src->trm_row_ptr;
    dest->trm_col_ind = src->trm_col_ind;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Destroy trm info.
 *******************************************************************************/
rocsparse_status rocsparse::destroy_trm_info(rocsparse_trm_info info)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    // Clean up
    if(info->row_map != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->row_map));
        info->row_map = nullptr;
    }

    if(info->trm_diag_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->trm_diag_ind));
        info->trm_diag_ind = nullptr;
    }

    // Clear trmt arrays
    if(info->trmt_perm != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->trmt_perm));
        info->trmt_perm = nullptr;
    }

    if(info->trmt_row_ptr != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->trmt_row_ptr));
        info->trmt_row_ptr = nullptr;
    }

    if(info->trmt_col_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->trmt_col_ind));
        info->trmt_col_ind = nullptr;
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

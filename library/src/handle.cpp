/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "handle.h"
#include "definitions.h"
#include "logging.h"
#include "utility.h"

#include <hip/hip_runtime.h>

ROCSPARSE_KERNEL void init_kernel(){};

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocsparse_handle::_rocsparse_handle()
{
    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));

    // Device wavefront size
    wavefront_size = properties.warpSize;

#if HIP_VERSION >= 307
    // ASIC revision
    asic_rev = properties.asicRevision;
#else
    asic_rev = 0;
#endif

    // Layer mode
    char* str_layer_mode;
    if((str_layer_mode = getenv("ROCSPARSE_LAYER")) == NULL)
    {
        layer_mode = rocsparse_layer_mode_none;
    }
    else
    {
        layer_mode = (rocsparse_layer_mode)(atoi(str_layer_mode));
    }

    // Obtain size for coomv device buffer
    rocsparse_int nthreads = properties.maxThreadsPerBlock;
    rocsparse_int nprocs   = 2 * properties.multiProcessorCount;
    rocsparse_int nblocks  = (nprocs * nthreads - 1) / 256 + 1;

    size_t coomv_size = (((sizeof(rocsparse_int) + 16) * nblocks - 1) / 256 + 1) * 256;

    // Allocate device buffer
    buffer_size = (coomv_size > 1024 * 1024) ? coomv_size : 1024 * 1024;
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

    // Device one
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&sone, sizeof(float)));
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&done, sizeof(double)));
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&cone, sizeof(rocsparse_float_complex)));
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&zone, sizeof(rocsparse_double_complex)));

    // Execute empty kernel for initialization
    hipLaunchKernelGGL(init_kernel, dim3(1), dim3(1), 0, stream);

    // Execute memset for initialization
    THROW_IF_HIP_ERROR(hipMemsetAsync(sone, 0, sizeof(float), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(done, 0, sizeof(double), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(cone, 0, sizeof(rocsparse_float_complex), stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(zone, 0, sizeof(rocsparse_double_complex), stream));

    float  hsone = 1.0f;
    double hdone = 1.0;

    rocsparse_float_complex  hcone = rocsparse_float_complex(1.0f, 0.0f);
    rocsparse_double_complex hzone = rocsparse_double_complex(1.0, 0.0);

    THROW_IF_HIP_ERROR(hipMemcpyAsync(sone, &hsone, sizeof(float), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(done, &hdone, sizeof(double), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        cone, &hcone, sizeof(rocsparse_float_complex), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(hipMemcpyAsync(
        zone, &hzone, sizeof(rocsparse_double_complex), hipMemcpyHostToDevice, stream));

    // Wait for device transfer to finish
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Open log file
    if(layer_mode & rocsparse_layer_mode_log_trace)
    {
        open_log_stream(&log_trace_os, &log_trace_ofs, "ROCSPARSE_LOG_TRACE_PATH");
    }

    // Open log_bench file
    if(layer_mode & rocsparse_layer_mode_log_bench)
    {
        open_log_stream(&log_bench_os, &log_bench_ofs, "ROCSPARSE_LOG_BENCH_PATH");
    }

    // Open log_debug file
    if(layer_mode & rocsparse_layer_mode_log_debug)
    {
        open_log_stream(&log_debug_os, &log_debug_ofs, "ROCSPARSE_LOG_DEBUG_PATH");
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocsparse_handle::~_rocsparse_handle()
{
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(buffer));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(sone));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(done));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(cone));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(zone));

    // Close log files
    if(log_trace_ofs.is_open())
    {
        log_trace_ofs.close();
    }
    if(log_bench_ofs.is_open())
    {
        log_bench_ofs.close();
    }
    if(log_debug_ofs.is_open())
    {
        log_debug_ofs.close();
    }
}

/*******************************************************************************
 * Exactly like cuSPARSE, rocSPARSE only uses one stream for one API routine
 ******************************************************************************/

/*******************************************************************************
 * set stream:
   This API assumes user has already created a valid stream
   Associate the following rocsparse API call with this user provided stream
 ******************************************************************************/
rocsparse_status _rocsparse_handle::set_stream(hipStream_t user_stream)
{
    // TODO check if stream is valid
    stream = user_stream;
    return rocsparse_status_success;
}

/*******************************************************************************
 * get stream
 ******************************************************************************/
rocsparse_status _rocsparse_handle::get_stream(hipStream_t* user_stream) const
{
    *user_stream = stream;
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * rocsparse_create_csrmv_info() routine. It should be destroyed at the end
 * using rocsparse_destroy_csrmv_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrmv_info(rocsparse_csrmv_info* info)
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
            *info = new _rocsparse_csrmv_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Copy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_csrmv_info(rocsparse_csrmv_info       dest,
                                           const rocsparse_csrmv_info src)
{
    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }

    // check if destination already contains data. If it does, verify its allocated arrays are the same size as source
    bool previously_created = false;
    previously_created |= (dest->size != 0);

    previously_created |= (dest->row_blocks != nullptr);
    previously_created |= (dest->wg_flags != nullptr);
    previously_created |= (dest->wg_ids != nullptr);

    previously_created |= (dest->trans != rocsparse_operation_none);
    previously_created |= (dest->m != 0);
    previously_created |= (dest->n != 0);
    previously_created |= (dest->nnz != 0);
    previously_created |= (dest->max_rows != 0);
    previously_created |= (dest->descr != nullptr);
    previously_created |= (dest->csr_row_ptr != nullptr);
    previously_created |= (dest->csr_col_ind != nullptr);
    previously_created |= (dest->index_type_I != rocsparse_indextype_u16);
    previously_created |= (dest->index_type_J != rocsparse_indextype_u16);

    if(previously_created)
    {
        // Sparsity pattern of dest and src must match
        bool invalid = false;
        invalid |= (dest->size != src->size);
        invalid |= (dest->trans != src->trans);
        invalid |= (dest->m != src->m);
        invalid |= (dest->n != src->n);
        invalid |= (dest->nnz != src->nnz);
        invalid |= (dest->max_rows != src->max_rows);
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

    if(src->row_blocks != nullptr)
    {
        if(dest->row_blocks == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc((void**)&dest->row_blocks, I_size * src->size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(
            dest->row_blocks, src->row_blocks, I_size * src->size, hipMemcpyDeviceToDevice));
    }

    if(src->wg_flags != nullptr)
    {
        if(dest->wg_flags == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc((void**)&dest->wg_flags, sizeof(unsigned int) * src->size));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->wg_flags,
                                      src->wg_flags,
                                      sizeof(unsigned int) * src->size,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->wg_ids != nullptr)
    {
        if(dest->wg_ids == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc((void**)&dest->wg_ids, J_size * src->size));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->wg_ids, src->wg_ids, J_size * src->size, hipMemcpyDeviceToDevice));
    }

    dest->size         = src->size;
    dest->trans        = src->trans;
    dest->m            = src->m;
    dest->n            = src->n;
    dest->nnz          = src->nnz;
    dest->max_rows     = src->max_rows;
    dest->index_type_I = src->index_type_I;
    dest->index_type_J = src->index_type_J;

    // Not owned by the info struct. Just pointers to externally allocated memory
    dest->descr       = src->descr;
    dest->csr_row_ptr = src->csr_row_ptr;
    dest->csr_col_ind = src->csr_col_ind;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrmv_info(rocsparse_csrmv_info info)
{
    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    // Clean up row blocks
    if(info->size > 0)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->row_blocks));
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->wg_flags));
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->wg_ids));
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
 * \brief rocsparse_trm_info is a structure holding the rocsparse bsrsv, csrsv,
 * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
 * csrilu0_analysis and csric0_analysis. It must be initialized using the
 * rocsparse_create_trm_info() routine. It should be destroyed at the end
 * using rocsparse_destroy_trm_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_trm_info(rocsparse_trm_info* info)
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
rocsparse_status rocsparse_copy_trm_info(rocsparse_trm_info dest, const rocsparse_trm_info src)
{
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
rocsparse_status rocsparse_destroy_trm_info(rocsparse_trm_info info)
{
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

/********************************************************************************
 * \brief rocsparse_check_trm_shared checks if the given trm info structure
 * shares its meta data with another trm info structure.
 *******************************************************************************/
bool rocsparse_check_trm_shared(const rocsparse_mat_info info, rocsparse_trm_info trm)
{
    if(info == nullptr)
    {
        return false;
    }

    int shared = -1;

    if(trm == info->bsrsv_lower_info)
        ++shared;
    if(trm == info->bsrsv_upper_info)
        ++shared;
    if(trm == info->bsrsvt_lower_info)
        ++shared;
    if(trm == info->bsrsvt_upper_info)
        ++shared;
    if(trm == info->bsrilu0_info)
        ++shared;
    if(trm == info->bsric0_info)
        ++shared;
    if(trm == info->csrilu0_info)
        ++shared;
    if(trm == info->csric0_info)
        ++shared;
    if(trm == info->csrsv_lower_info)
        ++shared;
    if(trm == info->csrsv_upper_info)
        ++shared;
    if(trm == info->csrsvt_lower_info)
        ++shared;
    if(trm == info->csrsvt_upper_info)
        ++shared;
    if(trm == info->csrsm_lower_info)
        ++shared;
    if(trm == info->csrsm_upper_info)
        ++shared;
    if(trm == info->bsrsm_lower_info)
        ++shared;
    if(trm == info->bsrsm_upper_info)
        ++shared;

    return (shared > 0) ? true : false;
}

/********************************************************************************
 * \brief rocsparse_csrgemm_info is a structure holding the rocsparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the rocsparse_create_csrgemm_info() routine. It should be destroyed at the
 * end using rocsparse_destroy_csrgemm_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrgemm_info(rocsparse_csrgemm_info* info)
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
            *info = new _rocsparse_csrgemm_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Copy csrgemm info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_csrgemm_info(rocsparse_csrgemm_info       dest,
                                             const rocsparse_csrgemm_info src)
{
    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }

    dest->mul = src->mul;
    dest->add = src->add;

    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Destroy csrgemm info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrgemm_info(rocsparse_csrgemm_info info)
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
 * \brief rocsparse_csritsv_info is a structure holding the rocsparse csritsv
 * info data gathered during csritsv_buffer_size. It must be initialized using
 * the rocsparse_create_csritsv_info() routine. It should be destroyed at the
 * end using rocsparse_destroy_csritsv_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csritsv_info(rocsparse_csritsv_info* info)
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
            *info = new _rocsparse_csritsv_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Copy csritsv info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_csritsv_info(rocsparse_csritsv_info       dest,
                                             const rocsparse_csritsv_info src)
{
    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }
    dest->is_submatrix      = src->is_submatrix;
    dest->ptr_end_size      = src->ptr_end_size;
    dest->ptr_end_indextype = src->ptr_end_indextype;
    dest->ptr_end           = src->ptr_end;
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Destroy csritsv info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csritsv_info(rocsparse_csritsv_info info)
{
    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    if(info->ptr_end != nullptr && info->is_submatrix)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->ptr_end));
        info->ptr_end = nullptr;
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

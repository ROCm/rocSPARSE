/*! \file */
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

#include "internal/precond/rocsparse_csrilu0.h"
#include "rocsparse_csrilu0.hpp"

#include "internal/level2/rocsparse_csrsv.h"

#include "../level2/rocsparse_csrsv.hpp"
#include "csrilu0_device.h"

namespace rocsparse
{
    template <typename T, typename U>
    static rocsparse_status csrilu0_numeric_boost_template(rocsparse_handle   handle,
                                                           rocsparse_mat_info info,
                                                           int                enable_boost,
                                                           const U*           boost_tol,
                                                           const T*           boost_val)
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrilu0_numeric_boost"),
                             (const void*&)info,
                             enable_boost,
                             (const void*&)boost_tol,
                             (const void*&)boost_val);

        ROCSPARSE_CHECKARG_POINTER(1, info);

        // Reset boost
        info->boost_enable        = 0;
        info->use_double_prec_tol = 0;

        // Numeric boost
        if(enable_boost)
        {
            // Check pointer arguments
            ROCSPARSE_CHECKARG_POINTER(3, boost_tol);
            ROCSPARSE_CHECKARG_POINTER(4, boost_val);

            info->boost_enable        = enable_boost;
            info->use_double_prec_tol = std::is_same<U, double>();
            info->boost_tol           = reinterpret_cast<const void*>(boost_tol);
            info->boost_val           = reinterpret_cast<const void*>(boost_val);
        }

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T, typename U, typename V>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrilu0_binsearch(rocsparse_int        m,
                           const rocsparse_int* csr_row_ptr,
                           const rocsparse_int* csr_col_ind,
                           T*                   csr_val,
                           const rocsparse_int* csr_diag_ind,
                           int*                 done,
                           const rocsparse_int* map,
                           rocsparse_int*       zero_pivot,
                           rocsparse_int*       singular_pivot,
                           double               tol,
                           rocsparse_index_base idx_base,
                           int                  enable_boost,
                           U                    boost_tol_device_host,
                           V                    boost_val_device_host)
    {
        auto boost_tol = (enable_boost) ? rocsparse::load_scalar_device_host(boost_tol_device_host)
                                        : rocsparse::zero_scalar_device_host(boost_tol_device_host);

        auto boost_val = (enable_boost) ? rocsparse::load_scalar_device_host(boost_val_device_host)
                                        : rocsparse::zero_scalar_device_host(boost_val_device_host);

        rocsparse::csrilu0_binsearch_kernel<BLOCKSIZE, WFSIZE, SLEEP>(m,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      csr_val,
                                                                      csr_diag_ind,
                                                                      done,
                                                                      map,
                                                                      zero_pivot,
                                                                      singular_pivot,
                                                                      tol,
                                                                      idx_base,
                                                                      enable_boost,
                                                                      boost_tol,
                                                                      boost_val);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASH,
              typename T,
              typename U,
              typename V>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrilu0_hash(rocsparse_int        m,
                      const rocsparse_int* csr_row_ptr,
                      const rocsparse_int* csr_col_ind,
                      T*                   csr_val,
                      const rocsparse_int* csr_diag_ind,
                      int*                 done,
                      const rocsparse_int* map,
                      rocsparse_int*       zero_pivot,
                      rocsparse_int*       singular_pivot,
                      double               tol,
                      rocsparse_index_base idx_base,
                      int                  enable_boost,
                      U                    boost_tol_device_host,
                      V                    boost_val_device_host)
    {
        auto boost_tol = (enable_boost) ? rocsparse::load_scalar_device_host(boost_tol_device_host)
                                        : rocsparse::zero_scalar_device_host(boost_tol_device_host);

        auto boost_val = (enable_boost) ? rocsparse::load_scalar_device_host(boost_val_device_host)
                                        : rocsparse::zero_scalar_device_host(boost_val_device_host);

        rocsparse::csrilu0_hash_kernel<BLOCKSIZE, WFSIZE, HASH>(m,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                csr_val,
                                                                csr_diag_ind,
                                                                done,
                                                                map,
                                                                zero_pivot,
                                                                singular_pivot,
                                                                tol,
                                                                idx_base,
                                                                enable_boost,
                                                                boost_tol,
                                                                boost_val);
    }

    template <typename T, typename U, typename V>
    static rocsparse_status csrilu0_dispatch(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             T*                        csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer,
                                             U                         boost_tol_device_host,
                                             V                         boost_val_device_host)
    {
        // Check for valid handle and matrix descriptor
        // Stream
        hipStream_t stream = handle->stream;

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);
        ptr += 256;

        // done array
        int* d_done_array = reinterpret_cast<int*>(ptr);

        // Initialize buffers
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_done_array, 0, sizeof(int) * m, stream));

        // Max nnz per row
        rocsparse_int max_nnz = info->csrilu0_info->max_nnz;

        // Determine gcnArch and ASIC revision
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);

#define CSRILU0_DIM 256
        dim3 csrilu0_blocks((m * handle->wavefront_size - 1) / CSRILU0_DIM + 1);
        dim3 csrilu0_threads(CSRILU0_DIM);

        if(gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrilu0_binsearch<CSRILU0_DIM, 64, true>),
                csrilu0_blocks,
                csrilu0_threads,
                0,
                stream,
                m,
                csr_row_ptr,
                csr_col_ind,
                csr_val,
                (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                d_done_array,
                (rocsparse_int*)info->csrilu0_info->row_map,
                (rocsparse_int*)info->zero_pivot,
                (rocsparse_int*)info->singular_pivot,
                info->singular_tol,
                descr->base,
                info->boost_enable,
                boost_tol_device_host,
                boost_val_device_host);
        }
        else
        {
            if(handle->wavefront_size == 32)
            {
                if(max_nnz < 32)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 1>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 64)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 2>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 128)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 4>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 256)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 8>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 512)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 16>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_binsearch<CSRILU0_DIM, 32, false>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
            }
            else if(handle->wavefront_size == 64)
            {
                if(max_nnz < 64)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 1>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 128)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 2>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 256)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 4>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 512)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 8>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else if(max_nnz < 1024)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 16>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_binsearch<CSRILU0_DIM, 64, false>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (rocsparse_int*)info->csrilu0_info->trm_diag_ind,
                        d_done_array,
                        (rocsparse_int*)info->csrilu0_info->row_map,
                        (rocsparse_int*)info->zero_pivot,
                        (rocsparse_int*)info->singular_pivot,
                        info->singular_tol,
                        descr->base,
                        info->boost_enable,
                        boost_tol_device_host,
                        boost_val_device_host);
                }
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
            }
        }
#undef CSRILU0_DIM

        return rocsparse_status_success;
    }

    static rocsparse_status csrilu0_quickreturn(rocsparse_handle          handle,
                                                int64_t                   m,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                void*                     csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_solve_policy    policy,
                                                void*                     temp_buffer)
    {
        if(m == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    static rocsparse_status csrilu0_checkarg(rocsparse_handle          handle, //0
                                             int64_t                   m, //1
                                             int64_t                   nnz, //2
                                             const rocsparse_mat_descr descr, //3
                                             void*                     csr_val, //4
                                             const void*               csr_row_ptr, //5
                                             const void*               csr_col_ind, //6
                                             rocsparse_mat_info        info, //7
                                             rocsparse_solve_policy    policy, //8
                                             void*                     temp_buffer) //9
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);

        const rocsparse_status status = rocsparse::csrilu0_quickreturn(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_SIZE(2, nnz);
        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);

        ROCSPARSE_CHECKARG_POINTER(7, info);
        ROCSPARSE_CHECKARG_ENUM(8, policy);
        ROCSPARSE_CHECKARG_ARRAY(9, m, temp_buffer);

        ROCSPARSE_CHECKARG(
            7, info, (info->csrilu0_info == nullptr), rocsparse_status_invalid_pointer);
        return rocsparse_status_continue;
    }

    template <typename T, typename U>
    static rocsparse_status csrilu0_core(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             nnz,
                                         const rocsparse_mat_descr descr,
                                         T*                        csr_val,
                                         const rocsparse_int*      csr_row_ptr,
                                         const rocsparse_int*      csr_col_ind,
                                         rocsparse_mat_info        info,
                                         rocsparse_solve_policy    policy,
                                         void*                     temp_buffer)
    {

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrilu0_dispatch(handle,
                                            m,
                                            nnz,
                                            descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            policy,
                                            temp_buffer,
                                            reinterpret_cast<const U*>(info->boost_tol),
                                            reinterpret_cast<const T*>(info->boost_val)));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_dispatch(
                handle,
                m,
                nnz,
                descr,
                csr_val,
                csr_row_ptr,
                csr_col_ind,
                info,
                policy,
                temp_buffer,
                (info->boost_enable != 0) ? *reinterpret_cast<const U*>(info->boost_tol)
                                          : static_cast<U>(0),
                (info->boost_enable != 0) ? *reinterpret_cast<const T*>(info->boost_val)
                                          : static_cast<T>(0)));
            return rocsparse_status_success;
        }
    }
}

template <typename T, typename U>
rocsparse_status rocsparse::csrilu0_template(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             T*                        csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
{

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsrilu0"),
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)info,
                         policy,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::csrilu0_checkarg(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_core<T, U>(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrilu0_numeric_boost(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             int                enable_boost,
                                                             const float*       boost_tol,
                                                             const float*       boost_val)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, boost_tol, boost_val));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_dcsrilu0_numeric_boost(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             int                enable_boost,
                                                             const double*      boost_tol,
                                                             const double*      boost_val)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, boost_tol, boost_val));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status
    rocsparse_ccsrilu0_numeric_boost(rocsparse_handle               handle,
                                     rocsparse_mat_info             info,
                                     int                            enable_boost,
                                     const float*                   boost_tol,
                                     const rocsparse_float_complex* boost_val)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, boost_tol, boost_val));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status
    rocsparse_zcsrilu0_numeric_boost(rocsparse_handle                handle,
                                     rocsparse_mat_info              info,
                                     int                             enable_boost,
                                     const double*                   boost_tol,
                                     const rocsparse_double_complex* boost_val)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, boost_tol, boost_val));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_dscsrilu0_numeric_boost(rocsparse_handle   handle,
                                                              rocsparse_mat_info info,
                                                              int                enable_boost,
                                                              const double*      boost_tol,
                                                              const float*       boost_val)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, boost_tol, boost_val));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status
    rocsparse_dccsrilu0_numeric_boost(rocsparse_handle               handle,
                                      rocsparse_mat_info             info,
                                      int                            enable_boost,
                                      const double*                  boost_tol,
                                      const rocsparse_float_complex* boost_val)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, boost_tol, boost_val));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_csrilu0_clear(rocsparse_handle   handle,
                                                    rocsparse_mat_info info)
try
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrilu0_clear", (const void*&)info);

    // If meta data is not shared, delete it
    if(!rocsparse::check_trm_shared(info, info->csrilu0_info))
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info(info->csrilu0_info));
    }

    info->csrilu0_info = nullptr;

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_scsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               float*                    csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    if(info != nullptr && info->use_double_prec_tol)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template<float, double>(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template<float, float>(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
        return rocsparse_status_success;
    }
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_dcsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               double*                   csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template<double, double>(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_ccsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_float_complex*  csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    if(info != nullptr && info->use_double_prec_tol)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template<rocsparse_float_complex, double>(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template<rocsparse_float_complex, float>(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
        return rocsparse_status_success;
    }
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_zcsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_double_complex* csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template<rocsparse_double_complex, double>(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle   handle,
                                                         rocsparse_mat_info info,
                                                         rocsparse_int*     position)
try
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(
        handle, "rocsparse_csrilu0_zero_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    // Stream
    hipStream_t stream = handle->stream;

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->csrilu0_info == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            const rocsparse_int neg_one = -1;
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, neg_one, sizeof(rocsparse_int), stream));
        }
        else
        {
            *position = -1;
        }

        return rocsparse_status_success;
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        rocsparse_int pivot;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        if(pivot == std::numeric_limits<rocsparse_int>::max())
        {
            const rocsparse_int neg_one = -1;
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, neg_one, sizeof(rocsparse_int), stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                               info->zero_pivot,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToDevice,
                                               stream));

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
        }
    }
    else
    {
        // rocsparse_pointer_mode_host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // If no zero pivot is found, set -1
        if(*position == std::numeric_limits<rocsparse_int>::max())
        {
            *position = -1;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
        }
    }

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_csrilu0_singular_pivot(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             rocsparse_int*     position)
try
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(
        handle, "rocsparse_csrilu0_singular_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    // Stream
    hipStream_t stream = handle->stream;

    // If m == 0 || nnz == 0 it can happen, that info structure is not created.
    // In this case, always return -1.
    if(info->csrilu0_info == nullptr)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            const rocsparse_int neg_one = -1;
            RETURN_IF_HIP_ERROR(hipMemsetAsync(position, neg_one, sizeof(rocsparse_int), stream));
        }
        else
        {
            *position = -1;
        }

        return rocsparse_status_success;
    }

    constexpr rocsparse_int max_int        = std::numeric_limits<rocsparse_int>::max();
    rocsparse_int           zero_pivot     = max_int;
    rocsparse_int           singular_pivot = max_int;

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &zero_pivot, info->zero_pivot, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&singular_pivot,
                                       info->singular_pivot,
                                       sizeof(rocsparse_int),
                                       hipMemcpyDeviceToHost,
                                       stream));

    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    singular_pivot = rocsparse::min(((zero_pivot == -1) ? max_int : zero_pivot),
                                    ((singular_pivot == -1) ? max_int : singular_pivot));

    if(singular_pivot == max_int)
    {
        singular_pivot = -1;
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {

        // rocsparse_pointer_mode_device
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            position, &singular_pivot, sizeof(rocsparse_int), hipMemcpyHostToDevice, stream));
    }
    else
    {
        // rocsparse_pointer_mode_host
        *position = singular_pivot;
    }

    return (rocsparse_status_success);
}
catch(...)
{
    return rocsparse::exception_to_rocsparse_status();
}

extern "C" rocsparse_status
    rocsparse_csrilu0_set_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double tol)
try
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(tol < 0)
    {
        return rocsparse_status_invalid_value;
    }

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrilu0_set_tolerance", (const void*&)info, tol);

    info->singular_tol = tol;

    return rocsparse_status_success;
}
catch(...)
{
    return rocsparse::exception_to_rocsparse_status();
}

extern "C" rocsparse_status
    rocsparse_csrilu0_get_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double* tol)
try
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(tol == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrilu0_get_tolerance", (const void*&)info, tol);

    *tol = info->singular_tol;

    return rocsparse_status_success;
}
catch(...)
{
    return rocsparse::exception_to_rocsparse_status();
}

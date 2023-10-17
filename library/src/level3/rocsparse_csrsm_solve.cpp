/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "common.h"
#include "definitions.h"
#include "utility.h"

#include "../level1/rocsparse_gthr.hpp"
#include "../level2/rocsparse_csrsv.hpp"
#include "csrsm_device.h"
#include <rocprim/rocprim.hpp>

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          bool         SLEEP,
          typename I,
          typename J,
          typename T,
          typename U>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrsm(rocsparse_operation transB,
           J                   m,
           J                   nrhs,
           U                   alpha,
           const I* __restrict__ csr_row_ptr,
           const J* __restrict__ csr_col_ind,
           const T* __restrict__ csr_val,
           T* __restrict__ B,
           int64_t ldb,
           int* __restrict__ done_array,
           J* __restrict__ map,
           J* __restrict__ zero_pivot,
           rocsparse_index_base idx_base,
           rocsparse_fill_mode  fill_mode,
           rocsparse_diag_type  diag_type)
{

    csrsm_device<BLOCKSIZE, WFSIZE, SLEEP>(transB,
                                           m,
                                           nrhs,
                                           load_scalar_device_host(alpha),
                                           csr_row_ptr,
                                           csr_col_ind,
                                           csr_val,
                                           B,
                                           ldb,
                                           done_array,
                                           map,
                                           zero_pivot,
                                           idx_base,
                                           fill_mode,
                                           diag_type);
}

template <unsigned int DIM_X, unsigned int DIM_Y, typename I, typename T>
ROCSPARSE_KERNEL(DIM_X* DIM_Y)
void csrsm_transpose(I m, I n, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb)
{
    dense_transpose_device<DIM_X, DIM_Y>(m, n, (T)1, A, lda, B, ldb);
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrsm_solve_dispatch(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                J                         m,
                                                J                         nrhs,
                                                I                         nnz,
                                                U                         alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                T*                        B,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_solve_policy    policy,
                                                void*                     temp_buffer)
{

    // Stream
    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    ptr += 256;

    // Each thread block performs at most blockdim columns of the
    // rhs matrix. Therefore, the number of blocks depend on nrhs
    // and the blocksize.
    // Because of this, we might need a larger done_array compared
    // to csrsv.
    int blockdim = 512;
    while(nrhs <= blockdim && blockdim > 32)
    {
        blockdim >>= 1;
    }
    blockdim <<= 1;

    int narrays = (nrhs - 1) / blockdim + 1;

    // done array
    int* done_array = reinterpret_cast<int*>(ptr);
    ptr += ((sizeof(int) * m * narrays - 1) / 256 + 1) * 256;

    // Temporary array to store transpose of B
    T* Bt = B;
    if(trans_B == rocsparse_operation_none)
    {
        Bt = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * nrhs - 1) / 256 + 1) * 256;
    }

    // Temporary array to store transpose of A
    T* At = nullptr;
    if(trans_A == rocsparse_operation_transpose
       || trans_A == rocsparse_operation_conjugate_transpose)
    {
        At = reinterpret_cast<T*>(ptr);
    }

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * m * narrays, stream));

    rocsparse_trm_info csrsm_info
        = (descr->fill_mode == rocsparse_fill_mode_upper)
              ? ((trans_A == rocsparse_operation_none) ? info->csrsm_upper_info
                                                       : info->csrsmt_upper_info)
              : ((trans_A == rocsparse_operation_none) ? info->csrsm_lower_info
                                                       : info->csrsmt_lower_info);

    // If diag type is unit, re-initialize zero pivot to remove structural zeros
    if(descr->diag_type == rocsparse_diag_type_unit)
    {
        RETURN_IF_HIP_ERROR(rocsparse_assign_async(
            static_cast<J*>(info->zero_pivot), std::numeric_limits<J>::max(), stream));
    }

    // Leading dimension
    int64_t ldimB = ldb;

    // Transpose B if B is not transposed yet to improve performance
    if(trans_B == rocsparse_operation_none)
    {
        // Leading dimension for transposed B
        ldimB = nrhs;

#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
        dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
        dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);

        hipLaunchKernelGGL((csrsm_transpose<CSRSM_DIM_X, CSRSM_DIM_Y>),
                           csrsm_blocks,
                           csrsm_threads,
                           0,
                           stream,
                           m,
                           nrhs,
                           B,
                           ldb,
                           Bt,
                           ldimB);
#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
    }

    // Pointers to differentiate between transpose mode
    const I* local_csr_row_ptr = csr_row_ptr;
    const J* local_csr_col_ind = csr_col_ind;
    const T* local_csr_val     = csr_val;

    rocsparse_fill_mode fill_mode = descr->fill_mode;

    // When computing transposed triangular solve, we first need to update the
    // transposed matrix values
    if(trans_A == rocsparse_operation_transpose
       || trans_A == rocsparse_operation_conjugate_transpose)
    {
        T* csrt_val = At;

        // Gather values
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_gthr_template(handle,
                                                          nnz,
                                                          csr_val,
                                                          csrt_val,
                                                          (const I*)csrsm_info->trmt_perm,
                                                          rocsparse_index_base_zero));

        if(trans_A == rocsparse_operation_conjugate_transpose)
        {
            // conjugate csrt_val
            hipLaunchKernelGGL((conjugate<256, I, T>),
                               dim3((nnz - 1) / 256 + 1),
                               dim3(256),
                               0,
                               stream,
                               nnz,
                               csrt_val);
        }

        local_csr_row_ptr = (const I*)csrsm_info->trmt_row_ptr;
        local_csr_col_ind = (const J*)csrsm_info->trmt_col_ind;
        local_csr_val     = (const T*)csrt_val;

        fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                             : rocsparse_fill_mode_lower;
    }
    {
        dim3 csrsm_blocks(((nrhs - 1) / blockdim + 1) * m);
        dim3 csrsm_threads(blockdim);

        // Determine gcnArch and ASIC revision

        const std::string gcn_arch_name = rocsparse_handle_get_arch_name(handle);
        int               asicRev       = handle->asic_rev;

        // rocsparse_pointer_mode_device

        if(blockdim == 64)
        {

            if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
            {
                hipLaunchKernelGGL((csrsm<64, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm<64, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 128)
        {
            if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
            {
                hipLaunchKernelGGL((csrsm<128, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm<128, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 256)
        {
            if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
            {
                hipLaunchKernelGGL((csrsm<256, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm<256, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 512)
        {
            if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
            {
                hipLaunchKernelGGL((csrsm<512, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm<512, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
        }
        else if(blockdim == 1024)
        {
            if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
            {
                hipLaunchKernelGGL((csrsm<1024, 64, true>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
            else
            {
                hipLaunchKernelGGL((csrsm<1024, 64, false>),
                                   csrsm_blocks,
                                   csrsm_threads,
                                   0,
                                   stream,
                                   trans_B,
                                   m,
                                   nrhs,
                                   alpha,
                                   local_csr_row_ptr,
                                   local_csr_col_ind,
                                   local_csr_val,
                                   Bt,
                                   ldimB,
                                   done_array,
                                   (J*)csrsm_info->row_map,
                                   (J*)info->zero_pivot,
                                   descr->base,
                                   fill_mode,
                                   descr->diag_type);
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }
    }
    // Transpose B back if B was not initially transposed
    if(trans_B == rocsparse_operation_none)
    {
#define CSRSM_DIM_X 32
#define CSRSM_DIM_Y 8
        dim3 csrsm_blocks((m - 1) / CSRSM_DIM_X + 1);
        dim3 csrsm_threads(CSRSM_DIM_X * CSRSM_DIM_Y);

        hipLaunchKernelGGL((dense_transpose_back<CSRSM_DIM_X, CSRSM_DIM_Y>),
                           csrsm_blocks,
                           csrsm_threads,
                           0,
                           stream,
                           m,
                           nrhs,
                           Bt,
                           ldimB,
                           B,
                           ldb);
#undef CSRSM_DIM_X
#undef CSRSM_DIM_Y
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrsm_solve_core(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         m,
                                            J                         nrhs,
                                            I                         nnz,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr,
                                            const T*                  csr_val,
                                            const I*                  csr_row_ptr,
                                            const J*                  csr_col_ind,
                                            T*                        B,
                                            int64_t                   ldb,
                                            rocsparse_mat_info        info,
                                            rocsparse_solve_policy    policy,
                                            void*                     temp_buffer)
{
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsm_solve_dispatch(handle,
                                                                 trans_A,
                                                                 trans_B,
                                                                 m,
                                                                 nrhs,
                                                                 nnz,
                                                                 alpha,
                                                                 descr,
                                                                 csr_val,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 B,
                                                                 ldb,
                                                                 info,
                                                                 policy,
                                                                 temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsm_solve_dispatch(handle,
                                                                 trans_A,
                                                                 trans_B,
                                                                 m,
                                                                 nrhs,
                                                                 nnz,
                                                                 *alpha,
                                                                 descr,
                                                                 csr_val,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 B,
                                                                 ldb,
                                                                 info,
                                                                 policy,
                                                                 temp_buffer));
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse_csrsm_solve_quickreturn(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int64_t                   m,
                                                   int64_t                   nrhs,
                                                   int64_t                   nnz,
                                                   const void*               alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const void*               csr_val,
                                                   const void*               csr_row_ptr,
                                                   const void*               csr_col_ind,
                                                   void*                     B,
                                                   int64_t                   ldb,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   void*                     temp_buffer)
{
    // Quick return if possible
    if(m == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

rocsparse_status rocsparse_csrsm_solve_checkarg(rocsparse_handle          handle, //0
                                                rocsparse_operation       trans_A, //1
                                                rocsparse_operation       trans_B, //2
                                                int64_t                   m, //3
                                                int64_t                   nrhs, //4
                                                int64_t                   nnz, //5
                                                const void*               alpha, //6
                                                const rocsparse_mat_descr descr, //7
                                                const void*               csr_val, //8
                                                const void*               csr_row_ptr, //9
                                                const void*               csr_col_ind, //10
                                                void*                     B, //11
                                                int64_t                   ldb, //12
                                                rocsparse_mat_info        info, //13
                                                rocsparse_solve_policy    policy, //14
                                                void*                     temp_buffer) //15
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_SIZE(3, m);
    ROCSPARSE_CHECKARG_SIZE(4, nrhs);
    ROCSPARSE_CHECKARG_SIZE(5, nnz);

    ROCSPARSE_CHECKARG(
        12, ldb, (trans_B == rocsparse_operation_none && ldb < m), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG(12,
                       ldb,
                       ((trans_B == rocsparse_operation_transpose
                         || trans_B == rocsparse_operation_conjugate_transpose)
                        && ldb < nrhs),
                       rocsparse_status_invalid_size);

    const rocsparse_status status = rocsparse_csrsm_solve_quickreturn(handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      m,
                                                                      nrhs,
                                                                      nnz,
                                                                      alpha,
                                                                      descr,
                                                                      csr_val,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      B,
                                                                      ldb,
                                                                      info,
                                                                      policy,
                                                                      temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(7, descr);

    ROCSPARSE_CHECKARG(
        7, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(7,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(10, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(13, info);
    ROCSPARSE_CHECKARG_ENUM(14, policy);

    ROCSPARSE_CHECKARG_POINTER(15, temp_buffer);
    ROCSPARSE_CHECKARG_POINTER(6, alpha);
    ROCSPARSE_CHECKARG_POINTER(11, B);
    return rocsparse_status_continue;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrsm_solve_impl(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         m,
                                            J                         nrhs,
                                            I                         nnz,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr,
                                            const T*                  csr_val,
                                            const I*                  csr_row_ptr,
                                            const J*                  csr_col_ind,
                                            T*                        B,
                                            int64_t                   ldb,
                                            rocsparse_mat_info        info,
                                            rocsparse_solve_policy    policy,
                                            void*                     temp_buffer)
{

    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrsm_solve"),
              trans_A,
              trans_B,
              m,
              nrhs,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)B,
              ldb,
              (const void*&)info,
              policy,
              (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse_csrsm_solve_checkarg(handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   m,
                                                                   nrhs,
                                                                   nnz,
                                                                   alpha,
                                                                   descr,
                                                                   csr_val,
                                                                   csr_row_ptr,
                                                                   csr_col_ind,
                                                                   B,
                                                                   ldb,
                                                                   info,
                                                                   policy,
                                                                   temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsm_solve_core(handle,
                                                         trans_A,
                                                         trans_B,
                                                         m,
                                                         nrhs,
                                                         nnz,
                                                         alpha,
                                                         descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind,
                                                         B,
                                                         ldb,
                                                         info,
                                                         policy,
                                                         temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                        \
    template rocsparse_status rocsparse_csrsm_solve_core(rocsparse_handle          handle,      \
                                                         rocsparse_operation       trans_A,     \
                                                         rocsparse_operation       trans_B,     \
                                                         JTYPE                     m,           \
                                                         JTYPE                     nrhs,        \
                                                         ITYPE                     nnz,         \
                                                         const TTYPE*              alpha,       \
                                                         const rocsparse_mat_descr descr,       \
                                                         const TTYPE*              csr_val,     \
                                                         const ITYPE*              csr_row_ptr, \
                                                         const JTYPE*              csr_col_ind, \
                                                         TTYPE*                    B,           \
                                                         int64_t                   ldb,         \
                                                         rocsparse_mat_info        info,        \
                                                         rocsparse_solve_policy    policy,      \
                                                         void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, ITYPE, JTYPE, TTYPE)                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_operation       trans_A,     \
                                     rocsparse_operation       trans_B,     \
                                     JTYPE                     m,           \
                                     JTYPE                     nrhs,        \
                                     ITYPE                     nnz,         \
                                     const TTYPE*              alpha,       \
                                     const rocsparse_mat_descr descr,       \
                                     const TTYPE*              csr_val,     \
                                     const ITYPE*              csr_row_ptr, \
                                     const JTYPE*              csr_col_ind, \
                                     TTYPE*                    B,           \
                                     JTYPE                     ldb,         \
                                     rocsparse_mat_info        info,        \
                                     rocsparse_solve_policy    policy,      \
                                     void*                     temp_buffer) \
    try                                                                     \
    {                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsm_solve_impl(handle,        \
                                                             trans_A,       \
                                                             trans_B,       \
                                                             m,             \
                                                             nrhs,          \
                                                             nnz,           \
                                                             alpha,         \
                                                             descr,         \
                                                             csr_val,       \
                                                             csr_row_ptr,   \
                                                             csr_col_ind,   \
                                                             B,             \
                                                             ldb,           \
                                                             info,          \
                                                             policy,        \
                                                             temp_buffer)); \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_scsrsm_solve, int32_t, int32_t, float);
C_IMPL(rocsparse_dcsrsm_solve, int32_t, int32_t, double);
C_IMPL(rocsparse_ccsrsm_solve, int32_t, int32_t, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsm_solve, int32_t, int32_t, rocsparse_double_complex);

#undef C_IMPL

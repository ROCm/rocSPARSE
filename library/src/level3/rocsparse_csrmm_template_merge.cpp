/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "utility.h"

#include "csrmm_device_merge.h"
#include "rocsparse_common.h"
#include "rocsparse_csrmm.hpp"

namespace rocsparse
{
    template <uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              uint32_t LOOPS,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename U>
    ROCSPARSE_KERNEL(WF_SIZE)
    void csrmmnt_merge_path_main_kernel(bool conj_A,
                                        bool conj_B,
                                        J    ncol_offset,
                                        J    ncol,
                                        J    m,
                                        J    n,
                                        J    k,
                                        I    nnz,
                                        U    alpha_device_host,
                                        const I* __restrict__ csr_row_ptr,
                                        const J* __restrict__ csr_col_ind,
                                        const A* __restrict__ csr_val,
                                        const coordinate_t<uint32_t>* __restrict__ coord0,
                                        const coordinate_t<uint32_t>* __restrict__ coord1,
                                        const B* __restrict__ dense_B,
                                        int64_t ldb,
                                        U       beta_device_host,
                                        C* __restrict__ dense_C,
                                        int64_t              ldc,
                                        rocsparse_order      order_C,
                                        rocsparse_index_base idx_base)
    {
        auto alpha = load_scalar_device_host(alpha_device_host);
        auto beta  = load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmmnt_merge_path_main_device<WF_SIZE, ITEMS_PER_THREAD, LOOPS>(conj_A,
                                                                                        conj_B,
                                                                                        ncol_offset,
                                                                                        ncol,
                                                                                        m,
                                                                                        n,
                                                                                        k,
                                                                                        nnz,
                                                                                        alpha,
                                                                                        csr_row_ptr,
                                                                                        csr_col_ind,
                                                                                        csr_val,
                                                                                        coord0,
                                                                                        coord1,
                                                                                        dense_B,
                                                                                        ldb,
                                                                                        beta,
                                                                                        dense_C,
                                                                                        ldc,
                                                                                        order_C,
                                                                                        idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmmnt_merge_path_main_multi_rows_kernel(
        bool conj_A,
        bool conj_B,
        J    ncol_offset,
        J    ncol,
        J    m,
        J    n,
        J    k,
        I    nnz,
        U    alpha_device_host,
        const I* __restrict__ csr_row_ptr,
        const J* __restrict__ csr_col_ind,
        const A* __restrict__ csr_val,
        const coordinate_t<uint32_t>* __restrict__ coord0,
        const coordinate_t<uint32_t>* __restrict__ coord1,
        const B* __restrict__ dense_B,
        int64_t ldb,
        U       beta_device_host,
        C* __restrict__ dense_C,
        int64_t              ldc,
        rocsparse_order      order_C,
        rocsparse_index_base idx_base)
    {
        auto alpha = load_scalar_device_host(alpha_device_host);
        auto beta  = load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmmnt_merge_path_main_multi_rows_device<BLOCKSIZE,
                                                                 WF_SIZE,
                                                                 ITEMS_PER_THREAD>(conj_A,
                                                                                   conj_B,
                                                                                   ncol_offset,
                                                                                   ncol,
                                                                                   m,
                                                                                   n,
                                                                                   k,
                                                                                   nnz,
                                                                                   alpha,
                                                                                   csr_row_ptr,
                                                                                   csr_col_ind,
                                                                                   csr_val,
                                                                                   coord0,
                                                                                   coord1,
                                                                                   dense_B,
                                                                                   ldb,
                                                                                   beta,
                                                                                   dense_C,
                                                                                   ldc,
                                                                                   order_C,
                                                                                   idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmmnt_merge_path_remainder_kernel(bool conj_A,
                                             bool conj_B,
                                             J    ncol_offset,
                                             J    m,
                                             J    n,
                                             J    k,
                                             I    nnz,
                                             U    alpha_device_host,
                                             const I* __restrict__ csr_row_ptr,
                                             const J* __restrict__ csr_col_ind,
                                             const A* __restrict__ csr_val,
                                             const coordinate_t<uint32_t>* __restrict__ coord0,
                                             const coordinate_t<uint32_t>* __restrict__ coord1,
                                             const B* __restrict__ dense_B,
                                             int64_t ldb,
                                             U       beta_device_host,
                                             C* __restrict__ dense_C,
                                             int64_t              ldc,
                                             rocsparse_order      order_C,
                                             rocsparse_index_base idx_base)
    {
        auto alpha = load_scalar_device_host(alpha_device_host);
        auto beta  = load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmmnt_merge_path_remainder_device<BLOCKSIZE, WF_SIZE, ITEMS_PER_THREAD>(
                conj_A,
                conj_B,
                ncol_offset,
                m,
                n,
                k,
                nnz,
                alpha,
                csr_row_ptr,
                csr_col_ind,
                csr_val,
                coord0,
                coord1,
                dense_B,
                ldb,
                beta,
                dense_C,
                ldc,
                order_C,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t ITEMS_PER_THREAD,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmmnn_merge_path_kernel(bool conj_A,
                                   bool conj_B,
                                   J    m,
                                   J    n,
                                   J    k,
                                   I    nnz,
                                   U    alpha_device_host,
                                   const I* __restrict__ csr_row_ptr,
                                   const J* __restrict__ csr_col_ind,
                                   const A* __restrict__ csr_val,
                                   const coordinate_t<uint32_t>* __restrict__ coord0,
                                   const coordinate_t<uint32_t>* __restrict__ coord1,
                                   const B* __restrict__ dense_B,
                                   int64_t ldb,
                                   U       beta_device_host,
                                   C* __restrict__ dense_C,
                                   int64_t              ldc,
                                   rocsparse_order      order_C,
                                   rocsparse_index_base idx_base)
    {
        auto alpha = load_scalar_device_host(alpha_device_host);
        auto beta  = load_scalar_device_host(beta_device_host);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmmnn_merge_path_device<BLOCKSIZE, WF_SIZE, ITEMS_PER_THREAD>(conj_A,
                                                                                       conj_B,
                                                                                       m,
                                                                                       n,
                                                                                       k,
                                                                                       nnz,
                                                                                       alpha,
                                                                                       csr_row_ptr,
                                                                                       csr_col_ind,
                                                                                       csr_val,
                                                                                       coord0,
                                                                                       coord1,
                                                                                       dense_B,
                                                                                       ldb,
                                                                                       beta,
                                                                                       dense_C,
                                                                                       ldc,
                                                                                       order_C,
                                                                                       idx_base);
        }
    }

    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_buffer_size_template_merge(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_csrmm_alg       alg,
                                                      J                         m,
                                                      J                         n,
                                                      J                         k,
                                                      I                         nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
                                                      size_t*                   buffer_size)
    {
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            constexpr uint32_t ITEM_PER_THREAD = 256;
            const uint64_t     total_work      = static_cast<uint64_t>(m) + nnz;
            const uint64_t     block_count     = (total_work - 1) / ITEM_PER_THREAD + 1;

            *buffer_size = 0;
            *buffer_size += sizeof(coordinate_t<uint32_t>) * ((block_count - 1) / 256 + 1) * 256;
            *buffer_size += sizeof(coordinate_t<uint32_t>) * ((block_count - 1) / 256 + 1) * 256;

            return rocsparse_status_success;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        }
    }

    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_analysis_template_merge(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_csrmm_alg       alg,
                                                   J                         m,
                                                   J                         n,
                                                   J                         k,
                                                   I                         nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const A*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   void*                     temp_buffer)
    {
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            if(temp_buffer == nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }

            constexpr uint32_t ITEM_PER_THREAD = 256;
            const uint64_t     total_work      = static_cast<uint64_t>(m) + nnz;
            const uint64_t     block_count     = (total_work - 1) / ITEM_PER_THREAD + 1;

            char*                   ptr    = reinterpret_cast<char*>(temp_buffer);
            coordinate_t<uint32_t>* coord0 = reinterpret_cast<coordinate_t<uint32_t>*>(ptr);
            ptr += sizeof(coordinate_t<uint32_t>) * ((block_count - 1) / 256 + 1) * 256;
            coordinate_t<uint32_t>* coord1 = reinterpret_cast<coordinate_t<uint32_t>*>(ptr);

            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrmmnn_merge_compute_coords<1, ITEM_PER_THREAD>),
                dim3(block_count),
                dim3(1),
                0,
                handle->stream,
                m,
                nnz,
                csr_row_ptr,
                coord0,
                coord1,
                descr->base);

            return rocsparse_status_success;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            return rocsparse_status_success;
        }
        }
    }

#define LAUNCH_CSRMMNN_MERGE_KERNEL(CSRMMNN_DIM, WF_SIZE, ITEM_PER_THREAD)                \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                   \
        (rocsparse::csrmmnn_merge_path_kernel<CSRMMNN_DIM, WF_SIZE, ITEM_PER_THREAD, T>), \
        dim3((block_count - 1) / (CSRMMNN_DIM / WF_SIZE) + 1),                            \
        dim3(CSRMMNN_DIM),                                                                \
        0,                                                                                \
        handle->stream,                                                                   \
        conj_A,                                                                           \
        conj_B,                                                                           \
        m,                                                                                \
        n,                                                                                \
        k,                                                                                \
        nnz,                                                                              \
        alpha_device_host,                                                                \
        csr_row_ptr,                                                                      \
        csr_col_ind,                                                                      \
        csr_val,                                                                          \
        coord0,                                                                           \
        coord1,                                                                           \
        dense_B,                                                                          \
        ldb,                                                                              \
        beta_device_host,                                                                 \
        dense_C,                                                                          \
        ldc,                                                                              \
        order_C,                                                                          \
        descr->base);

    template <uint32_t BLOCKSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename U>
    rocsparse_status csrmmnn_merge_dispatch(rocsparse_handle          handle,
                                            bool                      conj_A,
                                            bool                      conj_B,
                                            J                         m,
                                            J                         n,
                                            J                         k,
                                            I                         nnz,
                                            U                         alpha_device_host,
                                            const rocsparse_mat_descr descr,
                                            const A*                  csr_val,
                                            const I*                  csr_row_ptr,
                                            const J*                  csr_col_ind,
                                            const B*                  dense_B,
                                            int64_t                   ldb,
                                            U                         beta_device_host,
                                            C*                        dense_C,
                                            int64_t                   ldc,
                                            rocsparse_order           order_C,
                                            void*                     temp_buffer)
    {
        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::scale_2d_array(handle, m, n, ldc, 1, 0, beta_device_host, dense_C, order_C));

        constexpr uint32_t ITEM_PER_THREAD = 256;
        const uint64_t     total_work      = static_cast<uint64_t>(m) + nnz;
        const uint64_t     block_count     = (total_work - 1) / ITEM_PER_THREAD + 1;

        char*                   ptr    = reinterpret_cast<char*>(temp_buffer);
        coordinate_t<uint32_t>* coord0 = reinterpret_cast<coordinate_t<uint32_t>*>(ptr);
        ptr += sizeof(coordinate_t<uint32_t>) * ((block_count - 1) / 256 + 1) * 256;
        coordinate_t<uint32_t>* coord1 = reinterpret_cast<coordinate_t<uint32_t>*>(ptr);

        if(n <= 16)
        {
            LAUNCH_CSRMMNN_MERGE_KERNEL(256, 16, ITEM_PER_THREAD);
        }
        else if(n <= 32)
        {
            LAUNCH_CSRMMNN_MERGE_KERNEL(256, 32, ITEM_PER_THREAD);
        }
        else if(n <= 64)
        {
            LAUNCH_CSRMMNN_MERGE_KERNEL(256, 64, ITEM_PER_THREAD);
        }
        else
        {
            LAUNCH_CSRMMNN_MERGE_KERNEL(256, 256, ITEM_PER_THREAD);
        }

        return rocsparse_status_success;
    }

#define LAUNCH_CSRMMNT_MERGE_MAIN_KERNEL(WF_SIZE, ITEM_PER_THREAD, LOOPS)                \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                  \
        (rocsparse::csrmmnt_merge_path_main_kernel<WF_SIZE, ITEM_PER_THREAD, LOOPS, T>), \
        dim3(block_count),                                                               \
        dim3(WF_SIZE),                                                                   \
        0,                                                                               \
        handle->stream,                                                                  \
        conj_A,                                                                          \
        conj_B,                                                                          \
        (J)0,                                                                            \
        main,                                                                            \
        m,                                                                               \
        n,                                                                               \
        k,                                                                               \
        nnz,                                                                             \
        alpha_device_host,                                                               \
        csr_row_ptr,                                                                     \
        csr_col_ind,                                                                     \
        csr_val,                                                                         \
        coord0,                                                                          \
        coord1,                                                                          \
        dense_B,                                                                         \
        ldb,                                                                             \
        beta_device_host,                                                                \
        dense_C,                                                                         \
        ldc,                                                                             \
        order_C,                                                                         \
        descr->base);

#define LAUNCH_CSRMMNT_MERGE_MAIN_MULTI_ROWS_KERNEL(CSRMMNT_DIM, WF_SIZE, ITEM_PER_THREAD)         \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                            \
        (rocsparse::                                                                               \
             csrmmnt_merge_path_main_multi_rows_kernel<CSRMMNT_DIM, WF_SIZE, ITEM_PER_THREAD, T>), \
        dim3((block_count - 1) / (CSRMMNT_DIM / WF_SIZE) + 1),                                     \
        dim3(CSRMMNT_DIM),                                                                         \
        0,                                                                                         \
        handle->stream,                                                                            \
        conj_A,                                                                                    \
        conj_B,                                                                                    \
        (J)0,                                                                                      \
        main,                                                                                      \
        m,                                                                                         \
        n,                                                                                         \
        k,                                                                                         \
        nnz,                                                                                       \
        alpha_device_host,                                                                         \
        csr_row_ptr,                                                                               \
        csr_col_ind,                                                                               \
        csr_val,                                                                                   \
        coord0,                                                                                    \
        coord1,                                                                                    \
        dense_B,                                                                                   \
        ldb,                                                                                       \
        beta_device_host,                                                                          \
        dense_C,                                                                                   \
        ldc,                                                                                       \
        order_C,                                                                                   \
        descr->base);

#define LAUNCH_CSRMMNT_MERGE_REMAINDER_KERNEL(CSRMMNT_DIM, WF_SIZE, ITEM_PER_THREAD)         \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                      \
        (rocsparse::                                                                         \
             csrmmnt_merge_path_remainder_kernel<CSRMMNT_DIM, WF_SIZE, ITEM_PER_THREAD, T>), \
        dim3((block_count - 1) / (CSRMMNT_DIM / WF_SIZE) + 1),                               \
        dim3(CSRMMNT_DIM),                                                                   \
        0,                                                                                   \
        handle->stream,                                                                      \
        conj_A,                                                                              \
        conj_B,                                                                              \
        main,                                                                                \
        m,                                                                                   \
        n,                                                                                   \
        k,                                                                                   \
        nnz,                                                                                 \
        alpha_device_host,                                                                   \
        csr_row_ptr,                                                                         \
        csr_col_ind,                                                                         \
        csr_val,                                                                             \
        coord0,                                                                              \
        coord1,                                                                              \
        dense_B,                                                                             \
        ldb,                                                                                 \
        beta_device_host,                                                                    \
        dense_C,                                                                             \
        ldc,                                                                                 \
        order_C,                                                                             \
        descr->base);

    template <uint32_t BLOCKSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename U>
    rocsparse_status csrmmnt_merge_dispatch(rocsparse_handle          handle,
                                            bool                      conj_A,
                                            bool                      conj_B,
                                            J                         m,
                                            J                         n,
                                            J                         k,
                                            I                         nnz,
                                            U                         alpha_device_host,
                                            const rocsparse_mat_descr descr,
                                            const A*                  csr_val,
                                            const I*                  csr_row_ptr,
                                            const J*                  csr_col_ind,
                                            const B*                  dense_B,
                                            int64_t                   ldb,
                                            U                         beta_device_host,
                                            C*                        dense_C,
                                            int64_t                   ldc,
                                            rocsparse_order           order_C,
                                            void*                     temp_buffer)
    {
        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::scale_2d_array(handle, m, n, ldc, 1, 0, beta_device_host, dense_C, order_C));

        constexpr uint32_t ITEM_PER_THREAD = 256;
        const uint64_t     total_work      = static_cast<uint64_t>(m) + nnz;
        const uint64_t     block_count     = (total_work - 1) / ITEM_PER_THREAD + 1;

        char*                   ptr    = reinterpret_cast<char*>(temp_buffer);
        coordinate_t<uint32_t>* coord0 = reinterpret_cast<coordinate_t<uint32_t>*>(ptr);
        ptr += sizeof(coordinate_t<uint32_t>) * ((block_count - 1) / 256 + 1) * 256;
        coordinate_t<uint32_t>* coord1 = reinterpret_cast<coordinate_t<uint32_t>*>(ptr);

        J main      = 0;
        J remainder = n;

        if(n >= 256)
        {
            remainder = n % 256;
            main      = n - remainder;
            LAUNCH_CSRMMNT_MERGE_MAIN_KERNEL(256, ITEM_PER_THREAD, (256 / 256));
        }
        else if(n >= 128)
        {
            remainder = n % 128;
            main      = n - remainder;
            LAUNCH_CSRMMNT_MERGE_MAIN_KERNEL(128, ITEM_PER_THREAD, (128 / 128));
        }
        else if(n >= 64)
        {
            remainder = n % 64;
            main      = n - remainder;
            LAUNCH_CSRMMNT_MERGE_MAIN_KERNEL(64, ITEM_PER_THREAD, (64 / 64));
        }

        if(remainder > 0)
        {
            if(remainder <= 8)
            {
                LAUNCH_CSRMMNT_MERGE_REMAINDER_KERNEL(256, 8, ITEM_PER_THREAD);
            }
            else if(remainder <= 16)
            {
                LAUNCH_CSRMMNT_MERGE_REMAINDER_KERNEL(256, 16, ITEM_PER_THREAD);
            }
            else if(remainder <= 32)
            {
                LAUNCH_CSRMMNT_MERGE_REMAINDER_KERNEL(256, 32, ITEM_PER_THREAD);
            }
            else if(remainder <= 64)
            {
                LAUNCH_CSRMMNT_MERGE_REMAINDER_KERNEL(256, 64, ITEM_PER_THREAD);
            }
            else if(remainder <= 128)
            {
                LAUNCH_CSRMMNT_MERGE_REMAINDER_KERNEL(256, 128, ITEM_PER_THREAD);
            }
            else
            {
                LAUNCH_CSRMMNT_MERGE_REMAINDER_KERNEL(256, 256, ITEM_PER_THREAD);
            }
        }

        return rocsparse_status_success;
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C, typename U>
    rocsparse_status csrmm_template_merge(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          J                         m,
                                          J                         n,
                                          J                         k,
                                          I                         nnz,
                                          U                         alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const A*                  csr_val,
                                          const I*                  csr_row_ptr,
                                          const J*                  csr_col_ind,
                                          const B*                  dense_B,
                                          int64_t                   ldb,
                                          rocsparse_order           order_B,
                                          U                         beta_device_host,
                                          C*                        dense_C,
                                          int64_t                   ldc,
                                          rocsparse_order           order_C,
                                          void*                     temp_buffer,
                                          bool                      force_conj_A)
    {
        bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose || force_conj_A);
        bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

        // Run different csrmv kernels
        if(trans_A == rocsparse_operation_none)
        {
            if(temp_buffer == nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }

            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                RETURN_IF_ROCSPARSE_ERROR((csrmmnn_merge_dispatch<256, T>(handle,
                                                                          conj_A,
                                                                          conj_B,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          nnz,
                                                                          alpha_device_host,
                                                                          descr,
                                                                          csr_val,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          dense_B,
                                                                          ldb,
                                                                          beta_device_host,
                                                                          dense_C,
                                                                          ldc,
                                                                          order_C,
                                                                          temp_buffer)));
                return rocsparse_status_success;
            }
            else if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                RETURN_IF_ROCSPARSE_ERROR((csrmmnt_merge_dispatch<256, T>(handle,
                                                                          conj_A,
                                                                          conj_B,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          nnz,
                                                                          alpha_device_host,
                                                                          descr,
                                                                          csr_val,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          dense_B,
                                                                          ldb,
                                                                          beta_device_host,
                                                                          dense_C,
                                                                          ldc,
                                                                          order_C,
                                                                          temp_buffer)));
                return rocsparse_status_success;
            }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

#define INSTANTIATE_BUFFER_SIZE(TTYPE, ITYPE, JTYPE, ATYPE)                       \
    template rocsparse_status rocsparse::csrmm_buffer_size_template_merge<TTYPE>( \
        rocsparse_handle          handle,                                         \
        rocsparse_operation       trans_A,                                        \
        rocsparse_csrmm_alg       alg,                                            \
        JTYPE                     m,                                              \
        JTYPE                     n,                                              \
        JTYPE                     k,                                              \
        ITYPE                     nnz,                                            \
        const rocsparse_mat_descr descr,                                          \
        const ATYPE*              csr_val,                                        \
        const ITYPE*              csr_row_ptr,                                    \
        const JTYPE*              csr_col_ind,                                    \
        size_t*                   buffer_size)

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, int8_t);
#undef INSTANTIATE_BUFFER_SIZE

#define INSTANTIATE_ANALYSIS(TTYPE, ITYPE, JTYPE, ATYPE)                       \
    template rocsparse_status rocsparse::csrmm_analysis_template_merge<TTYPE>( \
        rocsparse_handle          handle,                                      \
        rocsparse_operation       trans_A,                                     \
        rocsparse_csrmm_alg       alg,                                         \
        JTYPE                     m,                                           \
        JTYPE                     n,                                           \
        JTYPE                     k,                                           \
        ITYPE                     nnz,                                         \
        const rocsparse_mat_descr descr,                                       \
        const ATYPE*              csr_val,                                     \
        const ITYPE*              csr_row_ptr,                                 \
        const JTYPE*              csr_col_ind,                                 \
        void*                     temp_buffer)

// Uniform precisions
INSTANTIATE_ANALYSIS(float, int32_t, int32_t, float);
INSTANTIATE_ANALYSIS(float, int64_t, int32_t, float);
INSTANTIATE_ANALYSIS(float, int64_t, int64_t, float);
INSTANTIATE_ANALYSIS(double, int32_t, int32_t, double);
INSTANTIATE_ANALYSIS(double, int64_t, int32_t, double);
INSTANTIATE_ANALYSIS(double, int64_t, int64_t, double);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_float_complex, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(rocsparse_double_complex, int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_ANALYSIS(int32_t, int32_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int32_t, int64_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int32_t, int64_t, int64_t, int8_t);
INSTANTIATE_ANALYSIS(float, int32_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(float, int64_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(float, int64_t, int64_t, int8_t);
#undef INSTANTIATE_ANALYSIS

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE, UTYPE)  \
    template rocsparse_status rocsparse::csrmm_template_merge<TTYPE>( \
        rocsparse_handle          handle,                             \
        rocsparse_operation       trans_A,                            \
        rocsparse_operation       trans_B,                            \
        JTYPE                     m,                                  \
        JTYPE                     n,                                  \
        JTYPE                     k,                                  \
        ITYPE                     nnz,                                \
        UTYPE                     alpha_device_host,                  \
        const rocsparse_mat_descr descr,                              \
        const ATYPE*              csr_val,                            \
        const ITYPE*              csr_row_ptr,                        \
        const JTYPE*              csr_col_ind,                        \
        const BTYPE*              dense_B,                            \
        int64_t                   ldb,                                \
        rocsparse_order           order_B,                            \
        UTYPE                     beta_device_host,                   \
        CTYPE*                    dense_C,                            \
        int64_t                   ldc,                                \
        rocsparse_order           order_C,                            \
        void*                     temp_buffer,                        \
        bool                      force_conj_A)

// Uniform precisions
INSTANTIATE(float, int32_t, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, int32_t, float, float, float, float);
INSTANTIATE(float, int64_t, int64_t, float, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, int32_t, double, double, double, double);
INSTANTIATE(double, int64_t, int64_t, double, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

INSTANTIATE(float, int32_t, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, int32_t, float, float, float, const float*);
INSTANTIATE(float, int64_t, int64_t, float, float, float, const float*);
INSTANTIATE(double, int32_t, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, int32_t, double, double, double, const double*);
INSTANTIATE(double, int64_t, int64_t, double, double, double, const double*);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex,
            const rocsparse_float_complex*);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex,
            const rocsparse_double_complex*);

// Mixed Precisions
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, float);

INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t, const int32_t*);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float, const float*);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float, const float*);
#undef INSTANTIATE

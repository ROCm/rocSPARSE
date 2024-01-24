/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "definitions.h"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_csrgemm.hpp"
#include "utility.h"

#include <rocprim/rocprim.hpp>

#include "rocsparse_csrgemm_mult.hpp"
#include "rocsparse_csrgemm_multadd.hpp"
#include "rocsparse_csrgemm_nnz_calc.hpp"
#include "rocsparse_csrgemm_scal.hpp"

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_nnz_calc(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             J                         m,
                                             J                         n,
                                             J                         k,
                                             const rocsparse_mat_descr descr_A,
                                             I                         nnz_A,
                                             const I*                  csr_row_ptr_A,
                                             const J*                  csr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             I                         nnz_B,
                                             const I*                  csr_row_ptr_B,
                                             const J*                  csr_col_ind_B,
                                             const rocsparse_mat_descr descr_D,
                                             I                         nnz_D,
                                             const I*                  csr_row_ptr_D,
                                             const J*                  csr_col_ind_D,
                                             const rocsparse_mat_descr descr_C,
                                             I*                        csr_row_ptr_C,
                                             I*                        nnz_C,
                                             const rocsparse_mat_info  info_C,
                                             void*                     temp_buffer)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D = info_C->csrgemm_info->add
                                      ? ((descr_D) ? descr_D->base : rocsparse_index_base_zero)
                                      : rocsparse_index_base_zero;

    bool mul = info_C->csrgemm_info->mul;
    bool add = info_C->csrgemm_info->add;

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Compute number of intermediate products for each row
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csrgemm_intermediate_products<CSRGEMM_DIM, CSRGEMM_SUB>),
        dim3((m - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
        dim3(CSRGEMM_DIM),
        0,
        stream,
        m,
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_row_ptr_B,
        csr_row_ptr_D,
        csr_row_ptr_C,
        base_A,
        mul,
        add);
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

    // Determine maximum of all intermediate products
    RETURN_IF_HIP_ERROR(rocprim::reduce(nullptr,
                                        rocprim_size,
                                        csr_row_ptr_C,
                                        csr_row_ptr_C + m,
                                        0,
                                        m,
                                        rocprim::maximum<I>(),
                                        stream));
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_HIP_ERROR(rocprim::reduce(rocprim_buffer,
                                        rocprim_size,
                                        csr_row_ptr_C,
                                        csr_row_ptr_C + m,
                                        0,
                                        m,
                                        rocprim::maximum<I>(),
                                        stream));

    I int_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&int_max, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToHost, stream));
    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    J* d_group_offset = reinterpret_cast<J*>(buffer);
    buffer += sizeof(J) * 256;

    // Group size buffer
    J h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(J) * CSRGEMM_MAXGROUPS);

    // Permutation array
    J* d_perm = nullptr;

    // If maximum of intermediate products exceeds 32, we process the rows in groups of
    // similar sized intermediate products
    if(int_max > 32)
    {
        // Group size buffer
        J* d_group_size = reinterpret_cast<J*>(buffer);
        buffer += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_group_reduce_part1<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
            dim3(CSRGEMM_DIM),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            m,
            csr_row_ptr_C,
            d_group_size);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
            dim3(1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            d_group_size);
#undef CSRGEMM_DIM

        // Exclusive sum to obtain group offsets
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<J>(),
                                                    stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<J>(),
                                                    stream));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(J) * CSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // Permutation temporary arrays
        J* tmp_vals = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        J* tmp_perm = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        I* tmp_keys = reinterpret_cast<I*>(buffer);
        buffer += ((sizeof(I) * m - 1) / 256 + 1) * 256;

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_template(handle, m, tmp_perm));

        rocprim::double_buffer<I> d_keys(csr_row_ptr_C, tmp_keys);
        rocprim::double_buffer<J> d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, rocprim_size, d_keys, d_vals, m, 0, 3, stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            rocprim_buffer, rocprim_size, d_keys, d_vals, m, 0, 3, stream));

        d_perm = d_vals.current();

        // Release tmp_keys buffer
        buffer -= ((sizeof(I) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(J), stream));
    }

    // Compute non-zero entries per row for each group

    // Group 0: 0 - 32 intermediate products
    if(h_group_size[0] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 4
#define CSRGEMM_HASHSIZE 32
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_wf_per_row<CSRGEMM_DIM,
                                               CSRGEMM_SUB,
                                               CSRGEMM_HASHSIZE,
                                               CSRGEMM_NNZ_HASH>),
            dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[0],
            &d_group_offset[0],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 1: 33 - 64 intermediate products
    if(h_group_size[1] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 64
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_wf_per_row<CSRGEMM_DIM,
                                               CSRGEMM_SUB,
                                               CSRGEMM_HASHSIZE,
                                               CSRGEMM_NNZ_HASH>),
            dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[1],
            &d_group_offset[1],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 2: 65 - 512 intermediate products
    if(h_group_size[2] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 512
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                                                 CSRGEMM_SUB,
                                                                                 CSRGEMM_HASHSIZE,
                                                                                 CSRGEMM_NNZ_HASH>),
                                           dim3(h_group_size[2]),
                                           dim3(CSRGEMM_DIM),
                                           0,
                                           stream,
                                           &d_group_offset[2],
                                           d_perm,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_row_ptr_D,
                                           csr_col_ind_D,
                                           csr_row_ptr_C,
                                           base_A,
                                           base_B,
                                           base_D,
                                           info_C->csrgemm_info->mul,
                                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 3: 513 - 1024 intermediate products
    if(h_group_size[3] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 1024
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                                                 CSRGEMM_SUB,
                                                                                 CSRGEMM_HASHSIZE,
                                                                                 CSRGEMM_NNZ_HASH>),
                                           dim3(h_group_size[3]),
                                           dim3(CSRGEMM_DIM),
                                           0,
                                           stream,
                                           &d_group_offset[3],
                                           d_perm,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_row_ptr_D,
                                           csr_col_ind_D,
                                           csr_row_ptr_C,
                                           base_A,
                                           base_B,
                                           base_D,
                                           info_C->csrgemm_info->mul,
                                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 4: 1025 - 2048 intermediate products
    if(h_group_size[4] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 2048
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                                                 CSRGEMM_SUB,
                                                                                 CSRGEMM_HASHSIZE,
                                                                                 CSRGEMM_NNZ_HASH>),
                                           dim3(h_group_size[4]),
                                           dim3(CSRGEMM_DIM),
                                           0,
                                           stream,
                                           &d_group_offset[4],
                                           d_perm,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_row_ptr_D,
                                           csr_col_ind_D,
                                           csr_row_ptr_C,
                                           base_A,
                                           base_B,
                                           base_D,
                                           info_C->csrgemm_info->mul,
                                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 5: 2049 - 4096 intermediate products
    if(h_group_size[5] > 0)
    {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 4096
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                                                 CSRGEMM_SUB,
                                                                                 CSRGEMM_HASHSIZE,
                                                                                 CSRGEMM_NNZ_HASH>),
                                           dim3(h_group_size[5]),
                                           dim3(CSRGEMM_DIM),
                                           0,
                                           stream,
                                           &d_group_offset[5],
                                           d_perm,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_row_ptr_D,
                                           csr_col_ind_D,
                                           csr_row_ptr_C,
                                           base_A,
                                           base_B,
                                           base_D,
                                           info_C->csrgemm_info->mul,
                                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 6: 4097 - 8192 intermediate products
    if(h_group_size[6] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 8192
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                                                 CSRGEMM_SUB,
                                                                                 CSRGEMM_HASHSIZE,
                                                                                 CSRGEMM_NNZ_HASH>),
                                           dim3(h_group_size[6]),
                                           dim3(CSRGEMM_DIM),
                                           0,
                                           stream,
                                           &d_group_offset[6],
                                           d_perm,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_row_ptr_D,
                                           csr_col_ind_D,
                                           csr_row_ptr_C,
                                           base_A,
                                           base_B,
                                           base_D,
                                           info_C->csrgemm_info->mul,
                                           info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 7: more than 8192 intermediate products
    if(h_group_size[7] > 0)
    {
        // Matrices B and D must be sorted in order to run this path
        if(descr_B->storage_mode == rocsparse_storage_mode_unsorted
           || (info_C->csrgemm_info->add ? descr_D->storage_mode == rocsparse_storage_mode_unsorted
                                         : false))
        {
            return rocsparse_status_requires_sorted_storage;
        }

#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
        I* workspace_B = nullptr;

        if(info_C->csrgemm_info->mul == true)
        {
            // Allocate additional buffer for C = alpha * A * B
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync((void**)&workspace_B, sizeof(I) * nnz_A, handle->stream));
        }

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::
                 csrgemm_nnz_block_per_row_multipass<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_CHUNKSIZE>),
            dim3(h_group_size[7]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            n,
            &d_group_offset[7],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            workspace_B,
            base_A,
            base_B,
            base_D,
            mul,
            add);

        if(info_C->csrgemm_info->mul == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace_B, handle->stream));
        }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Exclusive sum to obtain row pointers of C
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                static_cast<rocsparse_int>(descr_C->base),
                                                m + 1,
                                                rocprim::plus<I>(),
                                                stream));
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                static_cast<rocsparse_int>(descr_C->base),
                                                m + 1,
                                                rocprim::plus<I>(),
                                                stream));

    // Store nnz of C
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToDevice, stream));

        // Adjust nnz by index base
        if(descr_C->base == rocsparse_index_base_one)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_index_base<1>), dim3(1), dim3(1), 0, stream, nnz_C);
        }
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // Adjust nnz by index base
        *nnz_C -= descr_C->base;
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I>
    rocsparse_status csrgemm_nnz_checkarg(rocsparse_handle          handle, //0
                                          rocsparse_operation       trans_A, //1
                                          rocsparse_operation       trans_B, //2
                                          int64_t                   m, //3
                                          int64_t                   n, //4
                                          int64_t                   k, //5
                                          const rocsparse_mat_descr descr_A, //6
                                          int64_t                   nnz_A, //7
                                          const void*               csr_row_ptr_A, //8
                                          const void*               csr_col_ind_A, //9
                                          const rocsparse_mat_descr descr_B, //10
                                          int64_t                   nnz_B, //11
                                          const void*               csr_row_ptr_B, //12
                                          const void*               csr_col_ind_B, //13
                                          const rocsparse_mat_descr descr_D, //14
                                          int64_t                   nnz_D, //15
                                          const void*               csr_row_ptr_D, //16
                                          const void*               csr_col_ind_D, //17
                                          const rocsparse_mat_descr descr_C, //18
                                          I*                        csr_row_ptr_C, //19
                                          I*                        nnz_C, //20
                                          const rocsparse_mat_info  info_C, //21
                                          void*                     temp_buffer) //22
    {
        ROCSPARSE_CHECKARG_POINTER(21, info_C);
        ROCSPARSE_CHECKARG(
            21, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);

        const bool mul = info_C->csrgemm_info->mul;
        const bool add = info_C->csrgemm_info->add;

        if(mul == true && add == true)
        {
            ROCSPARSE_CHECKARG_HANDLE(0, handle);

            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(7, nnz_A);
            ROCSPARSE_CHECKARG_SIZE(11, nnz_B);
            ROCSPARSE_CHECKARG_SIZE(15, nnz_D);

            ROCSPARSE_CHECKARG_POINTER(6, descr_A);
            ROCSPARSE_CHECKARG_POINTER(10, descr_B);
            ROCSPARSE_CHECKARG_POINTER(14, descr_D);
            ROCSPARSE_CHECKARG_POINTER(18, descr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);

            ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(12, k, csr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(16, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);

            ROCSPARSE_CHECKARG_ARRAY(9, nnz_A, csr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(13, nnz_B, csr_col_ind_B);
            ROCSPARSE_CHECKARG_ARRAY(17, nnz_D, csr_col_ind_D);

            ROCSPARSE_CHECKARG(6,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(10,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(14,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(18,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status
                = rocsparse::csrgemm_multadd_nnz_quickreturn(handle,
                                                             trans_A,
                                                             trans_B,
                                                             m,
                                                             n,
                                                             k,
                                                             descr_A,
                                                             nnz_A,
                                                             csr_row_ptr_A,
                                                             csr_col_ind_A,
                                                             descr_B,
                                                             nnz_B,
                                                             csr_row_ptr_B,
                                                             csr_col_ind_B,
                                                             descr_D,
                                                             nnz_D,
                                                             csr_row_ptr_D,
                                                             csr_col_ind_D,
                                                             descr_C,
                                                             csr_row_ptr_C,
                                                             nnz_C,
                                                             info_C,
                                                             temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(1,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(2,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG_POINTER(22, temp_buffer);
            return rocsparse_status_continue;
        }
        else if(mul == true && add == false)
        {
            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(7, nnz_A);
            ROCSPARSE_CHECKARG_SIZE(11, nnz_B);
            ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr_A);
            ROCSPARSE_CHECKARG_ARRAY(12, k, csr_row_ptr_B);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);

            ROCSPARSE_CHECKARG_POINTER(6, descr_A);
            ROCSPARSE_CHECKARG_POINTER(10, descr_B);
            ROCSPARSE_CHECKARG_POINTER(18, descr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);

            ROCSPARSE_CHECKARG_ARRAY(9, nnz_A, csr_col_ind_A);
            ROCSPARSE_CHECKARG_ARRAY(13, nnz_B, csr_col_ind_B);
            ROCSPARSE_CHECKARG(6,
                               descr_A,
                               (descr_A->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(10,
                               descr_B,
                               (descr_B->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(18,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::csrgemm_mult_nnz_quickreturn(handle,
                                                                                    trans_A,
                                                                                    trans_B,
                                                                                    m,
                                                                                    n,
                                                                                    k,
                                                                                    descr_A,
                                                                                    nnz_A,
                                                                                    csr_row_ptr_A,
                                                                                    csr_col_ind_A,
                                                                                    descr_B,
                                                                                    nnz_B,
                                                                                    csr_row_ptr_B,
                                                                                    csr_col_ind_B,
                                                                                    descr_C,
                                                                                    csr_row_ptr_C,
                                                                                    nnz_C,
                                                                                    info_C,
                                                                                    temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            ROCSPARSE_CHECKARG(1,
                               trans_A,
                               (trans_A != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(2,
                               trans_B,
                               (trans_B != rocsparse_operation_none),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG_POINTER(22, temp_buffer);
            return rocsparse_status_continue;
        }
        else if(mul == false && add == true)
        {

            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_SIZE(15, nnz_D);
            ROCSPARSE_CHECKARG_POINTER(14, descr_D);
            ROCSPARSE_CHECKARG_POINTER(18, descr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);
            ROCSPARSE_CHECKARG_ARRAY(16, m, csr_row_ptr_D);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_ARRAY(17, nnz_D, csr_col_ind_D);
            ROCSPARSE_CHECKARG(18,
                               descr_C,
                               (descr_C->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);
            ROCSPARSE_CHECKARG(14,
                               descr_D,
                               (descr_D->type != rocsparse_matrix_type_general),
                               rocsparse_status_not_implemented);

            const rocsparse_status status = rocsparse::csrgemm_scal_nnz_quickreturn(handle,
                                                                                    m,
                                                                                    n,
                                                                                    descr_D,
                                                                                    nnz_D,
                                                                                    csr_row_ptr_D,
                                                                                    csr_col_ind_D,
                                                                                    descr_C,
                                                                                    csr_row_ptr_C,
                                                                                    nnz_C,
                                                                                    info_C,
                                                                                    temp_buffer);
            if(status != rocsparse_status_continue)
            {
                RETURN_IF_ROCSPARSE_ERROR(status);
                return rocsparse_status_success;
            }

            return rocsparse_status_continue;
        }
        else
        {
            assert(mul == false && add == false);
            ROCSPARSE_CHECKARG_HANDLE(0, handle);
            ROCSPARSE_CHECKARG_ENUM(1, trans_A);
            ROCSPARSE_CHECKARG_ENUM(2, trans_B);
            ROCSPARSE_CHECKARG_SIZE(3, m);
            ROCSPARSE_CHECKARG_SIZE(4, n);
            ROCSPARSE_CHECKARG_SIZE(5, k);
            ROCSPARSE_CHECKARG_POINTER(21, info_C);
            ROCSPARSE_CHECKARG(
                21, info_C, (info_C->csrgemm_info == nullptr), rocsparse_status_invalid_pointer);
            ROCSPARSE_CHECKARG_ARRAY(19, m, csr_row_ptr_C);
            ROCSPARSE_CHECKARG_POINTER(20, nnz_C);

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), handle->stream));
            }
            else
            {
                *nnz_C = 0;
            }

            if(m > 0)
            {
#define CSRGEMM_DIM 1024
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_set_base<CSRGEMM_DIM>),
                                                   dim3((m + 1) / CSRGEMM_DIM + 1),
                                                   dim3(CSRGEMM_DIM),
                                                   0,
                                                   handle->stream,
                                                   m + 1,
                                                   csr_row_ptr_C,
                                                   descr_C->base);
#undef CSRGEMM_DIM
            }
            return rocsparse_status_success;
        }
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_nnz_template(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 J                         m,
                                                 J                         n,
                                                 J                         k,
                                                 const rocsparse_mat_descr descr_A,
                                                 I                         nnz_A,
                                                 const I*                  csr_row_ptr_A,
                                                 const J*                  csr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 I                         nnz_B,
                                                 const I*                  csr_row_ptr_B,
                                                 const J*                  csr_col_ind_B,
                                                 const rocsparse_mat_descr descr_D,
                                                 I                         nnz_D,
                                                 const I*                  csr_row_ptr_D,
                                                 const J*                  csr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 I*                        csr_row_ptr_C,
                                                 I*                        nnz_C,
                                                 const rocsparse_mat_info  info_C,
                                                 void*                     temp_buffer)
{
    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;

    // Either mult, add or multadd need to be performed
    if(mul == true && add == true)
    {
        // C = alpha * A * B + beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_multadd_nnz_template(handle,
                                                                          trans_A,
                                                                          trans_B,
                                                                          m,
                                                                          n,
                                                                          k,
                                                                          descr_A,
                                                                          nnz_A,
                                                                          csr_row_ptr_A,
                                                                          csr_col_ind_A,
                                                                          descr_B,
                                                                          nnz_B,
                                                                          csr_row_ptr_B,
                                                                          csr_col_ind_B,
                                                                          descr_D,
                                                                          nnz_D,
                                                                          csr_row_ptr_D,
                                                                          csr_col_ind_D,
                                                                          descr_C,
                                                                          csr_row_ptr_C,
                                                                          nnz_C,
                                                                          info_C,
                                                                          temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == true && add == false)
    {
        // C = alpha * A * B
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_mult_nnz_template(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       descr_A,
                                                                       nnz_A,
                                                                       csr_row_ptr_A,
                                                                       csr_col_ind_A,
                                                                       descr_B,
                                                                       nnz_B,
                                                                       csr_row_ptr_B,
                                                                       csr_col_ind_B,
                                                                       descr_C,
                                                                       csr_row_ptr_C,
                                                                       nnz_C,
                                                                       info_C,
                                                                       temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == false && add == true)
    {
        // C = beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_scal_nnz_template(handle,
                                                                       m,
                                                                       n,
                                                                       descr_D,
                                                                       nnz_D,
                                                                       csr_row_ptr_D,
                                                                       csr_col_ind_D,
                                                                       descr_C,
                                                                       csr_row_ptr_C,
                                                                       nnz_C,
                                                                       info_C,
                                                                       temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        assert(mul == false && add == false);

        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), handle->stream));
        }
        else
        {
            *nnz_C = 0;
        }
        if(m > 0)
        {
#define CSRGEMM_DIM 1024
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_set_base<CSRGEMM_DIM>),
                                               dim3((m + 1) / CSRGEMM_DIM + 1),
                                               dim3(CSRGEMM_DIM),
                                               0,
                                               handle->stream,
                                               m + 1,
                                               csr_row_ptr_C,
                                               descr_C->base);
#undef CSRGEMM_DIM
        }
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE)                                            \
    template rocsparse_status rocsparse::csrgemm_nnz_template<ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_operation       trans_A,                                   \
        rocsparse_operation       trans_B,                                   \
        JTYPE                     m,                                         \
        JTYPE                     n,                                         \
        JTYPE                     k,                                         \
        const rocsparse_mat_descr descr_A,                                   \
        ITYPE                     nnz_A,                                     \
        const ITYPE*              csr_row_ptr_A,                             \
        const JTYPE*              csr_col_ind_A,                             \
        const rocsparse_mat_descr descr_B,                                   \
        ITYPE                     nnz_B,                                     \
        const ITYPE*              csr_row_ptr_B,                             \
        const JTYPE*              csr_col_ind_B,                             \
        const rocsparse_mat_descr descr_D,                                   \
        ITYPE                     nnz_D,                                     \
        const ITYPE*              csr_row_ptr_D,                             \
        const JTYPE*              csr_col_ind_D,                             \
        const rocsparse_mat_descr descr_C,                                   \
        ITYPE*                    csr_row_ptr_C,                             \
        ITYPE*                    nnz_C,                                     \
        const rocsparse_mat_info  info_C,                                    \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_nnz_core(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             J                         m,
                                             J                         n,
                                             J                         k,
                                             const rocsparse_mat_descr descr_A,
                                             I                         nnz_A,
                                             const I*                  csr_row_ptr_A,
                                             const J*                  csr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             I                         nnz_B,
                                             const I*                  csr_row_ptr_B,
                                             const J*                  csr_col_ind_B,
                                             const rocsparse_mat_descr descr_D,
                                             I                         nnz_D,
                                             const I*                  csr_row_ptr_D,
                                             const J*                  csr_col_ind_D,
                                             const rocsparse_mat_descr descr_C,
                                             I*                        csr_row_ptr_C,
                                             I*                        nnz_C,
                                             const rocsparse_mat_info  info_C,
                                             void*                     temp_buffer)
{
    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    // Either mult, add or multadd need to be performed
    if(mul == true && add == true)
    {
        // C = alpha * A * B + beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_multadd_nnz_core(handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      m,
                                                                      n,
                                                                      k,
                                                                      descr_A,
                                                                      nnz_A,
                                                                      csr_row_ptr_A,
                                                                      csr_col_ind_A,
                                                                      descr_B,
                                                                      nnz_B,
                                                                      csr_row_ptr_B,
                                                                      csr_col_ind_B,
                                                                      descr_D,
                                                                      nnz_D,
                                                                      csr_row_ptr_D,
                                                                      csr_col_ind_D,
                                                                      descr_C,
                                                                      csr_row_ptr_C,
                                                                      nnz_C,
                                                                      info_C,
                                                                      temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == true && add == false)
    {
        // C = alpha * A * B
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_mult_nnz_core(handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   descr_A,
                                                                   nnz_A,
                                                                   csr_row_ptr_A,
                                                                   csr_col_ind_A,
                                                                   descr_B,
                                                                   nnz_B,
                                                                   csr_row_ptr_B,
                                                                   csr_col_ind_B,
                                                                   descr_C,
                                                                   csr_row_ptr_C,
                                                                   nnz_C,
                                                                   info_C,
                                                                   temp_buffer));
        return rocsparse_status_success;
    }
    else if(mul == false && add == true)
    {

        // C = beta * D
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_scal_nnz_core(handle,
                                                                   m,
                                                                   n,
                                                                   descr_D,
                                                                   nnz_D,
                                                                   csr_row_ptr_D,
                                                                   csr_col_ind_D,
                                                                   descr_C,
                                                                   csr_row_ptr_C,
                                                                   nnz_C,
                                                                   info_C,
                                                                   temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        return rocsparse_status_success;
    }
}

namespace rocsparse
{
    template <typename... P>
    static rocsparse_status csrgemm_nnz_impl(P&&... p)
    {
        log_trace("rocsparse_csrgemm_nnz", p...);

        const rocsparse_status status = rocsparse::csrgemm_nnz_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_nnz_core(p...));
        return rocsparse_status_success;
    }
}

//
// rocsparse_xcsrgemm_nnz
//
extern "C" rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle          handle,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  rocsparse_int             k,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int             nnz_B,
                                                  const rocsparse_int*      csr_row_ptr_B,
                                                  const rocsparse_int*      csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_D,
                                                  rocsparse_int             nnz_D,
                                                  const rocsparse_int*      csr_row_ptr_D,
                                                  const rocsparse_int*      csr_col_ind_D,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int*            csr_row_ptr_C,
                                                  rocsparse_int*            nnz_C,
                                                  const rocsparse_mat_info  info_C,
                                                  void*                     temp_buffer)
try
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_nnz_impl(handle,
                                                          trans_A,
                                                          trans_B,
                                                          m,
                                                          n,
                                                          k,
                                                          descr_A,
                                                          nnz_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          descr_B,
                                                          nnz_B,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          descr_D,
                                                          nnz_D,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          descr_C,
                                                          csr_row_ptr_C,
                                                          nnz_C,
                                                          info_C,
                                                          temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

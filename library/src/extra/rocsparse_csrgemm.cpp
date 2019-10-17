/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include "rocsparse_csrgemm.hpp"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

__global__ void csrgemm_index_base(rocsparse_int* nnz)
{
    --(*nnz);
}

static rocsparse_status rocsparse_csrgemm_nnz_calc(rocsparse_handle          handle,
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
{
    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D
        = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Compute number of intermediate products for each row
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
    hipLaunchKernelGGL((csrgemm_intermediate_products<CSRGEMM_SUB>),
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
                       info_C->csrgemm_info->mul,
                       info_C->csrgemm_info->add);
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

    // Determine maximum of all intermediate products
    RETURN_IF_HIP_ERROR(rocprim::reduce(nullptr,
                                        rocprim_size,
                                        csr_row_ptr_C,
                                        csr_row_ptr_C + m,
                                        0,
                                        m,
                                        rocprim::maximum<rocsparse_int>(),
                                        stream));
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_HIP_ERROR(rocprim::reduce(rocprim_buffer,
                                        rocprim_size,
                                        csr_row_ptr_C,
                                        csr_row_ptr_C + m,
                                        0,
                                        m,
                                        rocprim::maximum<rocsparse_int>(),
                                        stream));

    rocsparse_int int_max;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &int_max, csr_row_ptr_C + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    rocsparse_int* d_group_offset = reinterpret_cast<rocsparse_int*>(buffer);
    buffer += sizeof(rocsparse_int) * 256;

    // Group size buffer
    rocsparse_int h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(rocsparse_int) * CSRGEMM_MAXGROUPS);

    // Permutation array
    rocsparse_int* d_perm = nullptr;

    // If maximum of intermediate products exceeds 32, we process the rows in groups of
    // similar sized intermediate products
    if(int_max > 32)
    {
        // Group size buffer
        rocsparse_int* d_group_size = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += sizeof(rocsparse_int) * 256 * CSRGEMM_MAXGROUPS;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        hipLaunchKernelGGL((csrgemm_group_reduce_part1<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
                           dim3(CSRGEMM_DIM),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           m,
                           csr_row_ptr_C,
                           d_group_size);

        hipLaunchKernelGGL((csrgemm_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
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
                                                    rocprim::plus<rocsparse_int>(),
                                                    stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<rocsparse_int>(),
                                                    stream));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(rocsparse_int) * CSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // Permutation temporary arrays
        rocsparse_int* tmp_vals = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        rocsparse_int* tmp_keys = reinterpret_cast<rocsparse_int*>(buffer);
        buffer += ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, m, tmp_perm));

        rocprim::double_buffer<rocsparse_int> d_keys(csr_row_ptr_C, tmp_keys);
        rocprim::double_buffer<rocsparse_int> d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, rocprim_size, d_keys, d_vals, m, 0, 3, stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            rocprim_buffer, rocprim_size, d_keys, d_vals, m, 0, 3, stream));

        d_perm = d_vals.current();

        // Release tmp_keys buffer
        buffer -= ((sizeof(rocsparse_int) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(rocsparse_int), stream));
    }

    // Compute non-zero entries per row for each group

    // Group 0: 0 - 32 intermediate products
    if(h_group_size[0] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 4
#define CSRGEMM_HASHSIZE 32
        hipLaunchKernelGGL(
            (csrgemm_nnz_wf_per_row<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_HASHSIZE, CSRGEMM_NNZ_HASH>),
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
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
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
        hipLaunchKernelGGL(
            (csrgemm_nnz_wf_per_row<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_HASHSIZE, CSRGEMM_NNZ_HASH>),
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
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
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
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
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
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
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
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
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
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
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
        hipLaunchKernelGGL((csrgemm_nnz_block_per_row<CSRGEMM_DIM,
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
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
        rocsparse_int* workspace_B = nullptr;

        if(info_C->csrgemm_info->mul == true)
        {
            // Allocate additional buffer for C = alpha * A * B
            RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace_B, sizeof(rocsparse_int) * nnz_A));
        }

        hipLaunchKernelGGL(
            (csrgemm_nnz_block_per_row_multipass<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_CHUNKSIZE>),
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
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);

        if(info_C->csrgemm_info->mul == true)
        {
            RETURN_IF_HIP_ERROR(hipFree(workspace_B));
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
                                                descr_C->base,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                rocprim_size,
                                                csr_row_ptr_C,
                                                csr_row_ptr_C,
                                                descr_C->base,
                                                m + 1,
                                                rocprim::plus<rocsparse_int>(),
                                                stream));

    // Store nnz of C
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, csr_row_ptr_C + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));

        // Adjust nnz by index base
        if(descr_C->base == rocsparse_index_base_one)
        {
            hipLaunchKernelGGL((csrgemm_index_base), dim3(1), dim3(1), 0, stream, nnz_C);
        }
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpy(nnz_C, csr_row_ptr_C + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // Adjust nnz by index base
        *nnz_C -= descr_C->base;
    }

    return rocsparse_status_success;
}

static rocsparse_status rocsparse_csrgemm_nnz_mult(rocsparse_handle          handle,
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
                                                   const rocsparse_mat_descr descr_C,
                                                   rocsparse_int*            csr_row_ptr_C,
                                                   rocsparse_int*            nnz_C,
                                                   const rocsparse_mat_info  info_C,
                                                   void*                     temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr
       || descr_B == nullptr || csr_row_ptr_B == nullptr || csr_col_ind_B == nullptr
       || descr_C == nullptr || csr_row_ptr_C == nullptr || nnz_C == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr_A->base != rocsparse_index_base_zero && descr_A->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_B->base != rocsparse_index_base_zero && descr_B->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_C->base != rocsparse_index_base_zero && descr_C->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(rocsparse_int), stream));
        }
        else
        {
            *nnz_C = 0;
        }

        return rocsparse_status_success;
    }

    // Perform nnz calculation
    return rocsparse_csrgemm_nnz_calc(handle,
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
                                      nullptr,
                                      0,
                                      nullptr,
                                      nullptr,
                                      descr_C,
                                      csr_row_ptr_C,
                                      nnz_C,
                                      info_C,
                                      temp_buffer);
}

static rocsparse_status rocsparse_csrgemm_nnz_scal(rocsparse_handle          handle,
                                                   rocsparse_int             m,
                                                   rocsparse_int             n,
                                                   const rocsparse_mat_descr descr_D,
                                                   rocsparse_int             nnz_D,
                                                   const rocsparse_int*      csr_row_ptr_D,
                                                   const rocsparse_int*      csr_col_ind_D,
                                                   const rocsparse_mat_descr descr_C,
                                                   rocsparse_int*            csr_row_ptr_C,
                                                   rocsparse_int*            nnz_C,
                                                   const rocsparse_mat_info  info_C,
                                                   void*                     temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_D == nullptr || csr_row_ptr_D == nullptr || csr_col_ind_D == nullptr
       || descr_C == nullptr || csr_row_ptr_C == nullptr || nnz_C == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr_C->base != rocsparse_index_base_zero && descr_C->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr_D->base != rocsparse_index_base_zero && descr_D->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0 || nnz_D == 0)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(rocsparse_int), stream));
        }
        else
        {
            *nnz_C = 0;
        }

        return rocsparse_status_success;
    }

    // When scaling a matrix, nnz of C will always be equal to nnz of D
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(nnz_C, &nnz_D, sizeof(rocsparse_int), hipMemcpyHostToDevice, stream));
    }
    else
    {
        *nnz_C = nnz_D;
    }

    // Copy row pointers
#define CSRGEMM_DIM 1024
    hipLaunchKernelGGL((csrgemm_copy),
                       dim3(m / CSRGEMM_DIM + 1),
                       dim3(CSRGEMM_DIM),
                       0,
                       stream,
                       m + 1,
                       csr_row_ptr_D,
                       csr_row_ptr_C,
                       descr_D->base,
                       descr_C->base);
#undef CSRGEMM_DIM

    return rocsparse_status_success;
}

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
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_csrgemm_nnz",
              trans_A,
              trans_B,
              m,
              n,
              k,
              (const void*&)descr_A,
              nnz_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)descr_B,
              nnz_B,
              (const void*&)csr_row_ptr_B,
              (const void*&)csr_col_ind_B,
              (const void*&)descr_D,
              nnz_D,
              (const void*&)csr_row_ptr_D,
              (const void*&)csr_col_ind_D,
              (const void*&)descr_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)nnz_C,
              (const void*&)info_C,
              (const void*&)temp_buffer);

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid rocsparse_csrgemm_info
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Either mult, add or multadd need to be performed
    if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
    {
        // C = alpha * A * B + beta * D
        // TODO
        return rocsparse_status_not_implemented;
    }
    else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
    {
        // C = alpha * A * B
        return rocsparse_csrgemm_nnz_mult(handle,
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
    }
    else if(info_C->csrgemm_info->mul == false && info_C->csrgemm_info->add == true)
    {
        // C = beta * D
        return rocsparse_csrgemm_nnz_scal(handle,
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
    }
    else
    {
        // C = 0
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrgemm_buffer_size(rocsparse_handle          handle,
                                                           rocsparse_operation       trans_A,
                                                           rocsparse_operation       trans_B,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           rocsparse_int             k,
                                                           const float*              alpha,
                                                           const rocsparse_mat_descr descr_A,
                                                           rocsparse_int             nnz_A,
                                                           const rocsparse_int*      csr_row_ptr_A,
                                                           const rocsparse_int*      csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int             nnz_B,
                                                           const rocsparse_int*      csr_row_ptr_B,
                                                           const rocsparse_int*      csr_col_ind_B,
                                                           const float*              beta,
                                                           const rocsparse_mat_descr descr_D,
                                                           rocsparse_int             nnz_D,
                                                           const rocsparse_int*      csr_row_ptr_D,
                                                           const rocsparse_int*      csr_col_ind_D,
                                                           rocsparse_mat_info        info_C,
                                                           size_t*                   buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
}

extern "C" rocsparse_status rocsparse_dcsrgemm_buffer_size(rocsparse_handle          handle,
                                                           rocsparse_operation       trans_A,
                                                           rocsparse_operation       trans_B,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           rocsparse_int             k,
                                                           const double*             alpha,
                                                           const rocsparse_mat_descr descr_A,
                                                           rocsparse_int             nnz_A,
                                                           const rocsparse_int*      csr_row_ptr_A,
                                                           const rocsparse_int*      csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int             nnz_B,
                                                           const rocsparse_int*      csr_row_ptr_B,
                                                           const rocsparse_int*      csr_col_ind_B,
                                                           const double*             beta,
                                                           const rocsparse_mat_descr descr_D,
                                                           rocsparse_int             nnz_D,
                                                           const rocsparse_int*      csr_row_ptr_D,
                                                           const rocsparse_int*      csr_col_ind_D,
                                                           rocsparse_mat_info        info_C,
                                                           size_t*                   buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
}

extern "C" rocsparse_status rocsparse_ccsrgemm_buffer_size(rocsparse_handle               handle,
                                                           rocsparse_operation            trans_A,
                                                           rocsparse_operation            trans_B,
                                                           rocsparse_int                  m,
                                                           rocsparse_int                  n,
                                                           rocsparse_int                  k,
                                                           const rocsparse_float_complex* alpha,
                                                           const rocsparse_mat_descr      descr_A,
                                                           rocsparse_int                  nnz_A,
                                                           const rocsparse_int*      csr_row_ptr_A,
                                                           const rocsparse_int*      csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int             nnz_B,
                                                           const rocsparse_int*      csr_row_ptr_B,
                                                           const rocsparse_int*      csr_col_ind_B,
                                                           const rocsparse_float_complex* beta,
                                                           const rocsparse_mat_descr      descr_D,
                                                           rocsparse_int                  nnz_D,
                                                           const rocsparse_int* csr_row_ptr_D,
                                                           const rocsparse_int* csr_col_ind_D,
                                                           rocsparse_mat_info   info_C,
                                                           size_t*              buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
}

extern "C" rocsparse_status rocsparse_zcsrgemm_buffer_size(rocsparse_handle                handle,
                                                           rocsparse_operation             trans_A,
                                                           rocsparse_operation             trans_B,
                                                           rocsparse_int                   m,
                                                           rocsparse_int                   n,
                                                           rocsparse_int                   k,
                                                           const rocsparse_double_complex* alpha,
                                                           const rocsparse_mat_descr       descr_A,
                                                           rocsparse_int                   nnz_A,
                                                           const rocsparse_int*      csr_row_ptr_A,
                                                           const rocsparse_int*      csr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           rocsparse_int             nnz_B,
                                                           const rocsparse_int*      csr_row_ptr_B,
                                                           const rocsparse_int*      csr_col_ind_B,
                                                           const rocsparse_double_complex* beta,
                                                           const rocsparse_mat_descr       descr_D,
                                                           rocsparse_int                   nnz_D,
                                                           const rocsparse_int* csr_row_ptr_D,
                                                           const rocsparse_int* csr_col_ind_D,
                                                           rocsparse_mat_info   info_C,
                                                           size_t*              buffer_size)
{
    return rocsparse_csrgemm_buffer_size_template(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
}

extern "C" rocsparse_status rocsparse_scsrgemm(rocsparse_handle          handle,
                                               rocsparse_operation       trans_A,
                                               rocsparse_operation       trans_B,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               rocsparse_int             k,
                                               const float*              alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int             nnz_A,
                                               const float*              csr_val_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int             nnz_B,
                                               const float*              csr_val_B,
                                               const rocsparse_int*      csr_row_ptr_B,
                                               const rocsparse_int*      csr_col_ind_B,
                                               const float*              beta,
                                               const rocsparse_mat_descr descr_D,
                                               rocsparse_int             nnz_D,
                                               const float*              csr_val_D,
                                               const rocsparse_int*      csr_row_ptr_D,
                                               const rocsparse_int*      csr_col_ind_D,
                                               const rocsparse_mat_descr descr_C,
                                               float*                    csr_val_C,
                                               const rocsparse_int*      csr_row_ptr_C,
                                               rocsparse_int*            csr_col_ind_C,
                                               const rocsparse_mat_info  info_C,
                                               void*                     temp_buffer)
{
    return rocsparse_csrgemm_template(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
}

extern "C" rocsparse_status rocsparse_dcsrgemm(rocsparse_handle          handle,
                                               rocsparse_operation       trans_A,
                                               rocsparse_operation       trans_B,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               rocsparse_int             k,
                                               const double*             alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int             nnz_A,
                                               const double*             csr_val_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int             nnz_B,
                                               const double*             csr_val_B,
                                               const rocsparse_int*      csr_row_ptr_B,
                                               const rocsparse_int*      csr_col_ind_B,
                                               const double*             beta,
                                               const rocsparse_mat_descr descr_D,
                                               rocsparse_int             nnz_D,
                                               const double*             csr_val_D,
                                               const rocsparse_int*      csr_row_ptr_D,
                                               const rocsparse_int*      csr_col_ind_D,
                                               const rocsparse_mat_descr descr_C,
                                               double*                   csr_val_C,
                                               const rocsparse_int*      csr_row_ptr_C,
                                               rocsparse_int*            csr_col_ind_C,
                                               const rocsparse_mat_info  info_C,
                                               void*                     temp_buffer)
{
    return rocsparse_csrgemm_template(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
}

extern "C" rocsparse_status rocsparse_ccsrgemm(rocsparse_handle               handle,
                                               rocsparse_operation            trans_A,
                                               rocsparse_operation            trans_B,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               rocsparse_int                  k,
                                               const rocsparse_float_complex* alpha,
                                               const rocsparse_mat_descr      descr_A,
                                               rocsparse_int                  nnz_A,
                                               const rocsparse_float_complex* csr_val_A,
                                               const rocsparse_int*           csr_row_ptr_A,
                                               const rocsparse_int*           csr_col_ind_A,
                                               const rocsparse_mat_descr      descr_B,
                                               rocsparse_int                  nnz_B,
                                               const rocsparse_float_complex* csr_val_B,
                                               const rocsparse_int*           csr_row_ptr_B,
                                               const rocsparse_int*           csr_col_ind_B,
                                               const rocsparse_float_complex* beta,
                                               const rocsparse_mat_descr      descr_D,
                                               rocsparse_int                  nnz_D,
                                               const rocsparse_float_complex* csr_val_D,
                                               const rocsparse_int*           csr_row_ptr_D,
                                               const rocsparse_int*           csr_col_ind_D,
                                               const rocsparse_mat_descr      descr_C,
                                               rocsparse_float_complex*       csr_val_C,
                                               const rocsparse_int*           csr_row_ptr_C,
                                               rocsparse_int*                 csr_col_ind_C,
                                               const rocsparse_mat_info       info_C,
                                               void*                          temp_buffer)
{
    return rocsparse_csrgemm_template(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
}

extern "C" rocsparse_status rocsparse_zcsrgemm(rocsparse_handle                handle,
                                               rocsparse_operation             trans_A,
                                               rocsparse_operation             trans_B,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               rocsparse_int                   k,
                                               const rocsparse_double_complex* alpha,
                                               const rocsparse_mat_descr       descr_A,
                                               rocsparse_int                   nnz_A,
                                               const rocsparse_double_complex* csr_val_A,
                                               const rocsparse_int*            csr_row_ptr_A,
                                               const rocsparse_int*            csr_col_ind_A,
                                               const rocsparse_mat_descr       descr_B,
                                               rocsparse_int                   nnz_B,
                                               const rocsparse_double_complex* csr_val_B,
                                               const rocsparse_int*            csr_row_ptr_B,
                                               const rocsparse_int*            csr_col_ind_B,
                                               const rocsparse_double_complex* beta,
                                               const rocsparse_mat_descr       descr_D,
                                               rocsparse_int                   nnz_D,
                                               const rocsparse_double_complex* csr_val_D,
                                               const rocsparse_int*            csr_row_ptr_D,
                                               const rocsparse_int*            csr_col_ind_D,
                                               const rocsparse_mat_descr       descr_C,
                                               rocsparse_double_complex*       csr_val_C,
                                               const rocsparse_int*            csr_row_ptr_C,
                                               rocsparse_int*                  csr_col_ind_C,
                                               const rocsparse_mat_info        info_C,
                                               void*                           temp_buffer)
{
    return rocsparse_csrgemm_template(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
}

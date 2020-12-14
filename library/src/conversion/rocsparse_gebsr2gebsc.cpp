/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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

#include "rocsparse_gebsr2gebsc.hpp"
#include "definitions.h"
#include "utility.h"

#include "gebsr2gebsc_device.h"
#include <rocprim/rocprim.hpp>

template <typename T>
rocsparse_status rocsparse_gebsr2gebsc_template(rocsparse_handle     handle,
                                                rocsparse_int        mb,
                                                rocsparse_int        nb,
                                                rocsparse_int        nnzb,
                                                const T*             bsr_val,
                                                const rocsparse_int* bsr_row_ptr,
                                                const rocsparse_int* bsr_col_ind,
                                                rocsparse_int        row_block_dim,
                                                rocsparse_int        col_block_dim,
                                                T*                   bsc_val,
                                                rocsparse_int*       bsc_row_ind,
                                                rocsparse_int*       bsc_col_ptr,
                                                rocsparse_action     copy_values,
                                                rocsparse_index_base idx_base,
                                                void*                temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgebsr2gebsc"),
              mb,
              nb,
              nnzb,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              row_block_dim,
              col_block_dim,
              (const void*&)bsc_val,
              (const void*&)bsc_row_ind,
              (const void*&)bsc_col_ptr,
              copy_values,
              idx_base,
              (const void*&)temp_buffer);

    log_bench(
        handle, "./rocsparse-bench -f gebsr2gebsc -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check rocsparse_action
    if(copy_values != rocsparse_action_symbolic && copy_values != rocsparse_action_numeric)
    {
        return rocsparse_status_invalid_value;
    }

    // Check index base
    if(idx_base != rocsparse_index_base_zero && idx_base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(mb < 0 || nb < 0 || nnzb < 0 || row_block_dim < 0 || col_block_dim < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0 || row_block_dim == 0 || col_block_dim == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if((bsr_val == nullptr && copy_values == rocsparse_action_numeric) || bsr_row_ptr == nullptr
       || bsr_col_ind == nullptr || (bsc_val == nullptr && copy_values == rocsparse_action_numeric)
       || bsc_row_ind == nullptr || bsc_col_ptr == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Stream
    hipStream_t stream = handle->stream;

    unsigned int startbit = 0;
    unsigned int endbit   = rocsparse_clz(nb);

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // work1 buffer
    rocsparse_int* tmp_work1 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnzb - 1) / 256 + 1) * 256;

    // work2 buffer
    rocsparse_int* tmp_work2 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnzb - 1) / 256 + 1) * 256;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnzb - 1) / 256 + 1) * 256;

    // rocprim buffer
    void* tmp_rocprim = reinterpret_cast<void*>(ptr);

    // Load CSR column indices into work1 buffer
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        tmp_work1, bsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyDeviceToDevice, stream));

    if(copy_values == rocsparse_action_symbolic)
    {
        // action symbolic

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, bsr_row_ptr, nnzb, mb, bsc_row_ind, idx_base));
        // Stable sort COO by columns
        rocprim::double_buffer<rocsparse_int> keys(tmp_work1, tmp_perm);
        rocprim::double_buffer<rocsparse_int> vals(bsc_row_ind, tmp_work2);

        size_t size = 0;

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnzb, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnzb, startbit, endbit, stream));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, keys.current(), nnzb, nb, bsc_col_ptr, idx_base));

        // Copy bsc_row_ind if not current
        if(vals.current() != bsc_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(bsc_row_ind,
                                               vals.current(),
                                               sizeof(rocsparse_int) * nnzb,
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
    }
    else
    {
        // action numeric

        // Create identitiy permutation
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnzb, tmp_perm));

        // Stable sort COO by columns
        rocprim::double_buffer<rocsparse_int> keys(tmp_work1, bsc_row_ind);
        rocprim::double_buffer<rocsparse_int> vals(tmp_perm, tmp_work2);

        size_t size = 0;

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnzb, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnzb, startbit, endbit, stream));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, keys.current(), nnzb, nb, bsc_col_ptr, idx_base));

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, bsr_row_ptr, nnzb, mb, tmp_work1, idx_base));

// Permute row indices and values
#define GEBSR2GEBSC_DIM 512
        dim3 gebsr2gebsc_blocks((nnzb - 1) / GEBSR2GEBSC_DIM + 1);
        dim3 gebsr2gebsc_threads(GEBSR2GEBSC_DIM);

        hipLaunchKernelGGL((gebsr2gebsc_permute_kernel<GEBSR2GEBSC_DIM>),
                           gebsr2gebsc_blocks,
                           gebsr2gebsc_threads,
                           0,
                           stream,
                           nnzb,
                           col_block_dim * row_block_dim,
                           tmp_work1,
                           bsr_val,
                           vals.current(),
                           bsc_row_ind,
                           bsc_val);
#undef GEBSR2GEBSC_DIM
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gebsr2gebsc_buffer_size_template(rocsparse_handle     handle,
                                                            rocsparse_int        mb,
                                                            rocsparse_int        nb,
                                                            rocsparse_int        nnzb,
                                                            const T*             bsr_val,
                                                            const rocsparse_int* bsr_row_ptr,
                                                            const rocsparse_int* bsr_col_ind,
                                                            rocsparse_int        row_block_dim,
                                                            rocsparse_int        col_block_dim,
                                                            size_t*              p_buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              "rocsparse_gebsr2gebsc_buffer_size",
              mb,
              nb,
              nnzb,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              row_block_dim,
              col_block_dim,
              (const void*&)p_buffer_size);

    // Check sizes
    if(mb < 0 || nb < 0 || nnzb < 0 || row_block_dim < 0 || col_block_dim < 0)
    {
        //      std::cout << "return invalid size "  << rocsparse_status_invalid_size << std::endl;
        return rocsparse_status_invalid_size;
    }

    // Check buffer size argument
    if(p_buffer_size == nullptr)
    {
        //      std::cout << "invalid pointer "<<std::endl;
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0 || row_block_dim == 0 || col_block_dim == 0)
    {
        // Do not return 0 as buffer size
        *p_buffer_size = 4;
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    hipStream_t stream = handle->stream;

    // Determine rocprim buffer size
    rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(p_buffer_size);

    rocprim::double_buffer<rocsparse_int> dummy(ptr, ptr);

    RETURN_IF_HIP_ERROR(
        rocprim::radix_sort_pairs(nullptr, *p_buffer_size, dummy, dummy, nnzb, 0, 32, stream));

    *p_buffer_size = ((*p_buffer_size - 1) / 256 + 1) * 256;

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays
    *p_buffer_size += sizeof(rocsparse_int) * ((nnzb - 1) / 256 + 1) * 256;
    *p_buffer_size += sizeof(rocsparse_int) * ((nnzb - 1) / 256 + 1) * 256;
    *p_buffer_size += sizeof(rocsparse_int) * ((nnzb - 1) / 256 + 1) * 256;

    // Do not return 0 as size
    if(*p_buffer_size == 0)
    {
        *p_buffer_size = 4;
    }

    return rocsparse_status_success;
}

//
// EXTERN C WRAPPING OF THE TEMPLATE.
//

#define C_IMPL(NAME, TYPE)                                                \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle,         \
                                     rocsparse_int        mb,             \
                                     rocsparse_int        nb,             \
                                     rocsparse_int        nnzb,           \
                                     const TYPE*          bsr_val,        \
                                     const rocsparse_int* bsr_row_ptr,    \
                                     const rocsparse_int* bsr_col_ind,    \
                                     rocsparse_int        row_block_dim,  \
                                     rocsparse_int        col_block_dim,  \
                                     size_t*              p_buffer_size)  \
    {                                                                     \
        return rocsparse_gebsr2gebsc_buffer_size_template(handle,         \
                                                          mb,             \
                                                          nb,             \
                                                          nnzb,           \
                                                          bsr_val,        \
                                                          bsr_row_ptr,    \
                                                          bsr_col_ind,    \
                                                          row_block_dim,  \
                                                          col_block_dim,  \
                                                          p_buffer_size); \
    }

C_IMPL(rocsparse_sgebsr2gebsc_buffer_size, float);
C_IMPL(rocsparse_dgebsr2gebsc_buffer_size, double);
C_IMPL(rocsparse_cgebsr2gebsc_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgebsr2gebsc_buffer_size, rocsparse_double_complex);

#undef C_IMPL

//
// EXTERN C WRAPPING OF THE TEMPLATE.
//
#define C_IMPL(NAME, TYPE)                                               \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle,        \
                                     rocsparse_int        mb,            \
                                     rocsparse_int        nb,            \
                                     rocsparse_int        nnzb,          \
                                     const TYPE*          bsr_val,       \
                                     const rocsparse_int* bsr_row_ptr,   \
                                     const rocsparse_int* bsr_col_ind,   \
                                     rocsparse_int        row_block_dim, \
                                     rocsparse_int        col_block_dim, \
                                     TYPE*                bsc_val,       \
                                     rocsparse_int*       bsc_row_ind,   \
                                     rocsparse_int*       bsc_col_ptr,   \
                                     rocsparse_action     copy_values,   \
                                     rocsparse_index_base idx_base,      \
                                     void*                buffer)        \
    {                                                                    \
        return rocsparse_gebsr2gebsc_template(handle,                    \
                                              mb,                        \
                                              nb,                        \
                                              nnzb,                      \
                                              bsr_val,                   \
                                              bsr_row_ptr,               \
                                              bsr_col_ind,               \
                                              row_block_dim,             \
                                              col_block_dim,             \
                                              bsc_val,                   \
                                              bsc_row_ind,               \
                                              bsc_col_ptr,               \
                                              copy_values,               \
                                              idx_base,                  \
                                              buffer);                   \
    }

C_IMPL(rocsparse_sgebsr2gebsc, float);
C_IMPL(rocsparse_dgebsr2gebsc, double);
C_IMPL(rocsparse_cgebsr2gebsc, rocsparse_float_complex);
C_IMPL(rocsparse_zgebsr2gebsc, rocsparse_double_complex);
#undef C_IMPL

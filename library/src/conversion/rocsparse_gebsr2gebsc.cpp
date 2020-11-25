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

/*! \file */
/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "rocsparse_gebsr2csr.hpp"
#include "definitions.h"
#include "utility.h"

#include "gebsr2csr_device.h"

template <typename T>
rocsparse_status rocsparse_gebsr2csr_template(rocsparse_handle          handle,
                                              rocsparse_direction       direction,
                                              rocsparse_int             mb,
                                              rocsparse_int             nb,
                                              const rocsparse_mat_descr bsr_descr,
                                              const T*                  bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             row_block_dim,
                                              rocsparse_int             col_block_dim,
                                              const rocsparse_mat_descr csr_descr,
                                              T*                        csr_val,
                                              rocsparse_int*            csr_row_ptr,
                                              rocsparse_int*            csr_col_ind)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid descriptors
    if(bsr_descr == nullptr || csr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgebsr2csr"),
              mb,
              nb,
              bsr_descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              row_block_dim,
              col_block_dim,
              csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind);

    log_bench(handle, "./rocsparse-bench -f gebsr2csr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check direction
    if(direction != rocsparse_direction_row && direction != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(mb < 0 || nb < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check block dimension
    if(row_block_dim <= 0 || col_block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0)
    {
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
    else if(csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(bsr_descr == nullptr || csr_descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check the description type of the matrix.
    if(rocsparse_matrix_type_general != bsr_descr->type
       || rocsparse_matrix_type_general != csr_descr->type)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    constexpr rocsparse_int block_size     = 256;
    rocsparse_int           wavefront_size = handle->wavefront_size;
    rocsparse_int           grid_size      = mb * row_block_dim / (block_size / wavefront_size);
    if(mb * row_block_dim % (block_size / wavefront_size) != 0)
    {
        grid_size++;
    }

    dim3 blocks(grid_size);
    dim3 threads(block_size);

    if(wavefront_size == 32)
    {
        if(direction == rocsparse_direction_row)
        {
            hipLaunchKernelGGL((gebsr2csr_kernel<rocsparse_direction_row, block_size, 32>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
        else
        {
            hipLaunchKernelGGL((gebsr2csr_kernel<rocsparse_direction_column, block_size, 32>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }
    else
    {
        assert(wavefront_size == 64);
        if(direction == rocsparse_direction_row)
        {
            hipLaunchKernelGGL((gebsr2csr_kernel<rocsparse_direction_row, block_size, 64>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
        else
        {
            hipLaunchKernelGGL((gebsr2csr_kernel<rocsparse_direction_column, block_size, 64>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }

    return rocsparse_status_success;
}

extern "C" {
#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_, typer_)                                         \
    rocsparse_status name_(rocsparse_handle          handle,        \
                           rocsparse_direction       dir,           \
                           rocsparse_int             mb,            \
                           rocsparse_int             nb,            \
                           const rocsparse_mat_descr bsr_descr,     \
                           const typer_*             bsr_val,       \
                           const rocsparse_int*      bsr_row_ptr,   \
                           const rocsparse_int*      bsr_col_ind,   \
                           rocsparse_int             row_block_dim, \
                           rocsparse_int             col_block_dim, \
                           const rocsparse_mat_descr csr_descr,     \
                           typer_*                   csr_val,       \
                           rocsparse_int*            csr_row_ptr,   \
                           rocsparse_int*            csr_col_ind)   \
    {                                                               \
        return rocsparse_gebsr2csr_template(handle,                 \
                                            dir,                    \
                                            mb,                     \
                                            nb,                     \
                                            bsr_descr,              \
                                            bsr_val,                \
                                            bsr_row_ptr,            \
                                            bsr_col_ind,            \
                                            row_block_dim,          \
                                            col_block_dim,          \
                                            csr_descr,              \
                                            csr_val,                \
                                            csr_row_ptr,            \
                                            csr_col_ind);           \
    }

IMPL(rocsparse_sgebsr2csr, float);
IMPL(rocsparse_dgebsr2csr, double);
IMPL(rocsparse_cgebsr2csr, rocsparse_float_complex);
IMPL(rocsparse_zgebsr2csr, rocsparse_double_complex);

#undef IMPL

} // extern "C"

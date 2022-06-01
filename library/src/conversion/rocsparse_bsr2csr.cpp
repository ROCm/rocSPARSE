/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_bsr2csr.hpp"
#include "definitions.h"
#include "utility.h"

#include "bsr2csr_device.h"

#define launch_bsr2csr_unroll_kernel(direction, block_size, bsr_block_dim)            \
    hipLaunchKernelGGL((bsr2csr_unroll_kernel<direction, block_size, bsr_block_dim>), \
                       blocks,                                                        \
                       threads,                                                       \
                       0,                                                             \
                       stream,                                                        \
                       mb,                                                            \
                       nb,                                                            \
                       bsr_descr->base,                                               \
                       bsr_val,                                                       \
                       bsr_row_ptr,                                                   \
                       bsr_col_ind,                                                   \
                       csr_descr->base,                                               \
                       csr_val,                                                       \
                       csr_row_ptr,                                                   \
                       csr_col_ind);

template <typename T>
rocsparse_status rocsparse_bsr2csr_template(rocsparse_handle          handle,
                                            rocsparse_direction       direction,
                                            rocsparse_int             mb,
                                            rocsparse_int             nb,
                                            const rocsparse_mat_descr bsr_descr,
                                            const T*                  bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
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
              replaceX<T>("rocsparse_Xbsr2csr"),
              mb,
              nb,
              bsr_descr,
              (const void*&)bsr_val,
              (const void*&)bsr_row_ptr,
              (const void*&)bsr_col_ind,
              block_dim,
              csr_descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind);

    log_bench(handle, "./rocsparse-bench -f bsr2csr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check direction
    if(rocsparse_enum_utils::is_invalid(direction))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix sorting mode
    if(bsr_descr->storage_mode != rocsparse_storage_mode_sorted
       || csr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || nb < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check block dimension
    if(block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr == nullptr || csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(bsr_val == nullptr && bsr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        rocsparse_int nnzb = (end - start);

        if(nnzb != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check the description type of the matrix.
    if(rocsparse_matrix_type_general != bsr_descr->type
       || rocsparse_matrix_type_general != csr_descr->type)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    if(block_dim == 1)
    {
        constexpr rocsparse_int block_size = 256;
        rocsparse_int           grid_size  = mb / block_size;
        if(mb % block_size != 0)
        {
            grid_size++;
        }

        dim3 blocks(grid_size);
        dim3 threads(block_size);

        hipLaunchKernelGGL((bsr2csr_block_dim_equals_one_kernel<block_size>),
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
                           csr_descr->base,
                           csr_val,
                           csr_row_ptr,
                           csr_col_ind);

        return rocsparse_status_success;
    }

    constexpr rocsparse_int block_size = 256;
    constexpr rocsparse_int warp_size  = 64;
    rocsparse_int           grid_size  = mb * block_dim / (block_size / warp_size);
    if(mb * block_dim % (block_size / warp_size) != 0)
    {
        grid_size++;
    }

    dim3 blocks(grid_size);
    dim3 threads(block_size);

    if(direction == rocsparse_direction_row)
    {
        if(block_dim == 2)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 2);
        }
        else if(block_dim == 3)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 3);
        }
        else if(block_dim == 4)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 4);
        }
        else if(block_dim == 5)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 5);
        }
        else if(block_dim == 6)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 6);
        }
        else if(block_dim == 7)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 7);
        }
        else if(block_dim == 8)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 8);
        }
        else if(block_dim == 9)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 9);
        }
        else if(block_dim == 10)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 10);
        }
        else if(block_dim == 11)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 11);
        }
        else if(block_dim == 12)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 12);
        }
        else if(block_dim == 13)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 13);
        }
        else if(block_dim == 14)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 14);
        }
        else if(block_dim == 15)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 15);
        }
        else if(block_dim == 16)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_row, block_size, 16);
        }
        else
        {
            hipLaunchKernelGGL((bsr2csr_kernel<rocsparse_direction_row, block_size>),
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
                               block_dim,
                               csr_descr->base,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }
    else
    {
        if(block_dim == 2)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 2);
        }
        else if(block_dim == 3)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 3);
        }
        else if(block_dim == 4)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 4);
        }
        else if(block_dim == 5)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 5);
        }
        else if(block_dim == 6)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 6);
        }
        else if(block_dim == 7)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 7);
        }
        else if(block_dim == 8)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 8);
        }
        else if(block_dim == 9)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 9);
        }
        else if(block_dim == 10)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 10);
        }
        else if(block_dim == 11)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 11);
        }
        else if(block_dim == 12)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 12);
        }
        else if(block_dim == 13)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 13);
        }
        else if(block_dim == 14)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 14);
        }
        else if(block_dim == 15)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 15);
        }
        else if(block_dim == 16)
        {
            launch_bsr2csr_unroll_kernel(rocsparse_direction_column, block_size, 16);
        }
        else
        {
            hipLaunchKernelGGL((bsr2csr_kernel<rocsparse_direction_column, block_size>),
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
                               block_dim,
                               csr_descr->base,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,      \
                                     rocsparse_direction       dir,         \
                                     rocsparse_int             mb,          \
                                     rocsparse_int             nb,          \
                                     const rocsparse_mat_descr bsr_descr,   \
                                     const TYPE*               bsr_val,     \
                                     const rocsparse_int*      bsr_row_ptr, \
                                     const rocsparse_int*      bsr_col_ind, \
                                     rocsparse_int             block_dim,   \
                                     const rocsparse_mat_descr csr_descr,   \
                                     TYPE*                     csr_val,     \
                                     rocsparse_int*            csr_row_ptr, \
                                     rocsparse_int*            csr_col_ind) \
    {                                                                       \
        return rocsparse_bsr2csr_template(handle,                           \
                                          dir,                              \
                                          mb,                               \
                                          nb,                               \
                                          bsr_descr,                        \
                                          bsr_val,                          \
                                          bsr_row_ptr,                      \
                                          bsr_col_ind,                      \
                                          block_dim,                        \
                                          csr_descr,                        \
                                          csr_val,                          \
                                          csr_row_ptr,                      \
                                          csr_col_ind);                     \
    }

C_IMPL(rocsparse_sbsr2csr, float);
C_IMPL(rocsparse_dbsr2csr, double);
C_IMPL(rocsparse_cbsr2csr, rocsparse_float_complex);
C_IMPL(rocsparse_zbsr2csr, rocsparse_double_complex);

#undef C_IMPL

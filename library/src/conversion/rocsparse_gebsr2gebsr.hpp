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

#pragma once
#ifndef ROCSPARSE_GEBSR2GEBSR_HPP
#define ROCSPARSE_GEBSR2GEBSR_HPP

#include "definitions.h"
#include "gebsr2gebsr_device.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include "rocsparse_csr2gebsr.hpp"
#include "rocsparse_gebsr2csr.hpp"

#include <hip/hip_runtime.h>

#define launch_gebsr2gebsr_fast_kernel(T, direction, block_size, segment_size)            \
    hipLaunchKernelGGL((gebsr2gebsr_fast_kernel<T, direction, block_size, segment_size>), \
                       grid_size,                                                         \
                       block_size,                                                        \
                       0,                                                                 \
                       stream,                                                            \
                       mb,                                                                \
                       nb,                                                                \
                       descr_A->base,                                                     \
                       bsr_val_A,                                                         \
                       bsr_row_ptr_A,                                                     \
                       bsr_col_ind_A,                                                     \
                       row_block_dim_A,                                                   \
                       col_block_dim_A,                                                   \
                       mb_c,                                                              \
                       nb_c,                                                              \
                       descr_C->base,                                                     \
                       bsr_val_C,                                                         \
                       bsr_row_ptr_C,                                                     \
                       bsr_col_ind_C,                                                     \
                       row_block_dim_C,                                                   \
                       col_block_dim_C);

template <typename T>
rocsparse_status rocsparse_gebsr2gebsr_buffer_size_template(rocsparse_handle          handle,
                                                            rocsparse_direction       dir,
                                                            rocsparse_int             mb,
                                                            rocsparse_int             nb,
                                                            rocsparse_int             nnzb,
                                                            const rocsparse_mat_descr descr_A,
                                                            const T*                  bsr_val_A,
                                                            const rocsparse_int*      bsr_row_ptr_A,
                                                            const rocsparse_int*      bsr_col_ind_A,
                                                            rocsparse_int row_block_dim_A,
                                                            rocsparse_int col_block_dim_A,
                                                            rocsparse_int row_block_dim_C,
                                                            rocsparse_int col_block_dim_C,
                                                            size_t*       buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid descriptor
    if(descr_A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgebsr2csr_buffer_size"),
              mb,
              nb,
              nnzb,
              descr_A,
              (const void*&)bsr_val_A,
              (const void*&)bsr_row_ptr_A,
              (const void*&)bsr_col_ind_A,
              row_block_dim_A,
              col_block_dim_A,
              row_block_dim_C,
              col_block_dim_C,
              (void*&)buffer_size);

    log_bench(handle,
              "./rocsparse-bench -f gebsr2gebsr_buffer_size -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(mb < 0 || nb < 0 | nnzb < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check block dimension
    if(row_block_dim_A <= 0 || col_block_dim_A <= 0 || row_block_dim_C <= 0 || col_block_dim_C <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(bsr_val_A == nullptr || bsr_row_ptr_A == nullptr || bsr_col_ind_A == nullptr
       || descr_A == nullptr || buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check the description type of the matrix.
    if(rocsparse_matrix_type_general != descr_A->type)
    {
        return rocsparse_status_not_implemented;
    }

    if(row_block_dim_C <= 32)
    {
        *buffer_size = 4;
    }
    else
    {
        // Perform the conversion gebsr->gebsr by performing gebsr->csr->gebsr.
        rocsparse_int m = mb * row_block_dim_A;

        *buffer_size = (m + 1) * sizeof(rocsparse_int)
                       + row_block_dim_A * col_block_dim_A * nnzb * sizeof(rocsparse_int)
                       + row_block_dim_A * col_block_dim_A * nnzb * sizeof(T);
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gebsr2gebsr_template(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_int             mb,
                                                rocsparse_int             nb,
                                                rocsparse_int             nnzb,
                                                const rocsparse_mat_descr descr_A,
                                                const T*                  bsr_val_A,
                                                const rocsparse_int*      bsr_row_ptr_A,
                                                const rocsparse_int*      bsr_col_ind_A,
                                                rocsparse_int             row_block_dim_A,
                                                rocsparse_int             col_block_dim_A,
                                                const rocsparse_mat_descr descr_C,
                                                T*                        bsr_val_C,
                                                rocsparse_int*            bsr_row_ptr_C,
                                                rocsparse_int*            bsr_col_ind_C,
                                                rocsparse_int             row_block_dim_C,
                                                rocsparse_int             col_block_dim_C,
                                                void*                     temp_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid descriptors
    if(descr_A == nullptr || descr_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgebsr2gebsr"),
              mb,
              nb,
              nnzb,
              descr_A,
              (const void*&)bsr_val_A,
              (const void*&)bsr_row_ptr_A,
              (const void*&)bsr_col_ind_A,
              row_block_dim_A,
              col_block_dim_A,
              descr_C,
              (const void*&)bsr_val_C,
              (const void*&)bsr_row_ptr_C,
              (const void*&)bsr_col_ind_C,
              row_block_dim_C,
              col_block_dim_C,
              (void*&)temp_buffer);

    log_bench(
        handle, "./rocsparse-bench -f gebsr2gebsr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check direction
    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(mb < 0 || nb < 0 || nnzb < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check block dimension
    if(row_block_dim_A <= 0 || col_block_dim_A <= 0 || row_block_dim_C <= 0 || col_block_dim_C <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_val_A == nullptr || bsr_row_ptr_A == nullptr || bsr_col_ind_A == nullptr
       || bsr_val_C == nullptr || bsr_row_ptr_C == nullptr || bsr_col_ind_C == nullptr
       || descr_A == nullptr || descr_C == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check the description type of the matrix.
    if(rocsparse_matrix_type_general != descr_A->type
       || rocsparse_matrix_type_general != descr_C->type)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

    rocsparse_int m    = mb * row_block_dim_A;
    rocsparse_int n    = nb * col_block_dim_A;
    rocsparse_int mb_c = (m + row_block_dim_C - 1) / row_block_dim_C;
    rocsparse_int nb_c = (n + col_block_dim_C - 1) / col_block_dim_C;

    rocsparse_int hstart = 0;
    rocsparse_int hend   = 0;
    RETURN_IF_HIP_ERROR(
        hipMemcpy(&hend, &bsr_row_ptr_C[mb_c], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(
        hipMemcpy(&hstart, &bsr_row_ptr_C[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    hipMemset(bsr_val_C, 0, (hend - hstart) * row_block_dim_C * col_block_dim_C * sizeof(T));

    // Common case where BSR block dimension is small
    if(row_block_dim_C <= 32)
    {
        // A 64 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        // row 2| 32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47 |
        // row 3| 48 49 50 51 | 52 53 54 55 | 56 57 58 59 | 60 61 62 63 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), 16 (full row),
        // 32 (half wavefront), or 64 (full wavefront). We assign one segment per
        // BSR block row where the segment size matches the block dimension as
        // closely as possible while still being greater than or equal to the
        // block dimension.

        // Note: block_size must be less than or equal to wavefront size
        rocsparse_int block_size   = row_block_dim_C > 16 ? 32 : 16;
        rocsparse_int segment_size = row_block_dim_C == 1 ? 2 : row_block_dim_C;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        rocsparse_int segments_per_block = block_size / segment_size;
        rocsparse_int grid_size          = (mb_c + segments_per_block - 1) / segments_per_block;

        if(dir == rocsparse_direction_row)
        {
            if(row_block_dim_C <= 2)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 2);
            }
            else if(row_block_dim_C <= 4)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 4);
            }
            else if(row_block_dim_C <= 8)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 8);
            }
            else if(row_block_dim_C <= 16)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 16);
            }
            else if(row_block_dim_C <= 32)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 32, 32);
            }
        }
        else
        {
            if(row_block_dim_C <= 2)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 2);
            }
            else if(row_block_dim_C <= 4)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 4);
            }
            else if(row_block_dim_C <= 8)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 8);
            }
            else if(row_block_dim_C <= 16)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 16);
            }
            else if(row_block_dim_C <= 32)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 32, 32);
            }
        }
    }
    else
    {
        rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(temp_buffer);

        rocsparse_int* csr_row_ptr = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += (m + 1);
        rocsparse_int* csr_col_ind = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += row_block_dim_A * col_block_dim_A * nnzb;
        T* csr_val = reinterpret_cast<T*>(ptr);
        ptr += row_block_dim_A * col_block_dim_A * nnzb;

        rocsparse_status status = rocsparse_gebsr2csr_template(handle,
                                                               dir,
                                                               mb,
                                                               nb,
                                                               descr_A,
                                                               bsr_val_A,
                                                               bsr_row_ptr_A,
                                                               bsr_col_ind_A,
                                                               row_block_dim_A,
                                                               col_block_dim_A,
                                                               descr_C,
                                                               csr_val,
                                                               csr_row_ptr,
                                                               csr_col_ind);

        if(status != rocsparse_status_success)
        {
            return status;
        }

        status = rocsparse_csr2gebsr_template(handle,
                                              dir,
                                              m,
                                              n,
                                              descr_C,
                                              csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              descr_C,
                                              bsr_val_C,
                                              bsr_row_ptr_C,
                                              bsr_col_ind_C,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              ptr);

        if(status != rocsparse_status_success)
        {
            return status;
        }
    }

    return rocsparse_status_success;
}

#endif // ROCSPARSE_GEBSR2GEBSR_HPP

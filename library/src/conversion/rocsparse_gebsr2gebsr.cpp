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

#include "rocsparse_gebsr2gebsr.hpp"
#include "definitions.h"
#include "utility.h"

#include "gebsr2csr_device.h"
#include "gebsr2gebsr_device.h"
#include "rocsparse_csr2gebsr.hpp"
#include "rocsparse_gebsr2csr.hpp"

#include <rocprim/rocprim.hpp>

#define launch_gebsr2gebsr_fast_kernel(T, direction, block_size, segment_size)         \
    hipLaunchKernelGGL((gebsr2gebsr_fast_kernel<direction, block_size, segment_size>), \
                       grid_size,                                                      \
                       block_size,                                                     \
                       0,                                                              \
                       stream,                                                         \
                       mb,                                                             \
                       nb,                                                             \
                       descr_A->base,                                                  \
                       bsr_val_A,                                                      \
                       bsr_row_ptr_A,                                                  \
                       bsr_col_ind_A,                                                  \
                       row_block_dim_A,                                                \
                       col_block_dim_A,                                                \
                       mb_c,                                                           \
                       nb_c,                                                           \
                       descr_C->base,                                                  \
                       bsr_val_C,                                                      \
                       bsr_row_ptr_C,                                                  \
                       bsr_col_ind_C,                                                  \
                       row_block_dim_C,                                                \
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
              (const void*&)buffer_size);

    log_bench(handle,
              "./rocsparse-bench -f gebsr2gebsr_buffer_size -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx>");

    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix sorting mode
    if(descr_A->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
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
    if(mb == 0 || nb == 0)
    {
        if(row_block_dim_C <= 32)
        {
            *buffer_size = 4;
        }
        else
        {
            // Perform the conversion gebsr->gebsr by performing gebsr->csr->gebsr.
            rocsparse_int m = mb * row_block_dim_A;

            *buffer_size = sizeof(rocsparse_int) * (m + 1)
                           + sizeof(rocsparse_int) * row_block_dim_A * col_block_dim_A * nnzb
                           + sizeof(T) * row_block_dim_A * col_block_dim_A * nnzb;
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr_A == nullptr || buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if((bsr_val_A == nullptr && bsr_col_ind_A != nullptr)
       || (bsr_val_A != nullptr && bsr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb != 0 && (bsr_val_A == nullptr && bsr_col_ind_A == nullptr))
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

        *buffer_size = sizeof(rocsparse_int) * (m + 1)
                       + sizeof(rocsparse_int) * row_block_dim_A * col_block_dim_A * nnzb
                       + sizeof(T) * row_block_dim_A * col_block_dim_A * nnzb;
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
              (const void*&)temp_buffer);

    log_bench(
        handle, "./rocsparse-bench -f gebsr2gebsr -r", replaceX<T>("X"), "--mtx <matrix.mtx>");

    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix sorting mode
    if(descr_A->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_C->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
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
    if(mb == 0 || nb == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr_A == nullptr || bsr_row_ptr_C == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if((bsr_val_A == nullptr && bsr_col_ind_A != nullptr)
       || (bsr_val_A != nullptr && bsr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if((bsr_val_C == nullptr && bsr_col_ind_C != nullptr)
       || (bsr_val_C != nullptr && bsr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb != 0 && (bsr_val_A == nullptr && bsr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_int m    = mb * row_block_dim_A;
    rocsparse_int n    = nb * col_block_dim_A;
    rocsparse_int mb_c = (m + row_block_dim_C - 1) / row_block_dim_C;
    rocsparse_int nb_c = (n + col_block_dim_C - 1) / col_block_dim_C;

    rocsparse_int start = 0;
    rocsparse_int end   = 0;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &end, &bsr_row_ptr_C[mb_c], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &start, &bsr_row_ptr_C[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    rocsparse_int nnzb_C = end - start;

    if(nnzb_C != 0 && (bsr_val_C == nullptr && bsr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    RETURN_IF_HIP_ERROR(hipMemsetAsync(
        bsr_val_C, 0, sizeof(T) * nnzb_C * row_block_dim_C * col_block_dim_C, handle->stream));

    // Check the description type of the matrix.
    if(rocsparse_matrix_type_general != descr_A->type
       || rocsparse_matrix_type_general != descr_C->type)
    {
        return rocsparse_status_not_implemented;
    }

    // Stream
    hipStream_t stream = handle->stream;

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
            else
            {
                // (row_block_dim_C <= 32)
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
            else
            {
                //  (row_block_dim_C <= 32)
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
        ptr += size_t(nnzb) * col_block_dim_A * row_block_dim_A;
        T* csr_val = reinterpret_cast<T*>(ptr);
        ptr += size_t(nnzb) * row_block_dim_A * col_block_dim_A;

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

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define launch_gebsr2gebsr_nnz_fast_kernel(block_size, segment_size)            \
    hipLaunchKernelGGL((gebsr2gebsr_nnz_fast_kernel<block_size, segment_size>), \
                       dim3(grid_size),                                         \
                       dim3(block_size),                                        \
                       0,                                                       \
                       handle->stream,                                          \
                       mb,                                                      \
                       nb,                                                      \
                       descr_A->base,                                           \
                       bsr_row_ptr_A,                                           \
                       bsr_col_ind_A,                                           \
                       row_block_dim_A,                                         \
                       col_block_dim_A,                                         \
                       mb_c,                                                    \
                       nb_c,                                                    \
                       descr_C->base,                                           \
                       bsr_row_ptr_C,                                           \
                       row_block_dim_C,                                         \
                       col_block_dim_C);

// Performs gebsr2csr conversion without computing the values. Used in rocsparse_gebsr2gebsr_nnz()
extern "C" rocsparse_status rocsparse_gebsr2csr_nnz(rocsparse_handle          handle,
                                                    rocsparse_direction       direction,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    const rocsparse_mat_descr bsr_descr,
                                                    const rocsparse_int*      bsr_row_ptr,
                                                    const rocsparse_int*      bsr_col_ind,
                                                    rocsparse_int             row_block_dim,
                                                    rocsparse_int             col_block_dim,
                                                    const rocsparse_mat_descr csr_descr,
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

    // Check matrix sorting mode
    if(bsr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }
    if(csr_descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_not_implemented;
    }

    // Check direction
    if(rocsparse_enum_utils::is_invalid(direction))
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
    if(bsr_row_ptr == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(bsr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                           &bsr_row_ptr[mb * row_block_dim],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
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
            hipLaunchKernelGGL((gebsr2csr_nnz_kernel<rocsparse_direction_row, block_size, 32>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_row_ptr,
                               csr_col_ind);
        }
        else
        {
            hipLaunchKernelGGL((gebsr2csr_nnz_kernel<rocsparse_direction_column, block_size, 32>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }
    else
    {
        assert(wavefront_size == 64);
        if(direction == rocsparse_direction_row)
        {
            hipLaunchKernelGGL((gebsr2csr_nnz_kernel<rocsparse_direction_row, block_size, 64>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_row_ptr,
                               csr_col_ind);
        }
        else
        {
            hipLaunchKernelGGL((gebsr2csr_nnz_kernel<rocsparse_direction_column, block_size, 64>),
                               blocks,
                               threads,
                               0,
                               stream,
                               mb,
                               nb,
                               bsr_descr->base,
                               bsr_row_ptr,
                               bsr_col_ind,
                               row_block_dim,
                               col_block_dim,
                               csr_descr->base,
                               csr_row_ptr,
                               csr_col_ind);
        }
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_gebsr2gebsr_nnz(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      const rocsparse_mat_descr descr_A,
                                                      const rocsparse_int*      bsr_row_ptr_A,
                                                      const rocsparse_int*      bsr_col_ind_A,
                                                      rocsparse_int             row_block_dim_A,
                                                      rocsparse_int             col_block_dim_A,
                                                      const rocsparse_mat_descr descr_C,
                                                      rocsparse_int*            bsr_row_ptr_C,
                                                      rocsparse_int             row_block_dim_C,
                                                      rocsparse_int             col_block_dim_C,
                                                      rocsparse_int* nnz_total_dev_host_ptr,
                                                      void*          temp_buffer)
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
              "rocsparse_gebsr2gebsr_nnz",
              mb,
              nb,
              nnzb,
              descr_A,
              (const void*&)bsr_row_ptr_A,
              (const void*&)bsr_col_ind_A,
              row_block_dim_A,
              col_block_dim_A,
              descr_C,
              (const void*&)bsr_row_ptr_C,
              row_block_dim_C,
              col_block_dim_C,
              (const void*&)nnz_total_dev_host_ptr,
              (const void*&)temp_buffer);

    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
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

    // Stream
    hipStream_t stream = handle->stream;

    rocsparse_int m    = mb * row_block_dim_A;
    rocsparse_int n    = nb * col_block_dim_A;
    rocsparse_int mb_c = (m + row_block_dim_C - 1) / row_block_dim_C;
    rocsparse_int nb_c = (n + col_block_dim_C - 1) / col_block_dim_C;

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0)
    {
        if(nullptr != nnz_total_dev_host_ptr)
        {
            rocsparse_pointer_mode mode;
            rocsparse_status       status = rocsparse_get_pointer_mode(handle, &mode);
            if(rocsparse_status_success != status)
            {
                return status;
            }

            constexpr rocsparse_int block_size = 1024;
            rocsparse_int           grid_size  = (mb_c + block_size - 1) / block_size;
            hipLaunchKernelGGL((gebsr2gebsr_fill_row_ptr_kernel<block_size>),
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               mb_c,
                               descr_C->base,
                               bsr_row_ptr_C);

            if(rocsparse_pointer_mode_device == mode)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    nnz_total_dev_host_ptr, 0, sizeof(rocsparse_int), handle->stream));
            }
            else
            {
                *nnz_total_dev_host_ptr = 0;
            }
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_row_ptr_A == nullptr || bsr_col_ind_A == nullptr || bsr_row_ptr_C == nullptr
       || descr_A == nullptr || descr_C == nullptr || nnz_total_dev_host_ptr == nullptr
       || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check the description type of the matrix.
    if(rocsparse_matrix_type_general != descr_A->type
       || rocsparse_matrix_type_general != descr_C->type)
    {
        return rocsparse_status_not_implemented;
    }

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
        // BSR block row of matrix C where the segment size matches the row block
        // dimension (row_block_dim_C) as closely as possible while still being
        // greater than or equal to the row block dimension.

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

        if(row_block_dim_C <= 2)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 2);
        }
        else if(row_block_dim_C <= 4)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 4);
        }
        else if(row_block_dim_C <= 8)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 8);
        }
        else if(row_block_dim_C <= 16)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 16);
        }
        else
        {
            // (row_block_dim_C <= 32)
            launch_gebsr2gebsr_nnz_fast_kernel(32, 32);
        }

        // Perform inclusive scan on bsr row pointer array
        auto   op = rocprim::plus<rocsparse_int>();
        size_t temp_storage_size_bytes;
        RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                    temp_storage_size_bytes,
                                                    bsr_row_ptr_C,
                                                    bsr_row_ptr_C,
                                                    mb_c + 1,
                                                    op,
                                                    handle->stream));

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= temp_storage_size_bytes)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                &temp_storage_ptr, temp_storage_size_bytes, handle->stream));
            temp_alloc = true;
        }

        RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(temp_storage_ptr,
                                                    temp_storage_size_bytes,
                                                    bsr_row_ptr_C,
                                                    bsr_row_ptr_C,
                                                    mb_c + 1,
                                                    op,
                                                    handle->stream));

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }

        // Compute nnz_total_dev_host_ptr
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipLaunchKernelGGL(gebsr2gebsr_compute_nnz_total_kernel<1>,
                               dim3(1),
                               dim3(1),
                               0,
                               handle->stream,
                               mb_c,
                               bsr_row_ptr_C,
                               nnz_total_dev_host_ptr);
        }
        else
        {
            rocsparse_int hstart = 0;
            rocsparse_int hend   = 0;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hend,
                                               &bsr_row_ptr_C[mb_c],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hstart,
                                               &bsr_row_ptr_C[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            *nnz_total_dev_host_ptr = hend - hstart;
        }
    }
    else
    {
        rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(temp_buffer);

        rocsparse_int* csr_row_ptr = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += (m + 1);
        rocsparse_int* csr_col_ind = reinterpret_cast<rocsparse_int*>(ptr);

        rocsparse_status status = rocsparse_gebsr2csr_nnz(handle,
                                                          dir,
                                                          mb,
                                                          nb,
                                                          descr_A,
                                                          bsr_row_ptr_A,
                                                          bsr_col_ind_A,
                                                          row_block_dim_A,
                                                          col_block_dim_A,
                                                          descr_C,
                                                          csr_row_ptr,
                                                          csr_col_ind);

        if(status != rocsparse_status_success)
        {
            return status;
        }

        status = rocsparse_csr2gebsr_nnz(handle,
                                         dir,
                                         m,
                                         n,
                                         descr_C,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         descr_C,
                                         bsr_row_ptr_C,
                                         row_block_dim_C,
                                         col_block_dim_C,
                                         nnz_total_dev_host_ptr,
                                         ptr);

        if(status != rocsparse_status_success)
        {
            return status;
        }
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_sgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                               rocsparse_direction       dir,
                                                               rocsparse_int             mb,
                                                               rocsparse_int             nb,
                                                               rocsparse_int             nnzb,
                                                               const rocsparse_mat_descr descr_A,
                                                               const float*              bsr_val_A,
                                                               const rocsparse_int* bsr_row_ptr_A,
                                                               const rocsparse_int* bsr_col_ind_A,
                                                               rocsparse_int        row_block_dim_A,
                                                               rocsparse_int        col_block_dim_A,
                                                               rocsparse_int        row_block_dim_C,
                                                               rocsparse_int        col_block_dim_C,
                                                               size_t*              buffer_size)
{
    return rocsparse_gebsr2gebsr_buffer_size_template(handle,
                                                      dir,
                                                      mb,
                                                      nb,
                                                      nnzb,
                                                      descr_A,
                                                      bsr_val_A,
                                                      bsr_row_ptr_A,
                                                      bsr_col_ind_A,
                                                      row_block_dim_A,
                                                      col_block_dim_A,
                                                      row_block_dim_C,
                                                      col_block_dim_C,
                                                      buffer_size);
}

extern "C" rocsparse_status rocsparse_dgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                               rocsparse_direction       dir,
                                                               rocsparse_int             mb,
                                                               rocsparse_int             nb,
                                                               rocsparse_int             nnzb,
                                                               const rocsparse_mat_descr descr_A,
                                                               const double*             bsr_val_A,
                                                               const rocsparse_int* bsr_row_ptr_A,
                                                               const rocsparse_int* bsr_col_ind_A,
                                                               rocsparse_int        row_block_dim_A,
                                                               rocsparse_int        col_block_dim_A,
                                                               rocsparse_int        row_block_dim_C,
                                                               rocsparse_int        col_block_dim_C,
                                                               size_t*              buffer_size)
{
    return rocsparse_gebsr2gebsr_buffer_size_template(handle,
                                                      dir,
                                                      mb,
                                                      nb,
                                                      nnzb,
                                                      descr_A,
                                                      bsr_val_A,
                                                      bsr_row_ptr_A,
                                                      bsr_col_ind_A,
                                                      row_block_dim_A,
                                                      col_block_dim_A,
                                                      row_block_dim_C,
                                                      col_block_dim_C,
                                                      buffer_size);
}

extern "C" rocsparse_status
    rocsparse_cgebsr2gebsr_buffer_size(rocsparse_handle               handle,
                                       rocsparse_direction            dir,
                                       rocsparse_int                  mb,
                                       rocsparse_int                  nb,
                                       rocsparse_int                  nnzb,
                                       const rocsparse_mat_descr      descr_A,
                                       const rocsparse_float_complex* bsr_val_A,
                                       const rocsparse_int*           bsr_row_ptr_A,
                                       const rocsparse_int*           bsr_col_ind_A,
                                       rocsparse_int                  row_block_dim_A,
                                       rocsparse_int                  col_block_dim_A,
                                       rocsparse_int                  row_block_dim_C,
                                       rocsparse_int                  col_block_dim_C,
                                       size_t*                        buffer_size)
{
    return rocsparse_gebsr2gebsr_buffer_size_template(handle,
                                                      dir,
                                                      mb,
                                                      nb,
                                                      nnzb,
                                                      descr_A,
                                                      bsr_val_A,
                                                      bsr_row_ptr_A,
                                                      bsr_col_ind_A,
                                                      row_block_dim_A,
                                                      col_block_dim_A,
                                                      row_block_dim_C,
                                                      col_block_dim_C,
                                                      buffer_size);
}

extern "C" rocsparse_status
    rocsparse_zgebsr2gebsr_buffer_size(rocsparse_handle                handle,
                                       rocsparse_direction             dir,
                                       rocsparse_int                   mb,
                                       rocsparse_int                   nb,
                                       rocsparse_int                   nnzb,
                                       const rocsparse_mat_descr       descr_A,
                                       const rocsparse_double_complex* bsr_val_A,
                                       const rocsparse_int*            bsr_row_ptr_A,
                                       const rocsparse_int*            bsr_col_ind_A,
                                       rocsparse_int                   row_block_dim_A,
                                       rocsparse_int                   col_block_dim_A,
                                       rocsparse_int                   row_block_dim_C,
                                       rocsparse_int                   col_block_dim_C,
                                       size_t*                         buffer_size)
{
    return rocsparse_gebsr2gebsr_buffer_size_template(handle,
                                                      dir,
                                                      mb,
                                                      nb,
                                                      nnzb,
                                                      descr_A,
                                                      bsr_val_A,
                                                      bsr_row_ptr_A,
                                                      bsr_col_ind_A,
                                                      row_block_dim_A,
                                                      col_block_dim_A,
                                                      row_block_dim_C,
                                                      col_block_dim_C,
                                                      buffer_size);
}

extern "C" rocsparse_status rocsparse_sgebsr2gebsr(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr_A,
                                                   const float*              bsr_val_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   rocsparse_int             row_block_dim_A,
                                                   rocsparse_int             col_block_dim_A,
                                                   const rocsparse_mat_descr descr_C,
                                                   float*                    bsr_val_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            bsr_col_ind_C,
                                                   rocsparse_int             row_block_dim_C,
                                                   rocsparse_int             col_block_dim_C,
                                                   void*                     temp_buffer)
{
    return rocsparse_gebsr2gebsr_template(handle,
                                          dir,
                                          mb,
                                          nb,
                                          nnzb,
                                          descr_A,
                                          bsr_val_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          row_block_dim_A,
                                          col_block_dim_A,
                                          descr_C,
                                          bsr_val_C,
                                          bsr_row_ptr_C,
                                          bsr_col_ind_C,
                                          row_block_dim_C,
                                          col_block_dim_C,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_dgebsr2gebsr(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr_A,
                                                   const double*             bsr_val_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   rocsparse_int             row_block_dim_A,
                                                   rocsparse_int             col_block_dim_A,
                                                   const rocsparse_mat_descr descr_C,
                                                   double*                   bsr_val_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            bsr_col_ind_C,
                                                   rocsparse_int             row_block_dim_C,
                                                   rocsparse_int             col_block_dim_C,
                                                   void*                     temp_buffer)
{
    return rocsparse_gebsr2gebsr_template(handle,
                                          dir,
                                          mb,
                                          nb,
                                          nnzb,
                                          descr_A,
                                          bsr_val_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          row_block_dim_A,
                                          col_block_dim_A,
                                          descr_C,
                                          bsr_val_C,
                                          bsr_row_ptr_C,
                                          bsr_col_ind_C,
                                          row_block_dim_C,
                                          col_block_dim_C,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_cgebsr2gebsr(rocsparse_handle               handle,
                                                   rocsparse_direction            dir,
                                                   rocsparse_int                  mb,
                                                   rocsparse_int                  nb,
                                                   rocsparse_int                  nnzb,
                                                   const rocsparse_mat_descr      descr_A,
                                                   const rocsparse_float_complex* bsr_val_A,
                                                   const rocsparse_int*           bsr_row_ptr_A,
                                                   const rocsparse_int*           bsr_col_ind_A,
                                                   rocsparse_int                  row_block_dim_A,
                                                   rocsparse_int                  col_block_dim_A,
                                                   const rocsparse_mat_descr      descr_C,
                                                   rocsparse_float_complex*       bsr_val_C,
                                                   rocsparse_int*                 bsr_row_ptr_C,
                                                   rocsparse_int*                 bsr_col_ind_C,
                                                   rocsparse_int                  row_block_dim_C,
                                                   rocsparse_int                  col_block_dim_C,
                                                   void*                          temp_buffer)
{
    return rocsparse_gebsr2gebsr_template(handle,
                                          dir,
                                          mb,
                                          nb,
                                          nnzb,
                                          descr_A,
                                          bsr_val_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          row_block_dim_A,
                                          col_block_dim_A,
                                          descr_C,
                                          bsr_val_C,
                                          bsr_row_ptr_C,
                                          bsr_col_ind_C,
                                          row_block_dim_C,
                                          col_block_dim_C,
                                          temp_buffer);
}

extern "C" rocsparse_status rocsparse_zgebsr2gebsr(rocsparse_handle                handle,
                                                   rocsparse_direction             dir,
                                                   rocsparse_int                   mb,
                                                   rocsparse_int                   nb,
                                                   rocsparse_int                   nnzb,
                                                   const rocsparse_mat_descr       descr_A,
                                                   const rocsparse_double_complex* bsr_val_A,
                                                   const rocsparse_int*            bsr_row_ptr_A,
                                                   const rocsparse_int*            bsr_col_ind_A,
                                                   rocsparse_int                   row_block_dim_A,
                                                   rocsparse_int                   col_block_dim_A,
                                                   const rocsparse_mat_descr       descr_C,
                                                   rocsparse_double_complex*       bsr_val_C,
                                                   rocsparse_int*                  bsr_row_ptr_C,
                                                   rocsparse_int*                  bsr_col_ind_C,
                                                   rocsparse_int                   row_block_dim_C,
                                                   rocsparse_int                   col_block_dim_C,
                                                   void*                           temp_buffer)
{
    return rocsparse_gebsr2gebsr_template(handle,
                                          dir,
                                          mb,
                                          nb,
                                          nnzb,
                                          descr_A,
                                          bsr_val_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          row_block_dim_A,
                                          col_block_dim_A,
                                          descr_C,
                                          bsr_val_C,
                                          bsr_row_ptr_C,
                                          bsr_col_ind_C,
                                          row_block_dim_C,
                                          col_block_dim_C,
                                          temp_buffer);
}

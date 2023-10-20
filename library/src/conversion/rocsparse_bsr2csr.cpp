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

#include "internal/conversion/rocsparse_bsr2csr.h"
#include "definitions.h"
#include "rocsparse_bsr2csr.hpp"
#include "utility.h"

#include "bsr2csr_device.h"

#define launch_bsr2csr_block_per_row_2_7_kernel(direction, block_size, bsr_block_dim) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                               \
        (bsr2csr_block_per_row_2_7_kernel<direction, block_size, bsr_block_dim>),     \
        dim3(mb),                                                                     \
        dim3(block_size),                                                             \
        0,                                                                            \
        stream,                                                                       \
        mb,                                                                           \
        nb,                                                                           \
        bsr_descr->base,                                                              \
        bsr_val,                                                                      \
        bsr_row_ptr,                                                                  \
        bsr_col_ind,                                                                  \
        block_dim,                                                                    \
        csr_descr->base,                                                              \
        csr_val,                                                                      \
        csr_row_ptr,                                                                  \
        csr_col_ind);

#define launch_bsr2csr_block_per_row_8_32_kernel(direction, block_size, bsr_block_dim) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                \
        (bsr2csr_block_per_row_8_32_kernel<direction, block_size, bsr_block_dim>),     \
        dim3(mb),                                                                      \
        dim3(block_size),                                                              \
        0,                                                                             \
        stream,                                                                        \
        mb,                                                                            \
        nb,                                                                            \
        bsr_descr->base,                                                               \
        bsr_val,                                                                       \
        bsr_row_ptr,                                                                   \
        bsr_col_ind,                                                                   \
        block_dim,                                                                     \
        csr_descr->base,                                                               \
        csr_val,                                                                       \
        csr_row_ptr,                                                                   \
        csr_col_ind);

#define launch_bsr2csr_block_per_row_33_256_kernel(                                          \
    direction, block_size, bsr_block_dim, sub_block_dim)                                     \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((bsr2csr_block_per_row_33_256_kernel<direction,       \
                                                                            block_size,      \
                                                                            bsr_block_dim,   \
                                                                            sub_block_dim>), \
                                       dim3(mb),                                             \
                                       dim3(block_size),                                     \
                                       0,                                                    \
                                       stream,                                               \
                                       mb,                                                   \
                                       nb,                                                   \
                                       bsr_descr->base,                                      \
                                       bsr_val,                                              \
                                       bsr_row_ptr,                                          \
                                       bsr_col_ind,                                          \
                                       block_dim,                                            \
                                       csr_descr->base,                                      \
                                       csr_val,                                              \
                                       csr_row_ptr,                                          \
                                       csr_col_ind);

template <typename T>
rocsparse_status rocsparse_bsr2csr_template_dispatch(rocsparse_handle          handle,
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
    // Stream
    hipStream_t stream = handle->stream;

    if(block_dim == 1)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((bsr2csr_block_dim_equals_one_kernel<1024>),
                                           dim3((mb - 1) / 1024 + 1),
                                           dim3(1024),
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

    if(direction == rocsparse_direction_row)
    {
        if(block_dim == 2)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_row, 256, 2);
        }
        else if(block_dim == 3)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_row, 256, 3);
        }
        else if(block_dim == 4)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_row, 256, 4);
        }
        else if(block_dim == 5)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_row, 256, 5);
        }
        else if(block_dim == 6)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_row, 256, 6);
        }
        else if(block_dim == 7)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_row, 256, 7);
        }
        else if(block_dim <= 8)
        {
            launch_bsr2csr_block_per_row_8_32_kernel(rocsparse_direction_row, 1024, 8);
        }
        else if(block_dim <= 16)
        {
            launch_bsr2csr_block_per_row_8_32_kernel(rocsparse_direction_row, 1024, 16);
        }
        else if(block_dim <= 32)
        {
            launch_bsr2csr_block_per_row_8_32_kernel(rocsparse_direction_row, 1024, 32);
        }
        else if(block_dim <= 64)
        {
            launch_bsr2csr_block_per_row_33_256_kernel(rocsparse_direction_row, 1024, 64, 32);
        }
        else if(block_dim <= 128)
        {
            launch_bsr2csr_block_per_row_33_256_kernel(rocsparse_direction_row, 1024, 128, 32);
        }
        else if(block_dim <= 256)
        {
            launch_bsr2csr_block_per_row_33_256_kernel(rocsparse_direction_row, 1024, 256, 32);
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
    }
    else
    {
        if(block_dim == 2)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_column, 256, 2);
        }
        else if(block_dim == 3)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_column, 256, 3);
        }
        else if(block_dim == 4)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_column, 256, 4);
        }
        else if(block_dim == 5)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_column, 256, 5);
        }
        else if(block_dim == 6)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_column, 256, 6);
        }
        else if(block_dim == 7)
        {
            launch_bsr2csr_block_per_row_2_7_kernel(rocsparse_direction_column, 256, 7);
        }
        else if(block_dim <= 8)
        {
            launch_bsr2csr_block_per_row_8_32_kernel(rocsparse_direction_column, 1024, 8);
        }
        else if(block_dim <= 16)
        {
            launch_bsr2csr_block_per_row_8_32_kernel(rocsparse_direction_column, 1024, 16);
        }
        else if(block_dim <= 32)
        {
            launch_bsr2csr_block_per_row_8_32_kernel(rocsparse_direction_column, 1024, 32);
        }
        else if(block_dim <= 64)
        {
            launch_bsr2csr_block_per_row_33_256_kernel(rocsparse_direction_column, 1024, 64, 32);
        }
        else if(block_dim <= 128)
        {
            launch_bsr2csr_block_per_row_33_256_kernel(rocsparse_direction_column, 1024, 128, 32);
        }
        else if(block_dim <= 256)
        {
            launch_bsr2csr_block_per_row_33_256_kernel(rocsparse_direction_column, 1024, 256, 32);
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_bsr2csr_template(rocsparse_handle          handle, //0
                                            rocsparse_direction       dir, //1
                                            rocsparse_int             mb, //2
                                            rocsparse_int             nb, //3
                                            const rocsparse_mat_descr bsr_descr, //4
                                            const T*                  bsr_val, //5
                                            const rocsparse_int*      bsr_row_ptr, //6
                                            const rocsparse_int*      bsr_col_ind, //7
                                            rocsparse_int             block_dim, //8
                                            const rocsparse_mat_descr csr_descr, //9
                                            T*                        csr_val, //10
                                            rocsparse_int*            csr_row_ptr, //11
                                            rocsparse_int*            csr_col_ind) //12
{
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

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

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);

    ROCSPARSE_CHECKARG_POINTER(4, bsr_descr);
    ROCSPARSE_CHECKARG(4,
                       bsr_descr,
                       (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(4,
                       bsr_descr,
                       (bsr_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);

    ROCSPARSE_CHECKARG_SIZE(8, block_dim);
    ROCSPARSE_CHECKARG(8, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_POINTER(9, csr_descr);
    ROCSPARSE_CHECKARG(9,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(9,
                       csr_descr,
                       (csr_descr->type != rocsparse_matrix_type_general),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ARRAY(11, mb, csr_row_ptr);

    // Quick return if possible
    if(mb == 0 || nb == 0)
    {
        if(csr_row_ptr != nullptr)
        {
            const rocsparse_int m = block_dim * mb;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((set_array_to_value<256>),
                                               dim3(((m + 1) - 1) / 256 + 1),
                                               dim3(256),
                                               0,
                                               handle->stream,
                                               (m + 1),
                                               csr_row_ptr,
                                               static_cast<rocsparse_int>(csr_descr->base));
        }
        return rocsparse_status_success;
    }

    if(csr_val == nullptr || csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        const rocsparse_int nnzb = (end - start);
        const rocsparse_int nnz  = nnzb * block_dim * block_dim;
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(12, nnz, csr_col_ind);
    }

    ROCSPARSE_CHECKARG_POINTER(5, bsr_val);
    ROCSPARSE_CHECKARG_POINTER(7, bsr_col_ind);

    //
    // Should we check pointers according to nnzb ?
    //

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsr2csr_template_dispatch(handle,
                                                                  dir,
                                                                  mb,
                                                                  nb,
                                                                  bsr_descr,
                                                                  bsr_val,
                                                                  bsr_row_ptr,
                                                                  bsr_col_ind,
                                                                  block_dim,
                                                                  csr_descr,
                                                                  csr_val,
                                                                  csr_row_ptr,
                                                                  csr_col_ind));
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
    try                                                                     \
    {                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_bsr2csr_template(handle,        \
                                                             dir,           \
                                                             mb,            \
                                                             nb,            \
                                                             bsr_descr,     \
                                                             bsr_val,       \
                                                             bsr_row_ptr,   \
                                                             bsr_col_ind,   \
                                                             block_dim,     \
                                                             csr_descr,     \
                                                             csr_val,       \
                                                             csr_row_ptr,   \
                                                             csr_col_ind)); \
        return rocsparse_status_success;                                    \
    }                                                                       \
    catch(...)                                                              \
    {                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                       \
    }

C_IMPL(rocsparse_sbsr2csr, float);
C_IMPL(rocsparse_dbsr2csr, double);
C_IMPL(rocsparse_cbsr2csr, rocsparse_float_complex);
C_IMPL(rocsparse_zbsr2csr, rocsparse_double_complex);

#undef C_IMPL

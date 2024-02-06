/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "gebsrmv_device.h"

#include "handle.h"

#include "utility.h"

#include <hip/hip_runtime.h>

namespace rocsparse
{
#define LAUNCH_GEBSRMV_GENERAL_KERNEL(BLOCKSIZE, WFSIZE)                            \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR((gebsrmvn_general_kernel<BLOCKSIZE, WFSIZE>), \
                                      dim3(mb),                                     \
                                      dim3(BLOCKSIZE),                              \
                                      0,                                            \
                                      handle->stream,                               \
                                      mb,                                           \
                                      dir,                                          \
                                      alpha,                                        \
                                      bsr_row_ptr,                                  \
                                      bsr_col_ind,                                  \
                                      bsr_val,                                      \
                                      row_block_dim,                                \
                                      col_block_dim,                                \
                                      x,                                            \
                                      beta,                                         \
                                      y,                                            \
                                      base);

#define LAUNCH_GEBSRMV_3XN_KERNEL(BLOCKSIZE, COLBSRDIM, WFSIZE)                            \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR((gebsrmvn_3xn_kernel<BLOCKSIZE, COLBSRDIM, WFSIZE>), \
                                      dim3((mb - 1) / (BLOCKSIZE / WFSIZE) + 1),           \
                                      dim3(BLOCKSIZE),                                     \
                                      0,                                                   \
                                      handle->stream,                                      \
                                      mb,                                                  \
                                      dir,                                                 \
                                      alpha,                                               \
                                      bsr_row_ptr,                                         \
                                      bsr_col_ind,                                         \
                                      bsr_val,                                             \
                                      x,                                                   \
                                      beta,                                                \
                                      y,                                                   \
                                      base);

    template <unsigned int BLOCKSIZE,
              unsigned int COLBSRDIM,
              unsigned int WFSIZE,
              typename T,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gebsrmvn_3xn_kernel(rocsparse_int       mb,
                             rocsparse_direction dir,
                             U                   alpha_device_host,
                             const rocsparse_int* __restrict__ bsr_row_ptr,
                             const rocsparse_int* __restrict__ bsr_col_ind,
                             const T* __restrict__ bsr_val,
                             const T* __restrict__ x,
                             U beta_device_host,
                             T* __restrict__ y,
                             rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::gebsrmvn_3xn_device<BLOCKSIZE, COLBSRDIM, WFSIZE>(
            mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
    }

    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gebsrmvn_general_kernel(rocsparse_int       mb,
                                 rocsparse_direction dir,
                                 U                   alpha_device_host,
                                 const rocsparse_int* __restrict__ bsr_row_ptr,
                                 const rocsparse_int* __restrict__ bsr_col_ind,
                                 const T* __restrict__ bsr_val,
                                 rocsparse_int row_block_dim,
                                 rocsparse_int col_block_dim,
                                 const T* __restrict__ x,
                                 U beta_device_host,
                                 T* __restrict__ y,
                                 rocsparse_index_base idx_base)
    {
        auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        auto beta  = rocsparse::load_scalar_device_host(beta_device_host);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::gebsrmvn_general_device<BLOCKSIZE, WFSIZE>(dir,
                                                              alpha,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              bsr_val,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              x,
                                                              beta,
                                                              y,
                                                              idx_base);
    }

    template <typename T, typename U>
    void launch_gebsrmv_row_block_dim_3(rocsparse_handle     handle,
                                        rocsparse_direction  dir,
                                        rocsparse_int        mb,
                                        rocsparse_int        nnzb,
                                        U                    alpha,
                                        const rocsparse_int* bsr_row_ptr,
                                        const rocsparse_int* bsr_col_ind,
                                        const T*             bsr_val,
                                        rocsparse_int        row_block_dim,
                                        rocsparse_int        col_block_dim,
                                        const T*             x,
                                        U                    beta,
                                        T*                   y,
                                        rocsparse_index_base base)
    {
        rocsparse_int blocks_per_row = nnzb / mb;

#define BSRMVN_DIM 128
        if(col_block_dim == 1)
        {
            if(blocks_per_row < 8)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 1, 4);
            }
            else if(blocks_per_row < 16)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 1, 8);
            }
            else if(blocks_per_row < 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 1, 16);
            }
            else if(blocks_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 1, 32);
            }
            else
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 1, 64);
            }
        }
        else if(col_block_dim == 2)
        {
            if(blocks_per_row < 8)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 2, 4);
            }
            else if(blocks_per_row < 16)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 2, 8);
            }
            else if(blocks_per_row < 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 2, 16);
            }
            else if(blocks_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 2, 32);
            }
            else
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 2, 64);
            }
        }
        else if(col_block_dim == 4)
        {
            if(blocks_per_row < 8)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 4, 4);
            }
            else if(blocks_per_row < 16)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 4, 8);
            }
            else if(blocks_per_row < 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 4, 16);
            }
            else if(blocks_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 4, 32);
            }
            else
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 4, 64);
            }
        }
        else if(col_block_dim == 5)
        {
            if(blocks_per_row < 8)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 5, 4);
            }
            else if(blocks_per_row < 16)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 5, 8);
            }
            else if(blocks_per_row < 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 5, 16);
            }
            else if(blocks_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 5, 32);
            }
            else
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 5, 64);
            }
        }
        else if(col_block_dim == 6)
        {
            if(blocks_per_row < 8)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 6, 4);
            }
            else if(blocks_per_row < 16)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 6, 8);
            }
            else if(blocks_per_row < 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 6, 16);
            }
            else if(blocks_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 6, 32);
            }
            else
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 6, 64);
            }
        }
        else if(col_block_dim == 7)
        {
            if(blocks_per_row < 8)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 7, 4);
            }
            else if(blocks_per_row < 16)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 7, 8);
            }
            else if(blocks_per_row < 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 7, 16);
            }
            else if(blocks_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 7, 32);
            }
            else
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 7, 64);
            }
        }
        else if(col_block_dim == 8)
        {
            if(blocks_per_row < 8)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 8, 4);
            }
            else if(blocks_per_row < 16)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 8, 8);
            }
            else if(blocks_per_row < 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 8, 16);
            }
            else if(blocks_per_row < 64 || handle->wavefront_size == 32)
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 8, 32);
            }
            else
            {
                LAUNCH_GEBSRMV_3XN_KERNEL(BSRMVN_DIM, 8, 64);
            }
        }
        else if(col_block_dim <= 16)
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 3, 16);
        }
        else
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 3, 32);
        }
#undef BSRMVN_DIM
    }

    template <typename T, typename U>
    rocsparse_status gebsrmv_template_row_block_dim_3(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      U                         alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             row_block_dim,
                                                      rocsparse_int             col_block_dim,
                                                      const T*                  x,
                                                      U                         beta,
                                                      T*                        y)
    {
        assert(row_block_dim == 3);

        if(trans == rocsparse_operation_none)
        {
            launch_gebsrmv_row_block_dim_3(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           alpha,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_val,
                                           row_block_dim,
                                           col_block_dim,
                                           x,
                                           beta,
                                           y,
                                           descr->base);
        }
        else
        {
            // TODO
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(T, U)                                                  \
                                                                           \
    template rocsparse_status rocsparse::gebsrmv_template_row_block_dim_3( \
        rocsparse_handle          handle,                                  \
        rocsparse_direction       dir,                                     \
        rocsparse_operation       trans,                                   \
        rocsparse_int             mb,                                      \
        rocsparse_int             nb,                                      \
        rocsparse_int             nnzb,                                    \
        U                         alpha,                                   \
        const rocsparse_mat_descr descr,                                   \
        const T*                  bsr_val,                                 \
        const rocsparse_int*      bsr_row_ptr,                             \
        const rocsparse_int*      bsr_col_ind,                             \
        rocsparse_int             row_block_dim,                           \
        rocsparse_int             col_block_dim,                           \
        const T*                  x,                                       \
        U                         beta,                                    \
        T*                        y)

INSTANTIATE(float, float);
INSTANTIATE(float, const float*);

INSTANTIATE(double, double);
INSTANTIATE(double, const double*);

INSTANTIATE(rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex, const rocsparse_float_complex*);

INSTANTIATE(rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex, const rocsparse_double_complex*);

#undef INSTANTIATE

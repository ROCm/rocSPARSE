/* ************************************************************************
 * Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse.h"
#include "utility.h"

#include <hip/hip_runtime.h>

#define LAUNCH_GEBSRMV_GENERAL_KERNEL(BLOCKSIZE, WFSIZE)             \
    hipLaunchKernelGGL((gebsrmvn_general_kernel<BLOCKSIZE, WFSIZE>), \
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

#define LAUNCH_GEBSRMV_MXN_16_KERNEL(BLOCKSIZE, ROWBSRDIM, COLBSRDIM)             \
    hipLaunchKernelGGL((gebsrmvn_mxn_16_kernel<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>), \
                       dim3(mb),                                                  \
                       dim3(BLOCKSIZE),                                           \
                       0,                                                         \
                       handle->stream,                                            \
                       mb,                                                        \
                       dir,                                                       \
                       alpha,                                                     \
                       bsr_row_ptr,                                               \
                       bsr_col_ind,                                               \
                       bsr_val,                                                   \
                       row_block_dim,                                             \
                       col_block_dim,                                             \
                       x,                                                         \
                       beta,                                                      \
                       y,                                                         \
                       base);

template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
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
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    gebsrmvn_general_device<BLOCKSIZE, WFSIZE>(dir,
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

template <unsigned int BLOCKSIZE,
          unsigned int ROWBSRDIM,
          unsigned int COLBSRDIM,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void gebsrmvn_mxn_16_kernel(rocsparse_int       mb,
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
    auto alpha = load_scalar_device_host(alpha_device_host);
    auto beta  = load_scalar_device_host(beta_device_host);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    gebsrmvn_mxn_16_device<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>(
        mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
}

template <typename T, typename U>
void launch_gebsrmvn_row_block_dim_13_16(rocsparse_handle     handle,
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
    if(row_block_dim == 13)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(52, 13, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(52, 13, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(39, 13, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(52, 13, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(65, 13, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(78, 13, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(91, 13, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(104, 13, 8);
        }
        else if(col_block_dim <= 16)
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
        }
        else
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
        }
    }
    else if(row_block_dim == 14)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(56, 14, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(56, 14, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(42, 14, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(56, 14, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(70, 14, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(84, 14, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(98, 14, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(112, 14, 8);
        }
        else if(col_block_dim <= 16)
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
        }
        else
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
        }
    }
    else if(row_block_dim == 15)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 15, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 15, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(45, 15, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 15, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(75, 15, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(90, 15, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(105, 15, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(120, 15, 8);
        }
        else if(col_block_dim <= 16)
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
        }
        else
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
        }
    }
    else if(row_block_dim == 16)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(64, 16, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(64, 16, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(48, 16, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(64, 16, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(80, 16, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(96, 16, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(112, 16, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(128, 16, 8);
        }
        else if(col_block_dim <= 16)
        {
            // BECAUSE HOOKED BY BSRMV
            // LCOV_EXCL_START
            LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
            // LCOV_EXCL_STOP
        }
        else
        {
            LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
        }
    }
}

template <typename T, typename U>
rocsparse_status rocsparse_gebsrmv_template_row_block_dim_13_16(rocsparse_handle          handle,
                                                                rocsparse_direction       dir,
                                                                rocsparse_operation       trans,
                                                                rocsparse_int             mb,
                                                                rocsparse_int             nb,
                                                                rocsparse_int             nnzb,
                                                                U                         alpha,
                                                                const rocsparse_mat_descr descr,
                                                                const T*                  bsr_val,
                                                                const rocsparse_int* bsr_row_ptr,
                                                                const rocsparse_int* bsr_col_ind,
                                                                rocsparse_int        row_block_dim,
                                                                rocsparse_int        col_block_dim,
                                                                const T*             x,
                                                                U                    beta,
                                                                T*                   y)
{
    assert(row_block_dim >= 13);
    assert(row_block_dim <= 16);

    if(trans == rocsparse_operation_none)
    {
        launch_gebsrmvn_row_block_dim_13_16(handle,
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
        return rocsparse_status_not_implemented;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(T, U)                                                     \
                                                                              \
    template rocsparse_status rocsparse_gebsrmv_template_row_block_dim_13_16( \
        rocsparse_handle          handle,                                     \
        rocsparse_direction       dir,                                        \
        rocsparse_operation       trans,                                      \
        rocsparse_int             mb,                                         \
        rocsparse_int             nb,                                         \
        rocsparse_int             nnzb,                                       \
        U                         alpha,                                      \
        const rocsparse_mat_descr descr,                                      \
        const T*                  bsr_val,                                    \
        const rocsparse_int*      bsr_row_ptr,                                \
        const rocsparse_int*      bsr_col_ind,                                \
        rocsparse_int             row_block_dim,                              \
        rocsparse_int             col_block_dim,                              \
        const T*                  x,                                          \
        U                         beta,                                       \
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

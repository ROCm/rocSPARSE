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
void launch_gebsrmvn_row_block_dim_9_12(rocsparse_handle     handle,
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
    if(row_block_dim == 9)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(63, 9, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(54, 9, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(54, 9, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(36, 9, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(45, 9, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(54, 9, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(63, 9, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(72, 9, 8);
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
    else if(row_block_dim == 10)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 10, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 10, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 10, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(40, 10, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(50, 10, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 10, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(70, 10, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(80, 10, 8);
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
    else if(row_block_dim == 11)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(55, 11, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(44, 11, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(33, 11, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(44, 11, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(55, 11, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(66, 11, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(77, 11, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(88, 11, 8);
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
    else if(row_block_dim == 12)
    {
        if(col_block_dim == 1)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 12, 1);
        }
        else if(col_block_dim == 2)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(48, 12, 2);
        }
        else if(col_block_dim == 3)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(48, 12, 3);
        }
        else if(col_block_dim == 4)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 12, 4);
        }
        else if(col_block_dim == 5)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 12, 5);
        }
        else if(col_block_dim == 6)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(72, 12, 6);
        }
        else if(col_block_dim == 7)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(84, 12, 7);
        }
        else if(col_block_dim == 8)
        {
            LAUNCH_GEBSRMV_MXN_16_KERNEL(96, 12, 8);
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
}

template <typename T, typename U>
rocsparse_status rocsparse_gebsrmv_template_row_block_dim_9_12(rocsparse_handle          handle,
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
    assert(row_block_dim >= 9);
    assert(row_block_dim <= 12);

    if(trans == rocsparse_operation_none)
    {
        launch_gebsrmvn_row_block_dim_9_12(handle,
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

#define INSTANTIATE(T, U)                                                    \
                                                                             \
    template rocsparse_status rocsparse_gebsrmv_template_row_block_dim_9_12( \
        rocsparse_handle          handle,                                    \
        rocsparse_direction       dir,                                       \
        rocsparse_operation       trans,                                     \
        rocsparse_int             mb,                                        \
        rocsparse_int             nb,                                        \
        rocsparse_int             nnzb,                                      \
        U                         alpha,                                     \
        const rocsparse_mat_descr descr,                                     \
        const T*                  bsr_val,                                   \
        const rocsparse_int*      bsr_row_ptr,                               \
        const rocsparse_int*      bsr_col_ind,                               \
        rocsparse_int             row_block_dim,                             \
        rocsparse_int             col_block_dim,                             \
        const T*                  x,                                         \
        U                         beta,                                      \
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

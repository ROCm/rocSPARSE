/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "bsrsm_device.h"
#include "bsrsm_device_large.h"
#include "definitions.h"
#include "rocsparse_bsrsm.hpp"
#include "utility.h"

#define LAUNCH_BSRSM_GTHR_DIM(bsize, wfsize, dim)                 \
    hipLaunchKernelGGL((bsr_gather<wfsize, bsize / wfsize, dim>), \
                       dim3((wfsize * nnzb - 1) / bsize + 1),     \
                       dim3(wfsize, bsize / wfsize),              \
                       0,                                         \
                       stream,                                    \
                       dir,                                       \
                       nnzb,                                      \
                       bsrsm_info->trmt_perm,                     \
                       bsr_val,                                   \
                       bsrt_val,                                  \
                       block_dim)

#define LAUNCH_BSRSM_GTHR(bsize, wfsize, dim) \
    if(dim <= 2)                              \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 4, 2);   \
    }                                         \
    else if(dim <= 4)                         \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else if(wfsize == 32)                     \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 16, 4);  \
    }                                         \
    else                                      \
    {                                         \
        LAUNCH_BSRSM_GTHR_DIM(bsize, 64, 8);  \
    }

template <unsigned int BLOCKSIZE, typename T, typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void bsrsm_copy_scale(rocsparse_int m,
                                                                    rocsparse_int n,
                                                                    U             alpha_device_host,
                                                                    const T*      B,
                                                                    rocsparse_int ldb,
                                                                    T*            X,
                                                                    rocsparse_int ldx)
{
    auto alpha = load_scalar_device_host(alpha_device_host);

    bsrsm_copy_scale_device(m, n, alpha, B, ldb, X, ldx);
}

template <unsigned int DIM_X, unsigned int DIM_Y, typename T, typename U>
__launch_bounds__(DIM_X* DIM_Y) ROCSPARSE_KERNEL void bsrsm_transpose(rocsparse_int m,
                                                                      rocsparse_int n,
                                                                      U alpha_device_host,
                                                                      const T* __restrict__ A,
                                                                      rocsparse_int lda,
                                                                      T* __restrict__ B,
                                                                      rocsparse_int ldb)
{
    auto alpha = load_scalar_device_host(alpha_device_host);

    dense_transpose_device<DIM_X, DIM_Y>(m, n, alpha, A, lda, B, ldb);
}

template <typename T, typename U>
rocsparse_status rocsparse_bsrsm_solve_template_large(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_X,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nrhs,
                                                      rocsparse_int             nnzb,
                                                      U                         alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             block_dim,
                                                      rocsparse_mat_info        info,
                                                      const T*                  B,
                                                      rocsparse_int             ldb,
                                                      T*                        X,
                                                      rocsparse_int             ldx,
                                                      void*                     temp_buffer)
{
#define LAUNCH_LARGE_KERNEL(K_, M_, S_)              \
    dim3 bsrsm_blocks(((nrhs - 1) / NCOL + 1) * mb); \
    dim3 bsrsm_threads(NCOL* M_);                    \
    hipLaunchKernelGGL((K_<NCOL * M_, NCOL, S_>),    \
                       bsrsm_blocks,                 \
                       bsrsm_threads,                \
                       0,                            \
                       stream,                       \
                       mb,                           \
                       nrhs,                         \
                       local_bsr_row_ptr,            \
                       local_bsr_col_ind,            \
                       local_bsr_val,                \
                       block_dim,                    \
                       Xt,                           \
                       ldimX,                        \
                       done_array,                   \
                       bsrsm_info->row_map,          \
                       info->zero_pivot,             \
                       descr->base,                  \
                       descr->diag_type,             \
                       dir);

    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    ptr += 256;

    // 16 columns per block seem to work very well
    static constexpr unsigned int NCOL = 16;

    int narrays = (nrhs - 1) / NCOL + 1;

    // done_array
    int* done_array = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * ((mb * narrays - 1) / 256 + 1) * 256;

    // Temporary array to store transpose of X
    T* Xt = X;
    if(trans_X == rocsparse_operation_none)
    {
        Xt = reinterpret_cast<T*>(ptr);
        ptr += sizeof(T) * ((mb * block_dim * nrhs - 1) / 256 + 1) * 256;
    }

    // Initialize buffers
    RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * mb * narrays, stream));

    rocsparse_trm_info bsrsm_info
        = (descr->fill_mode == rocsparse_fill_mode_upper)
              ? ((trans_A == rocsparse_operation_none) ? info->bsrsm_upper_info
                                                       : info->bsrsmt_upper_info)
              : ((trans_A == rocsparse_operation_none) ? info->bsrsm_lower_info
                                                       : info->bsrsmt_lower_info);

    // If diag type is unit, re-initialize zero pivot to remove structural zeros
    if(descr->diag_type == rocsparse_diag_type_unit)
    {
        rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            info->zero_pivot, &max, sizeof(rocsparse_int), hipMemcpyHostToDevice, stream));

        // Wait for device transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }

    rocsparse_fill_mode fill_mode = descr->fill_mode;

    // Transpose X if X is not transposed yet to improve performance
    rocsparse_int ldimX = ldx;
    if(trans_X == rocsparse_operation_none)
    {
        // Leading dimension for transposed X
        ldimX = nrhs;

#define BSRSM_DIM_X 32
#define BSRSM_DIM_Y 8
        dim3 bsrsm_blocks((mb * block_dim - 1) / BSRSM_DIM_X + 1);
        dim3 bsrsm_threads(BSRSM_DIM_X * BSRSM_DIM_Y);

        hipLaunchKernelGGL((bsrsm_transpose<BSRSM_DIM_X, BSRSM_DIM_Y>),
                           bsrsm_blocks,
                           bsrsm_threads,
                           0,
                           stream,
                           mb * block_dim,
                           nrhs,
                           alpha,
                           B,
                           ldb,
                           Xt,
                           ldimX);
#undef BSRSM_DIM_X
#undef BSRSM_DIM_Y
    }
    else
    {
        // Copy B into X and scale it with alpha
        hipLaunchKernelGGL((bsrsm_copy_scale<1024>),
                           dim3((mb * block_dim - 1) / 1024 + 1),
                           dim3(1024),
                           0,
                           stream,
                           mb * block_dim,
                           nrhs,
                           alpha,
                           B,
                           ldb,
                           X,
                           ldx);
    }

    // Pointers to differentiate between transpose mode
    const rocsparse_int* local_bsr_row_ptr = bsr_row_ptr;
    const rocsparse_int* local_bsr_col_ind = bsr_col_ind;
    const T*             local_bsr_val     = bsr_val;

    // When computing transposed triangular solve, we first need to update the
    // transposed matrix values
    if(trans_A == rocsparse_operation_transpose)
    {
        T* bsrt_val = reinterpret_cast<T*>(ptr);

        LAUNCH_BSRSM_GTHR(256, 64, block_dim);

        local_bsr_row_ptr = bsrsm_info->trmt_row_ptr;
        local_bsr_col_ind = bsrsm_info->trmt_col_ind;
        local_bsr_val     = bsrt_val;

        fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                             : rocsparse_fill_mode_lower;
    }

    // Determine gcnArch and ASIC revision
    int gcnArch = handle->properties.gcnArch;
    int asicRev = handle->asic_rev;
    int wfSize  = handle->wavefront_size;

    // gfx908 A0/1
    if(gcnArch == 908 && asicRev < 2)
    {
        if(fill_mode == rocsparse_fill_mode_upper)
        {
            LAUNCH_LARGE_KERNEL(bsrsm_upper_large_kernel, 16, true);
        }
        else
        {
            LAUNCH_LARGE_KERNEL(bsrsm_lower_large_kernel, 16, true);
        }
    }
    else
    {
        // Select tuned kernel
        unsigned int nbsr = std::max(4U, fnp2(block_dim));

        while(nbsr > wfSize)
        {
            nbsr >>= 1;
        }

        switch(nbsr)
        {
#define DEFINE_CASE(i)                                               \
    case i:                                                          \
    {                                                                \
        if(fill_mode == rocsparse_fill_mode_upper)                   \
        {                                                            \
            LAUNCH_LARGE_KERNEL(bsrsm_upper_large_kernel, i, false); \
        }                                                            \
        else                                                         \
        {                                                            \
            LAUNCH_LARGE_KERNEL(bsrsm_lower_large_kernel, i, false); \
        }                                                            \
        break;                                                       \
    }

            DEFINE_CASE(4);
            DEFINE_CASE(8);
            DEFINE_CASE(16);
            DEFINE_CASE(32);
            DEFINE_CASE(64);
#undef DEFINE_CASE
        }
    }
#undef LAUNCH_LARGE_KERNEL

    // Transpose X back if X was not initially transposed
    if(trans_X == rocsparse_operation_none)
    {
#define BSRSM_DIM_X 32
#define BSRSM_DIM_Y 8
        dim3 bsrsm_blocks((mb * block_dim - 1) / BSRSM_DIM_X + 1);
        dim3 bsrsm_threads(BSRSM_DIM_X * BSRSM_DIM_Y);

        hipLaunchKernelGGL((dense_transpose_back<BSRSM_DIM_X, BSRSM_DIM_Y>),
                           bsrsm_blocks,
                           bsrsm_threads,
                           0,
                           stream,
                           mb * block_dim,
                           nrhs,
                           Xt,
                           ldimX,
                           X,
                           ldx);
#undef BSRSM_DIM_X
#undef BSRSM_DIM_Y
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(real_type_, scalar_type_)                       \
    template rocsparse_status rocsparse_bsrsm_solve_template_large( \
        rocsparse_handle          handle,                           \
        rocsparse_direction       dir,                              \
        rocsparse_operation       trans_A,                          \
        rocsparse_operation       trans_X,                          \
        rocsparse_int             mb,                               \
        rocsparse_int             nrhs,                             \
        rocsparse_int             nnzb,                             \
        scalar_type_              alpha,                            \
        const rocsparse_mat_descr descr,                            \
        const real_type_*         bsr_val,                          \
        const rocsparse_int*      bsr_row_ptr,                      \
        const rocsparse_int*      bsr_col_ind,                      \
        rocsparse_int             block_dim,                        \
        rocsparse_mat_info        info,                             \
        const real_type_*         B,                                \
        rocsparse_int             ldb,                              \
        real_type_*               X,                                \
        rocsparse_int             ldx,                              \
        void*                     temp_buffer)

INSTANTIATE(float, float);
INSTANTIATE(float, const float*);

INSTANTIATE(double, double);
INSTANTIATE(double, const double*);

INSTANTIATE(rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex, const rocsparse_float_complex*);

INSTANTIATE(rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex, const rocsparse_double_complex*);
#undef INSTANTIATE

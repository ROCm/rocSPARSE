/*! \file */
/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_gtsv.hpp"

#include "gtsv_device.h"

template <typename T>
rocsparse_status rocsparse_gtsv_buffer_size_template(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     rocsparse_int    n,
                                                     const T*         dl,
                                                     const T*         d,
                                                     const T*         du,
                                                     const T*         B,
                                                     rocsparse_int    ldb,
                                                     size_t*          buffer_size)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgtsv_buffer_size"),
              m,
              n,
              (const void*&)dl,
              (const void*&)d,
              (const void*&)du,
              (const void*&)B,
              ldb,
              (const void*&)buffer_size);

    // Check sizes
    if(m <= 1 || n < 0 || ldb < std::max(1, m))
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid buffer_size pointer
    if(buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(n == 0)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(dl == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(d == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(du == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    constexpr unsigned int BLOCKSIZE = 256;

    rocsparse_int block_dim = 2;
    rocsparse_int m_pad     = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
    rocsparse_int gridsize  = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
    while(gridsize > 512)
    {
        block_dim *= 2;
        m_pad    = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
        gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
    }

    // round up to next power of 2
    gridsize = fnp2(gridsize);

    *buffer_size = 0;

    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // dl_pad
    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // d_pad
    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // du_pad
    *buffer_size += sizeof(T) * ((m_pad * n - 1) / 256 + 1) * 256; // rhs_pad
    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // w_pad
    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // v_pad
    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // w2_pad
    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // v2_pad
    *buffer_size += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256; // mt_pad

    *buffer_size += sizeof(T) * ((2 * gridsize * n - 1) / 256 + 1) * 256; // rhs_scratch
    *buffer_size += sizeof(T) * ((2 * gridsize - 1) / 256 + 1) * 256; // w_scratch
    *buffer_size += sizeof(T) * ((2 * gridsize - 1) / 256 + 1) * 256; // v_scratch

    *buffer_size += sizeof(rocsparse_int) * ((m_pad - 1) / 256 + 1) * 256; // pivot_pad

    return rocsparse_status_success;
}

template <unsigned int BLOCKSIZE, unsigned int BLOCKDIM, typename T>
rocsparse_status rocsparse_gtsv_spike_solver_template(rocsparse_handle handle,
                                                      rocsparse_int    m,
                                                      rocsparse_int    n,
                                                      rocsparse_int    m_pad,
                                                      rocsparse_int    gridsize,
                                                      const T*         dl,
                                                      const T*         d,
                                                      const T*         du,
                                                      T*               B,
                                                      rocsparse_int    ldb,
                                                      void*            temp_buffer)
{
    char* ptr    = reinterpret_cast<char*>(temp_buffer);
    T*    dl_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;
    T* d_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;
    T* du_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;
    T* rhs_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad * n - 1) / 256 + 1) * 256;
    T* w_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;
    T* v_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;
    T* w2_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;
    T* v2_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;
    T* mt_pad = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m_pad - 1) / 256 + 1) * 256;

    T* rhs_scratch = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((2 * gridsize * n - 1) / 256 + 1) * 256;
    T* w_scratch = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((2 * gridsize - 1) / 256 + 1) * 256;
    T* v_scratch = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((2 * gridsize - 1) / 256 + 1) * 256;

    rocsparse_int* pivot_pad = reinterpret_cast<rocsparse_int*>(ptr);
    //    ptr += sizeof(rocsparse_int) * ((m_pad - 1) / 256 + 1) * 256;

    hipLaunchKernelGGL((gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3((m_pad - 1) / BLOCKSIZE + 1),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m,
                       m_pad,
                       m_pad,
                       dl,
                       dl_pad,
                       static_cast<T>(0));

    hipLaunchKernelGGL((gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3((m_pad - 1) / BLOCKSIZE + 1),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m,
                       m_pad,
                       m_pad,
                       d,
                       d_pad,
                       static_cast<T>(1));

    hipLaunchKernelGGL((gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3((m_pad - 1) / BLOCKSIZE + 1),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m,
                       m_pad,
                       m_pad,
                       du,
                       du_pad,
                       static_cast<T>(0));

    hipLaunchKernelGGL((gtsv_transpose_and_pad_array_shared_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3((m_pad - 1) / BLOCKSIZE + 1, n),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m,
                       m_pad,
                       ldb,
                       B,
                       rhs_pad,
                       static_cast<T>(0));

    RETURN_IF_HIP_ERROR(hipMemsetAsync(w_pad, 0, m_pad * sizeof(T), handle->stream));
    RETURN_IF_HIP_ERROR(hipMemsetAsync(v_pad, 0, m_pad * sizeof(T), handle->stream));

    hipLaunchKernelGGL((gtsv_LBM_wv_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3(gridsize),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m_pad,
                       n,
                       ldb,
                       dl_pad,
                       d_pad,
                       du_pad,
                       w_pad,
                       v_pad,
                       mt_pad,
                       pivot_pad);

    if(n % 8 == 0)
    {
        hipLaunchKernelGGL((gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 8>),
                           dim3(gridsize, n / 8),
                           dim3(BLOCKSIZE),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           dl_pad,
                           d_pad,
                           du_pad,
                           rhs_pad,
                           mt_pad,
                           pivot_pad);
    }
    else if(n % 4 == 0)
    {
        hipLaunchKernelGGL((gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 4>),
                           dim3(gridsize, n / 4),
                           dim3(BLOCKSIZE),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           dl_pad,
                           d_pad,
                           du_pad,
                           rhs_pad,
                           mt_pad,
                           pivot_pad);
    }
    else if(n % 2 == 0)
    {
        hipLaunchKernelGGL((gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 2>),
                           dim3(gridsize, n / 2),
                           dim3(BLOCKSIZE),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           dl_pad,
                           d_pad,
                           du_pad,
                           rhs_pad,
                           mt_pad,
                           pivot_pad);
    }
    else
    {
        hipLaunchKernelGGL((gtsv_LBM_rhs_kernel<BLOCKSIZE, BLOCKDIM, 1>),
                           dim3(gridsize, n),
                           dim3(BLOCKSIZE),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           dl_pad,
                           d_pad,
                           du_pad,
                           rhs_pad,
                           mt_pad,
                           pivot_pad);
    }

    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(w2_pad, w_pad, m_pad * sizeof(T), hipMemcpyDeviceToDevice, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(v2_pad, v_pad, m_pad * sizeof(T), hipMemcpyDeviceToDevice, handle->stream));

    hipLaunchKernelGGL((gtsv_spike_block_level_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3(gridsize, n),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m_pad,
                       n,
                       ldb,
                       rhs_pad,
                       w_pad,
                       v_pad,
                       w2_pad,
                       v2_pad,
                       rhs_scratch,
                       w_scratch,
                       v_scratch);

    // gridsize is always a power of 2
    if(gridsize == 2)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<2>),
                           dim3(1, n),
                           dim3(2),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 4)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<4>),
                           dim3(1, n),
                           dim3(4),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 8)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<8>),
                           dim3(1, n),
                           dim3(8),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 16)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<16>),
                           dim3(1, n),
                           dim3(16),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 32)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<32>),
                           dim3(1, n),
                           dim3(32),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 64)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<64>),
                           dim3(1, n),
                           dim3(64),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 128)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<128>),
                           dim3(1, n),
                           dim3(128),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 256)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<256>),
                           dim3(1, n),
                           dim3(256),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }
    else if(gridsize == 512)
    {
        hipLaunchKernelGGL((gtsv_solve_spike_grid_level_kernel<512>),
                           dim3(1, n),
                           dim3(512),
                           0,
                           handle->stream,
                           m_pad,
                           n,
                           ldb,
                           rhs_scratch,
                           w_scratch,
                           v_scratch);
    }

    hipLaunchKernelGGL((gtsv_solve_spike_propagate_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3(gridsize, n),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m_pad,
                       n,
                       ldb,
                       rhs_pad,
                       w2_pad,
                       v2_pad,
                       rhs_scratch);

    hipLaunchKernelGGL((gtsv_spike_backward_substitution_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3(gridsize, n),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m_pad,
                       n,
                       ldb,
                       rhs_pad,
                       w2_pad,
                       v2_pad);

    hipLaunchKernelGGL((gtsv_transpose_back_array_kernel<BLOCKSIZE, BLOCKDIM>),
                       dim3((m_pad - 1) / BLOCKSIZE + 1, n),
                       dim3(BLOCKSIZE),
                       0,
                       handle->stream,
                       m,
                       m_pad,
                       ldb,
                       rhs_pad,
                       B);

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gtsv_template(rocsparse_handle handle,
                                         rocsparse_int    m,
                                         rocsparse_int    n,
                                         const T*         dl,
                                         const T*         d,
                                         const T*         du,
                                         T*               B,
                                         rocsparse_int    ldb,
                                         void*            temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xgtsv"),
              m,
              n,
              (const void*&)dl,
              (const void*&)d,
              (const void*&)du,
              (const void*&)B,
              ldb,
              (const void*&)temp_buffer);

    log_bench(handle, "./rocsparse-bench -f gtsv -r", replaceX<T>("X"), "--mtx <matrix.mtx> ");

    // Check sizes
    if(m <= 1 || n < 0 || ldb < std::max(1, m))
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(dl == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(d == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(du == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(B == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    constexpr unsigned int BLOCKSIZE = 256;

    rocsparse_int block_dim = 2;
    rocsparse_int m_pad     = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
    rocsparse_int gridsize  = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
    while(gridsize > 512)
    {
        block_dim *= 2;
        m_pad    = ((m - 1) / (block_dim * BLOCKSIZE) + 1) * (block_dim * BLOCKSIZE);
        gridsize = ((m_pad / block_dim - 1) / BLOCKSIZE + 1);
    }

    // round up to next power of 2
    gridsize = fnp2(gridsize);

    if(block_dim == 2)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 2>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else if(block_dim == 4)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 4>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else if(block_dim == 8)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 8>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else if(block_dim == 16)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 16>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else if(block_dim == 32)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 32>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else if(block_dim == 64)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 64>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else if(block_dim == 128)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 128>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else if(block_dim == 256)
    {
        return rocsparse_gtsv_spike_solver_template<BLOCKSIZE, 256>(
            handle, m, n, m_pad, gridsize, dl, d, du, B, ldb, temp_buffer);
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                                     \
                                     rocsparse_int    m,                                          \
                                     rocsparse_int    n,                                          \
                                     const TYPE*      dl,                                         \
                                     const TYPE*      d,                                          \
                                     const TYPE*      du,                                         \
                                     const TYPE*      B,                                          \
                                     rocsparse_int    ldb,                                        \
                                     size_t*          buffer_size)                                \
    {                                                                                             \
        return rocsparse_gtsv_buffer_size_template(handle, m, n, dl, d, du, B, ldb, buffer_size); \
    }

C_IMPL(rocsparse_sgtsv_buffer_size, float);
C_IMPL(rocsparse_dgtsv_buffer_size, double);
C_IMPL(rocsparse_cgtsv_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_buffer_size, rocsparse_double_complex);

#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                            \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                         \
                                     rocsparse_int    m,                              \
                                     rocsparse_int    n,                              \
                                     const TYPE*      dl,                             \
                                     const TYPE*      d,                              \
                                     const TYPE*      du,                             \
                                     TYPE*            B,                              \
                                     rocsparse_int    ldb,                            \
                                     void*            temp_buffer)                    \
    {                                                                                 \
        return rocsparse_gtsv_template(handle, m, n, dl, d, du, B, ldb, temp_buffer); \
    }

C_IMPL(rocsparse_sgtsv, float);
C_IMPL(rocsparse_dgtsv, double);
C_IMPL(rocsparse_cgtsv, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv, rocsparse_double_complex);

#undef C_IMPL

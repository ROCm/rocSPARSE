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

#include "rocsparse_gtsv_no_pivot.hpp"

#include "gtsv_nopivot_device.h"

#define LAUNCH_GTSV_NOPIVOT_CR_POW2_SHARED(T, block_size)                \
    hipLaunchKernelGGL((gtsv_nopivot_cr_pow2_shared_kernel<block_size>), \
                       dim3(n),                                          \
                       dim3(block_size),                                 \
                       0,                                                \
                       handle->stream,                                   \
                       m,                                                \
                       n,                                                \
                       ldb,                                              \
                       dl,                                               \
                       d,                                                \
                       du,                                               \
                       B);

#define LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(T, block_size)                \
    hipLaunchKernelGGL((gtsv_nopivot_pcr_pow2_shared_kernel<block_size>), \
                       dim3(n),                                           \
                       dim3(block_size),                                  \
                       0,                                                 \
                       handle->stream,                                    \
                       m,                                                 \
                       n,                                                 \
                       ldb,                                               \
                       dl,                                                \
                       d,                                                 \
                       du,                                                \
                       B);

#define LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(T, block_size, pcr_size)                \
    hipLaunchKernelGGL((gtsv_nopivot_crpcr_pow2_shared_kernel<block_size, pcr_size>), \
                       dim3(n),                                                       \
                       dim3(block_size),                                              \
                       0,                                                             \
                       handle->stream,                                                \
                       m,                                                             \
                       n,                                                             \
                       ldb,                                                           \
                       dl,                                                            \
                       d,                                                             \
                       du,                                                            \
                       B);

#define LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, block_size)                \
    hipLaunchKernelGGL((gtsv_nopivot_pcr_shared_kernel<block_size>), \
                       dim3(n),                                      \
                       dim3(block_size),                             \
                       0,                                            \
                       handle->stream,                               \
                       m,                                            \
                       n,                                            \
                       ldb,                                          \
                       dl,                                           \
                       d,                                            \
                       du,                                           \
                       B);

template <typename T>
rocsparse_status rocsparse_gtsv_no_pivot_buffer_size_template(rocsparse_handle handle,
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
              replaceX<T>("rocsparse_Xgtsv_no_pivot_buffer_size"),
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

    if(m <= 512)
    {
        // Do not return 0 as buffer size
        *buffer_size = 4;
    }
    else
    {
        *buffer_size = 0;

        *buffer_size += sizeof(T) * ((m - 1) / 256 + 1) * 256; // da0
        *buffer_size += sizeof(T) * ((m - 1) / 256 + 1) * 256; // da1
        *buffer_size += sizeof(T) * ((m - 1) / 256 + 1) * 256; // db0
        *buffer_size += sizeof(T) * ((m - 1) / 256 + 1) * 256; // db1
        *buffer_size += sizeof(T) * ((m - 1) / 256 + 1) * 256; // dc0
        *buffer_size += sizeof(T) * ((m - 1) / 256 + 1) * 256; // dc1
        *buffer_size += sizeof(T) * ((m * n - 1) / 256 + 1) * 256; // drhs0
        *buffer_size += sizeof(T) * ((m * n - 1) / 256 + 1) * 256; // drhs1
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gtsv_no_pivot_small_template(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        const T*         dl,
                                                        const T*         d,
                                                        const T*         du,
                                                        T*               B,
                                                        rocsparse_int    ldb,
                                                        void*            temp_buffer)
{
    assert(m <= 512);

    // Run special algorithm if m is power of 2
    if((m & (m - 1)) == 0)
    {
        if(m == 2)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(T, 2);
        }
        else if(m == 4)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(T, 4);
        }
        else if(m == 8)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(T, 8);
        }
        else if(m == 16)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(T, 16);
        }
        else if(m == 32)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(T, 32);
        }
        else if(m == 64)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_SHARED(T, 64);
        }
        else if(m == 128)
        {
            LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(T, 64, 64);
        }
        else if(m == 256)
        {
            LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(T, 128, 64);
        }
        else if(m == 512)
        {
            LAUNCH_GTSV_NOPIVOT_CRPCR_POW2_SHARED(T, 256, 64);
        }
    }
    else
    {
        if(m <= 4)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 4);
        }
        else if(m <= 8)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 8);
        }
        else if(m <= 16)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 16);
        }
        else if(m <= 32)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 32);
        }
        else if(m <= 64)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 64);
        }
        else if(m <= 128)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 128);
        }
        else if(m <= 256)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 256);
        }
        else if(m <= 512)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_SHARED(T, 512);
        }
    }

    return rocsparse_status_success;
}

#define LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1_N(T, block_size, stride, iter)      \
    hipLaunchKernelGGL((gtsv_nopivot_pcr_pow2_stage1_n_kernel<block_size>),     \
                       dim3(((m - 1) / block_size + 1), n, 1),                  \
                       dim3(block_size, 1, 1),                                  \
                       0,                                                       \
                       handle->stream,                                          \
                       stride,                                                  \
                       m,                                                       \
                       n,                                                       \
                       ((iter == 0) ? ldb : m),                                 \
                       ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),    \
                       ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),     \
                       ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),    \
                       ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)), \
                       (((iter & 1) == 0) ? da1 : da0),                         \
                       (((iter & 1) == 0) ? db1 : db0),                         \
                       (((iter & 1) == 0) ? dc1 : dc0),                         \
                       (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_PCR_STAGE1_N(T, block_size, stride, iter)           \
    hipLaunchKernelGGL((gtsv_nopivot_pcr_stage1_n_kernel<block_size>),          \
                       dim3(((m - 1) / block_size + 1), n, 1),                  \
                       dim3(block_size),                                        \
                       0,                                                       \
                       handle->stream,                                          \
                       stride,                                                  \
                       m,                                                       \
                       n,                                                       \
                       ((iter == 0) ? ldb : m),                                 \
                       ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),    \
                       ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),     \
                       ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),    \
                       ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)), \
                       (((iter & 1) == 0) ? da1 : da0),                         \
                       (((iter & 1) == 0) ? db1 : db0),                         \
                       (((iter & 1) == 0) ? dc1 : dc0),                         \
                       (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_CR_POW2_STAGE2(T, block_size, iter)          \
    hipLaunchKernelGGL((gtsv_nopivot_cr_pow2_stage2_kernel<block_size>), \
                       dim3(subsystem_count, n, 1),                      \
                       dim3(block_size),                                 \
                       0,                                                \
                       handle->stream,                                   \
                       m,                                                \
                       n,                                                \
                       ldb,                                              \
                       (((iter & 1) != 0) ? da1 : da0),                  \
                       (((iter & 1) != 0) ? db1 : db0),                  \
                       (((iter & 1) != 0) ? dc1 : dc0),                  \
                       (((iter & 1) != 0) ? drhs1 : drhs0),              \
                       B);

#define LAUNCH_GTSV_NOPIVOT_PCR_STAGE2(T, block_size, iter)          \
    hipLaunchKernelGGL((gtsv_nopivot_pcr_stage2_kernel<block_size>), \
                       dim3(subsystem_count, n, 1),                  \
                       dim3(block_size),                             \
                       0,                                            \
                       handle->stream,                               \
                       m,                                            \
                       n,                                            \
                       ldb,                                          \
                       (((iter & 1) != 0) ? da1 : da0),              \
                       (((iter & 1) != 0) ? db1 : db0),              \
                       (((iter & 1) != 0) ? dc1 : dc0),              \
                       (((iter & 1) != 0) ? drhs1 : drhs0),          \
                       B);

template <typename T>
rocsparse_status rocsparse_gtsv_no_pivot_medium_template(rocsparse_handle handle,
                                                         rocsparse_int    m,
                                                         rocsparse_int    n,
                                                         const T*         dl,
                                                         const T*         d,
                                                         const T*         du,
                                                         T*               B,
                                                         rocsparse_int    ldb,
                                                         void*            temp_buffer)
{
    assert(m > 512 && m <= 65536);

    char* ptr = reinterpret_cast<char*>(temp_buffer);
    T*    da0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* da1 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* db0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* db1 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* dc0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* dc1 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* drhs0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m * n - 1) / 256 + 1) * 256;
    T* drhs1 = reinterpret_cast<T*>(ptr);
    // ptr += sizeof(T) * ((m * n - 1) / 256 + 1) * 256;

    // Run special algorithm if m is power of 2
    if((m & (m - 1)) == 0)
    {
        // Stage1: Break large tridiagonal system into multiple smaller systems
        // using parallel cyclic reduction so that each sub system is of size 512.
        rocsparse_int iter
            = static_cast<rocsparse_int>(log2(m)) - static_cast<rocsparse_int>(log2(512));

        rocsparse_int stride = 1;
        for(rocsparse_int i = 0; i < iter; i++)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1_N(T, 256, stride, i);

            stride *= 2;
        }

        // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
        rocsparse_int subsystem_count = 1 << iter;

        LAUNCH_GTSV_NOPIVOT_CR_POW2_STAGE2(T, 256, iter);
    }
    else
    {
        // Stage1: Break large tridiagonal system into multiple smaller systems
        // using parallel cyclic reduction so that each sub system is of size 512 or less.
        rocsparse_int iter
            = static_cast<rocsparse_int>(log2(m)) - static_cast<rocsparse_int>(log2(512)) + 1;

        rocsparse_int stride = 1;
        for(rocsparse_int i = 0; i < iter; i++)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_STAGE1_N(T, 256, stride, i);

            stride *= 2;
        }

        // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
        rocsparse_int subsystem_count = 1 << iter;

        LAUNCH_GTSV_NOPIVOT_PCR_STAGE2(T, 512, iter);
    }

    return rocsparse_status_success;
}

#define LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1(T, block_size, stride, iter)        \
    hipLaunchKernelGGL((gtsv_nopivot_pcr_pow2_stage1_kernel<block_size>),       \
                       dim3(((m - 1) / block_size + 1), 1, 1),                  \
                       dim3(block_size, 1, 1),                                  \
                       0,                                                       \
                       handle->stream,                                          \
                       stride,                                                  \
                       m,                                                       \
                       n,                                                       \
                       ((iter == 0) ? ldb : m),                                 \
                       ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),    \
                       ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),     \
                       ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),    \
                       ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)), \
                       (((iter & 1) == 0) ? da1 : da0),                         \
                       (((iter & 1) == 0) ? db1 : db0),                         \
                       (((iter & 1) == 0) ? dc1 : dc0),                         \
                       (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_PCR_STAGE1(T, block_size, stride, iter)             \
    hipLaunchKernelGGL((gtsv_nopivot_pcr_stage1_kernel<block_size>),            \
                       dim3(((m - 1) / block_size + 1), 1, 1),                  \
                       dim3(block_size),                                        \
                       0,                                                       \
                       handle->stream,                                          \
                       stride,                                                  \
                       m,                                                       \
                       n,                                                       \
                       ((iter == 0) ? ldb : m),                                 \
                       ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),    \
                       ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),     \
                       ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),    \
                       ((iter == 0) ? B : (((iter & 1) == 0) ? drhs0 : drhs1)), \
                       (((iter & 1) == 0) ? da1 : da0),                         \
                       (((iter & 1) == 0) ? db1 : db0),                         \
                       (((iter & 1) == 0) ? dc1 : dc0),                         \
                       (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_THOMAS_POW2_STAGE2(T, block_size, system_size, iter)          \
    hipLaunchKernelGGL((gtsv_nopivot_thomas_pow2_stage2_kernel<block_size, system_size>), \
                       dim3(((subsystem_count - 1) / block_size + 1), n, 1),              \
                       dim3(block_size),                                                  \
                       0,                                                                 \
                       handle->stream,                                                    \
                       stride,                                                            \
                       m,                                                                 \
                       n,                                                                 \
                       ldb,                                                               \
                       (((iter & 1) != 0) ? da1 : da0),                                   \
                       (((iter & 1) != 0) ? db1 : db0),                                   \
                       (((iter & 1) != 0) ? dc1 : dc0),                                   \
                       (((iter & 1) != 0) ? drhs1 : drhs0),                               \
                       (((iter & 1) != 0) ? da0 : da1),                                   \
                       (((iter & 1) != 0) ? db0 : db1),                                   \
                       (((iter & 1) != 0) ? dc0 : dc1),                                   \
                       (((iter & 1) != 0) ? drhs0 : drhs1),                               \
                       B);

#define LAUNCH_GTSV_NOPIVOT_THOMAS_STAGE2(T, block_size, iter)               \
    hipLaunchKernelGGL((gtsv_nopivot_thomas_stage2_kernel<block_size>),      \
                       dim3(((subsystem_count - 1) / block_size + 1), n, 1), \
                       dim3(block_size),                                     \
                       0,                                                    \
                       handle->stream,                                       \
                       stride,                                               \
                       m,                                                    \
                       n,                                                    \
                       ldb,                                                  \
                       (((iter & 1) != 0) ? da1 : da0),                      \
                       (((iter & 1) != 0) ? db1 : db0),                      \
                       (((iter & 1) != 0) ? dc1 : dc0),                      \
                       (((iter & 1) != 0) ? drhs1 : drhs0),                  \
                       (((iter & 1) != 0) ? da0 : da1),                      \
                       (((iter & 1) != 0) ? db0 : db1),                      \
                       (((iter & 1) != 0) ? dc0 : dc1),                      \
                       (((iter & 1) != 0) ? drhs0 : drhs1),                  \
                       B);

template <typename T>
rocsparse_status rocsparse_gtsv_no_pivot_large_template(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        const T*         dl,
                                                        const T*         d,
                                                        const T*         du,
                                                        T*               B,
                                                        rocsparse_int    ldb,
                                                        void*            temp_buffer)
{
    assert(m > 65536);

    char* ptr = reinterpret_cast<char*>(temp_buffer);
    T*    da0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* da1 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* db0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* db1 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* dc0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* dc1 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m - 1) / 256 + 1) * 256;
    T* drhs0 = reinterpret_cast<T*>(ptr);
    ptr += sizeof(T) * ((m * n - 1) / 256 + 1) * 256;
    T* drhs1 = reinterpret_cast<T*>(ptr);
    // ptr += sizeof(T) * ((m * n - 1) / 256 + 1) * 256;

    // Run special algorithm if m is power of 2
    if((m & (m - 1)) == 0)
    {
        // Stage1: Break large tridiagonal system into multiple smaller systems
        // using parallel cyclic reduction so that each sub system is of size 512.
        rocsparse_int iter
            = static_cast<rocsparse_int>(log2(m)) - static_cast<rocsparse_int>(log2(512));

        rocsparse_int stride = 1;
        for(rocsparse_int i = 0; i < iter; i++)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_POW2_STAGE1(T, 256, stride, i);

            stride *= 2;
        }

        rocsparse_int subsystem_count = stride;

        // Stage2: Solve the many systems from stage1 in parallel using p-thread thomas algorithm.
        LAUNCH_GTSV_NOPIVOT_THOMAS_POW2_STAGE2(T, 256, 512, iter);
    }
    else
    {
        // Stage1: Break large tridiagonal system into multiple smaller systems
        // using parallel cyclic reduction so that each sub system is of size 512 or less.
        rocsparse_int iter
            = static_cast<rocsparse_int>(log2(m)) - static_cast<rocsparse_int>(log2(512)) + 1;

        rocsparse_int stride = 1;
        for(rocsparse_int i = 0; i < iter; i++)
        {
            LAUNCH_GTSV_NOPIVOT_PCR_STAGE1(T, 256, stride, i);

            stride *= 2;
        }

        // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
        rocsparse_int subsystem_count = 1 << iter;

        LAUNCH_GTSV_NOPIVOT_THOMAS_STAGE2(T, 256, iter);
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse_gtsv_no_pivot_template(rocsparse_handle handle,
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
              replaceX<T>("rocsparse_Xgtsv_no_pivot"),
              m,
              n,
              (const void*&)dl,
              (const void*&)d,
              (const void*&)du,
              (const void*&)B,
              ldb,
              (const void*&)temp_buffer);

    log_bench(
        handle, "./rocsparse-bench -f gtsv_no_pivot -r", replaceX<T>("X"), "--mtx <matrix.mtx> ");

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

    // If m is small we can solve the systems entirely in shared memory
    if(m <= 512)
    {
        return rocsparse_gtsv_no_pivot_small_template(handle, m, n, dl, d, du, B, ldb, temp_buffer);
    }
    else if(m <= 65536)
    {
        return rocsparse_gtsv_no_pivot_medium_template(
            handle, m, n, dl, d, du, B, ldb, temp_buffer);
    }

    return rocsparse_gtsv_no_pivot_large_template(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,      \
                                     rocsparse_int    m,           \
                                     rocsparse_int    n,           \
                                     const TYPE*      dl,          \
                                     const TYPE*      d,           \
                                     const TYPE*      du,          \
                                     const TYPE*      B,           \
                                     rocsparse_int    ldb,         \
                                     size_t*          buffer_size) \
    {                                                              \
        return rocsparse_gtsv_no_pivot_buffer_size_template(       \
            handle, m, n, dl, d, du, B, ldb, buffer_size);         \
    }

C_IMPL(rocsparse_sgtsv_no_pivot_buffer_size, float);
C_IMPL(rocsparse_dgtsv_no_pivot_buffer_size, double);
C_IMPL(rocsparse_cgtsv_no_pivot_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot_buffer_size, rocsparse_double_complex);

#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                                  \
                                     rocsparse_int    m,                                       \
                                     rocsparse_int    n,                                       \
                                     const TYPE*      dl,                                      \
                                     const TYPE*      d,                                       \
                                     const TYPE*      du,                                      \
                                     TYPE*            B,                                       \
                                     rocsparse_int    ldb,                                     \
                                     void*            temp_buffer)                             \
    {                                                                                          \
        return rocsparse_gtsv_no_pivot_template(handle, m, n, dl, d, du, B, ldb, temp_buffer); \
    }

C_IMPL(rocsparse_sgtsv_no_pivot, float);
C_IMPL(rocsparse_dgtsv_no_pivot, double);
C_IMPL(rocsparse_cgtsv_no_pivot, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot, rocsparse_double_complex);

#undef C_IMPL

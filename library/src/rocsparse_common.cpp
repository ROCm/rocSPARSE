/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.h"
#include "common.h"
#include "utility.h"

#include <hip/hip_runtime.h>

namespace rocsparse
{
    // Perform dense matrix transposition
    template <uint32_t DIMX, uint32_t DIMY, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dense_transpose_device(
        I m, I n, T alpha, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb)
    {
        int lid = threadIdx.x & (DIMX - 1);
        int wid = threadIdx.x / DIMX;

        I row_A = blockIdx.x * DIMX + lid;
        I row_B = blockIdx.x * DIMX + wid;

        __shared__ T sdata[DIMX][DIMX];

        for(I j = 0; j < n; j += DIMX)
        {
            __syncthreads();

            I col_A = j + wid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(row_A < m && col_A + k < n)
                {
                    sdata[wid + k][lid] = A[row_A + lda * (col_A + k)];
                }
            }

            __syncthreads();

            I col_B = j + lid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(col_B < n && row_B + k < m)
                {
                    B[col_B + ldb * (row_B + k)] = alpha * sdata[lid][wid + k];
                }
            }
        }
    }

    // Perform dense matrix back transposition
    template <uint32_t DIMX, uint32_t DIMY, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dense_transpose_back_device(
        I m, I n, const T* __restrict__ A, int64_t lda, T* __restrict__ B, int64_t ldb)
    {
        int lid = hipThreadIdx_x & (DIMX - 1);
        int wid = hipThreadIdx_x / DIMX;

        I row_A = hipBlockIdx_x * DIMX + wid;
        I row_B = hipBlockIdx_x * DIMX + lid;

        __shared__ T sdata[DIMX][DIMX];

        for(I j = 0; j < n; j += DIMX)
        {
            __syncthreads();

            I col_A = j + lid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(col_A < n && row_A + k < m)
                {
                    sdata[wid + k][lid] = A[col_A + lda * (row_A + k)];
                }
            }

            __syncthreads();

            I col_B = j + wid;

            for(uint32_t k = 0; k < DIMX; k += DIMY)
            {
                if(row_B < m && col_B + k < n)
                {
                    B[row_B + ldb * (col_B + k)] = sdata[lid][wid + k];
                }
            }
        }
    }

    // conjugate values in array
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void conjugate_device(I length, T* __restrict__ array)
    {
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(idx >= length)
        {
            return;
        }

        array[idx] = rocsparse::conj(array[idx]);
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void valset_device(I length, T value, T* __restrict__ array)
    {
        I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

        if(idx >= length)
        {
            return;
        }

        array[idx] = value;
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void valset_2d_device(
        I m, I n, int64_t ld, T value, T* __restrict__ array, rocsparse_order order)
    {
        I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= m * n)
        {
            return;
        }

        I wid = (order == rocsparse_order_column) ? gid / m : gid / n;
        I lid = (order == rocsparse_order_column) ? gid % m : gid % n;

        array[lid + ld * wid] = value;
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_DEVICE_ILF void scale_device(I length, T scalar, A* __restrict__ array)
    {
        const I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= length)
        {
            return;
        }

        if(scalar == static_cast<T>(0))
        {
            array[gid] = static_cast<A>(0);
        }
        else
        {
            array[gid] *= scalar;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_DEVICE_ILF void scale_2d_device(
        I m, I n, int64_t ld, int64_t stride, T value, A* __restrict__ array, rocsparse_order order)
    {
        I gid   = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        I batch = hipBlockIdx_y;

        if(gid >= m * n)
        {
            return;
        }

        I wid = (order == rocsparse_order_column) ? gid / m : gid / n;
        I lid = (order == rocsparse_order_column) ? gid % m : gid % n;

        if(value == static_cast<T>(0))
        {
            array[lid + ld * wid + stride * batch] = static_cast<A>(0);
        }
        else
        {
            array[lid + ld * wid + stride * batch] *= value;
        }
    }

    template <uint32_t DIM_X, uint32_t DIM_Y, typename I, typename T, typename U>
    ROCSPARSE_KERNEL(DIM_X* DIM_Y)
    void dense_transpose_kernel(
        I m, I n, U alpha_device_host, const T* A, int64_t lda, T* B, int64_t ldb)
    {
        const auto alpha = rocsparse::load_scalar_device_host(alpha_device_host);
        rocsparse::dense_transpose_device<DIM_X, DIM_Y>(m, n, alpha, A, lda, B, ldb);
    }

    template <uint32_t DIM_X, uint32_t DIM_Y, typename I, typename T>
    ROCSPARSE_KERNEL(DIM_X* DIM_Y)
    void dense_transpose_back_kernel(I m, I n, const T* A, int64_t lda, T* B, int64_t ldb)
    {
        rocsparse::dense_transpose_back_device<DIM_X, DIM_Y>(m, n, A, lda, B, ldb);
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void conjugate_kernel(I length, T* array)
    {
        rocsparse::conjugate_device<BLOCKSIZE>(length, array);
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void valset_kernel(I length, T value, T* array)
    {
        rocsparse::valset_device<BLOCKSIZE>(length, value, array);
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void valset_2d_kernel(I m, I n, int64_t ld, T value, T* array, rocsparse_order order)
    {
        rocsparse::valset_2d_device<BLOCKSIZE>(m, n, ld, value, array, order);
    }

    template <uint32_t BLOCKSIZE, typename I, typename T, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scale_kernel(I length, U scalar_device_host, T* array)
    {
        auto scalar = rocsparse::load_scalar_device_host(scalar_device_host);

        if(scalar != static_cast<T>(1))
        {
            rocsparse::scale_device<BLOCKSIZE>(length, scalar, array);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T, typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scale_2d_kernel(
        I m, I n, int64_t ld, int64_t stride, U scalar_device_host, T* array, rocsparse_order order)
    {
        auto scalar = rocsparse::load_scalar_device_host(scalar_device_host);

        if(scalar != static_cast<T>(1))
        {
            rocsparse::scale_2d_device<BLOCKSIZE>(m, n, ld, stride, scalar, array, order);
        }
    }
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse::dense_transpose(rocsparse_handle handle,
                                            I                m,
                                            I                n,
                                            U                alpha_device_host,
                                            const T*         A,
                                            int64_t          lda,
                                            T*               B,
                                            int64_t          ldb)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dense_transpose_kernel<32, 8>),
                                       dim3((m - 1) / 32 + 1),
                                       dim3(32 * 8),
                                       0,
                                       handle->stream,
                                       m,
                                       n,
                                       alpha_device_host,
                                       A,
                                       lda,
                                       B,
                                       ldb);

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse::dense_transpose_back(
    rocsparse_handle handle, I m, I n, const T* A, int64_t lda, T* B, int64_t ldb)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dense_transpose_back_kernel<32, 8>),
                                       dim3((m - 1) / 32 + 1),
                                       dim3(32 * 8),
                                       0,
                                       handle->stream,
                                       m,
                                       n,
                                       A,
                                       lda,
                                       B,
                                       ldb);

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse::conjugate(rocsparse_handle handle, I length, T* array)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::conjugate_kernel<256>),
                                       dim3((length - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       length,
                                       array);

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse::valset(rocsparse_handle handle, I length, T value, T* array)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::valset_kernel<256>),
                                       dim3((length - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       length,
                                       value,
                                       array);

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse::valset_2d(
    rocsparse_handle handle, I m, I n, int64_t ld, T value, T* array, rocsparse_order order)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::valset_2d_kernel<256>),
                                       dim3((int64_t(m) * n - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       m,
                                       n,
                                       ld,
                                       value,
                                       array,
                                       order);

    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status
    rocsparse::scale_array(rocsparse_handle handle, I length, U scalar_device_host, T* array)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::scale_kernel<256>),
                                       dim3((length - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       length,
                                       scalar_device_host,
                                       array);

    return rocsparse_status_success;
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse::scale_2d_array(rocsparse_handle handle,
                                           I                m,
                                           I                n,
                                           int64_t          ld,
                                           int64_t          batch_count,
                                           int64_t          stride,
                                           U                scalar_device_host,
                                           T*               array,
                                           rocsparse_order  order)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::scale_2d_kernel<256>),
                                       dim3((int64_t(m) * n - 1) / 256 + 1, batch_count),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       m,
                                       n,
                                       ld,
                                       stride,
                                       scalar_device_host,
                                       array,
                                       order);

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE, UTYPE)                                                     \
    template rocsparse_status rocsparse::dense_transpose(rocsparse_handle handle,            \
                                                         ITYPE            m,                 \
                                                         ITYPE            n,                 \
                                                         UTYPE            alpha_device_host, \
                                                         const TTYPE*     A,                 \
                                                         int64_t          lda,               \
                                                         TTYPE*           B,                 \
                                                         int64_t          ldb);

INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, float, const float*);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, double, const double*);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, const rocsparse_double_complex*);

INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, float, const float*);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, double, const double*);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, const rocsparse_double_complex*);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                                      \
    template rocsparse_status rocsparse::dense_transpose_back(rocsparse_handle handle, \
                                                              ITYPE            m,      \
                                                              ITYPE            n,      \
                                                              const TTYPE*     A,      \
                                                              int64_t          lda,    \
                                                              TTYPE*           B,      \
                                                              int64_t          ldb);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                   \
    template rocsparse_status rocsparse::conjugate( \
        rocsparse_handle handle, ITYPE length, TTYPE* array);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                \
    template rocsparse_status rocsparse::valset( \
        rocsparse_handle handle, ITYPE length, TTYPE value, TTYPE* array);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                           \
    template rocsparse_status rocsparse::valset_2d(rocsparse_handle handle, \
                                                   ITYPE            m,      \
                                                   ITYPE            n,      \
                                                   int64_t          ld,     \
                                                   TTYPE            value,  \
                                                   TTYPE*           array,  \
                                                   rocsparse_order  order);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE, UTYPE)              \
    template rocsparse_status rocsparse::scale_array( \
        rocsparse_handle handle, ITYPE length, UTYPE scalar_device_host, TTYPE* array);

INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, float, const float*);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, double, const double*);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, const rocsparse_double_complex*);

INSTANTIATE(int64_t, int32_t, int32_t);
INSTANTIATE(int64_t, int32_t, const int32_t*);
INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, float, const float*);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, double, const double*);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, const rocsparse_double_complex*);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE, UTYPE)                                                     \
    template rocsparse_status rocsparse::scale_2d_array(rocsparse_handle handle,             \
                                                        ITYPE            m,                  \
                                                        ITYPE            n,                  \
                                                        int64_t          ld,                 \
                                                        int64_t          batch_count,        \
                                                        int64_t          stride,             \
                                                        UTYPE            scalar_device_host, \
                                                        TTYPE*           array,              \
                                                        rocsparse_order  order);

INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int32_t, const int32_t*);
INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, float, const float*);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, double, const double*);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, const rocsparse_double_complex*);

INSTANTIATE(int64_t, int32_t, int32_t);
INSTANTIATE(int64_t, int32_t, const int32_t*);
INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, float, const float*);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, double, const double*);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, const rocsparse_double_complex*);
#undef INSTANTIATE

/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include "definitions.h"
#include "utility.h"

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T>
static ROCSPARSE_DEVICE_ILF void coommnn_segmented_main_device(bool conj_A,
                                                               bool conj_B,
                                                               I    M,
                                                               I    N,
                                                               I    K,
                                                               I    nnz,
                                                               I    batch_stride_A,
                                                               T    alpha,
                                                               I* __restrict__ row_block_red,
                                                               T* __restrict__ val_block_red,
                                                               const I* __restrict__ coo_row_ind,
                                                               const I* __restrict__ coo_col_ind,
                                                               const T* __restrict__ coo_val,
                                                               const T* __restrict__ B,
                                                               I ldb,
                                                               I batch_stride_B,
                                                               T* __restrict__ C,
                                                               I                    ldc,
                                                               I                    batch_stride_C,
                                                               rocsparse_order      order,
                                                               rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    int bid = hipBlockIdx_x;
    int lid = tid & (WF_SIZE - 1);
    int wid = tid / WF_SIZE;

    int batch = hipBlockIdx_z;

    __shared__ I shared_row[BLOCKSIZE];
    __shared__ T shared_val_prev[WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

    I colB = WF_SIZE * hipBlockIdx_y;

    I offset = bid * LOOPS * BLOCKSIZE;
    I idx    = offset + tid;

    I row_ind;
    T valB[WF_SIZE];

    while(idx < (offset + LOOPS * BLOCKSIZE))
    {
        I row = (idx < nnz) ? rocsparse_nontemporal_load(&coo_row_ind[idx + batch_stride_A * batch])
                                  - idx_base
                            : -1;
        I col = (idx < nnz) ? rocsparse_nontemporal_load(&coo_col_ind[idx + batch_stride_A * batch])
                                  - idx_base
                            : 0;
        T val = (idx < nnz) ? alpha
                                  * conj_val(rocsparse_nontemporal_load(
                                                 &coo_val[idx + batch_stride_A * batch]),
                                             conj_A)
                            : static_cast<T>(0);

        row_ind = row;

        for(I i = 0; i < WF_SIZE; ++i)
        {
            T v = rocsparse_shfl(val, i, WF_SIZE);
            I c = rocsparse_shfl(col, i, WF_SIZE);

            if(!TRANSB)
            {
                valB[i] = v * conj_val(B[c + ldb * (colB + lid) + batch_stride_B * batch], conj_B);
            }
            else
            {
                valB[i] = v * conj_val(B[ldb * c + (colB + lid) + batch_stride_B * batch], conj_B);
            }
        }

        // Transpose
        __syncthreads();
        for(I i = 0; i < WF_SIZE; ++i)
        {
            shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
        }
        __syncthreads();
        for(I i = 0; i < WF_SIZE; ++i)
        {
            valB[i] = shared_val[BLOCKSIZE * i + tid];
        }

        // First thread in block checks row index from previous loop
        // if it has been completed or if additional rows have to be
        // appended.
        if(idx > offset && tid == 0)
        {
            I prevrow = shared_row[BLOCKSIZE - 1];
            if(row_ind == prevrow)
            {
                for(I i = 0; i < WF_SIZE; ++i)
                {
                    valB[i] += shared_val_prev[i];
                }
            }
            else if(prevrow >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        C[prevrow + ldc * (colB + i) + batch_stride_C * batch]
                            += shared_val_prev[i];
                    }
                }
                else
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        C[(colB + i) + ldc * prevrow + batch_stride_C * batch]
                            += shared_val_prev[i];
                    }
                }
            }
        }

        __syncthreads();
        shared_row[tid] = row_ind;
        for(I i = 0; i < WF_SIZE; ++i)
        {
            shared_val[BLOCKSIZE * i + tid] = valB[i];
        }
        __syncthreads();

        // segmented reduction
        for(I j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row_ind == shared_row[tid - j])
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                    }
                }
            }
            __syncthreads();
            for(I i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * i + tid] = valB[i];
            }
            __syncthreads();
        }

        shared_val_prev[lid] = shared_val[BLOCKSIZE * lid + (BLOCKSIZE - 1)];
        __syncthreads();

        // All lanes but the last one write their result in C.
        // The last value might need to be appended by the next iteration.
        if(tid < BLOCKSIZE - 1)
        {
            if(row_ind != shared_row[tid + 1] && row_ind >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        C[row_ind + ldc * (colB + i) + batch_stride_C * batch] += valB[i];
                    }
                }
                else
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        C[(colB + i) + ldc * row_ind + batch_stride_C * batch] += valB[i];
                    }
                }
            }
        }

        idx += BLOCKSIZE;
    }

    if(tid == BLOCKSIZE - 1)
    {
        row_block_red[bid + hipGridDim_x * batch] = row_ind;
        for(I i = 0; i < WF_SIZE; ++i)
        {
            val_block_red[hipGridDim_x * (colB + i) + bid + (hipGridDim_x * N) * batch] = valB[i];
        }
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T>
static ROCSPARSE_DEVICE_ILF void
    coommnn_segmented_remainder_device(bool conj_A,
                                       bool conj_B,
                                       I    colB_offset,
                                       I    M,
                                       I    N,
                                       I    K,
                                       I    nnz,
                                       I    batch_stride_A,
                                       T    alpha,
                                       I* __restrict__ row_block_red,
                                       T* __restrict__ val_block_red,
                                       const I* __restrict__ coo_row_ind,
                                       const I* __restrict__ coo_col_ind,
                                       const T* __restrict__ coo_val,
                                       const T* __restrict__ B,
                                       I ldb,
                                       I batch_stride_B,
                                       T* __restrict__ C,
                                       I                    ldc,
                                       I                    batch_stride_C,
                                       rocsparse_order      order,
                                       rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;
    int bid = hipBlockIdx_x;
    int lid = tid & (WF_SIZE - 1);
    int wid = tid / WF_SIZE;

    int batch = hipBlockIdx_z;

    __shared__ I shared_row[BLOCKSIZE];
    __shared__ T shared_val_prev[WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE * WF_SIZE];

    I colB = colB_offset;

    I offset = bid * LOOPS * BLOCKSIZE;
    I idx    = offset + tid;

    I row_ind;
    T valB[WF_SIZE];

    while(idx < (offset + LOOPS * BLOCKSIZE))
    {
        I row = (idx < nnz) ? rocsparse_nontemporal_load(&coo_row_ind[idx + batch_stride_A * batch])
                                  - idx_base
                            : -1;
        I col = (idx < nnz) ? rocsparse_nontemporal_load(&coo_col_ind[idx + batch_stride_A * batch])
                                  - idx_base
                            : 0;
        T val = (idx < nnz) ? alpha
                                  * conj_val(rocsparse_nontemporal_load(
                                                 &coo_val[idx + batch_stride_A * batch]),
                                             conj_A)
                            : static_cast<T>(0);

        row_ind = row;

        for(I i = 0; i < WF_SIZE; ++i)
        {
            T v = rocsparse_shfl(val, i, WF_SIZE);
            I c = __shfl(col, i, WF_SIZE);

            if(!TRANSB)
            {
                valB[i]
                    = (colB + lid) < N
                          ? v * conj_val(B[c + ldb * (colB + lid) + batch_stride_B * batch], conj_B)
                          : static_cast<T>(0);
            }
            else
            {
                valB[i]
                    = (colB + lid) < N
                          ? v * conj_val(B[ldb * c + (colB + lid) + batch_stride_B * batch], conj_B)
                          : static_cast<T>(0);
            }
        }

        // Transpose
        __syncthreads();
        for(I i = 0; i < WF_SIZE; ++i)
        {
            shared_val[BLOCKSIZE * lid + WF_SIZE * wid + i] = valB[i];
        }
        __syncthreads();
        for(I i = 0; i < WF_SIZE; ++i)
        {
            valB[i] = shared_val[BLOCKSIZE * i + tid];
        }

        // First thread in block checks row index from previous loop
        // if it has been completed or if additional rows have to be
        // appended.
        if(idx > offset && tid == 0)
        {
            I prevrow = shared_row[BLOCKSIZE - 1];
            if(row_ind == prevrow)
            {
                for(I i = 0; i < WF_SIZE; ++i)
                {
                    valB[i] += shared_val_prev[i];
                }
            }
            else if(prevrow >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        if((colB + i) < N)
                        {
                            C[prevrow + ldc * (colB + i) + batch_stride_C * batch]
                                += shared_val_prev[i];
                        }
                    }
                }
                else
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        if((colB + i) < N)
                        {
                            C[colB + i + ldc * prevrow + batch_stride_C * batch]
                                += shared_val_prev[i];
                        }
                    }
                }
            }
        }

        __syncthreads();
        shared_row[tid] = row_ind;
        for(I i = 0; i < WF_SIZE; ++i)
        {
            shared_val[BLOCKSIZE * i + tid] = valB[i];
        }
        __syncthreads();

        // segmented reduction
        for(I j = 1; j < BLOCKSIZE; j <<= 1)
        {
            if(tid >= j)
            {
                if(row_ind == shared_row[tid - j])
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        valB[i] = valB[i] + shared_val[BLOCKSIZE * i + tid - j];
                    }
                }
            }
            __syncthreads();
            for(I i = 0; i < WF_SIZE; ++i)
            {
                shared_val[BLOCKSIZE * i + tid] = valB[i];
            }
            __syncthreads();
        }

        shared_val_prev[lid] = shared_val[BLOCKSIZE * lid + (BLOCKSIZE - 1)];
        __syncthreads();

        // All lanes but the last one write their result in C.
        // The last value might need to be appended by the next iteration.
        if(tid < BLOCKSIZE - 1)
        {
            if(row_ind != shared_row[tid + 1] && row_ind >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        if((colB + i) < N)
                        {
                            C[row_ind + ldc * (colB + i) + batch_stride_C * batch] += valB[i];
                        }
                    }
                }
                else
                {
                    for(I i = 0; i < WF_SIZE; ++i)
                    {
                        if((colB + i) < N)
                        {
                            C[(colB + i) + ldc * row_ind + batch_stride_C * batch] += valB[i];
                        }
                    }
                }
            }
        }

        idx += BLOCKSIZE;
    }

    if(tid == BLOCKSIZE - 1)
    {
        row_block_red[bid + hipGridDim_x * batch] = row_ind;
        for(I i = 0; i < WF_SIZE; ++i)
        {
            if((colB + i) < N)
            {
                val_block_red[hipGridDim_x * (colB + i) + bid + (hipGridDim_x * N) * batch]
                    = valB[i];
            }
        }
    }
}

// Segmented block reduction kernel
template <unsigned int BLOCKSIZE, typename I, typename T>
static ROCSPARSE_DEVICE_ILF void segmented_blockreduce(const I* rows, T* vals)
{
    int tid = hipThreadIdx_x;

#pragma unroll
    for(int j = 1; j < BLOCKSIZE; j <<= 1)
    {
        T val = static_cast<T>(0);
        if(tid >= j)
        {
            if(rows[tid] == rows[tid - j])
            {
                val = vals[tid - j];
            }
        }
        __syncthreads();

        vals[tid] = vals[tid] + val;
        __syncthreads();
    }
}

// Do the final block reduction of the block reduction buffers back into global memory
template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coommnn_general_block_reduce(I n,
                                      I nblocks,
                                      const I* __restrict__ row_block_red,
                                      const T* __restrict__ val_block_red,
                                      T*              C,
                                      I               ldc,
                                      I               batch_stride_C,
                                      rocsparse_order order)
{
    int tid   = hipThreadIdx_x;
    int batch = hipBlockIdx_z;

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ I shared_row[BLOCKSIZE];
    __shared__ T shared_val[BLOCKSIZE];

    shared_row[tid] = -1;
    shared_val[tid] = static_cast<T>(0);

    __syncthreads();

    I col = hipBlockIdx_x;

    for(I i = tid; i < nblocks; i += BLOCKSIZE)
    {
        // Copy data to reduction buffers
        shared_row[tid] = row_block_red[i + nblocks * batch];
        shared_val[tid] = val_block_red[i + nblocks * col + nblocks * n * batch];

        __syncthreads();

        // Do segmented block reduction
        segmented_blockreduce<BLOCKSIZE>(shared_row, shared_val);

        // Add reduced sum to C if valid
        I row   = shared_row[tid];
        I rowp1 = (tid < BLOCKSIZE - 1) ? shared_row[tid + 1] : -1;

        if(row != rowp1 && row >= 0)
        {
            if(order == rocsparse_order_column)
            {
                C[row + ldc * col + batch_stride_C * batch] += shared_val[tid];
            }
            else
            {
                C[col + ldc * row + batch_stride_C * batch] += shared_val[tid];
            }
        }

        __syncthreads();
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coommnn_segmented_main_kernel(bool conj_A,
                                       bool conj_B,
                                       I    M,
                                       I    N,
                                       I    K,
                                       I    nnz,
                                       I    batch_stride_A,
                                       U    alpha_device_host,
                                       I* __restrict__ row_block_red,
                                       T* __restrict__ val_block_red,
                                       const I* __restrict__ coo_row_ind,
                                       const I* __restrict__ coo_col_ind,
                                       const T* __restrict__ coo_val,
                                       const T* __restrict__ B,
                                       I ldb,
                                       I batch_stride_B,
                                       T* __restrict__ C,
                                       I                    ldc,
                                       I                    batch_stride_C,
                                       rocsparse_order      order,
                                       rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);

    if(alpha != static_cast<T>(0))
    {
        coommnn_segmented_main_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(conj_A,
                                                                         conj_B,
                                                                         M,
                                                                         N,
                                                                         K,
                                                                         nnz,
                                                                         batch_stride_A,
                                                                         alpha,
                                                                         row_block_red,
                                                                         val_block_red,
                                                                         coo_row_ind,
                                                                         coo_col_ind,
                                                                         coo_val,
                                                                         B,
                                                                         ldb,
                                                                         batch_stride_B,
                                                                         C,
                                                                         ldc,
                                                                         batch_stride_C,
                                                                         order,
                                                                         idx_base);
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          unsigned int LOOPS,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coommnn_segmented_remainder_kernel(bool conj_A,
                                            bool conj_B,
                                            I    colB_offset,
                                            I    M,
                                            I    N,
                                            I    K,
                                            I    nnz,
                                            I    batch_stride_A,
                                            U    alpha_device_host,
                                            I* __restrict__ row_block_red,
                                            T* __restrict__ val_block_red,
                                            const I* __restrict__ coo_row_ind,
                                            const I* __restrict__ coo_col_ind,
                                            const T* __restrict__ coo_val,
                                            const T* __restrict__ B,
                                            I ldb,
                                            I batch_stride_B,
                                            T* __restrict__ C,
                                            I                    ldc,
                                            I                    batch_stride_C,
                                            rocsparse_order      order,
                                            rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);

    if(alpha != static_cast<T>(0))
    {
        coommnn_segmented_remainder_device<BLOCKSIZE, WF_SIZE, LOOPS, TRANSB>(conj_A,
                                                                              conj_B,
                                                                              colB_offset,
                                                                              M,
                                                                              N,
                                                                              K,
                                                                              nnz,
                                                                              batch_stride_A,
                                                                              alpha,
                                                                              row_block_red,
                                                                              val_block_red,
                                                                              coo_row_ind,
                                                                              coo_col_ind,
                                                                              coo_val,
                                                                              B,
                                                                              ldb,
                                                                              batch_stride_B,
                                                                              C,
                                                                              ldc,
                                                                              batch_stride_C,
                                                                              order,
                                                                              idx_base);
    }
}

template <typename I, typename T>
rocsparse_status rocsparse_coomm_buffer_size_template_segmented(rocsparse_handle    handle,
                                                                rocsparse_operation trans_A,
                                                                I                   m,
                                                                I                   n,
                                                                I                   k,
                                                                I                   nnz,
                                                                I                   batch_count,
                                                                const rocsparse_mat_descr descr,
                                                                const T*                  coo_val,
                                                                const I* coo_row_ind,
                                                                const I* coo_col_ind,
                                                                size_t*  buffer_size)
{
#define LOOPS 4
#define COOMMN_DIM 256
    I nblocks    = (nnz - 1) / (COOMMN_DIM * LOOPS) + 1;
    *buffer_size = size_t(256)
                   + ((sizeof(I) * nblocks * batch_count - 1) / COOMMN_DIM + 1) * COOMMN_DIM
                   + ((sizeof(T) * nblocks * n * batch_count - 1) / COOMMN_DIM + 1) * COOMMN_DIM;
#undef COOMMN_DIM
#undef LOOPS

    return rocsparse_status_success;
}

#define LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)            \
    hipLaunchKernelGGL((coommnn_segmented_main_kernel<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB>), \
                       dim3(nblocks, (main - 1) / WF_SIZE + 1, batch_count_C),               \
                       dim3(COOMMNN_DIM),                                                    \
                       0,                                                                    \
                       stream,                                                               \
                       conj_A,                                                               \
                       conj_B,                                                               \
                       m,                                                                    \
                       n,                                                                    \
                       k,                                                                    \
                       nnz,                                                                  \
                       batch_stride_A,                                                       \
                       alpha_device_host,                                                    \
                       row_block_red,                                                        \
                       val_block_red,                                                        \
                       coo_row_ind,                                                          \
                       coo_col_ind,                                                          \
                       coo_val,                                                              \
                       B,                                                                    \
                       ldb,                                                                  \
                       batch_stride_B,                                                       \
                       C,                                                                    \
                       ldc,                                                                  \
                       batch_stride_C,                                                       \
                       order,                                                                \
                       descr->base);

#define LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)            \
    hipLaunchKernelGGL((coommnn_segmented_remainder_kernel<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB>), \
                       dim3(nblocks, 1, batch_count_C),                                           \
                       dim3(COOMMNN_DIM),                                                         \
                       0,                                                                         \
                       stream,                                                                    \
                       conj_A,                                                                    \
                       conj_B,                                                                    \
                       main,                                                                      \
                       m,                                                                         \
                       n,                                                                         \
                       k,                                                                         \
                       nnz,                                                                       \
                       batch_stride_A,                                                            \
                       alpha_device_host,                                                         \
                       row_block_red,                                                             \
                       val_block_red,                                                             \
                       coo_row_ind,                                                               \
                       coo_col_ind,                                                               \
                       coo_val,                                                                   \
                       B,                                                                         \
                       ldb,                                                                       \
                       batch_stride_B,                                                            \
                       C,                                                                         \
                       ldc,                                                                       \
                       batch_stride_C,                                                            \
                       order,                                                                     \
                       descr->base);

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_segmented(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_order           order,
                                                    I                         m,
                                                    I                         n,
                                                    I                         k,
                                                    I                         nnz,
                                                    I                         batch_count_A,
                                                    I                         batch_stride_A,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  coo_val,
                                                    const I*                  coo_row_ind,
                                                    const I*                  coo_col_ind,
                                                    const T*                  B,
                                                    I                         ldb,
                                                    I                         batch_count_B,
                                                    I                         batch_stride_B,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    I                         ldc,
                                                    I                         batch_count_C,
                                                    I                         batch_stride_C,
                                                    void*                     temp_buffer)
{
    bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose);
    bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

    // Stream
    hipStream_t stream = handle->stream;

    // Run different coomm kernels
    if(trans_A == rocsparse_operation_none)
    {
#define LOOPS 4
#define COOMMN_DIM 256
        I nblocks = (nnz - 1) / (COOMMN_DIM * LOOPS) + 1;

        // row and val block reduction buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);
        ptr += 256;
        I* row_block_red = reinterpret_cast<I*>(reinterpret_cast<void*>(ptr));
        ptr += ((sizeof(I) * nblocks * batch_count_C - 1) / COOMMN_DIM + 1) * COOMMN_DIM;
        T* val_block_red = reinterpret_cast<T*>(reinterpret_cast<void*>(ptr));
        // ptr += ((sizeof(T) * nblocks * n * batch_count_C - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            row_block_red,
            0XFF,
            ((sizeof(I) * nblocks * batch_count_C - 1) / COOMMN_DIM + 1) * COOMMN_DIM,
            stream));

        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_conjugate_transpose))
        {
            I main      = 0;
            I remainder = 0;

            if(n >= 8)
            {
                remainder = n % 8;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 8, LOOPS, false);
            }
            else if(n >= 4)
            {
                remainder = n % 4;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 4, LOOPS, false);
            }
            else if(n >= 2)
            {
                remainder = n % 2;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 2, LOOPS, false);
            }
            else if(n >= 1)
            {
                remainder = n % 1;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 1, LOOPS, false);
            }
            else
            {
                remainder = n;
            }

            if(remainder > 0)
            {
                if(remainder <= 1)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 1, LOOPS, false);
                }
                else if(remainder <= 2)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 2, LOOPS, false);
                }
                else if(remainder <= 4)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 4, LOOPS, false);
                }
                else if(remainder <= 8)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 8, LOOPS, false);
                }
            }
        }
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_column
                    && trans_B == rocsparse_operation_conjugate_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            I main      = 0;
            I remainder = 0;

            if(n >= 8)
            {
                remainder = n % 8;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 8, LOOPS, true);
            }
            else if(n >= 4)
            {
                remainder = n % 4;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 4, LOOPS, true);
            }
            else if(n >= 2)
            {
                remainder = n % 2;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 2, LOOPS, true);
            }
            else if(n >= 1)
            {
                remainder = n % 1;
                main      = n - remainder;
                LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMN_DIM, 1, LOOPS, true);
            }
            else
            {
                remainder = n;
            }

            if(remainder > 0)
            {
                if(remainder <= 1)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 1, LOOPS, true);
                }
                else if(remainder <= 2)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 2, LOOPS, true);
                }
                else if(remainder <= 4)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 4, LOOPS, true);
                }
                else if(remainder <= 8)
                {
                    LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMN_DIM, 8, LOOPS, true);
                }
            }
        }
#undef COOMMN_DIM
#undef LOOPS

        hipLaunchKernelGGL((coommnn_general_block_reduce<1024>),
                           dim3(n, 1, batch_count_C),
                           1024,
                           0,
                           stream,
                           n,
                           nblocks,
                           row_block_red,
                           val_block_red,
                           C,
                           ldc,
                           batch_stride_C,
                           order);
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                                           \
    template rocsparse_status rocsparse_coomm_buffer_size_template_segmented<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                                   \
        rocsparse_operation       trans_A,                                                  \
        ITYPE                     m,                                                        \
        ITYPE                     n,                                                        \
        ITYPE                     k,                                                        \
        ITYPE                     nnz,                                                      \
        ITYPE                     batch_count,                                              \
        const rocsparse_mat_descr descr,                                                    \
        const TTYPE*              coo_val,                                                  \
        const ITYPE*              coo_row_ind,                                              \
        const ITYPE*              coo_col_ind,                                              \
        size_t*                   buffer_size);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE, UTYPE)                                               \
    template rocsparse_status rocsparse_coomm_template_segmented<ITYPE, TTYPE, UTYPE>( \
        rocsparse_handle          handle,                                              \
        rocsparse_operation       trans_A,                                             \
        rocsparse_operation       trans_B,                                             \
        rocsparse_order           order,                                               \
        ITYPE                     m,                                                   \
        ITYPE                     n,                                                   \
        ITYPE                     k,                                                   \
        ITYPE                     nnz,                                                 \
        ITYPE                     batch_count_A,                                       \
        ITYPE                     batch_stride_A,                                      \
        UTYPE                     alpha_device_host,                                   \
        const rocsparse_mat_descr descr,                                               \
        const TTYPE*              coo_val,                                             \
        const ITYPE*              coo_row_ind,                                         \
        const ITYPE*              coo_col_ind,                                         \
        const TTYPE*              B,                                                   \
        ITYPE                     ldb,                                                 \
        ITYPE                     batch_count_B,                                       \
        ITYPE                     batch_stride_B,                                      \
        UTYPE                     beta_device_host,                                    \
        TTYPE*                    C,                                                   \
        ITYPE                     ldc,                                                 \
        ITYPE                     batch_count_C,                                       \
        ITYPE                     batch_stride_C,                                      \
        void*                     temp_buffer);

INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int32_t, float, const float*);
INSTANTIATE(int32_t, double, const double*);
INSTANTIATE(int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, rocsparse_double_complex, const rocsparse_double_complex*);

INSTANTIATE(int64_t, float, const float*);
INSTANTIATE(int64_t, double, const double*);
INSTANTIATE(int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, rocsparse_double_complex, const rocsparse_double_complex*);
#undef INSTANTIATE

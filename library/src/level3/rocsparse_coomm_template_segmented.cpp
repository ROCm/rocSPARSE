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

#include "rocsparse_coomm_template_segmented.hpp"
#include "common.h"
#include "definitions.h"
#include "utility.h"

template <unsigned int BLOCKSIZE, unsigned int WF_SIZE, bool TRANSB, typename I, typename T>
static __device__ void coommnn_general_wf_segmented(I                    nnz,
                                                    I                    n,
                                                    I                    loops,
                                                    T                    alpha,
                                                    const I*             coo_row_ind,
                                                    const I*             coo_col_ind,
                                                    const T*             coo_val,
                                                    const T*             B,
                                                    I                    ldb,
                                                    T*                   C,
                                                    I                    ldc,
                                                    I*                   row_block_red,
                                                    T*                   val_block_red,
                                                    rocsparse_order      order,
                                                    rocsparse_index_base idx_base)
{
    int tid = hipThreadIdx_x;

    // Lane index (0,...,WF_SIZE)
    int lid = tid & (WF_SIZE - 1);
    // Wavefront index
    I wid = tid / WF_SIZE;

    // Shared memory to hold row indices and values for segmented reduction
    __shared__ I shared_row[WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE + 1];

    I col    = BLOCKSIZE / WF_SIZE * hipBlockIdx_y + wid;
    I offset = hipBlockIdx_x * loops * WF_SIZE;

    // Current threads index into COO structure
    I idx = offset + lid;

    I row;
    T val;

    // Each thread processes 'loop' COO entries
    while(idx < offset + loops * WF_SIZE)
    {
        // Get corresponding COO entry
        I r = (idx < nnz) ? rocsparse_nontemporal_load(coo_row_ind + idx) - idx_base : -1;
        I c = (idx < nnz) ? rocsparse_nontemporal_load(coo_col_ind + idx) - idx_base : 0;
        T v = (idx < nnz) ? alpha * rocsparse_nontemporal_load(coo_val + idx) : static_cast<T>(0);

        row = r;

        if(!TRANSB)
        {
            val = (col < n) ? v * rocsparse_ldg(B + col * ldb + c) : static_cast<T>(0);
        }
        else
        {
            val = (col < n) ? v * rocsparse_ldg(B + c * ldb + col) : static_cast<T>(0);
        }

        // First thread in wavefront checks row index from previous loop
        // if it has been completed or if additional rows have to be
        // appended.
        if(idx > offset && lid == 0 && col < n)
        {
            I prevrow = shared_row[WF_SIZE - 1];
            if(row == prevrow)
            {
                val = val + shared_val[wid][WF_SIZE - 1];
            }
            else if(prevrow >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    C[prevrow + col * ldc] = C[prevrow + col * ldc] + shared_val[wid][WF_SIZE - 1];
                }
                else
                {
                    C[col + prevrow * ldc] = C[col + prevrow * ldc] + shared_val[wid][WF_SIZE - 1];
                }
            }
        }

        __syncthreads();

        shared_val[wid][lid] = val;
        shared_row[lid]      = row;

        __syncthreads();

#pragma unroll
        // Segmented wavefront reduction
        for(int j = 1; j < WF_SIZE; j <<= 1)
        {
            if(lid >= j)
            {
                if(row == shared_row[lid - j])
                {
                    val = val + shared_val[wid][lid - j];
                }
            }
            __syncthreads();

            shared_val[wid][lid] = val;

            __syncthreads();
        }

        // All lanes but the last one write their result in C.
        // The last value might need to be appended by the next iteration.
        if(lid < WF_SIZE - 1 && col < n)
        {
            if(row != shared_row[lid + 1] && row >= 0)
            {
                if(order == rocsparse_order_column)
                {
                    C[row + col * ldc] = C[row + col * ldc] + val;
                }
                else
                {
                    C[col + row * ldc] = C[col + row * ldc] + val;
                }
            }
        }

        idx += WF_SIZE;
    }

    // Write last entries into buffers for segmented block reduction
    if(lid == WF_SIZE - 1 && col < n)
    {
        rocsparse_nontemporal_store(row, row_block_red + hipBlockIdx_x + hipGridDim_x * col);
        rocsparse_nontemporal_store(val, val_block_red + hipBlockIdx_x + hipGridDim_x * col);
    }
}

// Segmented block reduction kernel
template <unsigned int BLOCKSIZE, typename I, typename T>
static __device__ void segmented_blockreduce(const I* rows, T* vals)
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
    void coommn_general_block_reduce(I nblocks,
                                     const I* __restrict__ row_block_red,
                                     const T* __restrict__ val_block_red,
                                     T*              C,
                                     I               ldc,
                                     rocsparse_order order)
{
    int tid = hipThreadIdx_x;

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
        shared_row[tid] = row_block_red[i + nblocks * col];
        shared_val[tid] = val_block_red[i + nblocks * col];

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
                C[row + ldc * col] = C[row + ldc * col] + shared_val[tid];
            }
            else
            {
                C[col + ldc * row] = C[col + ldc * row] + shared_val[tid];
            }
        }

        __syncthreads();
    }
}

template <unsigned int BLOCKSIZE,
          unsigned int WF_SIZE,
          bool         TRANSB,
          typename I,
          typename T,
          typename U>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void coommnn_wf_segmented(I nnz,
                              I n,
                              I nloops,
                              U alpha_device_host,
                              const I* __restrict__ coo_row_ind,
                              const I* __restrict__ coo_col_ind,
                              const T* __restrict__ coo_val,
                              const T* __restrict__ B,
                              I ldb,
                              T* __restrict__ C,
                              I ldc,
                              I* __restrict__ row_block_red,
                              T* __restrict__ val_block_red,
                              rocsparse_order      order,
                              rocsparse_index_base idx_base)
{
    auto alpha = load_scalar_device_host(alpha_device_host);
    coommnn_general_wf_segmented<BLOCKSIZE, WF_SIZE, TRANSB>(nnz,
                                                             n,
                                                             nloops,
                                                             alpha,
                                                             coo_row_ind,
                                                             coo_col_ind,
                                                             coo_val,
                                                             B,
                                                             ldb,
                                                             C,
                                                             ldc,
                                                             row_block_red,
                                                             val_block_red,
                                                             order,
                                                             idx_base);
}

template <typename I, typename T, typename U>
rocsparse_status rocsparse_coomm_template_segmented(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_order           order,
                                                    I                         m,
                                                    I                         n,
                                                    I                         k,
                                                    I                         nnz,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  coo_val,
                                                    const I*                  coo_row_ind,
                                                    const I*                  coo_col_ind,
                                                    const T*                  B,
                                                    I                         ldb,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    I                         ldc)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Run different coomm kernels
    if(trans_A == rocsparse_operation_none)
    {
#define COOMMN_DIM 256
        I nloops  = 16;
        I nblocks = (nnz - 1) / (handle->wavefront_size * nloops) + 1;

        size_t required_size = 256 + ((sizeof(I) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM
                               + ((sizeof(T) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= required_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMalloc(&temp_storage_ptr, required_size));
            temp_alloc = true;
        }

        dim3 coommn_blocks(nblocks, (n - 1) / (COOMMN_DIM / handle->wavefront_size) + 1);
        dim3 coommn_threads(COOMMN_DIM);

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_storage_ptr);
        ptr += 256;

        // row block reduction buffer
        I* row_block_red = reinterpret_cast<I*>(reinterpret_cast<void*>(ptr));
        ptr += ((sizeof(I) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        // val block reduction buffer
        T* val_block_red = reinterpret_cast<T*>(reinterpret_cast<void*>(ptr));
        // ptr += ((sizeof(T) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        int len1 = ((sizeof(I) * nblocks * n - 1) / COOMMN_DIM + 1) * COOMMN_DIM;

        hipMemset(row_block_red, 0XFF, len1);

        if((order == rocsparse_order_column && trans_B == rocsparse_operation_none)
           || (order == rocsparse_order_row && trans_B == rocsparse_operation_transpose))
        {
            if(handle->wavefront_size == 32)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 32, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
            else if(handle->wavefront_size == 64)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 64, false>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
        }
        else if((order == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                || (order == rocsparse_order_row && trans_B == rocsparse_operation_none))
        {
            if(handle->wavefront_size == 32)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 32, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
            else if(handle->wavefront_size == 64)
            {
                hipLaunchKernelGGL((coommnn_wf_segmented<COOMMN_DIM, 64, true>),
                                   coommn_blocks,
                                   coommn_threads,
                                   0,
                                   stream,
                                   nnz,
                                   n,
                                   nloops,
                                   alpha_device_host,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   B,
                                   ldb,
                                   C,
                                   ldc,
                                   row_block_red,
                                   val_block_red,
                                   order,
                                   descr->base);
            }
        }
#undef COOMMN_DIM

        hipLaunchKernelGGL((coommn_general_block_reduce<1024>),
                           dim3(n),
                           1024,
                           0,
                           stream,
                           nblocks,
                           row_block_red,
                           val_block_red,
                           C,
                           ldc,
                           order);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(hipFree(temp_storage_ptr));
        }
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

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
        UTYPE                     alpha_device_host,                                   \
        const rocsparse_mat_descr descr,                                               \
        const TTYPE*              coo_val,                                             \
        const ITYPE*              coo_row_ind,                                         \
        const ITYPE*              coo_col_ind,                                         \
        const TTYPE*              B,                                                   \
        ITYPE                     ldb,                                                 \
        UTYPE                     beta_device_host,                                    \
        TTYPE*                    C,                                                   \
        ITYPE                     ldc);

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

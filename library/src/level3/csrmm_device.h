#pragma once
#ifndef CSRMM_DEVICE_H
#define CSRMM_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int SUBWAVE_SIZE>
static __device__ void csrmmnn_general_device(rocsparse_int M,
                                              rocsparse_int N,
                                              rocsparse_int K,
                                              rocsparse_int nnz,
                                              T alpha,
                                              const rocsparse_int* csr_row_ptr,
                                              const rocsparse_int* csr_col_ind,
                                              const T* csr_val,
                                              const T* B,
                                              rocsparse_int ldb,
                                              T beta,
                                              T* C,
                                              rocsparse_int ldc,
                                              rocsparse_index_base idx_base)
{
    rocsparse_int tid    = hipThreadIdx_x;
    rocsparse_int gid    = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int warpid = gid / SUBWAVE_SIZE;
    rocsparse_int laneid = gid & (SUBWAVE_SIZE - 1);
    rocsparse_int subid  = tid / SUBWAVE_SIZE;
    rocsparse_int nwarps = hipGridDim_x * hipBlockDim_x / SUBWAVE_SIZE;
    rocsparse_int col    = laneid + hipBlockIdx_y * SUBWAVE_SIZE;
    rocsparse_int colB   = col * ldb;
    rocsparse_int colC   = col * ldc;

    __shared__ rocsparse_int shared_col[BLOCKSIZE/SUBWAVE_SIZE][SUBWAVE_SIZE];
    __shared__ T shared_val[BLOCKSIZE/SUBWAVE_SIZE][SUBWAVE_SIZE];

    for(rocsparse_int row = warpid; row < M; row += nwarps)
    {
        rocsparse_int row_start = __ldg(csr_row_ptr + row) - idx_base;
        rocsparse_int row_end   = __ldg(csr_row_ptr + row + 1) - idx_base;

        T sum = static_cast<T>(0);

        for(rocsparse_int j = row_start; j < row_end; j += SUBWAVE_SIZE)
        {
            rocsparse_int k = j + laneid;

            __syncthreads();

            shared_col[subid][laneid] = (k < row_end) ? __ldg(csr_col_ind + k) - idx_base : 0;
            shared_val[subid][laneid] = (k < row_end) ? alpha * __ldg(csr_val + k) : static_cast<T>(0);

            __syncthreads();

            for(rocsparse_int i = 0; i < SUBWAVE_SIZE && col < N; ++i)
            {
                sum += shared_val[subid][i] * __ldg(&B[shared_col[subid][i] + colB]);
            }
        }

        if(col < N)
        {
            if(beta == 0.0)
            {
                C[row + colC] = sum;
            }
            else
            {
                C[row + colC] = __ldg(&C[row + colC]) * beta + sum;
            }
        }
    }
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int SUBWAVE_SIZE>
static __device__ void csrmmnt_general_device(rocsparse_int offset,
                                              rocsparse_int ncol,
                                              rocsparse_int M,
                                              rocsparse_int N,
                                              rocsparse_int K,
                                              rocsparse_int nnz,
                                              T alpha,
                                              const rocsparse_int* csr_row_ptr,
                                              const rocsparse_int* csr_col_ind,
                                              const T* csr_val,
                                              const T* B,
                                              rocsparse_int ldb,
                                              T beta,
                                              T* C,
                                              rocsparse_int ldc,
                                              rocsparse_index_base idx_base)
{
    rocsparse_int tid    = hipThreadIdx_x;
    rocsparse_int gid    = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int row    = gid / SUBWAVE_SIZE;
    rocsparse_int laneid = tid & (SUBWAVE_SIZE - 1);
    rocsparse_int subid  = hipThreadIdx_x / SUBWAVE_SIZE;

    if(row >= M)
    {
        return;
    }

    __shared__ rocsparse_int shared_col[BLOCKSIZE/SUBWAVE_SIZE][SUBWAVE_SIZE];
    __shared__ T shared_val[BLOCKSIZE/SUBWAVE_SIZE][SUBWAVE_SIZE];

    rocsparse_int row_start = __ldg(csr_row_ptr + row) - idx_base;
    rocsparse_int row_end   = __ldg(csr_row_ptr + row + 1) - idx_base;

    for(rocsparse_int l = offset; l < ncol; l += SUBWAVE_SIZE)
    {
        rocsparse_int col = l + laneid;
        T sum = static_cast<T>(0);

        for(rocsparse_int j = row_start; j < row_end; j += SUBWAVE_SIZE)
        {
            rocsparse_int k = j + laneid;

            __syncthreads();

            shared_col[subid][laneid] = (k < row_end) ? N * (__ldg(csr_col_ind + k) - idx_base) : 0;
            shared_val[subid][laneid] = (k < row_end) ? alpha * __ldg(csr_val + k) : static_cast<T>(0);

            __syncthreads();

            for(rocsparse_int i = 0; i < SUBWAVE_SIZE; ++i)
            {
                T val_B = (col < ncol) ? __ldg(B + col + shared_col[subid][i]) : static_cast<T>(0);
                sum += shared_val[subid][i] * val_B;
            }
        }

        if(col < ncol)
        {
            if(beta == static_cast<T>(0))
            {
                C[row + col * ldc] = sum;
            }
            else
            {
                C[row + col * ldc] = beta * __ldg(C + row + col * ldc) + sum;
            }
        }
    }
}

#endif // CSRMM_DEVICE_H

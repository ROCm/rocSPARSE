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

#pragma once
#ifndef GTSV_DEVICE_H
#define GTSV_DEVICE_H

#include "common.h"

// clang-format off
// Consider tridiagonal linear system A * x = rhs where A is m x m. Matrix A is represented by the three
// arrays (the diagonal, upper diagonal, and lower diagonal) each of length m. The first entry in the
// lower diagonal must be zero and the last entry in the upper diagonal must be zero. We solve this linear
// system using the "Spike-Diagonal Pivoting" algorithm as detailed in the thesis:
//
// "Scalable Parallel Tridiagonal Algorithms with diagonal pivoting and their optimizations for many-core
// architectures by L. Chang"
//
// See also:
//
// "L. Chang, J. A. Stratton, H. Kim and W. W. Hwu, "A scalable, numerically stable, high-performance tridiagonal
// solver using GPUs," SC '12: Proceedings of the International Conference on High Performance Computing, Networking,
// Storage and Analysis, Salt Lake City, UT, USA, 2012, pp. 1-11, doi: 10.1109/SC.2012.12."
//
// Here we give a rough outline:
//
// Given the tridiagonal linear system A * x = rhs. We first decompose this into A * x = D * S * x = rhs where
// D is a block diagonal matrix and S is the "spike" matrix. We then define y = S * x which then allows us to
// first solve D * y = rhs and then solve S * x = y. Because each block in the block diagonal matrix D is indendent,
// we can solve each block in parallel. Specifically we use one thread for each diagonal block in D. We could use
// the Thomas algorithm here, however because we want to incorporate some pivoting mechanism, we instead have each thread
// decompose its diagonal block in to L * B * M^T where both L and M are lower triangular matrices and B is a diagonal
// matrix. Specifically if the matrix D has the form:
//
// D = |D1  0  0  0  0  .  0 |
//     |0   D2 0  0  0  .  0 |
//     |0   0  D3 0  0  .  0 |
//     |0   0  0  D4 0  .  0 |
//     |.   .  .  .  .  .  . |
//     |.               .  . |
//     |0   .  .  .  .  .  Dm|
//
// Then D_i = L_i * B_i * M_i^T for i = 1..m. Note that each Di is itself tridiagonal. The matrices L, B, and M can
// be computed by noting that:
//
// D_i = |Pd C | = |Id      0   | |Pd  0 | |Id Pd^-1*C|
//       |A  Tr|   |A*Pd^-1 In-d| |0   Ts| |0  In-d   |
//
// where Pd is either 1x1 or 2x2 and Id is either 1x1 or 2x2 identity matrix. Ts is computed as Ts = Tr - A*Pd^-1*C.
// For example consider one of the block diagonal matrices:
//
// Di = |2 1 0 0 0 0|
//      |1 2 1 0 0 0|
//      |0 1 2 1 0 0|
//      |0 0 1 2 1 0|
//      |0 0 0 1 2 1|
//      |0 0 0 0 1 2|
//
// Then if using no pivoting (i.e. Pd is 1x1) we get:
//
// Pd = 2, Pd^-1 = 1/2, A = |1|, C = |1 0 0 0 0|, and Ts = |3/2 1  0  0  0|
//                          |0|                            |1   2  1  0  0|
//                          |0|                            |0   1  2  1  0|
//                          |0|                            |0   0  1  2  1|
//                          |0|                            |0   0  0  1  2|
//
// We can then recursively perform this on each subsequent Ts until we get:
//
// Di = |1   0   0   0   0   0| |2  0   0   0   0   0  | |1   1/2 0   0   0   0  |
//      |1/2 1   0   0   0   0| |0  3/2 0   0   0   0  | |0   1   2/3 0   0   0  |
//      |0   2/3 1   0   0   0| |0  0   4/3 0   0   0  | |0   0   1   3/4 0   0  |
//      |0   0   3/4 1   0   0| |0  0   0   5/4 0   0  | |0   0   0   1   4/5 0  |
//      |0   0   0   4/5 1   0| |0  0   0   0   6/5 0  | |0   0   0   0   1   5/6|
//      |0   0   0   0   5/6 1| |0  0   0   0   0   7/6| |0   0   0   0   0   1  |
//
// Solving each of these systems then is just a matter of solving L * B * yi = rhsi
// followed by M^T * xi = yi. The determination of whether we should use Pd as 1x1 or 2x2
// is based off the Bunch-Kaufmann pivoting criteria. See cited sources above.
//
// Let us now return to our factoization of the the original tridiagonal linear system,
// A * x = D * S * x = rhs. We broke up finding the solution into the two phases. First
// solving D * y = rhs and then secondly solving S * x = y. We now know how to solve
// the first phase which is also the phase that performs the pivoting. We therefore focus on
// solving the "spike" linear system. If the original matrix A is:
//
// A = |2 1 0 0 0 0 0 0| = |2 1 0 0 0 0 0 0| |1   0   v11   0   0   0   0   0|
//     |1 2 1 0 0 0 0 0|   |1 2 0 0 0 0 0 0| |0   1   v12   0   0   0   0   0|
//     |0 1 2 1 0 0 0 0|   |0 0 2 1 0 0 0 0| |0   w21 1     0   v21 0   0   0|
//     |0 0 1 2 1 0 0 0|   |0 0 1 2 0 0 0 0| |0   w22 0     1   v22 0   0   0|
//     |0 0 0 1 2 1 0 0|   |0 0 0 0 2 1 0 0| |0   0   0     w31 1   0   v31 0|
//     |0 0 0 0 1 2 0 0|   |0 0 0 0 1 2 0 0| |0   0   0     w32 0   1   v32 0|
//     |0 0 0 0 0 1 2 1|   |0 0 0 0 0 0 2 1| |0   0   0     0   0   w41 1   0|
//     |0 0 0 0 0 0 1 2|   |0 0 0 0 0 0 1 2| |0   0   0     0   0   w42 0   1|
//                         --------D-------- ----------------S----------------
//
// Here we use 2x2 blocks in D but in practice we use much larger blocks, for example 128x128.
// We can solve for all the v and w unknowns by solving the following:
//
// |2 1||v11| = |0| and |2 1||w21| = |1| etc.
// |1 2||v12|   |1|     |1 2||w22|   |0|
//
// Solving S * x = y then involves recursively decomposing the "spike" matrix like so:
//
// S = |1   0   v11   0   0   0   0   0| = |1  0   v11 0  0  0  0   0| |1 0 0 0 v11' 0 0 0|
//     |0   1   v12   0   0   0   0   0|   |0  1   v12 0  0  0  0   0| |0 1 0 0 v12' 0 0 0|
//     |0   w21 1     0   v21 0   0   0|   |0  w21 1   0  0  0  0   0| |0 0 1 0 v13' 0 0 0|
//     |0   w22 0     1   v22 0   0   0|   |0  w22 0   1  0  0  0   0| |0 0 0 1 v14' 0 0 0|
//     |0   0   0     w31 1   0   v31 0|   |0  0   0   0  1  0  v31 0| |0 0 0 w21' 1 0 0 0|
//     |0   0   0     w32 0   1   v32 0|   |0  0   0   0  0  1  v32 0| |0 0 0 w22' 0 1 0 0|
//     |0   0   0     0   0   w41 1   0|   |0  0   0   0  0  w41 1  0| |0 0 0 w23' 0 0 1 0|
//     |0   0   0     0   0   w42 0   1|   |0  0   0   0  0  w42 0  0| |0 0 0 w24' 0 0 0 1|
//
// In the above the non-prime w and v values (i.e. v11, v12, w21, w22 etc) have been previously
// computed. The primed w and v values (i.e. v11', v12', w21', w22' etc) can be found by solving:
//
// |1   0   v11 0||v11'| = |0| etc...
// |0   1   v12 0||v12'|   |0|
// |0   w21 1   0||v13'|   |0|
// |0   w22 0   1||v14'|   |1|
// clang-format on

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void gtsv_copy_result_array_kernel(rocsparse_int m,
                                       rocsparse_int m_pad,
                                       rocsparse_int ldb,
                                       rocsparse_int block_dim,
                                       const T* __restrict__ input,
                                       T* __restrict__ output)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int bidy = hipBlockIdx_y;

    rocsparse_int gid = tidx + BLOCKSIZE * bidx;

    if(gid < m)
    {
        output[gid + ldb * bidy] = input[gid + bidy * m_pad];
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void gtsv_fill_padded_array_kernel(rocsparse_int m,
                                       rocsparse_int m_pad,
                                       rocsparse_int ldb,
                                       const T* __restrict__ input,
                                       T* __restrict__ output,
                                       T pad_value)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int bidy = hipBlockIdx_y;

    rocsparse_int gid = tidx + BLOCKSIZE * bidx;

    if(gid < m)
    {
        output[gid + bidy * m_pad] = input[gid + bidy * ldb];
    }
    else if(gid < m_pad)
    {
        output[gid + bidy * m_pad] = pad_value;
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__ void gtsv_LBM_1x1_kernel(rocsparse_int m,
                                                                 rocsparse_int n,
                                                                 rocsparse_int ldb,
                                                                 rocsparse_int block_dim,
                                                                 const T* __restrict__ a,
                                                                 const T* __restrict__ b,
                                                                 const T* __restrict__ c,
                                                                 T* __restrict__ rhs,
                                                                 T* __restrict__ w,
                                                                 T* __restrict__ v,
                                                                 T* __restrict__ mt)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

    rocsparse_int stride = gid * block_dim;

    if(stride >= m)
    {
        return;
    }

    T bk                      = b[stride];
    w[stride]                 = a[stride];
    v[stride + block_dim - 1] = c[stride + block_dim - 1];

    // Forward solve
    rocsparse_int k = 0;
    while(k < block_dim)
    {
        T ck   = c[k + stride];
        T ak_1 = (k < block_dim - 1) ? a[k + 1 + stride] : static_cast<T>(0);
        T bk_1 = (k < block_dim - 1) ? b[k + 1 + stride] : static_cast<T>(0);

        T iBk = static_cast<T>(1) / bk;

        bk_1 = bk_1 - ak_1 * ck * iBk;
        ak_1 = ak_1 * iBk;
        ck   = ck * iBk;

        T rhsk = rhs[k + stride];
        T wk   = w[k + stride];
        T vk   = v[k + stride];

        rhs[k + stride] = rhsk * iBk;
        w[k + stride]   = wk * iBk;
        v[k + stride]   = vk * iBk;
        mt[k + stride]  = ck;

        if(k < block_dim - 1)
        {
            rhs[k + 1 + stride] = rhs[k + 1 + stride] - ak_1 * rhsk;
            w[k + 1 + stride]   = w[k + 1 + stride] - ak_1 * wk;
        }

        bk = bk_1;
        k  = k + 1;
    }

    // backward solve
    k = block_dim - 2;
    while(k >= 0)
    {
        rhs[k + stride] = rhs[k + stride] - mt[k + stride] * rhs[k + 1 + stride];
        v[k + stride]   = v[k + stride] - mt[k + stride] * v[k + 1 + stride];
        w[k + stride]   = w[k + stride] - mt[k + stride] * w[k + 1 + stride];

        k = k - 1;
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__ void gtsv_LBM_2x2_kernel(rocsparse_int m,
                                                                 rocsparse_int n,
                                                                 rocsparse_int ldb,
                                                                 rocsparse_int block_dim,
                                                                 const T* __restrict__ a,
                                                                 const T* __restrict__ b,
                                                                 const T* __restrict__ c,
                                                                 T* __restrict__ rhs,
                                                                 T* __restrict__ w,
                                                                 T* __restrict__ v,
                                                                 T* __restrict__ mt)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

    rocsparse_int stride = gid * block_dim;

    if(stride >= m)
    {
        return;
    }

    T bk                      = b[stride];
    w[stride]                 = a[stride];
    v[stride + block_dim - 1] = c[stride + block_dim - 1];

    rocsparse_int k = 0;
    while(k < block_dim)
    {
        T ck   = c[k + stride];
        T ak_1 = (k < block_dim - 1) ? a[k + 1 + stride] : static_cast<T>(0);
        T bk_1 = (k < block_dim - 1) ? b[k + 1 + stride] : static_cast<T>(0);
        T ck_1 = (k < block_dim - 1) ? c[k + 1 + stride] : static_cast<T>(0);
        T ak_2 = (k < block_dim - 2) ? a[k + 2 + stride] : static_cast<T>(0);

        T det = bk * bk_1 - ak_1 * ck;
        det   = static_cast<T>(1) / det;

        T rhsk   = rhs[k + stride];
        T rhsk_1 = rhs[k + 1 + stride];
        T wk     = w[k + stride];
        T wk_1   = w[k + 1 + stride];
        T vk     = v[k + stride];
        T vk_1   = v[k + 1 + stride];

        rhs[k + stride] = (bk_1 * rhsk - ck * rhsk_1) * det;
        w[k + stride]   = (bk_1 * wk - ck * wk_1) * det;
        v[k + stride]   = (bk_1 * vk - ck * vk_1) * det;
        mt[k + stride]  = -ck * ck_1 * det;

        if(k < block_dim - 1)
        {
            rhs[k + 1 + stride] = (-ak_1 * rhsk + bk * rhsk_1) * det;
            w[k + 1 + stride]   = (-ak_1 * wk + bk * wk_1) * det;
            v[k + 1 + stride]   = (-ak_1 * vk + bk * vk_1) * det;
            mt[k + 1 + stride]  = bk * ck_1 * det;
        }

        T bk_2 = static_cast<T>(0);

        if(k < block_dim - 2)
        {
            rhs[k + 2 + stride]
                = rhs[k + 2 + stride] - (-ak_1 * ak_2 * det) * rhsk - (bk * ak_2 * det) * rhsk_1;
            w[k + 2 + stride]
                = w[k + 2 + stride] - (-ak_1 * ak_2 * det) * wk - (bk * ak_2 * det) * wk_1;

            bk_2 = b[k + 2 + stride];
            bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
        }

        bk = bk_2;
        k  = k + 2;
    }

    // backward solve
    k = block_dim - 4;
    while(k >= 0)
    {
        rhs[k + stride]     = rhs[k + stride] - mt[k + stride] * rhs[k + 2 + stride];
        rhs[k + 1 + stride] = rhs[k + 1 + stride] - mt[k + 1 + stride] * rhs[k + 2 + stride];
        w[k + stride]       = w[k + stride] - mt[k + stride] * w[k + 2 + stride];
        w[k + 1 + stride]   = w[k + 1 + stride] - mt[k + 1 + stride] * w[k + 2 + stride];
        v[k + stride]       = v[k + stride] - mt[k + stride] * v[k + 2 + stride];
        v[k + 1 + stride]   = v[k + 1 + stride] - mt[k + 1 + stride] * v[k + 2 + stride];

        k = k - 2;
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__ void gtsv_LBM_wv_kernel(rocsparse_int m,
                                                                rocsparse_int n,
                                                                rocsparse_int ldb,
                                                                rocsparse_int block_dim,
                                                                const T* __restrict__ a,
                                                                const T* __restrict__ b,
                                                                const T* __restrict__ c,
                                                                T* __restrict__ w,
                                                                T* __restrict__ v,
                                                                T* __restrict__ mt,
                                                                rocsparse_int* __restrict__ pivot)
{
    // From Bunch-Kaufman pivoting criteria
    const double kappa = 0.5 * (sqrt(5.0) - 1.0);

    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

    rocsparse_int stride = gid * block_dim;

    if(stride >= m)
    {
        return;
    }

    T bk                      = b[stride];
    w[stride]                 = a[stride];
    v[stride + block_dim - 1] = c[stride + block_dim - 1];

    // forward solve (L* B * w = w)
    rocsparse_int k = 0;
    while(k < block_dim)
    {
        T ck   = c[k + stride];
        T ak_1 = (k < block_dim - 1) ? a[k + 1 + stride] : static_cast<T>(0);
        T bk_1 = (k < block_dim - 1) ? b[k + 1 + stride] : static_cast<T>(0);
        T ck_1 = (k < block_dim - 1) ? c[k + 1 + stride] : static_cast<T>(0);
        T ak_2 = (k < block_dim - 2) ? a[k + 2 + stride] : static_cast<T>(0);

        // decide whether we should use 1x1 or 2x2 pivoting using Bunch-Kaufman
        // pivoting criteria
        double sigma = 0;
        sigma        = max(rocsparse_abs(ak_1), rocsparse_abs(ak_2));
        sigma        = max(rocsparse_abs(bk_1), sigma);
        sigma        = max(rocsparse_abs(ck), sigma);
        sigma        = max(rocsparse_abs(ck_1), sigma);

        // 1x1 pivoting
        if(rocsparse_abs(bk) * sigma >= kappa * rocsparse_abs(ak_1 * ck) || k == block_dim - 1)
        {
            T iBk = static_cast<T>(1) / bk;

            bk_1 = bk_1 - ak_1 * ck * iBk;
            ak_1 = ak_1 * iBk;
            ck   = ck * iBk;

            T wk = w[k + stride];
            T vk = v[k + stride];

            w[k + stride]     = wk * iBk;
            v[k + stride]     = vk * iBk;
            mt[k + stride]    = ck;
            pivot[k + stride] = 1;

            if(k < block_dim - 1)
            {
                w[k + 1 + stride] = w[k + 1 + stride] - ak_1 * wk;
            }

            bk = bk_1;
            k  = k + 1;
        }
        // 2x2 pivoting
        else
        {
            T det = bk * bk_1 - ak_1 * ck;
            det   = static_cast<T>(1) / det;

            T wk   = w[k + stride];
            T wk_1 = w[k + 1 + stride];
            T vk   = v[k + stride];
            T vk_1 = v[k + 1 + stride];

            w[k + stride]     = (bk_1 * wk - ck * wk_1) * det;
            v[k + stride]     = (bk_1 * vk - ck * vk_1) * det;
            mt[k + stride]    = -ck * ck_1 * det;
            pivot[k + stride] = 2;

            if(k < block_dim - 1)
            {
                w[k + 1 + stride]     = (-ak_1 * wk + bk * wk_1) * det;
                v[k + 1 + stride]     = (-ak_1 * vk + bk * vk_1) * det;
                mt[k + 1 + stride]    = bk * ck_1 * det;
                pivot[k + 1 + stride] = 2;
            }

            T bk_2 = static_cast<T>(0);

            if(k < block_dim - 2)
            {
                w[k + 2 + stride]
                    = w[k + 2 + stride] - (-ak_1 * ak_2 * det) * wk - (bk * ak_2 * det) * wk_1;

                bk_2 = b[k + 2 + stride];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k  = k + 2;
        }
    }

    __threadfence();

    // at this point k = block_dim
    k -= 1;

    k -= pivot[k + stride];

    // backward solve (M^T * w = w)
    while(k >= 0)
    {
        if(pivot[k + stride] == 1)
        {
            w[k + stride] = w[k + stride] - mt[k + stride] * w[k + 1 + stride];
            v[k + stride] = v[k + stride] - mt[k + stride] * v[k + 1 + stride];

            k -= 1;
        }
        else
        {
            w[k + stride]     = w[k + stride] - mt[k + stride] * w[k + 1 + stride];
            w[k - 1 + stride] = w[k - 1 + stride] - mt[k - 1 + stride] * w[k + 1 + stride];
            v[k + stride]     = v[k + stride] - mt[k + stride] * v[k + 1 + stride];
            v[k - 1 + stride] = v[k - 1 + stride] - mt[k - 1 + stride] * v[k + 1 + stride];

            k -= 2;
        }
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void gtsv_LBM_rhs_kernel(rocsparse_int m,
                             rocsparse_int n,
                             rocsparse_int ldb,
                             rocsparse_int block_dim,
                             const T* __restrict__ a,
                             const T* __restrict__ b,
                             const T* __restrict__ c,
                             T* __restrict__ rhs,
                             const T* __restrict__ w,
                             const T* __restrict__ v,
                             const T* __restrict__ mt,
                             const rocsparse_int* __restrict__ pivot)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int bidy = hipBlockIdx_y;
    rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

    rocsparse_int stride = gid * block_dim;

    if(stride >= m)
    {
        return;
    }

    T bk = b[stride];

    // forward solve (L* B * rhs = rhs)
    rocsparse_int k = 0;
    while(k < block_dim)
    {
        T ck   = c[k + stride];
        T ak_1 = (k < block_dim - 1) ? a[k + 1 + stride] : static_cast<T>(0);
        T bk_1 = (k < block_dim - 1) ? b[k + 1 + stride] : static_cast<T>(0);
        T ck_1 = (k < block_dim - 1) ? c[k + 1 + stride] : static_cast<T>(0);
        T ak_2 = (k < block_dim - 2) ? a[k + 2 + stride] : static_cast<T>(0);

        // 1x1 pivoting
        if(pivot[k + stride] == 1 || k == block_dim - 1)
        {
            T iBk = static_cast<T>(1) / bk;

            bk_1 = bk_1 - ak_1 * ck * iBk;
            ak_1 = ak_1 * iBk;
            ck   = ck * iBk;

            T rhsk = rhs[k + stride + m * bidy];

            rhs[k + stride + m * bidy] = rhsk * iBk;

            if(k < block_dim - 1)
            {
                rhs[k + 1 + stride + m * bidy] = rhs[k + 1 + stride + m * bidy] - ak_1 * rhsk;
            }

            bk = bk_1;
            k  = k + 1;
        }
        // 2x2 pivoting
        else
        {
            T det = bk * bk_1 - ak_1 * ck;
            det   = static_cast<T>(1) / det;

            T rhsk   = rhs[k + stride + m * bidy];
            T rhsk_1 = rhs[k + 1 + stride + m * bidy];

            rhs[k + stride + m * bidy] = (bk_1 * rhsk - ck * rhsk_1) * det;

            if(k < block_dim - 1)
            {
                rhs[k + 1 + stride + m * bidy] = (-ak_1 * rhsk + bk * rhsk_1) * det;
            }

            T bk_2 = static_cast<T>(0);

            if(k < block_dim - 2)
            {
                rhs[k + 2 + stride + m * bidy] = rhs[k + 2 + stride + m * bidy]
                                                 - (-ak_1 * ak_2 * det) * rhsk
                                                 - (bk * ak_2 * det) * rhsk_1;

                bk_2 = b[k + 2 + stride];
                bk_2 = bk_2 - ak_2 * bk * ck_1 * det;
            }

            bk = bk_2;
            k  = k + 2;
        }
    }

    __threadfence();

    // at this point k = block_dim
    k -= 1;

    k -= pivot[k + stride];

    // backward solve (M^T * rhs = rhs)
    while(k >= 0)
    {
        if(pivot[k + stride] == 1)
        {
            rhs[k + stride + m * bidy]
                = rhs[k + stride + m * bidy] - mt[k + stride] * rhs[k + 1 + stride + m * bidy];

            k -= 1;
        }
        else
        {
            rhs[k + stride + m * bidy]
                = rhs[k + stride + m * bidy] - mt[k + stride] * rhs[k + 1 + stride + m * bidy];
            rhs[k - 1 + stride + m * bidy] = rhs[k - 1 + stride + m * bidy]
                                             - mt[k - 1 + stride] * rhs[k + 1 + stride + m * bidy];

            k -= 2;
        }
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void gtsv_spike_block_level_kernel(rocsparse_int m,
                                       rocsparse_int n,
                                       rocsparse_int ldb,
                                       rocsparse_int block_dim,
                                       T* __restrict__ rhs,
                                       T* __restrict__ w,
                                       T* __restrict__ v,
                                       T* __restrict__ w2,
                                       T* __restrict__ v2,
                                       T* __restrict__ rhs_scratch,
                                       T* __restrict__ w_scratch,
                                       T* __restrict__ v_scratch)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int bidy = hipBlockIdx_y;
    rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

    __shared__ T sw[2 * BLOCKSIZE];
    __shared__ T sv[2 * BLOCKSIZE];
    __shared__ T srhs[2 * BLOCKSIZE];

    sw[tidx] = (gid * block_dim < m) ? w[gid * block_dim] : static_cast<T>(0);
    sw[tidx + BLOCKSIZE]
        = (gid * block_dim < m) ? w[gid * block_dim + block_dim - 1] : static_cast<T>(0);

    sv[tidx] = (gid * block_dim < m) ? v[gid * block_dim] : static_cast<T>(0);
    sv[tidx + BLOCKSIZE]
        = (gid * block_dim < m) ? v[gid * block_dim + block_dim - 1] : static_cast<T>(0);

    srhs[tidx] = (gid * block_dim < m) ? rhs[gid * block_dim + m * bidy] : static_cast<T>(0);
    srhs[tidx + BLOCKSIZE] = (gid * block_dim < m) ? rhs[gid * block_dim + block_dim - 1 + m * bidy]
                                                   : static_cast<T>(0);

    __syncthreads();

    rocsparse_int stride = 2;

    while(stride <= BLOCKSIZE)
    {
        if(tidx < BLOCKSIZE / stride)
        {
            rocsparse_int index = stride * tidx + stride / 2 - 1;
            rocsparse_int minus = index - stride / 2;
            rocsparse_int plus  = index + stride / 2;

            T det = static_cast<T>(1) - sw[index + 1] * sv[index + BLOCKSIZE];
            det   = static_cast<T>(1) / det;

            T tmp1 = srhs[index + BLOCKSIZE];
            T tmp2 = srhs[index + 1];

            srhs[index + BLOCKSIZE] = (tmp1 - sv[index + BLOCKSIZE] * tmp2) * det;
            srhs[index + 1]         = (tmp2 - tmp1 * sw[index + 1]) * det;
            srhs[minus + 1]         = srhs[minus + 1] - sv[minus + 1] * srhs[index + 1];
            srhs[plus + BLOCKSIZE]
                = srhs[plus + BLOCKSIZE] - sw[plus + BLOCKSIZE] * srhs[index + BLOCKSIZE];

            sv[index + BLOCKSIZE] = -det * (sv[index + BLOCKSIZE] * sv[index + 1]);
            sv[index + 1]         = det * sv[index + 1];
            sw[index + 1]         = -det * (sw[index + BLOCKSIZE] * sw[index + 1]);
            sw[index + BLOCKSIZE] = det * sw[index + BLOCKSIZE];

            sw[minus + 1] = sw[minus + 1] - sv[minus + 1] * sw[index + 1];
            sv[minus + 1] = -sv[minus + 1] * sv[index + 1];
            sv[plus + BLOCKSIZE]
                = sv[plus + BLOCKSIZE] - sv[index + BLOCKSIZE] * sw[plus + BLOCKSIZE];
            sw[plus + BLOCKSIZE] = -sw[plus + BLOCKSIZE] * sw[index + BLOCKSIZE];
        }

        stride *= 2;

        __syncthreads();
    }

    if(gid * block_dim < m)
    {
        if(bidy == 0)
        {
            w2[gid * block_dim]                 = sw[tidx];
            w2[gid * block_dim + block_dim - 1] = sw[tidx + BLOCKSIZE];
            v2[gid * block_dim]                 = sv[tidx];
            v2[gid * block_dim + block_dim - 1] = sv[tidx + BLOCKSIZE];
        }

        rhs[gid * block_dim + m * bidy]                 = srhs[tidx];
        rhs[gid * block_dim + block_dim - 1 + m * bidy] = srhs[tidx + BLOCKSIZE];
    }

    if(tidx == 0)
    {
        if(bidy == 0)
        {
            w_scratch[bidx]                = sw[0];
            w_scratch[hipGridDim_x + bidx] = sw[2 * BLOCKSIZE - 1];

            v_scratch[bidx]                = sv[0];
            v_scratch[hipGridDim_x + bidx] = sv[2 * BLOCKSIZE - 1];
        }

        rhs_scratch[bidx + 2 * hipGridDim_x * bidy]                = srhs[0];
        rhs_scratch[hipGridDim_x + bidx + 2 * hipGridDim_x * bidy] = srhs[2 * BLOCKSIZE - 1];
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void gtsv_solve_spike_grid_level_kernel(rocsparse_int m,
                                            rocsparse_int n,
                                            rocsparse_int ldb,
                                            rocsparse_int block_dim,
                                            T* __restrict__ rhs_scratch,
                                            const T* __restrict__ w_scratch,
                                            const T* __restrict__ v_scratch)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidy = hipBlockIdx_y;

    __shared__ T sw[2 * BLOCKSIZE];
    __shared__ T sv[2 * BLOCKSIZE];
    __shared__ T srhs[2 * BLOCKSIZE];

    sw[tidx]               = w_scratch[tidx];
    sw[tidx + BLOCKSIZE]   = w_scratch[tidx + BLOCKSIZE];
    sv[tidx]               = v_scratch[tidx];
    sv[tidx + BLOCKSIZE]   = v_scratch[tidx + BLOCKSIZE];
    srhs[tidx]             = rhs_scratch[tidx + 2 * BLOCKSIZE * bidy];
    srhs[tidx + BLOCKSIZE] = rhs_scratch[tidx + BLOCKSIZE + 2 * BLOCKSIZE * bidy];

    __syncthreads();

    rocsparse_int stride = 2;

    while(stride <= BLOCKSIZE)
    {
        rocsparse_int i = tidx;
        if(i < BLOCKSIZE / stride)
        {
            rocsparse_int index = stride * i + stride / 2 - 1;
            rocsparse_int minus = index - stride / 2;
            rocsparse_int plus  = index + stride / 2;

            T det = static_cast<T>(1) - sw[index + 1] * sv[index + BLOCKSIZE];
            det   = static_cast<T>(1) / det;

            T tmp1 = srhs[index + BLOCKSIZE];
            T tmp2 = srhs[index + 1];

            srhs[index + BLOCKSIZE] = (tmp1 - sv[index + BLOCKSIZE] * tmp2) * det;
            srhs[index + 1]         = (tmp2 - tmp1 * sw[index + 1]) * det;
            srhs[minus + 1]         = srhs[minus + 1] - sv[minus + 1] * srhs[index + 1];
            srhs[plus + BLOCKSIZE]
                = srhs[plus + BLOCKSIZE] - sw[plus + BLOCKSIZE] * srhs[index + BLOCKSIZE];

            sv[index + BLOCKSIZE] = -det * (sv[index + BLOCKSIZE] * sv[index + 1]);
            sv[index + 1]         = det * sv[index + 1];
            sw[index + 1]         = -det * (sw[index + BLOCKSIZE] * sw[index + 1]);
            sw[index + BLOCKSIZE] = det * sw[index + BLOCKSIZE];

            sw[minus + 1] = sw[minus + 1] - sv[minus + 1] * sw[index + 1];
            sv[minus + 1] = -sv[minus + 1] * sv[index + 1];
            sv[plus + BLOCKSIZE]
                = sv[plus + BLOCKSIZE] - sv[index + BLOCKSIZE] * sw[plus + BLOCKSIZE];
            sw[plus + BLOCKSIZE] = -sw[plus + BLOCKSIZE] * sw[index + BLOCKSIZE];
        }

        stride *= 2;

        __syncthreads();
    }

    stride = BLOCKSIZE / 2;
    while(stride >= 2)
    {
        rocsparse_int i = tidx;
        if(i < BLOCKSIZE / stride)
        {
            rocsparse_int index = stride * i + stride / 2 - 1;
            rocsparse_int minus = index - stride / 2;
            rocsparse_int plus  = index + stride / 2 + 1;

            minus = (minus < 0) ? 0 : minus;
            plus  = plus < BLOCKSIZE ? plus : BLOCKSIZE - 1;

            srhs[index + BLOCKSIZE]
                = srhs[index + BLOCKSIZE] - sw[index + BLOCKSIZE] * srhs[minus + BLOCKSIZE];
            srhs[index + BLOCKSIZE] = srhs[index + BLOCKSIZE] - sv[index + BLOCKSIZE] * srhs[plus];
            srhs[index + 1]         = srhs[index + 1] - sw[index + 1] * srhs[minus + BLOCKSIZE];
            srhs[index + 1]         = srhs[index + 1] - sv[index + 1] * srhs[plus];
        }

        stride /= 2;

        __syncthreads();
    }

    rhs_scratch[tidx + 2 * BLOCKSIZE * bidy]             = srhs[tidx];
    rhs_scratch[tidx + BLOCKSIZE + 2 * BLOCKSIZE * bidy] = srhs[tidx + BLOCKSIZE];
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void gtsv_solve_spike_propagate_kernel(rocsparse_int m,
                                           rocsparse_int n,
                                           rocsparse_int ldb,
                                           rocsparse_int block_dim,
                                           T* __restrict__ rhs,
                                           T* __restrict__ w,
                                           T* __restrict__ v,
                                           T* __restrict__ rhs_scratch)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int bidy = hipBlockIdx_y;
    rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

    __shared__ T sw[2 * BLOCKSIZE];
    __shared__ T sv[2 * BLOCKSIZE];
    __shared__ T srhs[2 * BLOCKSIZE + 2];

    sw[tidx] = (gid * block_dim < m) ? w[gid * block_dim] : static_cast<T>(0);
    sw[tidx + BLOCKSIZE]
        = (gid * block_dim < m) ? w[gid * block_dim + block_dim - 1] : static_cast<T>(0);

    sv[tidx] = (gid * block_dim < m) ? v[gid * block_dim] : static_cast<T>(0);
    sv[tidx + BLOCKSIZE]
        = (gid * block_dim < m) ? v[gid * block_dim + block_dim - 1] : static_cast<T>(0);

    srhs[tidx + 1] = (gid * block_dim < m) ? rhs[gid * block_dim + block_dim - 1 + m * bidy]
                                           : static_cast<T>(0);
    srhs[tidx + 1 + BLOCKSIZE]
        = (gid * block_dim < m) ? rhs[gid * block_dim + m * bidy] : static_cast<T>(0);

    __syncthreads();

    // load in boundary values from scratch pad
    if(tidx == 0)
    {
        srhs[0] = (bidx > 0) ? rhs_scratch[bidx + hipGridDim_x - 1 + 2 * hipGridDim_x * bidy]
                             : static_cast<T>(0);
        srhs[2 * BLOCKSIZE + 1] = (bidx < hipGridDim_x - 1)
                                      ? rhs_scratch[bidx + 1 + 2 * hipGridDim_x * bidy]
                                      : static_cast<T>(0);

        srhs[BLOCKSIZE + 1] = rhs_scratch[bidx + 2 * hipGridDim_x * bidy];
        srhs[BLOCKSIZE]     = rhs_scratch[bidx + hipGridDim_x + 2 * hipGridDim_x * bidy];
    }

    __syncthreads();

    rocsparse_int stride = BLOCKSIZE;

    while(stride >= 2)
    {
        if(tidx < BLOCKSIZE / stride)
        {
            rocsparse_int index = stride * tidx + stride / 2 - 1;
            rocsparse_int minus = index - stride / 2;
            rocsparse_int plus  = index + stride / 2;

            srhs[index + 1] = srhs[index + 1] - sv[index + BLOCKSIZE] * srhs[plus + 2 + BLOCKSIZE];
            srhs[index + 1] = srhs[index + 1] - sw[index + BLOCKSIZE] * srhs[minus + 1];
            srhs[index + BLOCKSIZE + 2]
                = srhs[index + BLOCKSIZE + 2] - sv[index + 1] * srhs[plus + 2 + BLOCKSIZE];
            srhs[index + BLOCKSIZE + 2]
                = srhs[index + BLOCKSIZE + 2] - sw[index + 1] * srhs[minus + 1];
        }

        stride /= 2;

        __syncthreads();
    }

    if(gid * block_dim < m)
    {
        rhs[gid * block_dim + m * bidy]                 = srhs[tidx + 1 + BLOCKSIZE];
        rhs[gid * block_dim + block_dim - 1 + m * bidy] = srhs[tidx + 1];
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) __global__
    void gtsv_spike_backward_substitution_kernel(rocsparse_int m,
                                                 rocsparse_int n,
                                                 rocsparse_int ldb,
                                                 rocsparse_int block_dim,
                                                 T* __restrict__ rhs,
                                                 const T* __restrict__ w,
                                                 const T* __restrict__ v)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int bidx = hipBlockIdx_x;
    rocsparse_int bidy = hipBlockIdx_y;
    rocsparse_int gid  = tidx + BLOCKSIZE * bidx;

    if(gid * block_dim >= m)
    {
        return;
    }

    T tmp1 = (gid > 0) ? rhs[gid * block_dim - 1 + m * bidy] : static_cast<T>(0);
    T tmp2 = (gid * block_dim + block_dim < m) ? rhs[gid * block_dim + block_dim + m * bidy]
                                               : static_cast<T>(0);

    for(rocsparse_int i = 1; i < block_dim - 1; i++)
    {
        rhs[gid * block_dim + i + m * bidy]
            = rhs[gid * block_dim + i + m * bidy] - w[gid * block_dim + i] * tmp1;
        rhs[gid * block_dim + i + m * bidy]
            = rhs[gid * block_dim + i + m * bidy] - v[gid * block_dim + i] * tmp2;
    }
}

#endif // GTSV_DEVICE_H

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
#ifndef GPSV_INTERLEAVED_BATCH_DEVICE_H
#define GPSV_INTERLEAVED_BATCH_DEVICE_H

#include "common.h"

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void gpsv_strided_gather(rocsparse_int m,
                                                                       rocsparse_int batch_count,
                                                                       rocsparse_int batch_stride,
                                                                       const T* __restrict__ in,
                                                                       T* __restrict__ out)
{
    // Current batch this thread works on
    rocsparse_int b = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not run out of bounds
    if(b >= batch_count)
    {
        return;
    }

    // Process all entries of the batch
    for(rocsparse_int i = 0; i < m; ++i)
    {
        out[batch_count * i + b] = in[batch_stride * i + b];
    }
}

template <unsigned int BLOCKSIZE, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void gpsv_interleaved_batch_kernel(rocsparse_int m,
                                       rocsparse_int batch_count,
                                       rocsparse_int batch_stride,
                                       T* __restrict__ ds,
                                       T* __restrict__ dl,
                                       T* __restrict__ d,
                                       T* __restrict__ du,
                                       T* __restrict__ dw,
                                       T* __restrict__ X,
                                       T* __restrict__ t1,
                                       T* __restrict__ t2,
                                       T* __restrict__ B)
{
    // Current batch this thread works on
    rocsparse_int b = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not run out of bounds
    if(b >= batch_count)
    {
        return;
    }

    // Process all rows but the last four
    for(rocsparse_int row = 0; row < m - 1; ++row)
    {
        bool second_last_row = (row == m - 2);

        // Some indices
        rocsparse_int idp0  = batch_stride * (row + 0) + b;
        rocsparse_int idp1  = batch_stride * (row + 1) + b;
        rocsparse_int idp2  = batch_stride * (row + 2) + b;
        rocsparse_int idp0c = batch_count * (row + 0) + b;
        rocsparse_int idp1c = batch_count * (row + 1) + b;
        rocsparse_int idp2c = batch_count * (row + 2) + b;

        // Some values
        T dlp1 = dl[idp1];
        T dp1  = d[idp1];
        T dup1 = du[idp1];
        T dwp1 = dw[idp1];
        T Bp1  = B[idp1c];

        // Prefetch
        T dsp2 = static_cast<T>(0);
        T dlp2 = static_cast<T>(0);
        T dp2  = static_cast<T>(0);
        T dup2 = static_cast<T>(0);
        T dwp2 = static_cast<T>(0);
        T Bp2  = static_cast<T>(0);

        if(!second_last_row)
        {
            dsp2 = ds[idp2];
            dlp2 = dl[idp2];
            dp2  = d[idp2];
            dup2 = du[idp2];
            dwp2 = dw[idp2];
            Bp2  = B[idp2c];
        }

        T v1 = dlp1;
        T v2 = dsp2;

        T v1v2sq = rocsparse_fma(v1, v1, v2 * v2);

        if(v1v2sq != static_cast<T>(0))
        {
            T diag = d[idp0];
            T val  = rocsparse_sqrt(rocsparse_fma(diag, diag, v1v2sq));
            T a_ii = (rocsparse_gt(diag, static_cast<T>(0))) ? diag + val : diag - val;

            v1 = v1 / a_ii;
            v2 = v2 / a_ii;

            T sq   = a_ii * a_ii;
            T beta = static_cast<T>(2) * sq / (v1v2sq + sq);
            T tau  = static_cast<T>(2)
                    / rocsparse_fma(v2, v2, (rocsparse_fma(v1, v1, static_cast<T>(1))));

            // Process the five diagonals
            T d1 = rocsparse_fma(v2, dsp2, rocsparse_fma(v1, dlp1, diag)) * beta;
            T d2 = rocsparse_fma(v2, dlp2, rocsparse_fma(v1, dp1, du[idp0])) * beta;
            T d3 = rocsparse_fma(v2, dp2, rocsparse_fma(v1, dup1, dw[idp0])) * beta;
            T d4 = rocsparse_fma(v2, dup2, rocsparse_fma(v1, dwp1, t1[idp0c])) * beta;
            T d5 = rocsparse_fma(v2, dwp2, rocsparse_fma(v1, t1[idp1c], t2[idp0c])) * beta;
            T fs = rocsparse_fma(v2, Bp2, rocsparse_fma(v1, Bp1, B[idp0c])) * tau;

            // Update
            d[idp0] -= d1;
            du[idp0] -= d2;
            dw[idp0] -= d3;
            t1[idp0c] -= d4;
            t2[idp0c] -= d5;
            B[idp0c] -= fs;

            dl[idp1]  = v1;
            d[idp1]   = rocsparse_fma(-d2, v1, dp1);
            du[idp1]  = rocsparse_fma(-d3, v1, dup1);
            dw[idp1]  = rocsparse_fma(-d4, v1, dwp1);
            t1[idp1c] = rocsparse_fma(-d5, v1, t1[idp1c]);
            B[idp1c]  = rocsparse_fma(-v1, fs, Bp1);

            if(!second_last_row)
            {
                ds[idp2] = v2;
                dl[idp2] = rocsparse_fma(-d2, v2, dlp2);
                d[idp2]  = rocsparse_fma(-d3, v2, dp2);
                du[idp2] = rocsparse_fma(-d4, v2, dup2);
                dw[idp2] = rocsparse_fma(-d5, v2, dwp2);
                B[idp2c] = rocsparse_fma(-v2, fs, Bp2);
            }
        }
    }

    // Backsolve
    for(rocsparse_int row = m - 1; row >= 0; --row)
    {
        rocsparse_int idp0  = batch_stride * row + b;
        rocsparse_int idp0c = batch_count * row + b;

        T sum = static_cast<T>(0);

        if(row + 1 < m)
            sum += du[idp0] * X[batch_stride * (row + 1) + b];
        if(row + 2 < m)
            sum += dw[idp0] * X[batch_stride * (row + 2) + b];
        if(row + 3 < m)
            sum += t1[idp0c] * X[batch_stride * (row + 3) + b];
        if(row + 4 < m)
            sum += t2[idp0c] * X[batch_stride * (row + 4) + b];

        X[idp0] = (B[idp0c] - sum) / d[idp0];
    }
}

#endif // GPSV_INTERLEAVED_BATCH_DEVICE_H

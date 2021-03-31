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
#ifndef ROCSPARSE_REDUCE_HPP
#define ROCSPARSE_REDUCE_HPP

inline constexpr int rocsparse_reduce_log2ui(int x)
{
    unsigned int ax = (unsigned int)x;
    int          v  = 0;
    while(ax >>= 1)
    {
        v++;
    }
    return v;
}

template <int N, typename T>
__inline__ __device__ T rocsparse_reduce_wavefront(T val)
{
    constexpr int WFBITS = rocsparse_reduce_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        val += __shfl_down(val, offset);
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocsparse_float_complex
    rocsparse_reduce_wavefront(rocsparse_float_complex val)
{
    constexpr int WFBITS = rocsparse_reduce_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        val += {__shfl_down(std::real(val), offset), __shfl_down(std::imag(val), offset)};
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocsparse_double_complex
    rocsparse_reduce_wavefront(rocsparse_double_complex val)
{
    constexpr int WFBITS = rocsparse_reduce_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
    for(int i = 0; i < WFBITS; i++)
    {
        val += {__shfl_down(std::real(val), offset), __shfl_down(std::imag(val), offset)};
        offset >>= 1;
    }
    return val;
}

template <rocsparse_int NB, typename T>
__inline__ __device__ T rocsparse_reduce_block(T val)
{
    __shared__ T psums[warpSize];

    rocsparse_int wavefront = hipThreadIdx_x / warpSize;
    rocsparse_int wavelet   = hipThreadIdx_x % warpSize;

    if(wavefront == 0)
        psums[wavelet] = 0;
    __syncthreads();

    val = rocsparse_reduce_wavefront<warpSize>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocsparse_int num_wavefronts = NB / warpSize;
    val = (hipThreadIdx_x < num_wavefronts) ? psums[wavelet] : 0;
    if(wavefront == 0)
        val = rocsparse_reduce_wavefront<num_wavefronts>(val); // sum wavefront sums

    return val;
}

template <typename T>
inline constexpr int rocsparse_reduce_WIN()
{
    size_t nb = sizeof(T);

    int n = 8;
    if(nb >= 8)
        n = 2;
    else if(nb >= 4)
        n = 4;

    return n;
}

inline constexpr int rocsparse_reduce_WIN(size_t nb)
{
    int n = 8;
    if(nb >= 8)
        n = 2;
    else if(nb >= 4)
        n = 4;

    return n;
}

inline size_t rocsparse_reduce_block_count(rocsparse_int n, rocsparse_int NB)
{
    if(n <= 0)
        n = 1; // avoid sign loss issues
    return size_t(n - 1) / NB + 1;
}

#endif

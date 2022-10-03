/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse.h"
#ifdef WIN32
#include <intrin.h>
#endif
#include <hip/hip_runtime.h>

// clang-format off

#ifndef ROCSPARSE_USE_MOVE_DPP
#if defined(__gfx803__) || \
    defined(__gfx900__) || \
    defined(__gfx906__) || \
    defined(__gfx908__) || \
    defined(__gfx90a__)
#define ROCSPARSE_USE_MOVE_DPP 1
#else
#define ROCSPARSE_USE_MOVE_DPP 0
#endif
#endif

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) ((dir == rocsparse_direction_row) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (block_dim * block_dim * (j) + (bi) * block_dim + (bj))
#define BSR_IND_C(j, bi, bj) (block_dim * block_dim * (j) + (bi) + (bj) * block_dim)

#define GEBSR_IND(j, bi, bj, dir) ((dir == rocsparse_direction_row) ? GEBSR_IND_R(j, bi, bj) : GEBSR_IND_C(j, bi, bj))
#define GEBSR_IND_R(j, bi, bj) (row_block_dim * col_block_dim * (j) + (bi) * col_block_dim + (bj))
#define GEBSR_IND_C(j, bi, bj) (row_block_dim * col_block_dim * (j) + (bi) + (bj) * row_block_dim)

// find next power of 2
__device__ __host__ __forceinline__ unsigned int fnp2(unsigned int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

__device__ __forceinline__ float rocsparse_ldg(const float* ptr) { return __ldg(ptr); }
__device__ __forceinline__ double rocsparse_ldg(const double* ptr) { return __ldg(ptr); }
__device__ __forceinline__ rocsparse_float_complex rocsparse_ldg(const rocsparse_float_complex* ptr) { return rocsparse_float_complex(__ldg((const float*)ptr), __ldg((const float*)ptr + 1)); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_ldg(const rocsparse_double_complex* ptr) { return rocsparse_double_complex(__ldg((const double*)ptr), __ldg((const double*)ptr + 1)); }
__device__ __forceinline__ int32_t rocsparse_ldg(const int32_t* ptr) { return __ldg(ptr); }
__device__ __forceinline__ int64_t rocsparse_ldg(const int64_t* ptr) { return __ldg(ptr); }

__device__ __forceinline__ float rocsparse_fma(float p, float q, float r) { return fma(p, q, r); }
__device__ __forceinline__ double rocsparse_fma(double p, double q, double r) { return fma(p, q, r); }
__device__ __forceinline__ rocsparse_float_complex rocsparse_fma(rocsparse_float_complex p, rocsparse_float_complex q, rocsparse_float_complex r) { return std::fma(p, q, r); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_fma(rocsparse_double_complex p, rocsparse_double_complex q, rocsparse_double_complex r) { return std::fma(p, q, r); }

__device__ __forceinline__ float rocsparse_abs(float x) { return x < 0.0f ? -x : x; }
__device__ __forceinline__ double rocsparse_abs(double x) { return x < 0.0 ? -x : x; }
__device__ __forceinline__ float rocsparse_abs(rocsparse_float_complex x) { return std::abs(x); }
__device__ __forceinline__ double rocsparse_abs(rocsparse_double_complex x) { return std::abs(x); }

__device__ __forceinline__ float rocsparse_sqrt(float val) { return sqrt(val); }
__device__ __forceinline__ double rocsparse_sqrt(double val) { return sqrt(val); }
__device__ __forceinline__ rocsparse_float_complex rocsparse_sqrt(rocsparse_float_complex val)
{
    float x = std::real(val);
    float y = std::imag(val);

    float sgnp = (y < 0.0f) ? -1.0f : 1.0f;
    float absz = rocsparse_abs(val);

    return rocsparse_float_complex(sqrt((absz + x) * 0.5f), sgnp * sqrt((absz - x) * 0.5f));
}
__device__ __forceinline__ rocsparse_double_complex rocsparse_sqrt(rocsparse_double_complex val)
{
    double x = std::real(val);
    double y = std::imag(val);

    double sgnp = (y < 0.0) ? -1.0 : 1.0;
    double absz = rocsparse_abs(val);

    return rocsparse_double_complex(sqrt((absz + x) * 0.5), sgnp * sqrt((absz - x) * 0.5));
}

__device__ __forceinline__ float rocsparse_conj(const float& x) { return x; }
__device__ __forceinline__ double rocsparse_conj(const double& x) { return x; }
__device__ __forceinline__ rocsparse_float_complex rocsparse_conj(const rocsparse_float_complex& x) { return std::conj(x); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_conj(const rocsparse_double_complex& x) { return std::conj(x); }

__device__ __forceinline__ float rocsparse_real(const float& x) { return x; }
__device__ __forceinline__ double rocsparse_real(const double& x) { return x; }
__device__ __forceinline__ float rocsparse_real(const rocsparse_float_complex& x) { return std::real(x); }
__device__ __forceinline__ double rocsparse_real(const rocsparse_double_complex& x) { return std::real(x); }

__device__ __forceinline__ bool rocsparse_gt(const float& x, const float& y) { return x > y; }
__device__ __forceinline__ bool rocsparse_gt(const double& x, const double& y) { return x > y; }
__device__ __forceinline__ bool rocsparse_gt(const rocsparse_float_complex& x, const rocsparse_float_complex& y)
{
    if(&x == &y)
    {
        return false;
    }

    assert(std::imag(x) == std::imag(y) && std::imag(x) == 0.0f);

    return std::real(x) > std::real(y);
}
__device__ __forceinline__ bool rocsparse_gt(const rocsparse_double_complex& x, const rocsparse_double_complex& y)
{
    if(&x == &y)
    {
        return false;
    }

    assert(std::imag(x) == std::imag(y) && std::imag(x) == 0.0);

    return std::real(x) > std::real(y);
}

__device__ __forceinline__ float rocsparse_nontemporal_load(const float* ptr) { return __builtin_nontemporal_load(ptr); }
__device__ __forceinline__ double rocsparse_nontemporal_load(const double* ptr) { return __builtin_nontemporal_load(ptr); }
__device__ __forceinline__ rocsparse_float_complex rocsparse_nontemporal_load(const rocsparse_float_complex* ptr) { return rocsparse_float_complex(__builtin_nontemporal_load((const float*)ptr), __builtin_nontemporal_load((const float*)ptr + 1)); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_nontemporal_load(const rocsparse_double_complex* ptr) { return rocsparse_double_complex(__builtin_nontemporal_load((const double*)ptr), __builtin_nontemporal_load((const double*)ptr + 1)); }
__device__ __forceinline__ int32_t rocsparse_nontemporal_load(const int32_t* ptr) { return __builtin_nontemporal_load(ptr); }
__device__ __forceinline__ int64_t rocsparse_nontemporal_load(const int64_t* ptr) { return __builtin_nontemporal_load(ptr); }

__device__ __forceinline__ void rocsparse_nontemporal_store(float val, float* ptr) { __builtin_nontemporal_store(val, ptr); }
__device__ __forceinline__ void rocsparse_nontemporal_store(double val, double* ptr) { __builtin_nontemporal_store(val, ptr); }
__device__ __forceinline__ void rocsparse_nontemporal_store(rocsparse_float_complex val, rocsparse_float_complex* ptr) { __builtin_nontemporal_store(std::real(val), (float*)ptr); __builtin_nontemporal_store(std::imag(val), (float*)ptr + 1); }
__device__ __forceinline__ void rocsparse_nontemporal_store(rocsparse_double_complex val, rocsparse_double_complex* ptr) { __builtin_nontemporal_store(std::real(val), (double*)ptr); __builtin_nontemporal_store(std::imag(val), (double*)ptr + 1); }
__device__ __forceinline__ void rocsparse_nontemporal_store(int32_t val, int32_t* ptr) { __builtin_nontemporal_store(val, ptr); }
__device__ __forceinline__ void rocsparse_nontemporal_store(int64_t val, int64_t* ptr) { __builtin_nontemporal_store(val, ptr); }

__device__ __forceinline__ int32_t rocsparse_shfl(int32_t var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }
__device__ __forceinline__ int64_t rocsparse_shfl(int64_t var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }
__device__ __forceinline__ float rocsparse_shfl(float var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }
__device__ __forceinline__ double rocsparse_shfl(double var, int src_lane, int width = warpSize) { return __shfl(var, src_lane, width); }
__device__ __forceinline__ rocsparse_float_complex rocsparse_shfl(rocsparse_float_complex var, int src_lane, int width = warpSize) { return rocsparse_float_complex(__shfl(std::real(var), src_lane, width), __shfl(std::imag(var), src_lane, width)); }
__device__ __forceinline__ rocsparse_double_complex rocsparse_shfl(rocsparse_double_complex var, int src_lane, int width = warpSize) { return rocsparse_double_complex(__shfl(std::real(var), src_lane, width), __shfl(std::imag(var), src_lane, width)); }

__device__ __forceinline__ int64_t atomicMin(int64_t* ptr, int64_t val) { return atomicMin((unsigned long long*)ptr, val); }
__device__ __forceinline__ int64_t atomicMax(int64_t* ptr, int64_t val) { return atomicMax((unsigned long long*)ptr, val); }
__device__ __forceinline__ int64_t atomicAdd(int64_t* ptr, int64_t val) { return atomicAdd((unsigned long long*)ptr, val); }
__device__ __forceinline__ int64_t atomicCAS(int64_t* ptr, int64_t cmp, int64_t val) { return atomicCAS((unsigned long long*)ptr, cmp, val); }

__device__ __forceinline__ rocsparse_float_complex atomicAdd(rocsparse_float_complex* ptr, rocsparse_float_complex val)
{
    return rocsparse_float_complex(atomicAdd((float*)ptr, std::real(val)),
                                   atomicAdd((float*)ptr + 1, std::imag(val)));
}
__device__ __forceinline__ rocsparse_double_complex atomicAdd(rocsparse_double_complex* ptr, rocsparse_double_complex val)
{
    return rocsparse_double_complex(atomicAdd((double*)ptr, std::real(val)),
                                    atomicAdd((double*)ptr + 1, std::imag(val)));
}

__device__ __forceinline__ bool rocsparse_is_inf(float val){ return (val == std::numeric_limits<float>::infinity()); }
__device__ __forceinline__ bool rocsparse_is_inf(double val){ return (val == std::numeric_limits<double>::infinity()); }
__device__ __forceinline__ bool rocsparse_is_inf(rocsparse_float_complex val){ return (std::real(val) == std::numeric_limits<float>::infinity() || std::imag(val) == std::numeric_limits<float>::infinity()); }
__device__ __forceinline__ bool rocsparse_is_inf(rocsparse_double_complex val){ return (std::real(val) == std::numeric_limits<double>::infinity() || std::imag(val) == std::numeric_limits<double>::infinity()); }

__device__ __forceinline__ bool rocsparse_is_nan(float val){ return (val != val); }
__device__ __forceinline__ bool rocsparse_is_nan(double val){ return (val != val); }
__device__ __forceinline__ bool rocsparse_is_nan(rocsparse_float_complex val){ return (std::real(val) != std::real(val) || std::imag(val) != std::imag(val)); }
__device__ __forceinline__ bool rocsparse_is_nan(rocsparse_double_complex val){ return (std::real(val) != std::real(val) || std::imag(val) != std::imag(val)); }

template <typename T>
__device__ __forceinline__ T conj_val(T val, bool conj)
{
    return conj ? rocsparse_conj(val) : val;
}

// Block reduce kernel computing block sum
template <unsigned int BLOCKSIZE, typename T>
__device__ __forceinline__ void rocsparse_blockreduce_sum(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = data[i] + data[i + 512]; } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = data[i] + data[i + 256]; } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = data[i] + data[i + 128]; } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = data[i] + data[i +  64]; } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = data[i] + data[i +  32]; } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = data[i] + data[i +  16]; } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = data[i] + data[i +   8]; } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = data[i] + data[i +   4]; } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = data[i] + data[i +   2]; } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = data[i] + data[i +   1]; } __syncthreads(); }
}

// Block reduce kernel computing blockwide maximum entry
template <unsigned int BLOCKSIZE, typename T>
__device__ __forceinline__ void rocsparse_blockreduce_max(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = max(data[i], data[i + 512]); } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = max(data[i], data[i + 256]); } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = max(data[i], data[i + 128]); } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = max(data[i], data[i +  64]); } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = max(data[i], data[i +  32]); } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = max(data[i], data[i +  16]); } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = max(data[i], data[i +   8]); } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = max(data[i], data[i +   4]); } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = max(data[i], data[i +   2]); } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = max(data[i], data[i +   1]); } __syncthreads(); }
}

// Block reduce kernel computing blockwide minimum entry
template <unsigned int BLOCKSIZE, typename T>
__device__ __forceinline__ void rocsparse_blockreduce_min(int i, T* data)
{
    if(BLOCKSIZE > 512) { if(i < 512 && i + 512 < BLOCKSIZE) { data[i] = min(data[i], data[i + 512]); } __syncthreads(); }
    if(BLOCKSIZE > 256) { if(i < 256 && i + 256 < BLOCKSIZE) { data[i] = min(data[i], data[i + 256]); } __syncthreads(); }
    if(BLOCKSIZE > 128) { if(i < 128 && i + 128 < BLOCKSIZE) { data[i] = min(data[i], data[i + 128]); } __syncthreads(); }
    if(BLOCKSIZE >  64) { if(i <  64 && i +  64 < BLOCKSIZE) { data[i] = min(data[i], data[i +  64]); } __syncthreads(); }
    if(BLOCKSIZE >  32) { if(i <  32 && i +  32 < BLOCKSIZE) { data[i] = min(data[i], data[i +  32]); } __syncthreads(); }
    if(BLOCKSIZE >  16) { if(i <  16 && i +  16 < BLOCKSIZE) { data[i] = min(data[i], data[i +  16]); } __syncthreads(); }
    if(BLOCKSIZE >   8) { if(i <   8 && i +   8 < BLOCKSIZE) { data[i] = min(data[i], data[i +   8]); } __syncthreads(); }
    if(BLOCKSIZE >   4) { if(i <   4 && i +   4 < BLOCKSIZE) { data[i] = min(data[i], data[i +   4]); } __syncthreads(); }
    if(BLOCKSIZE >   2) { if(i <   2 && i +   2 < BLOCKSIZE) { data[i] = min(data[i], data[i +   2]); } __syncthreads(); }
    if(BLOCKSIZE >   1) { if(i <   1 && i +   1 < BLOCKSIZE) { data[i] = min(data[i], data[i +   1]); } __syncthreads(); }
}

#if ROCSPARSE_USE_MOVE_DPP
// DPP-based wavefront reduction maximum
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_max(int* maximum)
{
    if(WFSIZE >  1) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x111, 0xf, 0xf, 0));
    if(WFSIZE >  2) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x112, 0xf, 0xf, 0));
    if(WFSIZE >  4) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x114, 0xf, 0xe, 0));
    if(WFSIZE >  8) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x118, 0xf, 0xc, 0));
    if(WFSIZE > 16) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x142, 0xa, 0xf, 0));
    if(WFSIZE > 32) *maximum = max(*maximum, __hip_move_dpp(*maximum, 0x143, 0xc, 0xf, 0));
}

// DPP-based wavefront reduction minimum
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int* minimum)
{
    if(WFSIZE >  1) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x111, 0xf, 0xf, 0));
    if(WFSIZE >  2) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x112, 0xf, 0xf, 0));
    if(WFSIZE >  4) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x114, 0xf, 0xe, 0));
    if(WFSIZE >  8) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x118, 0xf, 0xc, 0));
    if(WFSIZE > 16) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x142, 0xa, 0xf, 0));
    if(WFSIZE > 32) *minimum = min(*minimum, __hip_move_dpp(*minimum, 0x143, 0xc, 0xf, 0));
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int64_t* minimum)
{
    typedef union i64_b32
    {
        int64_t i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_min;
    i64_b32_t temp_min;
    temp_min.i64 = *minimum;

    if(WFSIZE > 1)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x111, 0xf, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x111, 0xf, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 2)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x112, 0xf, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x112, 0xf, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 4)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x114, 0xf, 0xe, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x114, 0xf, 0xe, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 8)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x118, 0xf, 0xc, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x118, 0xf, 0xc, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 16)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x142, 0xa, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x142, 0xa, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    if(WFSIZE > 32)
    {
        upper_min.b32[0] = __hip_move_dpp(temp_min.b32[0], 0x143, 0xc, 0xf, false);
        upper_min.b32[1] = __hip_move_dpp(temp_min.b32[1], 0x143, 0xc, 0xf, false);
        temp_min.i64 = min(temp_min.i64, upper_min.i64);
    }

    *minimum = temp_min.i64;
}

// DPP-based wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int* sum)
{
    if(WFSIZE >  1) *sum += __hip_move_dpp(*sum, 0x111, 0xf, 0xf, 0);
    if(WFSIZE >  2) *sum += __hip_move_dpp(*sum, 0x112, 0xf, 0xf, 0);
    if(WFSIZE >  4) *sum += __hip_move_dpp(*sum, 0x114, 0xf, 0xe, 0);
    if(WFSIZE >  8) *sum += __hip_move_dpp(*sum, 0x118, 0xf, 0xc, 0);
    if(WFSIZE > 16) *sum += __hip_move_dpp(*sum, 0x142, 0xa, 0xf, 0);
    if(WFSIZE > 32) *sum += __hip_move_dpp(*sum, 0x143, 0xc, 0xf, 0);
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int64_t* sum)
{
    typedef union i64_b32
    {
        int64_t i64;
        uint32_t b32[2];
    } i64_b32_t;

    i64_b32_t upper_sum;
    i64_b32_t temp_sum;
    temp_sum.i64 = *sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
        temp_sum.i64 += upper_sum.i64;
    }

    *sum = temp_sum.i64;
}

// DPP-based float wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ float rocsparse_wfreduce_sum(float sum)
{
    typedef union flt_b32
    {
        float val;
        uint32_t b32;
    } flt_b32_t;

    flt_b32_t upper_sum;
    flt_b32_t temp_sum;
    temp_sum.val = sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32 = __hip_move_dpp(temp_sum.b32, 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}

// DPP-based double wavefront reduction
template <unsigned int WFSIZE>
__device__ __forceinline__ double rocsparse_wfreduce_sum(double sum)
{
    typedef union dbl_b32
    {
        double val;
        uint32_t b32[2];
    } dbl_b32_t;

    dbl_b32_t upper_sum;
    dbl_b32_t temp_sum;
    temp_sum.val = sum;

    if(WFSIZE > 1)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x111, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x111, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 2)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x112, 0xf, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x112, 0xf, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 4)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x114, 0xf, 0xe, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x114, 0xf, 0xe, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 8)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x118, 0xf, 0xc, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x118, 0xf, 0xc, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 16)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x142, 0xa, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x142, 0xa, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    if(WFSIZE > 32)
    {
        upper_sum.b32[0] = __hip_move_dpp(temp_sum.b32[0], 0x143, 0xc, 0xf, false);
        upper_sum.b32[1] = __hip_move_dpp(temp_sum.b32[1], 0x143, 0xc, 0xf, false);
        temp_sum.val += upper_sum.val;
    }

    sum = temp_sum.val;
    return sum;
}
#else /* ROCSPARSE_USE_MOVE_DPP */
template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_max(int* maximum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *maximum = max(*maximum, __shfl_xor(*maximum, i));
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int* minimum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *minimum = min(*minimum, __shfl_xor(*minimum, i));
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_min(int64_t* minimum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *minimum = min(*minimum, __shfl_xor(*minimum, i));
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int* sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *sum += __shfl_xor(*sum, i);
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void rocsparse_wfreduce_sum(int64_t* sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *sum += __shfl_xor(*sum, i);
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ float rocsparse_wfreduce_sum(float sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ double rocsparse_wfreduce_sum(double sum)
{
    for(int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}
#endif /* ROCSPARSE_USE_MOVE_DPP */

// DPP-based complex float wavefront reduction sum
template <unsigned int WFSIZE>
__device__ __forceinline__ rocsparse_float_complex rocsparse_wfreduce_sum(rocsparse_float_complex sum)
{
    return rocsparse_float_complex(rocsparse_wfreduce_sum<WFSIZE>(std::real(sum)),
                                   rocsparse_wfreduce_sum<WFSIZE>(std::imag(sum)));
}

// DPP-based complex double wavefront reduction
template <unsigned int WFSIZE>
__device__ __forceinline__ rocsparse_double_complex rocsparse_wfreduce_sum(rocsparse_double_complex sum)
{
    return rocsparse_double_complex(rocsparse_wfreduce_sum<WFSIZE>(std::real(sum)),
                                    rocsparse_wfreduce_sum<WFSIZE>(std::imag(sum)));
}
// clang-format on

// Perform dense matrix transposition
template <unsigned int DIMX, unsigned int DIMY, typename I, typename T>
__device__ void dense_transpose_device(
    I m, I n, T alpha, const T* __restrict__ A, I lda, T* __restrict__ B, I ldb)
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

        for(I k = 0; k < DIMX; k += DIMY)
        {
            if(row_A < m && col_A + k < n)
            {
                sdata[wid + k][lid] = A[row_A + lda * (col_A + k)];
            }
        }

        __syncthreads();

        I col_B = j + lid;

        for(I k = 0; k < DIMX; k += DIMY)
        {
            if(col_B < n && row_B + k < m)
            {
                B[col_B + ldb * (row_B + k)] = alpha * sdata[lid][wid + k];
            }
        }
    }
}

// Perform dense matrix back transposition
template <unsigned int DIMX, unsigned int DIMY, typename I, typename T>
__launch_bounds__(DIMX* DIMY) ROCSPARSE_KERNEL
    void dense_transpose_back(I m, I n, const T* __restrict__ A, I lda, T* __restrict__ B, I ldb)
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

        for(I k = 0; k < DIMX; k += DIMY)
        {
            if(col_A < n && row_A + k < m)
            {
                sdata[wid + k][lid] = A[col_A + lda * (row_A + k)];
            }
        }

        __syncthreads();

        I col_B = j + wid;

        for(I k = 0; k < DIMX; k += DIMY)
        {
            if(row_B < m && col_B + k < n)
            {
                B[row_B + ldb * (col_B + k)] = sdata[lid][wid + k];
            }
        }
    }
}

// BSR gather functionality to permute the BSR values array
template <unsigned int WFSIZE, unsigned int DIMY, unsigned int BSRDIM, typename I, typename T>
__launch_bounds__(WFSIZE* DIMY) ROCSPARSE_KERNEL void bsr_gather(rocsparse_direction dir,
                                                                 I                   nnzb,
                                                                 const I* __restrict__ perm,
                                                                 const T* __restrict__ bsr_val_A,
                                                                 T* __restrict__ bsr_val_T,
                                                                 I block_dim)
{
    int lid = threadIdx.x & (BSRDIM - 1);
    int wid = threadIdx.x / BSRDIM;

    // Non-permuted nnz index
    I j = blockIdx.x * DIMY + threadIdx.y;

    // Do not exceed the number of elements
    if(j >= nnzb)
    {
        return;
    }

    // Load the permuted nnz index
    I p = perm[j];

    // Gather values from A and store them to T with respect to the
    // given row / column permutation
    for(I bi = lid; bi < block_dim; bi += BSRDIM)
    {
        for(I bj = wid; bj < block_dim; bj += BSRDIM)
        {
            bsr_val_T[BSR_IND(j, bi, bj, dir)] = bsr_val_A[BSR_IND(p, bj, bi, dir)];
        }
    }
}

// Set array to be filled with value
template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void set_array_to_value(I m, T* __restrict__ array, T value)
{
    I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

    if(idx >= m)
    {
        return;
    }

    array[idx] = value;
}

// Scale array by value
template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void scale_array(I m, T* __restrict__ array, T value)
{
    I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

    if(idx >= m)
    {
        return;
    }

    array[idx] *= value;
}

template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void scale_array(I m, T* __restrict__ array, const T* value)
{
    I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

    if(idx >= m)
    {
        return;
    }

    if(*value != static_cast<T>(1))
    {
        array[idx] *= (*value);
    }
}

// conjugate values in array
template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void conjugate(I m, T* __restrict__ array)
{
    I idx = hipThreadIdx_x + BLOCKSIZE * hipBlockIdx_x;

    if(idx >= m)
    {
        return;
    }

    array[idx] = rocsparse_conj(array[idx]);
}

template <unsigned int BLOCKSIZE, typename I, typename J>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csr_max_nnz_per_row(J m, const I* __restrict__ csr_row_ptr, J* __restrict__ max_nnz)
{
    int tid = hipThreadIdx_x;
    J   gid = tid + BLOCKSIZE * hipBlockIdx_x;

    __shared__ I shared[BLOCKSIZE];

    if(gid < m)
    {
        shared[tid] = csr_row_ptr[gid + 1] - csr_row_ptr[gid];
    }
    else
    {
        shared[tid] = 0;
    }

    __syncthreads();

    rocsparse_blockreduce_max<BLOCKSIZE>(tid, shared);

    if(tid == 0)
    {
        atomicMax(max_nnz, shared[0]);
    }
}

/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CHECK_HPP
#define ROCSPARSE_CHECK_HPP

#include <cassert>
#include <rocsparse.h>
#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#include "rocsparse_math.hpp"

#ifndef GOOGLE_TEST
#define ASSERT_TRUE(cond) assert(cond);
#define UNIT_ASSERT_EQ(state1, state2) assert(state1 == state2);
#define ASSERT_NEAR(state1, state2, eps) assert(std::abs(state1 - state2) < eps);
#endif

#define UNIT_CHECK(M, N, lda, hCPU, hGPU, UNIT_ASSERT_EQ)                 \
    do                                                                    \
    {                                                                     \
        for(rocsparse_int j = 0; j < N; ++j)                              \
            for(rocsparse_int i = 0; i < M; ++i)                          \
                if(rocsparse_isnan(hCPU[i + j * lda]))                    \
                {                                                         \
                    ASSERT_TRUE(rocsparse_isnan(hGPU[i + j * lda]));      \
                }                                                         \
                else                                                      \
                {                                                         \
                    UNIT_ASSERT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]); \
                }                                                         \
    } while(0)

template <typename T>
void unit_check_general(rocsparse_int M, rocsparse_int N, rocsparse_int lda, T* hCPU, T* hGPU);

template <>
inline void unit_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, float* hCPU, float* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void unit_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, double* hCPU, double* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, rocsparse_int* hCPU, rocsparse_int* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
inline void unit_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, size_t* hCPU, size_t* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_EQ);
}

template <typename T>
void near_check_general(rocsparse_int M, rocsparse_int N, rocsparse_int lda, T* hCPU, T* hGPU);

template <>
inline void near_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, float* hCPU, float* hGPU)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            float compare_val = std::max(std::abs(hCPU[i + j * lda] * 1e-3f),
                                         10 * std::numeric_limits<float>::epsilon());
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(hCPU[i + j * lda]))
            {
                ASSERT_TRUE(rocsparse_isnan(hGPU[i + j * lda]));
            }
            else if(rocsparse_isinf(hCPU[i + j * lda]))
            {
                ASSERT_TRUE(rocsparse_isinf(hGPU[i + j * lda]));
            }
            else
            {
                ASSERT_NEAR(hCPU[i + j * lda], hGPU[i + j * lda], compare_val);
            }
#else
            assert(std::abs(hCPU[i + j * lda] - hGPU[i + j * lda]) < compare_val);
#endif
        }
    }
}

template <>
inline void near_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, double* hCPU, double* hGPU)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            double compare_val = std::max(std::abs(hCPU[i + j * lda] * 1e-10),
                                          10 * std::numeric_limits<double>::epsilon());
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(hCPU[i + j * lda]))
            {
                ASSERT_TRUE(rocsparse_isnan(hGPU[i + j * lda]));
            }
            else if(rocsparse_isinf(hCPU[i + j * lda]))
            {
                ASSERT_TRUE(rocsparse_isinf(hGPU[i + j * lda]));
            }
            else
            {
                ASSERT_NEAR(hCPU[i + j * lda], hGPU[i + j * lda], compare_val);
            }
#else
            assert(std::abs(hCPU[i + j * lda] - hGPU[i + j * lda]) < compare_val);
#endif
        }
    }
}

#endif // ROCSPARSE_CHECK_HPP

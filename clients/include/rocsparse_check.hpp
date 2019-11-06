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
#define ASSERT_TRUE(cond)                                      \
    do                                                         \
    {                                                          \
        if(!cond)                                              \
        {                                                      \
            std::cerr << "ASSERT_TRUE() failed." << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while(0)

#define ASSERT_EQ(state1, state2)                                                              \
    do                                                                                         \
    {                                                                                          \
        if(state1 != state2)                                                                   \
        {                                                                                      \
            std::cerr.precision(16);                                                           \
            std::cerr << "ASSERT_EQ(" << state1 << ", " << state2 << ") failed." << std::endl; \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

#define ASSERT_FLOAT_EQ ASSERT_EQ
#define ASSERT_DOUBLE_EQ ASSERT_EQ
#endif

#define ASSERT_FLOAT_COMPLEX_EQ(a, b)                \
    do                                               \
    {                                                \
        ASSERT_FLOAT_EQ(std::real(a), std::real(b)); \
        ASSERT_FLOAT_EQ(std::imag(a), std::imag(b)); \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b)                \
    do                                                \
    {                                                 \
        ASSERT_DOUBLE_EQ(std::real(a), std::real(b)); \
        ASSERT_DOUBLE_EQ(std::imag(a), std::imag(b)); \
    } while(0)

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
inline void unit_check_general(rocsparse_int            M,
                               rocsparse_int            N,
                               rocsparse_int            lda,
                               rocsparse_float_complex* hCPU,
                               rocsparse_float_complex* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(rocsparse_int             M,
                               rocsparse_int             N,
                               rocsparse_int             lda,
                               rocsparse_double_complex* hCPU,
                               rocsparse_double_complex* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
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
            if(std::abs(hCPU[i + j * lda] - hGPU[i + j * lda]) >= compare_val)
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_NEAR(" << hCPU[i + j * lda] << ", " << hGPU[i + j * lda]
                          << ") failed: " << std::abs(hCPU[i + j * lda] - hGPU[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
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
            if(std::abs(hCPU[i + j * lda] - hGPU[i + j * lda]) >= compare_val)
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << hCPU[i + j * lda] << ", " << hGPU[i + j * lda]
                          << ") failed: " << std::abs(hCPU[i + j * lda] - hGPU[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
#endif
        }
    }
}

template <>
inline void near_check_general(rocsparse_int            M,
                               rocsparse_int            N,
                               rocsparse_int            lda,
                               rocsparse_float_complex* hCPU,
                               rocsparse_float_complex* hGPU)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_float_complex compare_val
                = rocsparse_float_complex(std::max(std::abs(std::real(hCPU[i + j * lda]) * 1e-3f),
                                                   10 * std::numeric_limits<float>::epsilon()),
                                          std::max(std::abs(std::imag(hCPU[i + j * lda]) * 1e-3f),
                                                   10 * std::numeric_limits<float>::epsilon()));
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
                ASSERT_NEAR(std::real(hCPU[i + j * lda]),
                            std::real(hGPU[i + j * lda]),
                            std::real(compare_val));
                ASSERT_NEAR(std::imag(hCPU[i + j * lda]),
                            std::imag(hGPU[i + j * lda]),
                            std::imag(compare_val));
            }
#else
            if(std::abs(std::real(hCPU[i + j * lda]) - std::real(hGPU[i + j * lda]))
                   >= std::real(compare_val)
               || std::abs(std::imag(hCPU[i + j * lda]) - std::imag(hGPU[i + j * lda]))
                      >= std::imag(compare_val))
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << hCPU[i + j * lda] << ", " << hGPU[i + j * lda]
                          << ") failed: " << std::abs(hCPU[i + j * lda] - hGPU[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
#endif
        }
    }
}

template <>
inline void near_check_general(rocsparse_int             M,
                               rocsparse_int             N,
                               rocsparse_int             lda,
                               rocsparse_double_complex* hCPU,
                               rocsparse_double_complex* hGPU)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_double_complex compare_val
                = rocsparse_double_complex(std::max(std::abs(std::real(hCPU[i + j * lda]) * 1e-10),
                                                    10 * std::numeric_limits<double>::epsilon()),
                                           std::max(std::abs(std::imag(hCPU[i + j * lda]) * 1e-10),
                                                    10 * std::numeric_limits<double>::epsilon()));
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
                ASSERT_NEAR(std::real(hCPU[i + j * lda]),
                            std::real(hGPU[i + j * lda]),
                            std::real(compare_val));
                ASSERT_NEAR(std::imag(hCPU[i + j * lda]),
                            std::imag(hGPU[i + j * lda]),
                            std::imag(compare_val));
            }
#else
            if(std::abs(std::real(hCPU[i + j * lda]) - std::real(hGPU[i + j * lda]))
                   >= std::real(compare_val)
               || std::abs(std::imag(hCPU[i + j * lda]) - std::imag(hGPU[i + j * lda]))
                      >= std::imag(compare_val))
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << hCPU[i + j * lda] << ", " << hGPU[i + j * lda]
                          << ") failed: " << std::abs(hCPU[i + j * lda] - hGPU[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
#endif
        }
    }
}

#endif // ROCSPARSE_CHECK_HPP

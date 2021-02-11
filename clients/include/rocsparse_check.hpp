/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_math.hpp"

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#ifndef GOOGLE_TEST

#include <iostream>

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
void unit_check_general(int64_t M, int64_t N, int64_t lda, const T* hCPU, const T* hGPU);

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const float* hCPU, const float* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const double* hCPU, const double* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
inline void unit_check_general(int64_t                        M,
                               int64_t                        N,
                               int64_t                        lda,
                               const rocsparse_float_complex* hCPU,
                               const rocsparse_float_complex* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
inline void unit_check_general(int64_t                         M,
                               int64_t                         N,
                               int64_t                         lda,
                               const rocsparse_double_complex* hCPU,
                               const rocsparse_double_complex* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int32_t* hCPU, const int32_t* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const int64_t* hCPU, const int64_t* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
inline void
    unit_check_general(int64_t M, int64_t N, int64_t lda, const size_t* hCPU, const size_t* hGPU)
{
    UNIT_CHECK(M, N, lda, hCPU, hGPU, ASSERT_EQ);
}

template <typename U, typename V>
void near_check_general_template(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, const U* hCPU, const U* hGPU, V tol);

template <>
inline void near_check_general_template(rocsparse_int M,
                                        rocsparse_int N,
                                        rocsparse_int lda,
                                        const float*  hCPU,
                                        const float*  hGPU,
                                        float         tol)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            float compare_val = std::max(std::abs(hCPU[i + j * lda] * tol),
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
inline void near_check_general_template(rocsparse_int M,
                                        rocsparse_int N,
                                        rocsparse_int lda,
                                        const double* hCPU,
                                        const double* hGPU,
                                        double        tol)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            double compare_val = std::max(std::abs(hCPU[i + j * lda] * tol),
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
inline void near_check_general_template(rocsparse_int                  M,
                                        rocsparse_int                  N,
                                        rocsparse_int                  lda,
                                        const rocsparse_float_complex* hCPU,
                                        const rocsparse_float_complex* hGPU,
                                        float                          tol)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_float_complex compare_val
                = rocsparse_float_complex(std::max(std::abs(std::real(hCPU[i + j * lda]) * tol),
                                                   10 * std::numeric_limits<float>::epsilon()),
                                          std::max(std::abs(std::imag(hCPU[i + j * lda]) * tol),
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
inline void near_check_general_template(rocsparse_int                   M,
                                        rocsparse_int                   N,
                                        rocsparse_int                   lda,
                                        const rocsparse_double_complex* hCPU,
                                        const rocsparse_double_complex* hGPU,
                                        double                          tol)
{
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_double_complex compare_val
                = rocsparse_double_complex(std::max(std::abs(std::real(hCPU[i + j * lda]) * tol),
                                                    10 * std::numeric_limits<double>::epsilon()),
                                           std::max(std::abs(std::imag(hCPU[i + j * lda]) * tol),
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

template <typename T>
struct default_tolerance;

template <>
struct default_tolerance<float>
{
    static constexpr float value = 1.0e-3f;
};

template <>
struct default_tolerance<double>
{
    static constexpr double value = 1.0e-10;
};

template <>
struct default_tolerance<rocsparse_float_complex>
{
    static constexpr float value = default_tolerance<float>::value;
};

template <>
struct default_tolerance<rocsparse_double_complex>
{
    static constexpr double value = default_tolerance<double>::value;
};

template <typename T>
inline void near_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, const T* hCPU, const T* hGPU)
{
    near_check_general_template<T>(M, N, lda, hCPU, hGPU, default_tolerance<T>::value);
}

template <typename U, typename V>
inline void near_check_general(
    rocsparse_int M, rocsparse_int N, rocsparse_int lda, const U* hCPU, const U* hGPU, V tol)
{
    near_check_general_template<U>(M, N, lda, hCPU, hGPU, tol);
}

#endif // ROCSPARSE_CHECK_HPP

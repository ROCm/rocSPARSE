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
#include "rocsparse_check.hpp"

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

#ifndef GOOGLE_TEST

#include <iostream>

#define ASSERT_TRUE(cond)                                      \
    do                                                         \
    {                                                          \
        if(!(cond))                                            \
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

#define ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, UNIT_ASSERT_EQ)  \
    do                                                              \
    {                                                               \
        for(rocsparse_int j = 0; j < N; ++j)                        \
            for(rocsparse_int i = 0; i < M; ++i)                    \
                if(rocsparse_isnan(A[i + j * LDA]))                 \
                {                                                   \
                    ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));   \
                }                                                   \
                else                                                \
                {                                                   \
                    UNIT_ASSERT_EQ(A[i + j * LDA], B[i + j * LDB]); \
                }                                                   \
    } while(0)

template <>
void unit_check_general(
    int64_t M, int64_t N, const float* A, int64_t LDA, const float* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const double* A, int64_t LDA, const double* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(int64_t                        M,
                        int64_t                        N,
                        const rocsparse_float_complex* A,
                        int64_t                        LDA,
                        const rocsparse_float_complex* B,
                        int64_t                        LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(int64_t                         M,
                        int64_t                         N,
                        const rocsparse_double_complex* A,
                        int64_t                         LDA,
                        const rocsparse_double_complex* B,
                        int64_t                         LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const int32_t* A, int64_t LDA, const int32_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void unit_check_general(
    int64_t M, int64_t N, const int64_t* A, int64_t LDA, const int64_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}
template <>
void unit_check_general(
    int64_t M, int64_t N, const size_t* A, int64_t LDA, const size_t* B, int64_t LDB)
{
    ROCSPARSE_UNIT_CHECK(M, N, A, LDA, B, LDB, ASSERT_EQ);
}

template <>
void unit_check_enum(const rocsparse_index_base a, const rocsparse_index_base b)
{
    ASSERT_TRUE(a == b);
}

template <>
void unit_check_enum(const rocsparse_order a, const rocsparse_order b)
{
    ASSERT_TRUE(a == b);
}

template <>
void unit_check_enum(const rocsparse_direction a, const rocsparse_direction b)
{
    ASSERT_TRUE(a == b);
}

#define MAX_TOL_MULTIPLIER 4

template <typename T>
void near_check_general_template(rocsparse_int      M,
                                 rocsparse_int      N,
                                 const T*           A,
                                 rocsparse_int      LDA,
                                 const T*           B,
                                 rocsparse_int      LDB,
                                 floating_data_t<T> tol = default_tolerance<T>::value)
{
    int tolm = 1;
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            T compare_val
                = std::max(std::abs(A[i + j * LDA] * tol), 10 * std::numeric_limits<T>::epsilon());
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));
            }
            else if(rocsparse_isinf(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isinf(B[i + j * LDB]));
            }
            else
            {
                int k;
                for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
                {
                    if(std::abs(A[i + j * LDA] - B[i + j * LDB]) <= compare_val * k)
                    {
                        break;
                    }
                }

                if(k > MAX_TOL_MULTIPLIER)
                {
                    ASSERT_NEAR(A[i + j * LDA], B[i + j * LDB], compare_val);
                }
                tolm = std::max(tolm, k);
            }
#else

            int k;
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                if(std::abs(A[i + j * LDA] - B[i + j * LDB]) <= compare_val * k)
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_NEAR(" << A[i + j * LDA] << ", " << B[i + j * LDB]
                          << ") failed: " << std::abs(A[i + j * LDA] - B[i + j * LDB])
                          << " exceeds permissive range [" << compare_val << ","
                          << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                exit(EXIT_FAILURE);
            }
            tolm = std::max(tolm, k);
#endif
        }
    }

    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <>
void near_check_general_template(rocsparse_int                  M,
                                 rocsparse_int                  N,
                                 const rocsparse_float_complex* A,
                                 rocsparse_int                  LDA,
                                 const rocsparse_float_complex* B,
                                 rocsparse_int                  LDB,
                                 float                          tol)
{
    int tolm = 1;
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_float_complex compare_val
                = rocsparse_float_complex(std::max(std::abs(std::real(A[i + j * LDA]) * tol),
                                                   10 * std::numeric_limits<float>::epsilon()),
                                          std::max(std::abs(std::imag(A[i + j * LDA]) * tol),
                                                   10 * std::numeric_limits<float>::epsilon()));
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));
            }
            else if(rocsparse_isinf(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isinf(B[i + j * LDB]));
            }
            else
            {
                int k;
                for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
                {
                    if(std::abs(std::real(A[i + j * LDA]) - std::real(B[i + j * LDB]))
                           <= std::real(compare_val) * k
                       && std::abs(std::imag(A[i + j * LDA]) - std::imag(B[i + j * LDB]))
                              <= std::imag(compare_val) * k)
                    {
                        break;
                    }
                }

                if(k > MAX_TOL_MULTIPLIER)
                {
                    ASSERT_NEAR(std::real(A[i + j * LDA]),
                                std::real(B[i + j * LDB]),
                                std::real(compare_val));
                    ASSERT_NEAR(std::imag(A[i + j * LDA]),
                                std::imag(B[i + j * LDB]),
                                std::imag(compare_val));
                }
                tolm = std::max(tolm, k);
            }
#else

            int k;
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                if(std::abs(std::real(A[i + j * LDA]) - std::real(B[i + j * LDB]))
                       <= std::real(compare_val) * k
                   && std::abs(std::imag(A[i + j * LDA]) - std::imag(B[i + j * LDB]))
                          <= std::imag(compare_val) * k)
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << A[i + j * LDA] << ", " << B[i + j * LDB]
                          << ") failed: " << std::abs(A[i + j * LDA] - B[i + j * LDB])
                          << " exceeds permissive range [" << compare_val << ","
                          << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                exit(EXIT_FAILURE);
            }
            tolm = std::max(tolm, k);
#endif
        }
    }

    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <>
void near_check_general_template(rocsparse_int                   M,
                                 rocsparse_int                   N,
                                 const rocsparse_double_complex* A,
                                 rocsparse_int                   LDA,
                                 const rocsparse_double_complex* B,
                                 rocsparse_int                   LDB,
                                 double                          tol)
{
    int tolm = 1;
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            rocsparse_double_complex compare_val
                = rocsparse_double_complex(std::max(std::abs(std::real(A[i + j * LDA]) * tol),
                                                    10 * std::numeric_limits<double>::epsilon()),
                                           std::max(std::abs(std::imag(A[i + j * LDA]) * tol),
                                                    10 * std::numeric_limits<double>::epsilon()));
#ifdef GOOGLE_TEST
            if(rocsparse_isnan(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isnan(B[i + j * LDB]));
            }
            else if(rocsparse_isinf(A[i + j * LDA]))
            {
                ASSERT_TRUE(rocsparse_isinf(B[i + j * LDB]));
            }
            else
            {
                int k;
                for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
                {
                    if(std::abs(std::real(A[i + j * LDA]) - std::real(B[i + j * LDB]))
                           <= std::real(compare_val) * k
                       && std::abs(std::imag(A[i + j * LDA]) - std::imag(B[i + j * LDB]))
                              <= std::imag(compare_val) * k)
                    {
                        break;
                    }
                }

                if(k > MAX_TOL_MULTIPLIER)
                {
                    ASSERT_NEAR(std::real(A[i + j * LDA]),
                                std::real(B[i + j * LDB]),
                                std::real(compare_val));
                    ASSERT_NEAR(std::imag(A[i + j * LDA]),
                                std::imag(B[i + j * LDB]),
                                std::imag(compare_val));
                }
                tolm = std::max(tolm, k);
            }
#else

            int k;
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                if(std::abs(std::real(A[i + j * LDA]) - std::real(B[i + j * LDB]))
                       <= std::real(compare_val) * k
                   && std::abs(std::imag(A[i + j * LDA]) - std::imag(B[i + j * LDB]))
                          <= std::imag(compare_val) * k)
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << A[i + j * LDA] << ", " << B[i + j * LDB]
                          << ") failed: " << std::abs(A[i + j * LDA] - B[i + j * LDB])
                          << " exceeds permissive range [" << compare_val << ","
                          << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                exit(EXIT_FAILURE);
            }
            tolm = std::max(tolm, k);
#endif
        }
    }

    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <typename T>
void near_check_general(rocsparse_int      M,
                        rocsparse_int      N,
                        const T*           A,
                        rocsparse_int      LDA,
                        const T*           B,
                        rocsparse_int      LDB,
                        floating_data_t<T> tol)
{
    near_check_general_template(M, N, A, LDA, B, LDB, tol);
}

#define INSTANTIATE(TYPE)                                       \
    template void near_check_general(rocsparse_int         M,   \
                                     rocsparse_int         N,   \
                                     const TYPE*           A,   \
                                     rocsparse_int         LDA, \
                                     const TYPE*           B,   \
                                     rocsparse_int         LDB, \
                                     floating_data_t<TYPE> tol)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE

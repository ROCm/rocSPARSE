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

template <typename T>
struct floating_traits
{
    using data_t = T;
};

template <>
struct floating_traits<rocsparse_float_complex>
{
    using data_t = float;
};

template <>
struct floating_traits<rocsparse_double_complex>
{
    using data_t = double;
};

template <typename T>
using floating_data_t = typename floating_traits<T>::data_t;

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
void unit_check_general(int64_t M, int64_t N, int64_t lda, const T* hCPU, const T* hGPU);

template <typename T>
void near_check_general(rocsparse_int      M,
                        rocsparse_int      N,
                        rocsparse_int      lda,
                        const T*           hCPU,
                        const T*           hGPU,
                        floating_data_t<T> tol = default_tolerance<T>::value);

#endif // ROCSPARSE_CHECK_HPP

/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATH_HPP
#define ROCSPARSE_MATH_HPP

#include <cmath>
#include <rocsparse.h>

/* =================================================================================== */
/*! \brief  returns true if value is NaN */
template <typename T>
inline bool rocsparse_isnan(T arg);

template <>
inline bool rocsparse_isnan(int arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(int64_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(uint64_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(double arg)
{
    return std::isnan(arg);
}
template <>
inline bool rocsparse_isnan(float arg)
{
    return std::isnan(arg);
}

template <>
inline bool rocsparse_isnan(rocsparse_float_complex arg)
{
    return std::isnan(std::real(arg)) || std::isnan(std::imag(arg));
}

template <>
inline bool rocsparse_isnan(rocsparse_double_complex arg)
{
    return std::isnan(std::real(arg)) || std::isnan(std::imag(arg));
}

/* =================================================================================== */
/*! \brief  returns true if value is inf */
template <typename T>
inline bool rocsparse_isinf(T arg)
{
    return std::isinf(arg);
}

template <>
inline bool rocsparse_isinf(rocsparse_float_complex arg)
{
    return std::isinf(std::real(arg)) || std::isinf(std::imag(arg));
}

template <>
inline bool rocsparse_isinf(rocsparse_double_complex arg)
{
    return std::isinf(std::real(arg)) || std::isinf(std::imag(arg));
}

/* =================================================================================== */
/*! \brief  returns complex conjugate */
template <typename T>
inline T rocsparse_conj(T arg)
{
    return arg;
}

template <>
inline rocsparse_float_complex rocsparse_conj(rocsparse_float_complex arg)
{
    return std::conj(arg);
}

template <>
inline rocsparse_double_complex rocsparse_conj(rocsparse_double_complex arg)
{
    return std::conj(arg);
}

#endif // ROCSPARSE_MATH_HPP

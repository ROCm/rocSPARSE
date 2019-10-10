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
#ifndef ROCSPARSE_MATH_HPP
#define ROCSPARSE_MATH_HPP

#include <cmath>
#include <complex>
#include <rocsparse.h>

/* =================================================================================== */
/*! \brief  returns true if value is NaN */
template <typename T>
inline bool rocsparse_isnan(T arg)
{
    return std::isnan(arg);
}

template <>
inline bool rocsparse_isnan(std::complex<float> arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}

template <>
inline bool rocsparse_isnan(std::complex<double> arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}

/* =================================================================================== */
/*! \brief  returns true if value is inf */
template <typename T>
inline bool rocsparse_isinf(T arg)
{
    return std::isinf(arg);
}

template <>
inline bool rocsparse_isinf(std::complex<float> arg)
{
    return std::isinf(arg.real()) || std::isinf(arg.imag());
}

template <>
inline bool rocsparse_isinf(std::complex<double> arg)
{
    return std::isinf(arg.real()) || std::isinf(arg.imag());
}

/* =================================================================================== */
/*! \brief inject fma for rocsparse complex types into namespace std */
namespace std
{
    inline std::complex<float> fma(std::complex<float> p,
                                   std::complex<float> q,
                                   std::complex<float> r)
    {
        float real = std::fma(-p.imag(), q.imag(), std::fma(p.real(), q.real(), r.real()));
        float imag = std::fma(p.real(), q.imag(), std::fma(p.imag(), q.real(), r.imag()));

        return std::complex<float>(real, imag);
    }

    inline std::complex<double> fma(std::complex<double> p,
                                    std::complex<double> q,
                                    std::complex<double> r)
    {
        double real = std::fma(-p.imag(), q.imag(), std::fma(p.real(), q.real(), r.real()));
        double imag = std::fma(p.real(), q.imag(), std::fma(p.imag(), q.real(), r.imag()));

        return std::complex<double>(real, imag);
    }
}

#endif // ROCSPARSE_MATH_HPP

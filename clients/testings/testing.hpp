/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "flops.hpp"
#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_matrix_factory.hpp"
#include "utility.hpp"
#include <rocsparse.hpp>

template <typename T>
inline T* rocsparse_fake_pointer()
{
    return static_cast<T*>((void*)0x4);
}

template <typename T>
inline T rocsparse_nan()
{
    return std::numeric_limits<T>::quiet_NaN();
}

template <>
inline rocsparse_float_complex rocsparse_nan<rocsparse_float_complex>()
{
    return rocsparse_float_complex(std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::quiet_NaN());
}

template <>
inline rocsparse_double_complex rocsparse_nan<rocsparse_double_complex>()
{
    return rocsparse_double_complex(std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN());
}

template <typename T>
inline T rocsparse_inf()
{
    return std::numeric_limits<T>::infinity();
}

template <>
inline rocsparse_float_complex rocsparse_inf<rocsparse_float_complex>()
{
    return rocsparse_float_complex(std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::infinity());
}

template <>
inline rocsparse_double_complex rocsparse_inf<rocsparse_double_complex>()
{
    return rocsparse_double_complex(std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity());
}

template <typename T>
floating_data_t<T> get_near_check_tol(const Arguments& arg)
{
    return static_cast<floating_data_t<T>>(arg.tolm) * default_tolerance<T>::value;
}

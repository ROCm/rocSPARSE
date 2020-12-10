/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
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
#ifndef TYPE_DISPATCH_HPP
#define TYPE_DISPATCH_HPP

#include "rocsparse_arguments.hpp"

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto rocsparse_simple_dispatch(const Arguments& arg)
{
    switch(arg.compute_type)
    {
    case rocsparse_datatype_f32_r:
        return TEST<float>{}(arg);
    case rocsparse_datatype_f64_r:
        return TEST<double>{}(arg);
    case rocsparse_datatype_f32_c:
        return TEST<rocsparse_float_complex>{}(arg);
    case rocsparse_datatype_f64_c:
        return TEST<rocsparse_double_complex>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

template <template <typename...> class TEST>
auto rocsparse_it_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;

    if(I == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int32_t, rocsparse_double_complex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, rocsparse_double_complex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ijt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    if(I == rocsparse_indextype_i32 && J == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int32_t, int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int32_t, int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int32_t, int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int32_t, int32_t, rocsparse_double_complex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64 && J == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, int32_t, rocsparse_double_complex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64 && J == rocsparse_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, int64_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, int64_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, int64_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, int64_t, rocsparse_double_complex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

#endif // TYPE_DISPATCH_HPP

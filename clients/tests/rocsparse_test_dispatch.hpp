/*! \file */
/* ************************************************************************
* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_test.hpp"
#include "rocsparse_test_dispatch_enum.hpp"
#include "type_dispatch.hpp"

template <rocsparse_test_dispatch_enum::value_type TYPE_DISPATCH = rocsparse_test_dispatch_enum::t>
struct rocsparse_test_dispatch;

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::t>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_simple_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::it>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_it_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ijt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ijt_dispatch<TEST>(arg);
    }
};

/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief rocsparse_enum.hpp provides common testing utilities.
 */

#pragma once
#ifndef ROCSPARSE_ENUM_HPP
#define ROCSPARSE_ENUM_HPP

#include "rocsparse_test.hpp"
#include <hip/hip_runtime_api.h>
#include <vector>

struct rocsparse_itilu0_alg_t
{
    using value_t                                 = rocsparse_itilu0_alg;
    static constexpr unsigned int nvalues         = 5;
    static constexpr value_t      values[nvalues] = {rocsparse_itilu0_alg_default,
                                                rocsparse_itilu0_alg_async_inplace,
                                                rocsparse_itilu0_alg_async_split,
                                                rocsparse_itilu0_alg_sync_split,
                                                rocsparse_itilu0_alg_sync_split_fusion};

    static void info(std::ostream& out_)
    {
        for(int i = 0; i < nvalues; ++i)
        {
            if(i > 0)
                out_ << ", ";
            const value_t v = values[i];
            switch(v)
            {
#define LOCAL_CASE(TOKEN)                 \
    case TOKEN:                           \
    {                                     \
        out_ << TOKEN << " : " << #TOKEN; \
        break;                            \
    }

            case rocsparse_itilu0_alg_default:
            {
                out_ << rocsparse_itilu0_alg_default << " : "
                     << "rocsparse_itilu0_alg_default";
            }
                LOCAL_CASE(rocsparse_itilu0_alg_async_inplace);
                LOCAL_CASE(rocsparse_itilu0_alg_async_split);
                LOCAL_CASE(rocsparse_itilu0_alg_sync_split);
                LOCAL_CASE(rocsparse_itilu0_alg_sync_split_fusion);
#undef LOCAL_CASE
            }
        }
    };

    static constexpr bool is_invalid(rocsparse_int value_)
    {
        return is_invalid((value_t)value_);
    };

    static constexpr bool is_invalid(value_t value_)
    {
        switch(value_)
        {
        case rocsparse_itilu0_alg_default:
        case rocsparse_itilu0_alg_async_inplace:
        case rocsparse_itilu0_alg_async_split:
        case rocsparse_itilu0_alg_sync_split:
        case rocsparse_itilu0_alg_sync_split_fusion:
        {
            return false;
        }
        }
        return true;
    }
};

struct rocsparse_matrix_type_t
{
    using value_t                                 = rocsparse_matrix_type;
    static constexpr unsigned int nvalues         = 4;
    static constexpr value_t      values[nvalues] = {rocsparse_matrix_type_general,
                                                rocsparse_matrix_type_symmetric,
                                                rocsparse_matrix_type_hermitian,
                                                rocsparse_matrix_type_triangular};
};

struct rocsparse_operation_t
{
    using value_t                                 = rocsparse_operation;
    static constexpr unsigned int nvalues         = 3;
    static constexpr value_t      values[nvalues] = {rocsparse_operation_none,
                                                rocsparse_operation_transpose,
                                                rocsparse_operation_conjugate_transpose};
};

struct rocsparse_storage_mode_t
{
    using value_t                         = rocsparse_storage_mode;
    static constexpr unsigned int nvalues = 2;
    static constexpr value_t      values[nvalues]
        = {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted};
};

std::ostream& operator<<(std::ostream& out, const rocsparse_operation& v);
std::ostream& operator<<(std::ostream& out, const rocsparse_direction& v);

struct rocsparse_datatype_t
{
    using value_t = rocsparse_datatype;
    template <typename T>
    static inline rocsparse_datatype get();
};

template <>
inline rocsparse_datatype rocsparse_datatype_t::get<float>()
{
    return rocsparse_datatype_f32_r;
}
template <>
inline rocsparse_datatype rocsparse_datatype_t::get<double>()
{
    return rocsparse_datatype_f64_r;
}

template <>
inline rocsparse_datatype rocsparse_datatype_t::get<rocsparse_float_complex>()
{
    return rocsparse_datatype_f32_c;
}

template <>
inline rocsparse_datatype rocsparse_datatype_t::get<rocsparse_double_complex>()
{
    return rocsparse_datatype_f64_c;
}

#endif // ROCSPARSE_ENUM_HPP

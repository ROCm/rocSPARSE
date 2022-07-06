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

#endif // ROCSPARSE_ENUM_HPP

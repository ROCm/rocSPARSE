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
#ifndef ROCSPARSE_DATATYPE2STRING_HPP
#define ROCSPARSE_DATATYPE2STRING_HPP

#include <rocsparse.h>

typedef enum rocsparse_datatype_
{
    rocsparse_datatype_f32_r = 151, /**< 32 bit floating point, real */
    rocsparse_datatype_f64_r = 152, /**< 64 bit floating point, real */
    rocsparse_datatype_f32_c = 154, /**< 32 bit floating point, complex real */
    rocsparse_datatype_f64_c = 155 /**< 64 bit floating point, complex real */
} rocsparse_datatype;

typedef enum rocsparse_matrix_init_
{
    rocsparse_matrix_random          = 0, /**< Random initialization */
    rocsparse_matrix_laplace_2d      = 1, /**< Initialize 2D laplacian matrix */
    rocsparse_matrix_laplace_3d      = 2, /**< Initialize 3D laplacian matrix */
    rocsparse_matrix_file_mtx        = 3, /**< Read from .mtx (matrix market) file */
    rocsparse_matrix_file_rocalution = 4 /**< Read from .csr (rocALUTION) file */
} rocsparse_matrix_init;

constexpr auto rocsparse_matrix2string(rocsparse_matrix_init matrix)
{
    switch(matrix)
    {
    case rocsparse_matrix_random:
        return "rand";
    case rocsparse_matrix_laplace_2d:
        return "L2D";
    case rocsparse_matrix_laplace_3d:
        return "L3D";
    case rocsparse_matrix_file_mtx:
        return "mtx";
    case rocsparse_matrix_file_rocalution:
        return "csr";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_datatype2string(rocsparse_datatype type)
{
    switch(type)
    {
    case rocsparse_datatype_f32_r:
        return "f32_r";
    case rocsparse_datatype_f64_r:
        return "f64_r";
    case rocsparse_datatype_f32_c:
        return "f32_c";
    case rocsparse_datatype_f64_c:
        return "f64_c";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_indexbase2string(rocsparse_index_base base)
{
    switch(base)
    {
    case rocsparse_index_base_zero:
        return "0b";
    case rocsparse_index_base_one:
        return "1b";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_operation2string(rocsparse_operation trans)
{
    switch(trans)
    {
    case rocsparse_operation_none:
        return "NT";
    case rocsparse_operation_transpose:
        return "T";
    case rocsparse_operation_conjugate_transpose:
        return "CT";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_diagtype2string(rocsparse_diag_type diag)
{
    switch(diag)
    {
    case rocsparse_diag_type_non_unit:
        return "ND";
    case rocsparse_diag_type_unit:
        return "UD";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_fillmode2string(rocsparse_fill_mode uplo)
{
    switch(uplo)
    {
    case rocsparse_fill_mode_lower:
        return "L";
    case rocsparse_fill_mode_upper:
        return "U";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_action2string(rocsparse_action action)
{
    switch(action)
    {
    case rocsparse_action_symbolic:
        return "sym";
    case rocsparse_action_numeric:
        return "num";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_partition2string(rocsparse_hyb_partition part)
{
    switch(part)
    {
    case rocsparse_hyb_partition_auto:
        return "auto";
    case rocsparse_hyb_partition_user:
        return "user";
    case rocsparse_hyb_partition_max:
        return "max";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_analysis2string(rocsparse_analysis_policy policy)
{
    switch(policy)
    {
    case rocsparse_analysis_policy_reuse:
        return "reuse";
    case rocsparse_analysis_policy_force:
        return "force";
    default:
        return "invalid";
    }
}

constexpr auto rocsparse_solve2string(rocsparse_solve_policy policy)
{
    switch(policy)
    {
    case rocsparse_solve_policy_auto:
        return "auto";
    default:
        return "invalid";
    }
}

#endif // ROCSPARSE_DATATYPE2STRING_HPP

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

/*! \file
 *  \brief gbyte.hpp provides data transfer counts of Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef GBYTE_HPP
#define GBYTE_HPP

#include <rocsparse.h>

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double axpyi_gbyte_count(rocsparse_int nnz)
{
    return (nnz * sizeof(rocsparse_int) + (3.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double doti_gbyte_count(rocsparse_int nnz)
{
    return (nnz * sizeof(rocsparse_int) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gthr_gbyte_count(rocsparse_int nnz)
{
    return (nnz * sizeof(rocsparse_int) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double gthrz_gbyte_count(rocsparse_int nnz)
{
    return (nnz * sizeof(rocsparse_int) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double roti_gbyte_count(rocsparse_int nnz)
{
    return (nnz * sizeof(rocsparse_int) + (3.0 * nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double sctr_gbyte_count(rocsparse_int nnz)
{
    return (nnz * sizeof(rocsparse_int) + (2.0 * nnz) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double
    coomv_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz, bool beta = false)
{
    return (2.0 * nnz * sizeof(rocsparse_int) + (M + N + nnz + (beta ? M : 0)) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double
    csrmv_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz, bool beta = false)
{
    return ((M + 1 + nnz) * sizeof(rocsparse_int) + (M + N + nnz + (beta ? M : 0)) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double csrsv_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return ((M + 1 + nnz) * sizeof(rocsparse_int) + (M + M + nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double
    ellmv_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz, bool beta = false)
{
    return (nnz * sizeof(rocsparse_int) + (M + N + nnz + (beta ? M : 0)) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double csrmm_gbyte_count(rocsparse_int M,
                                   rocsparse_int nnz_A,
                                   rocsparse_int nnz_B,
                                   rocsparse_int nnz_C,
                                   bool          beta = false)
{
    return ((M + 1 + nnz_A) * sizeof(rocsparse_int)
            + (nnz_A + nnz_B + nnz_C + (beta ? nnz_C : 0)) * sizeof(T))
           / 1e9;
}

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double csrgemm_gbyte_count(rocsparse_int M,
                                     rocsparse_int N,
                                     rocsparse_int K,
                                     rocsparse_int nnz_A,
                                     rocsparse_int nnz_B,
                                     rocsparse_int nnz_C,
                                     rocsparse_int nnz_D,
                                     const T*      alpha,
                                     const T*      beta)
{
    double size_A = alpha ? (M + 1.0 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T) : 0.0;
    double size_B = alpha ? (K + 1.0 + nnz_B) * sizeof(rocsparse_int) + nnz_B * sizeof(T) : 0.0;
    double size_C = (M + 1.0 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);
    double size_D = beta ? (M + 1.0 + nnz_D) * sizeof(rocsparse_int) + nnz_D * sizeof(T) : 0.0;

    return (size_A + size_B + size_C + size_D) / 1e9;
}

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double csrilu0_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return ((M + 1 + nnz) * sizeof(rocsparse_int) + 2.0 * nnz * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double csr2coo_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return (M + 1 + nnz) * sizeof(rocsparse_int) / 1e9;
}

template <typename T>
constexpr double coo2csr_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return (M + 1 + nnz) * sizeof(rocsparse_int) / 1e9;
}

template <typename T>
constexpr double csr2csc_gbyte_count(rocsparse_int    M,
                                     rocsparse_int    N,
                                     rocsparse_int    nnz,
                                     rocsparse_action action)
{
    return ((M + N + 2 + 2.0 * nnz) * sizeof(rocsparse_int)
            + (action == rocsparse_action_numeric ? (2.0 * nnz) * sizeof(T) : 0.0))
           / 1e9;
}

template <typename T>
constexpr double csr2ell_gbyte_count(rocsparse_int M, rocsparse_int nnz, rocsparse_int ell_nnz)
{
    return ((M + 1.0 + ell_nnz) * sizeof(rocsparse_int) + (nnz + ell_nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double ell2csr_gbyte_count(rocsparse_int M, rocsparse_int csr_nnz, rocsparse_int ell_nnz)
{
    return ((M + 1.0 + ell_nnz) * sizeof(rocsparse_int) + (csr_nnz + ell_nnz) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double csr2hyb_gbyte_count(rocsparse_int M,
                                     rocsparse_int nnz,
                                     rocsparse_int ell_nnz,
                                     rocsparse_int coo_nnz)
{
    return ((M + 1.0 + ell_nnz + 2.0 * coo_nnz) * sizeof(rocsparse_int)
            + (nnz + ell_nnz + coo_nnz) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double identity_gbyte_count(rocsparse_int N)
{
    return N * sizeof(rocsparse_int) / 1e9;
}

template <typename T>
constexpr double csrsort_gbyte_count(rocsparse_int M, rocsparse_int nnz, bool permute)
{
    return ((2.0 * M + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocsparse_int))
           / 1e9;
}

template <typename T>
constexpr double cscsort_gbyte_count(rocsparse_int N, rocsparse_int nnz, bool permute)
{
    return ((2.0 * N + 2.0 + 2.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocsparse_int))
           / 1e9;
}

template <typename T>
constexpr double coosort_gbyte_count(rocsparse_int nnz, bool permute)
{
    return ((4.0 * nnz + (permute ? 2.0 * nnz : 0.0)) * sizeof(rocsparse_int)) / 1e9;
}

#endif // GBYTE_HPP

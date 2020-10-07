/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
constexpr double bsrmv_gbyte_count(rocsparse_int mb,
                                   rocsparse_int nb,
                                   rocsparse_int nnzb,
                                   rocsparse_int bsr_dim,
                                   bool          beta = false)
{
    return ((mb + 1 + nnzb) * sizeof(rocsparse_int)
            + ((mb + nb) * bsr_dim + nnzb * bsr_dim * bsr_dim + (beta ? mb * bsr_dim : 0))
                  * sizeof(T))
           / 1e9;
}

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
constexpr double bsrsv_gbyte_count(rocsparse_int mb, rocsparse_int nnzb, rocsparse_int bsr_dim)
{
    return ((mb + 1 + nnzb) * sizeof(rocsparse_int)
            + (bsr_dim * (mb + mb + nnzb * bsr_dim)) * sizeof(T))
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
constexpr double bsrmm_gbyte_count(rocsparse_int Mb,
                                   rocsparse_int nnzb,
                                   rocsparse_int block_dim,
                                   rocsparse_int nnz_B,
                                   rocsparse_int nnz_C,
                                   bool          beta = false)
{
    //reads
    size_t reads = (Mb + 1 + nnzb) * sizeof(rocsparse_int)
                   + (block_dim * block_dim * nnzb + nnz_B + (beta ? nnz_C : 0)) * sizeof(T);

    //writes
    size_t writes = nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

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
constexpr double csrgeam_gbyte_count(rocsparse_int M,
                                     rocsparse_int nnz_A,
                                     rocsparse_int nnz_B,
                                     rocsparse_int nnz_C,
                                     const T*      alpha,
                                     const T*      beta)
{
    double size_A = alpha ? (M + 1.0 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T) : 0.0;
    double size_B = alpha ? (M + 1.0 + nnz_B) * sizeof(rocsparse_int) + nnz_B * sizeof(T) : 0.0;
    double size_C = (M + 1.0 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);

    return (size_A + size_B + size_C) / 1e9;
}

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
constexpr double bsric0_gbyte_count(rocsparse_int Mb, rocsparse_int block_dim, rocsparse_int nnzb)
{
    return ((Mb + 1 + nnzb) * sizeof(rocsparse_int)
            + 2.0 * block_dim * block_dim * nnzb * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double bsrilu0_gbyte_count(rocsparse_int Mb, rocsparse_int block_dim, rocsparse_int nnzb)
{
    return ((Mb + 1 + nnzb) * sizeof(rocsparse_int)
            + 2.0 * block_dim * block_dim * nnzb * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double csric0_gbyte_count(rocsparse_int M, rocsparse_int nnz)
{
    return ((M + 1 + nnz) * sizeof(rocsparse_int) + 2.0 * nnz * sizeof(T)) / 1e9;
}

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
constexpr double nnz_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_direction dir)
{
    return ((M * N) * sizeof(T)
            + ((rocsparse_direction_row == dir) ? M : N) * sizeof(rocsparse_int))
           / 1e9;
}

template <rocsparse_direction DIRA, typename T>
constexpr double dense2csx_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz)
{
    const rocsparse_int L             = (rocsparse_direction_row == DIRA) ? M : N;
    const rocsparse_int write_csx_ptr = (L + 1) * sizeof(rocsparse_int);
    const rocsparse_int read_csx_ptr  = (L + 1) * sizeof(rocsparse_int);
    const rocsparse_int build_csx_ptr = write_csx_ptr + read_csx_ptr;

    const rocsparse_int write_csx
        = nnz * sizeof(T) + nnz * sizeof(rocsparse_int) + (L + 1) * sizeof(rocsparse_int);
    const rocsparse_int read_dense = M * N * sizeof(T);
    return (read_dense + build_csx_ptr + write_csx) / 1e9;
}

template <typename T>
constexpr double prune_dense2csr_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz)
{
    size_t reads = M * N * sizeof(T);

    size_t writes = (M + 1 + nnz) * sizeof(rocsparse_int) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double
    prune_dense2csr_by_percentage_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz)
{
    size_t reads = M * N * sizeof(T);

    size_t writes = (M + 1 + nnz) * sizeof(rocsparse_int) + nnz * sizeof(T);

    return (reads + writes) / 1e9;
}

template <rocsparse_direction DIRA, typename T>
constexpr double csx2dense_gbyte_count(rocsparse_int M, rocsparse_int N, rocsparse_int nnz)
{
    const rocsparse_int L = (rocsparse_direction_row == DIRA) ? M : N;
    const rocsparse_int read_csx
        = nnz * sizeof(T) + nnz * sizeof(rocsparse_int) + (L + 1) * sizeof(rocsparse_int);
    const rocsparse_int write_dense
        = M * N * sizeof(T) + nnz * sizeof(T); // set to zero + nnz assignments.
    return (read_csx + write_dense) / 1e9;
}

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
constexpr double hyb2csr_gbyte_count(rocsparse_int M,
                                     rocsparse_int csr_nnz,
                                     rocsparse_int ell_nnz,
                                     rocsparse_int coo_nnz)
{
    return ((M + 1.0 + csr_nnz + ell_nnz + 2.0 * coo_nnz) * sizeof(rocsparse_int)
            + (csr_nnz + ell_nnz + coo_nnz) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double bsr2csr_gbyte_count(rocsparse_int Mb, rocsparse_int block_dim, rocsparse_int nnzb)
{
    // reads
    size_t reads
        = nnzb * block_dim * block_dim * sizeof(T) + (Mb + 1 + nnzb) * sizeof(rocsparse_int);

    // writes
    size_t writes = nnzb * block_dim * block_dim * sizeof(T)
                    + (Mb * block_dim + 1 + nnzb * block_dim * block_dim) * sizeof(rocsparse_int);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double gebsr2csr_gbyte_count(rocsparse_int Mb,
                                       rocsparse_int row_block_dim,
                                       rocsparse_int col_block_dim,
                                       rocsparse_int nnzb)
{
    // reads
    size_t reads = nnzb * row_block_dim * col_block_dim * sizeof(T)
                   + (Mb + 1 + nnzb) * sizeof(rocsparse_int);

    // writes
    size_t writes
        = nnzb * row_block_dim * col_block_dim * sizeof(T)
          + (Mb * row_block_dim + 1 + nnzb * row_block_dim * col_block_dim) * sizeof(rocsparse_int);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double csr2bsr_gbyte_count(rocsparse_int M,
                                     rocsparse_int Mb,
                                     rocsparse_int nnz,
                                     rocsparse_int nnzb,
                                     rocsparse_int block_dim)
{
    // reads
    size_t reads = (M + 1 + nnz) * sizeof(rocsparse_int) + nnz * sizeof(T);

    // writes
    size_t writes = (Mb + 1 + nnzb * block_dim * block_dim) * sizeof(rocsparse_int)
                    + (nnzb * block_dim * block_dim) * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double
    csr2csr_compress_gbyte_count(rocsparse_int M, rocsparse_int nnz_A, rocsparse_int nnz_C)
{
    size_t reads = (M + 1 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T);

    size_t writes = (M + 1 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double
    prune_csr2csr_gbyte_count(rocsparse_int M, rocsparse_int nnz_A, rocsparse_int nnz_C)
{
    // reads
    size_t reads = (M + 1 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T);

    // writes
    size_t writes = (M + 1 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
}

template <typename T>
constexpr double prune_csr2csr_by_percentage_gbyte_count(rocsparse_int M,
                                                         rocsparse_int nnz_A,
                                                         rocsparse_int nnz_C)
{
    // reads
    size_t reads = (M + 1 + nnz_A) * sizeof(rocsparse_int) + nnz_A * sizeof(T);

    // writes
    size_t writes = (M + 1 + nnz_C) * sizeof(rocsparse_int) + nnz_C * sizeof(T);

    return (reads + writes) / 1e9;
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

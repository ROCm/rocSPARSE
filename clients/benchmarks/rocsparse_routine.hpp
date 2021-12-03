/*! \file */
/* ************************************************************************
* Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include "rocsparse_arguments.hpp"

// clang-format off
#define LIST_OF_ROUTINES			\
  bellmm,					\
    bsric0,					\
    bsrilu0,					\
    bsrmm,					\
    bsrmv,					\
    bsrsm,					\
    bsrsv,					\
    bsrxmv,					\
    bsr2csr,					\
    coomm,					\
    coomv,					\
    coosort,					\
    coosv,					\
    coomv_aos,					\
    coosm,					\
    coo2csr,					\
    coo2dense,					\
    cscsort,					\
    csc2dense,					\
    csrcolor,					\
    csric0,					\
    csrilu0,					\
    csrgeam,					\
    csrgemm,					\
    csrgemm_reuse,				\
    csrmv,					\
    csrmv_managed,				\
    csrmm,					\
    csrsm,					\
    csrsort,					\
    csrsv,					\
    csr2dense,					\
    csr2bsr,					\
    csr2coo,					\
    csr2csc,					\
    csr2csr_compress,				\
    csr2ell,					\
    csr2gebsr,					\
    csr2hyb,					\
    dense2coo,					\
    dense2csc,					\
    dense2csr,					\
    dense_to_sparse_coo,			\
    dense_to_sparse_csc,			\
    dense_to_sparse_csr,			\
    doti,					\
    dotci,					\
    ellmv,					\
    ell2csr,					\
    gebsr2csr,					\
    gebsr2gebsr,				\
    gthr,					\
    gthrz,					\
    gebsr2gebsc,				\
    gebsrmv,					\
    gebsrmm,					\
    gemmi,					\
    gemvi,					\
    gpsv_interleaved_batch, \
    gtsv,					\
    gtsv_no_pivot,				\
    gtsv_no_pivot_strided_batch,		\
    gtsv_interleaved_batch,		\
    hybmv,					\
    hyb2csr,					\
    identity,					\
    nnz,					\
    prune_csr2csr,				\
    prune_csr2csr_by_percentage,		\
    prune_dense2csr,				\
    prune_dense2csr_by_percentage,		\
    roti,					\
    sctr,					\
    sddmm,					\
    sparse_to_dense_coo,			\
    sparse_to_dense_csc,			\
    sparse_to_dense_csr
// clang-format on

template <std::size_t N, typename T>
static constexpr std::size_t countof(T (&)[N])
{
    return N;
}

struct rocsparse_routine
{
private:
public:
    typedef enum _ : rocsparse_int
    {
        axpyi = 0,
        LIST_OF_ROUTINES
    } value_type;
    value_type                   value{};
    static constexpr value_type  all_routines[] = {axpyi, LIST_OF_ROUTINES};
    static constexpr std::size_t num_routines   = countof(all_routines);

private:
    static constexpr const char* s_routine_names[num_routines]{"axpyi",
                                                               "bellmm",
                                                               "bsric0",
                                                               "bsrilu0",
                                                               "bsrmm",
                                                               "bsrmv",
                                                               "bsrsm",
                                                               "bsrsv",
                                                               "bsrxmv",
                                                               "bsr2csr",
                                                               "coomm",
                                                               "coomv",
                                                               "coosort",
                                                               "coosv",
                                                               "coomv_aos",
                                                               "coosm",
                                                               "coo2csr",
                                                               "coo2dense",
                                                               "cscsort",
                                                               "csc2dense",
                                                               "csrcolor",
                                                               "csric0",
                                                               "csrilu0",
                                                               "csrgeam",
                                                               "csrgemm",
                                                               "csrgemm_reuse",
                                                               "csrmv",
                                                               "csrmv_managed",
                                                               "csrmm",
                                                               "csrsm",
                                                               "csrsort",
                                                               "csrsv",
                                                               "csr2dense",
                                                               "csr2bsr",
                                                               "csr2coo",
                                                               "csr2csc",
                                                               "csr2csr_compress",
                                                               "csr2ell",
                                                               "csr2gebsr",
                                                               "csr2hyb",
                                                               "dense2coo",
                                                               "dense2csc",
                                                               "dense2csr",
                                                               "dense_to_sparse_coo",
                                                               "dense_to_sparse_csc",
                                                               "dense_to_sparse_csr",
                                                               "doti",
                                                               "dotci",
                                                               "ellmv",
                                                               "ell2csr",
                                                               "gebsr2csr",
                                                               "gebsr2gebsr",
                                                               "gthr",
                                                               "gthrz",
                                                               "gebsr2gebsc",
                                                               "gebsrmv",
                                                               "gebsrmm",
                                                               "gemmi",
                                                               "gemvi",
                                                               "gtsv",
                                                               "gtsv_no_pivot",
                                                               "gtsv_no_pivot_strided_batch",
                                                               "gtsv_interleaved_batch",
                                                               "gpsv_interleaved_batch",
                                                               "hybmv",
                                                               "hyb2csr",
                                                               "identity",
                                                               "nnz",
                                                               "prune_csr2csr",
                                                               "prune_csr2csr_by_percentage",
                                                               "prune_dense2csr",
                                                               "prune_dense2csr_by_percentage",
                                                               "roti",
                                                               "sctr",
                                                               "sddmm",
                                                               "sparse_to_dense_coo",
                                                               "sparse_to_dense_csc",
                                                               "sparse_to_dense_csr"};

public:
    rocsparse_routine();
    rocsparse_routine& operator()(const char* function);
    explicit rocsparse_routine(const char* function);
    rocsparse_status
        dispatch(const char precision, const char indextype, const Arguments& arg) const;
    constexpr const char* to_string() const;

private:
    template <rocsparse_routine::value_type FNAME, typename T, typename I, typename J = I>
    static rocsparse_status dispatch_call(const Arguments& arg);

    template <rocsparse_routine::value_type FNAME, typename T>
    static rocsparse_status dispatch_indextype(const char cindextype, const Arguments& arg);

    template <rocsparse_routine::value_type FNAME>
    static rocsparse_status
        dispatch_precision(const char precision, const char indextype, const Arguments& arg);
};

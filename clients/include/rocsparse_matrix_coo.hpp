/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_MATRIX_COO_HPP
#define ROCSPARSE_MATRIX_COO_HPP

#include "rocsparse_vector.hpp"

template <memory_mode::value_t MODE,
          typename T,
          typename I = rocsparse_int,
          typename J = rocsparse_int>
struct coo_matrix
{
    template <typename S>
    using array_t = typename memory_traits<MODE>::template array_t<S>;

    J                    m{};
    J                    n{};
    I                    nnz{};
    rocsparse_index_base base{};
    array_t<J>           row_ind{};
    array_t<J>           col_ind{};
    array_t<T>           val{};
    coo_matrix(){};
    coo_matrix(J m_, J n_, I nnz_, rocsparse_index_base base_)
        : m(m_)
        , n(n_)
        , nnz(nnz_)
        , base(base_)
        , row_ind(nnz_)
        , col_ind(nnz_)
        , val(nnz_)
    {
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using host_coo_matrix = coo_matrix<memory_mode::host, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using device_coo_matrix = coo_matrix<memory_mode::device, T, I, J>;
template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
using managed_coo_matrix = coo_matrix<memory_mode::managed, T, I, J>;

#endif // ROCSPARSE_MATRIX_COO_HPP

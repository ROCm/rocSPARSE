/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_IMPORTER_HPP
#define ROCSPARSE_IMPORTER_HPP

#include "rocsparse_check.hpp"
#include "rocsparse_traits.hpp"

#include "rocsparse_vector.hpp"

#include "rocsparse_matrix_coo.hpp"
#include "rocsparse_matrix_csx.hpp"
#include "rocsparse_matrix_dense.hpp"
#include "rocsparse_matrix_gebsx.hpp"

template <typename X, typename Y>
rocsparse_status rocsparse_type_conversion(const X& x, Y& y);

template <typename X, typename Y>
inline void
    rocsparse_importer_copy_mixed_arrays(size_t size, X* __restrict__ x, const Y* __restrict__ y)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(size_t i = 0; i < size; ++i)
    {
        x[i] = static_cast<X>(y[i]);
    }
}

template <>
inline void rocsparse_importer_copy_mixed_arrays(size_t size,
                                                 rocsparse_float_complex* __restrict__ x,
                                                 const rocsparse_double_complex* __restrict__ y)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(size_t i = 0; i < size; ++i)
    {
        x[i] = rocsparse_float_complex(static_cast<float>(std::real(y[i])),
                                       static_cast<float>(std::imag(y[i])));
    }
}

template <>
inline void rocsparse_importer_copy_mixed_arrays(size_t size,
                                                 rocsparse_double_complex* __restrict__ x,
                                                 const rocsparse_float_complex* __restrict__ y)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(size_t i = 0; i < size; ++i)
    {
        x[i] = rocsparse_double_complex(static_cast<double>(std::real(y[i])),
                                        static_cast<double>(std::imag(y[i])));
    }
}

template <>
inline void rocsparse_importer_copy_mixed_arrays(size_t size,
                                                 float* __restrict__ x,
                                                 const rocsparse_float_complex* __restrict__ y)
{
    throw rocsparse_status_not_implemented;
}

template <>
inline void rocsparse_importer_copy_mixed_arrays(size_t size,
                                                 float* __restrict__ x,
                                                 const rocsparse_double_complex* __restrict__ y)
{
    throw rocsparse_status_not_implemented;
}

template <>
inline void rocsparse_importer_copy_mixed_arrays(size_t size,
                                                 double* __restrict__ x,
                                                 const rocsparse_float_complex* __restrict__ y)
{
    throw rocsparse_status_not_implemented;
}

template <>
inline void rocsparse_importer_copy_mixed_arrays(size_t size,
                                                 double* __restrict__ x,
                                                 const rocsparse_double_complex* __restrict__ y)
{
    throw rocsparse_status_not_implemented;
}

template <typename U>
rocsparse_status rocsparse_importer_switch_base(size_t               size,
                                                U&                   u,
                                                rocsparse_index_base base,
                                                rocsparse_index_base newbase)
{

    if(base != newbase)
    {
        switch(newbase)
        {
        case rocsparse_index_base_one:
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(size_t i = 0; i < size; ++i)
            {
                ++u[i];
            }
            return rocsparse_status_success;
        }

        case rocsparse_index_base_zero:
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(size_t i = 0; i < size; ++i)
            {
                --u[i];
            }

            return rocsparse_status_success;
        }
        }
        return rocsparse_status_invalid_value;
    }

    return rocsparse_status_success;
}

template <typename IMPL>
class rocsparse_importer
{

public:
    template <typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status
        import_sparse_csx(rocsparse_direction* dir, J* m, J* n, I* nnz, rocsparse_index_base* base)
    {
        return static_cast<IMPL&>(*this).import_sparse_csx(dir, m, n, nnz, base);
    }

    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_csx(I* ptr, J* ind, T* val)
    {
        return static_cast<IMPL&>(*this).import_sparse_csx(ptr, ind, val);
    }

    template <typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_gebsx(rocsparse_direction*  dir,
                                         rocsparse_direction*  dirb,
                                         J*                    mb,
                                         J*                    nb,
                                         I*                    nnzb,
                                         J*                    block_dim_row,
                                         J*                    block_dim_column,
                                         rocsparse_index_base* base)
    {
        return static_cast<IMPL&>(*this).import_sparse_gebsx(
            dir, dirb, mb, nb, nnzb, block_dim_row, block_dim_column, base);
    }

    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_gebsx(I* ptr, J* ind, T* val)
    {
        return static_cast<IMPL&>(*this).import_sparse_gebsx(ptr, ind, val);
    }

    rocsparse_status import_dense_vector(size_t* size)
    {
        return static_cast<IMPL&>(*this).import_dense_vector(size);
    }
    template <typename T>
    rocsparse_status import_dense_vector(T* data, size_t incy)
    {
        return static_cast<IMPL&>(*this).import_dense_vector(data, incy);
    }

    template <typename I = rocsparse_int>
    rocsparse_status import_dense_matrix(rocsparse_order* order, I* m, I* n)
    {
        return static_cast<IMPL&>(*this).import_dense_matrix(order, m, n);
    }

    template <typename T, typename I = rocsparse_int>
    rocsparse_status import_dense_matrix(T* data, I ld)
    {
        return static_cast<IMPL&>(*this).import_dense_matrix(data, ld);
    }

    template <typename I = rocsparse_int>
    rocsparse_status import_sparse_coo(I* m, I* n, I* nnz, rocsparse_index_base* base)
    {
        return static_cast<IMPL&>(*this).import_sparse_coo(m, n, nnz, base);
    }

    template <typename T, typename I = rocsparse_int>
    rocsparse_status import_sparse_coo(I* row_ind, I* col_ind, T* val)
    {
        return static_cast<IMPL&>(*this).import_sparse_csx(row_ind, col_ind, val);
    }

public:
    template <rocsparse_direction DIRECTION,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    rocsparse_status import(host_csx_matrix<DIRECTION, T, I, J>& csx_);

    template <typename T, typename I = rocsparse_int>
    rocsparse_status import(host_coo_matrix<T, I>& matrix_);

    template <rocsparse_direction DIRECTION,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    rocsparse_status import(host_gebsx_matrix<DIRECTION, T, I, J>& that_);

    template <typename T, typename I = rocsparse_int>
    inline rocsparse_status import(host_dense_matrix<T, I>& that_);
    template <typename T>
    inline rocsparse_status import(host_dense_vector<T>& that_);
};

#endif

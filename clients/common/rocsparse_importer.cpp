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

#include "rocsparse_importer.hpp"

template <typename IMPL>
template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
rocsparse_status rocsparse_importer<IMPL>::import(host_csx_matrix<DIRECTION, T, I, J>& csx_)
{

    //
    // Define
    //
    {
        J                    M;
        J                    N;
        I                    nnz;
        rocsparse_index_base base;
        rocsparse_direction  dir;
        rocsparse_status     status = this->import_sparse_csx(&dir, &M, &N, &nnz, &base);
        if(status != rocsparse_status_success)
        {
            return status;
        }
        csx_.define(M, N, nnz, base);
    }

    //
    // Import
    //
    {
        rocsparse_status status = this->import_sparse_csx<T, I, J>(csx_.ptr, csx_.ind, csx_.val);
        if(status != rocsparse_status_success)
        {
            return status;
        }
    }
    return rocsparse_status_success;
}

template <typename IMPL>
template <typename T, typename I>
rocsparse_status rocsparse_importer<IMPL>::import(host_coo_matrix<T, I>& matrix_)
{
    //
    // Define.
    //
    {
        I                    M, N, nnz;
        rocsparse_index_base base;
        rocsparse_status     status = this->import_sparse_coo(&M, &N, &nnz, &base);
        if(status != rocsparse_status_success)
        {
            return status;
        }

        matrix_.define(M, N, nnz, base);
    }

    //
    // Import.
    //
    {
        rocsparse_status status
            = this->import_sparse_coo(matrix_.ptr, matrix_.ind, matrix_.val, matrix_.base);
        if(status != rocsparse_status_success)
        {
            return status;
        }
    }

    return rocsparse_status_success;
}

template <typename IMPL>
template <rocsparse_direction DIRECTION, typename T, typename I, typename J>
rocsparse_status rocsparse_importer<IMPL>::import(host_gebsx_matrix<DIRECTION, T, I, J>& that_)
{

    //
    // Define
    //
    {
        J                    Mb;
        J                    Nb;
        I                    nnzb;
        rocsparse_index_base base;
        rocsparse_direction  dir;
        rocsparse_direction  dirb;
        J                    block_dim_row, block_dim_column;
        rocsparse_status     status = this->import_sparse_gebsx(
            &dir, &dirb, &Mb, &Nb, &nnzb, &block_dim_row, &block_dim_column, &base);
        if(status != rocsparse_status_success)
        {
            return status;
        }
        if(dir != DIRECTION)
        {
            std::cerr << "dir != DIRECTION" << std::endl;
            return rocsparse_status_invalid_value;
        }

        that_.define(dirb, Mb, Nb, nnzb, block_dim_row, block_dim_column, base);
    }

    //
    // Import
    //
    {
        rocsparse_status status = this->import_sparse_csx<T, I, J>(that_.ptr, that_.ind, that_.val);
        if(status != rocsparse_status_success)
        {
            return status;
        }
    }
    return rocsparse_status_success;
}

template <typename IMPL>
template <typename T, typename I>
rocsparse_status rocsparse_importer<IMPL>::import(host_dense_matrix<T, I>& that_)
{

    //
    // Define
    //
    {
        rocsparse_order  order;
        I                M;
        I                N;
        rocsparse_status status = this->import_dense_matrix(&order, &M, &N);
        if(status != rocsparse_status_success)
        {
            return status;
        }
        that_.define(M, N, order);
    }

    //
    // Import
    //
    {
        rocsparse_status status = this->import_dense_matrix<T, I>(that_.val, that_.ld);
        if(status != rocsparse_status_success)
        {
            return status;
        }
    }
    return rocsparse_status_success;
}

template <typename IMPL>
template <typename T>
rocsparse_status rocsparse_importer<IMPL>::import(host_dense_vector<T>& that_)
{

    //
    // Define
    //
    {
        size_t           M;
        rocsparse_status status = this->import_dense_vector(&M);
        if(status != rocsparse_status_success)
        {
            return status;
        }
        that_.resize(M);
    }

    //
    // Import
    //
    {
        static constexpr size_t ld     = 1;
        rocsparse_status        status = this->import_dense_vector<T>(that_.data(), ld);
        if(status != rocsparse_status_success)
        {
            return status;
        }
    }
    return rocsparse_status_success;
}

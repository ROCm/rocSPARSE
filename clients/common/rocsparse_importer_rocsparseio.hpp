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

#pragma once
#ifndef ROCSPARSE_IMPORTER_ROCSPARSEIO_HPP
#define ROCSPARSE_IMPORTER_ROCSPARSEIO_HPP

#include "rocsparse_importer.hpp"

#ifdef ROCSPARSEIO
#include "rocsparseio.h"
#endif

class rocsparse_importer_rocsparseio : public rocsparse_importer<rocsparse_importer_rocsparseio>
{
protected:
    std::string m_filename{};
#ifdef ROCSPARSEIO
    rocsparseio_type   m_ptr_type{};
    rocsparseio_type   m_ind_type{};
    rocsparseio_type   m_val_type{};
    rocsparseio_handle m_handle{};
#endif
public:
    ~rocsparse_importer_rocsparseio();
    using IMPL = rocsparse_importer_rocsparseio;
    rocsparse_importer_rocsparseio(const std::string& filename_);

public:
private:
#ifdef ROCSPARSEIO
    rocsparseio_type m_row_ind_type;
    rocsparseio_type m_col_ind_type;
#endif
public:
    template <typename I = rocsparse_int>
    rocsparse_status import_sparse_coo(I* m, I* n, I* nnz, rocsparse_index_base* base);

private:
#ifdef ROCSPARSEIO

    size_t m_m;
    size_t m_nnz;
#endif
public:
    template <typename T, typename I = rocsparse_int>
    rocsparse_status import_sparse_coo(I* row_ind, I* col_ind, T* val);

public:
    template <typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_gebsx(rocsparse_direction*  dir,
                                         rocsparse_direction*  dirb,
                                         J*                    mb,
                                         J*                    nb,
                                         I*                    nnzb,
                                         J*                    block_dim_row,
                                         J*                    block_dim_column,
                                         rocsparse_index_base* base);

private:
#ifdef ROCSPARSEIO

    size_t m_mb{};
    size_t m_nnzb{};
    size_t m_row_block_dim{};
    size_t m_col_block_dim{};
#endif
public:
    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_gebsx(I* ptr, J* ind, T* val);

public:
    template <typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status
        import_sparse_csx(rocsparse_direction* dir, J* m, J* n, I* nnz, rocsparse_index_base* base);

    template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
    rocsparse_status import_sparse_csx(I* ptr, J* ind, T* val);
};

#endif // HEADER

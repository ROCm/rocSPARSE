/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_MATRIX_FACTORY_HPP
#define ROCSPARSE_MATRIX_FACTORY_HPP

#include "rocsparse_import.hpp"
#include "rocsparse_matrix_utils.hpp"

std::string rocsparse_exepath();

#include "rocsparse_matrix_factory_file.hpp"
#include "rocsparse_matrix_factory_laplace2d.hpp"
#include "rocsparse_matrix_factory_laplace3d.hpp"
#include "rocsparse_matrix_factory_random.hpp"
#include "rocsparse_matrix_factory_zero.hpp"

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory : public rocsparse_matrix_factory_base<T, I, J>
{
public:
    const Arguments& m_arg;

private:
    rocsparse_matrix_factory_base<T, I, J>* m_instance;

public:
    ~rocsparse_matrix_factory();
    rocsparse_matrix_factory(const Arguments&      arg,
                             rocsparse_matrix_init matrix,
                             bool                  to_int    = false,
                             bool                  full_rank = false,
                             bool                  noseed    = false);

    rocsparse_matrix_factory(const rocsparse_matrix_factory& that) = delete;
    rocsparse_matrix_factory& operator=(const rocsparse_matrix_factory& that) = delete;
    explicit rocsparse_matrix_factory(const Arguments& arg,
                                      bool             to_int    = false,
                                      bool             full_rank = false,
                                      bool             noseed    = false);

    virtual void init_csr(std::vector<I>&        csr_row_ptr,
                          std::vector<J>&        csr_col_ind,
                          std::vector<T>&        csr_val,
                          J&                     m,
                          J&                     n,
                          I&                     nnz,
                          rocsparse_index_base   base,
                          rocsparse_matrix_type  matrix_type,
                          rocsparse_fill_mode    uplo,
                          rocsparse_storage_mode storage)
    {
        return this->m_instance->init_csr(
            csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, base, matrix_type, uplo, storage);
    }

    virtual void init_gebsr(std::vector<I>&        bsr_row_ptr,
                            std::vector<J>&        bsr_col_ind,
                            std::vector<T>&        bsr_val,
                            rocsparse_direction    dirb,
                            J&                     mb,
                            J&                     nb,
                            I&                     nnzb,
                            J&                     row_block_dim,
                            J&                     col_block_dim,
                            rocsparse_index_base   base,
                            rocsparse_matrix_type  matrix_type,
                            rocsparse_fill_mode    uplo,
                            rocsparse_storage_mode storage)
    {
        return this->m_instance->init_gebsr(bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_val,
                                            dirb,
                                            mb,
                                            nb,
                                            nnzb,
                                            row_block_dim,
                                            col_block_dim,
                                            base,
                                            matrix_type,
                                            uplo,
                                            storage);
    }

    virtual void init_coo(std::vector<I>&        coo_row_ind,
                          std::vector<I>&        coo_col_ind,
                          std::vector<T>&        coo_val,
                          I&                     m,
                          I&                     n,
                          I&                     nnz,
                          rocsparse_index_base   base,
                          rocsparse_matrix_type  matrix_type,
                          rocsparse_fill_mode    uplo,
                          rocsparse_storage_mode storage)
    {
        return this->m_instance->init_coo(
            coo_row_ind, coo_col_ind, coo_val, m, n, nnz, base, matrix_type, uplo, storage);
    }

    //
    // COO
    //
    void init_coo(std::vector<I>&      coo_row_ind,
                  std::vector<I>&      coo_col_ind,
                  std::vector<T>&      coo_val,
                  I&                   m,
                  I&                   n,
                  I&                   nnz,
                  rocsparse_index_base base);

    void init_coo(host_coo_matrix<T, I>& that);
    void init_coo(host_coo_matrix<T, I>& that, I& m, I& n);
    void init_coo(host_coo_matrix<T, I>& that, I& m, I& n, rocsparse_index_base base);

    //
    // CSR
    //
    void init_csr(std::vector<I>&      csr_row_ptr,
                  std::vector<J>&      csr_col_ind,
                  std::vector<T>&      csr_val,
                  J&                   m,
                  J&                   n,
                  I&                   nnz,
                  rocsparse_index_base base);

    void init_csr(host_csr_matrix<T, I, J>& that);
    void init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n);
    void init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n, rocsparse_index_base base);

    //
    // CSC
    //
    void init_csc(std::vector<I>&      csc_col_ptr,
                  std::vector<J>&      csc_row_ind,
                  std::vector<T>&      csc_val,
                  J&                   m,
                  J&                   n,
                  I&                   nnz,
                  rocsparse_index_base base);

    void init_csc(host_csc_matrix<T, I, J>& that, J& m, J& n, rocsparse_index_base base);

    //
    // GEBSR
    //
    void init_gebsr(std::vector<I>&      bsr_row_ptr,
                    std::vector<J>&      bsr_col_ind,
                    std::vector<T>&      bsr_val,
                    rocsparse_direction  dirb,
                    J&                   mb,
                    J&                   nb,
                    I&                   nnzb,
                    J&                   row_block_dim,
                    J&                   col_block_dim,
                    rocsparse_index_base base);

    void init_gebsr(host_gebsr_matrix<T, I, J>& that);

    void init_gebsr(host_gebsr_matrix<T, I, J>& that,
                    J&                          mb,
                    J&                          nb,
                    J&                          row_block_dim,
                    J&                          col_block_dim,
                    rocsparse_index_base        base_);

    void init_gebsr(host_gebsr_matrix<T, I, J>& that,
                    rocsparse_direction         block_dir_,
                    J&                          mb_,
                    J&                          nb_,
                    I&                          nnzb_,
                    J&                          row_block_dim_,
                    J&                          col_block_dim_,
                    rocsparse_index_base        base_);

    void init_gebsr_spezial(host_gebsr_matrix<T, I, J>& that, J& mb, J& nb);

    void init_gebsc(std::vector<I>&      bsc_col_ptr,
                    std::vector<J>&      bsc_row_ind,
                    std::vector<T>&      bsc_val,
                    rocsparse_direction  dirb,
                    J&                   Mb,
                    J&                   Nb,
                    I&                   nnzb,
                    J&                   row_block_dim,
                    J&                   col_block_dim,
                    rocsparse_index_base base);

    //
    // BSR
    //
    void init_bsr(std::vector<I>&      bsr_row_ptr,
                  std::vector<J>&      bsr_col_ind,
                  std::vector<T>&      bsr_val,
                  rocsparse_direction  dirb,
                  J&                   mb,
                  J&                   nb,
                  I&                   nnzb,
                  J&                   block_dim,
                  rocsparse_index_base base);

    void init_bsr(host_gebsr_matrix<T, I, J>& that_, J& mb_, J& nb_, rocsparse_index_base base_);

    void init_bsr(host_gebsr_matrix<T, I, J>&   that,
                  device_gebsr_matrix<T, I, J>& that_on_device,
                  J&                            mb_,
                  J&                            nb_,
                  rocsparse_index_base          base_);

    //
    // COO AOS
    //
    void init_coo_aos(host_coo_aos_matrix<T, I>& that, I& m, I& n, rocsparse_index_base base);

    //
    // ELL
    //
    void init_ell(host_ell_matrix<T, I>& that, I& m, I& n, rocsparse_index_base base);

    //
    // HYBRID
    //
    void init_hyb(
        rocsparse_hyb_mat hyb, I& m, I& n, I& nnz, rocsparse_index_base base, bool& conform);
};

//
// Transform a csr matrix in a general bsr matrix.
// It fills the values such as the conversion to the csr matrix
// will give to 1,2,3,4,5,6,7,8,9, etc...
//
template <typename T>
inline void rocsparse_init_gebsr_matrix_from_csr(rocsparse_matrix_factory<T>& matrix_factory,
                                                 std::vector<rocsparse_int>&  bsr_row_ptr,
                                                 std::vector<rocsparse_int>&  bsr_col_ind,
                                                 std::vector<T>&              bsr_val,
                                                 rocsparse_direction          direction,
                                                 rocsparse_int&               mb,
                                                 rocsparse_int&               nb,
                                                 rocsparse_int                row_block_dim,
                                                 rocsparse_int                col_block_dim,
                                                 rocsparse_int&               nnzb,
                                                 rocsparse_index_base         bsr_base)
{
    // Uncompressed CSR matrix on host
    std::vector<T> hcsr_val_A;

    // Generate uncompressed CSR matrix on host (or read from file)

    matrix_factory.init_csr(bsr_row_ptr, bsr_col_ind, hcsr_val_A, mb, nb, nnzb, bsr_base);

    bsr_val.resize(row_block_dim * col_block_dim * nnzb);
    rocsparse_int idx = 0;

    switch(direction)
    {
    case rocsparse_direction_column:
    {
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            for(rocsparse_int r = 0; r < row_block_dim; ++r)
            {
                for(rocsparse_int k = bsr_row_ptr[i] - bsr_base; k < bsr_row_ptr[i + 1] - bsr_base;
                    ++k)
                {
                    for(rocsparse_int c = 0; c < col_block_dim; ++c)
                    {
                        bsr_val[k * row_block_dim * col_block_dim + c * row_block_dim + r]
                            = static_cast<T>(++idx);
                    }
                }
            }
        }
        break;
    }

    case rocsparse_direction_row:
    {
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            for(rocsparse_int r = 0; r < row_block_dim; ++r)
            {
                for(rocsparse_int k = bsr_row_ptr[i] - bsr_base; k < bsr_row_ptr[i + 1] - bsr_base;
                    ++k)
                {
                    for(rocsparse_int c = 0; c < col_block_dim; ++c)
                    {
                        bsr_val[k * row_block_dim * col_block_dim + r * col_block_dim + c]
                            = static_cast<T>(++idx);
                    }
                }
            }
        }
        break;
    }
    }
}

#endif // ROCSPARSE_MATRIX_FACTORY_HPP

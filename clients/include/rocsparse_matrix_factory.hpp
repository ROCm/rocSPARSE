/*! \file */
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

#pragma once
#ifndef ROCSPARSE_MATRIX_FACTORY_HPP
#define ROCSPARSE_MATRIX_FACTORY_HPP

#include "rocsparse_matrix.hpp"

std::string rocsparse_exepath();

template <typename T>
struct rocsparse_matrix_factory_base
{
protected:
    rocsparse_matrix_factory_base(){};

public:
    virtual ~rocsparse_matrix_factory_base(){};
    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
        = 0;

    virtual void init_gebsr(std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int&              nnzb,
                            rocsparse_int&              row_block_dim,
                            rocsparse_int&              col_block_dim,
                            rocsparse_index_base        base)
        = 0;

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
        = 0;
};

template <typename T>
struct rocsparse_matrix_factory_random : public rocsparse_matrix_factory_base<T>
{
private:
    bool m_fullrank;

public:
    rocsparse_matrix_factory_random(bool fullrank)
        : m_fullrank(fullrank)
    {
        rocsparse_seedrand();
    };

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        // Compute non-zero entries of the matrix
        nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

        // Sample random matrix
        std::vector<rocsparse_int> row_ind(nnz);

        // Sample COO matrix
        rocsparse_init_coo_matrix(row_ind, csr_col_ind, csr_val, M, N, nnz, base, this->m_fullrank);

        // Convert to CSR
        host_coo_to_csr(M, nnz, row_ind, csr_row_ptr, base);
    }

    virtual void init_gebsr(std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int&              nnzb,
                            rocsparse_int&              row_block_dim,
                            rocsparse_int&              col_block_dim,
                            rocsparse_index_base        base)
    {
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        rocsparse_int nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(rocsparse_int i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        // Compute non-zero entries of the matrix
        nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

        // Sample random matrix
        rocsparse_init_coo_matrix(
            coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, this->m_fullrank);
    }
};

template <typename T>
struct rocsparse_matrix_factory_rocalution : public rocsparse_matrix_factory_base<T>
{
private:
    std::string m_filename;
    bool        m_toint;

public:
    rocsparse_matrix_factory_rocalution(const char* filename, bool toint = false)
        : m_filename(filename)
        , m_toint(toint){};

    virtual void init_gebsr(std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int&              nnzb,
                            rocsparse_int&              row_block_dim,
                            rocsparse_int&              col_block_dim,
                            rocsparse_index_base        base)
    {
        //
        // Temporarily the file contains a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //
        rocsparse_int nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(rocsparse_int i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {

        rocsparse_init_csr_rocalution(this->m_filename.c_str(),
                                      csr_row_ptr,
                                      csr_col_ind,
                                      csr_val,
                                      M,
                                      N,
                                      nnz,
                                      base,
                                      this->m_toint);
    }

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_rocalution(this->m_filename.c_str(),
                                      coo_row_ind,
                                      coo_col_ind,
                                      coo_val,
                                      M,
                                      N,
                                      nnz,
                                      base,
                                      this->m_toint);
    }
};

template <typename T>
struct rocsparse_matrix_factory_mtx : public rocsparse_matrix_factory_base<T>
{
private:
    std::string m_filename;

public:
    rocsparse_matrix_factory_mtx(const char* filename)
        : m_filename(filename){};

    virtual void init_gebsr(std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int&              nnzb,
                            rocsparse_int&              row_block_dim,
                            rocsparse_int&              col_block_dim,
                            rocsparse_index_base        base)
    {
        //
        // Temporarily the file contains a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //
        rocsparse_int nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(rocsparse_int i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_csr_mtx(
            this->m_filename.c_str(), csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);
    }

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_mtx(
            this->m_filename.c_str(), coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    }
};

template <typename T>
struct rocsparse_matrix_factory_laplace2d : public rocsparse_matrix_factory_base<T>
{
private:
    rocsparse_int m_dimx, m_dimy;

public:
    rocsparse_matrix_factory_laplace2d(rocsparse_int dimx, rocsparse_int dimy)
        : m_dimx(dimx)
        , m_dimy(dimy){};

    virtual void init_gebsr(std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int&              nnzb,
                            rocsparse_int&              row_block_dim,
                            rocsparse_int&              col_block_dim,
                            rocsparse_index_base        base)
    {
        //
        // Temporarily laplace2d generates a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //
        rocsparse_int nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(rocsparse_int i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_csr_laplace2d(
            csr_row_ptr, csr_col_ind, csr_val, this->m_dimx, this->m_dimy, M, N, nnz, base);
    }

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_laplace2d(
            coo_row_ind, coo_col_ind, coo_val, this->m_dimx, this->m_dimy, M, N, nnz, base);
    }
};

template <typename T>
struct rocsparse_matrix_factory_laplace3d : public rocsparse_matrix_factory_base<T>
{
private:
    rocsparse_int m_dimx, m_dimy, m_dimz;

public:
    rocsparse_matrix_factory_laplace3d(rocsparse_int dimx, rocsparse_int dimy, rocsparse_int dimz)
        : m_dimx(dimx)
        , m_dimy(dimy)
        , m_dimz(dimz){};

    virtual void init_gebsr(std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int&              nnzb,
                            rocsparse_int&              row_block_dim,
                            rocsparse_int&              col_block_dim,
                            rocsparse_index_base        base)
    {
        //
        // Temporarily laplace3d generates a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //
        rocsparse_int nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(rocsparse_int i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_csr_laplace3d(csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     this->m_dimx,
                                     this->m_dimy,
                                     this->m_dimz,
                                     M,
                                     N,
                                     nnz,
                                     base);
    }

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        rocsparse_init_coo_laplace3d(coo_row_ind,
                                     coo_col_ind,
                                     coo_val,
                                     this->m_dimx,
                                     this->m_dimy,
                                     this->m_dimz,
                                     M,
                                     N,
                                     nnz,
                                     base);
    }
};

template <typename T>
struct rocsparse_matrix_factory : public rocsparse_matrix_factory_base<T>
{
private:
    rocsparse_direction  arg_dir;
    rocsparse_int        arg_m;
    rocsparse_int        arg_n;
    rocsparse_int        arg_row_block_dim;
    rocsparse_int        arg_col_block_dim;
    rocsparse_int        arg_block_dim;
    rocsparse_index_base arg_base;

    rocsparse_matrix_factory_base<T>* m_instance;

public:
    virtual ~rocsparse_matrix_factory()
    {
        if(this->m_instance)
        {
            delete this->m_instance;
            this->m_instance = nullptr;
        }
    }

    rocsparse_matrix_factory(const Arguments& arg, bool to_int = false, bool full_rank = false)
        : arg_dir(arg.direction)
        , arg_m(arg.M)
        , arg_n(arg.N)
        , arg_row_block_dim(arg.col_block_dimA)
        , arg_col_block_dim(arg.row_block_dimA)
        , arg_block_dim(arg.block_dim)
        , arg_base(arg.baseA)
    {
        //
        // FORCE REINIT.
        //
        rocsparse_seedrand();

        switch(arg.matrix)
        {
        case rocsparse_matrix_random:
        {
            this->m_instance = new rocsparse_matrix_factory_random<T>(full_rank);
            break;
        }

        case rocsparse_matrix_laplace_2d:
        {
            this->m_instance = new rocsparse_matrix_factory_laplace2d<T>(arg.dimx, arg.dimy);
            break;
        }

        case rocsparse_matrix_laplace_3d:
        {
            this->m_instance
                = new rocsparse_matrix_factory_laplace3d<T>(arg.dimx, arg.dimy, arg.dimz);
            break;
        }

        case rocsparse_matrix_file_rocalution:
        {
            std::string filename
                = arg.timing ? arg.filename
                             : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";
            this->m_instance = new rocsparse_matrix_factory_rocalution<T>(filename.c_str(), to_int);
            break;
        }

        case rocsparse_matrix_file_mtx:
        {
            std::string filename
                = arg.timing ? arg.filename
                             : rocsparse_exepath() + "../matrices/" + arg.filename + ".mtx";
            this->m_instance = new rocsparse_matrix_factory_mtx<T>(filename.c_str());
            break;
        }

        default:
        {
            this->m_instance = nullptr;
            break;
        }
        }
        assert(this->m_instance != nullptr);
    }

    virtual void init_csr(std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind,
                          std::vector<T>&             csr_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        this->m_instance->init_csr(csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);
    }

    void init_csr(host_csr_matrix<T>& that)
    {
        that.base = this->arg_base;
        that.m    = this->arg_m;
        that.n    = this->arg_n;
        this->m_instance->init_csr(
            that.ptr, that.ind, that.val, that.m, that.n, that.nnz, that.base);
    }

    virtual void init_gebsr(std::vector<rocsparse_int>& bsr_row_ptr,
                            std::vector<rocsparse_int>& bsr_col_ind,
                            std::vector<T>&             bsr_val,
                            rocsparse_int&              Mb,
                            rocsparse_int&              Nb,
                            rocsparse_int&              nnzb,
                            rocsparse_int&              row_block_dim,
                            rocsparse_int&              col_block_dim,
                            rocsparse_index_base        base)
    {
        this->m_instance->init_gebsr(
            bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, row_block_dim, col_block_dim, base);
    }

    void init_gebsr(host_gebsr_matrix<T>& that)
    {
        that.base          = this->arg_base;
        that.mb            = this->arg_m;
        that.nb            = this->arg_n;
        that.row_block_dim = this->arg_row_block_dim;
        that.col_block_dim = this->arg_col_block_dim;
        this->m_instance->init_gebsr(that.ptr,
                                     that.ind,
                                     that.val,
                                     that.mb,
                                     that.nb,
                                     that.nnzb,
                                     that.row_block_dim,
                                     that.col_block_dim,
                                     that.base);
    }

    virtual void init_coo(std::vector<rocsparse_int>& coo_row_ind,
                          std::vector<rocsparse_int>& coo_col_ind,
                          std::vector<T>&             coo_val,
                          rocsparse_int&              M,
                          rocsparse_int&              N,
                          rocsparse_int&              nnz,
                          rocsparse_index_base        base)
    {
        this->m_instance->init_coo(coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    }

    void init_coo(host_coo_matrix<T>& that)
    {
        that.base = this->arg.base;
        that.m    = this->arg.m;
        that.n    = this->arg.n;
        this->m_instance->init_coo(
            that.row_ind, that.col_ind, that.val, that.m, that.n, that.nnz, that.base);
    }
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
                                                 rocsparse_int&               Mb,
                                                 rocsparse_int&               Nb,
                                                 rocsparse_int                row_block_dim,
                                                 rocsparse_int                col_block_dim,
                                                 rocsparse_int&               nnzb,
                                                 rocsparse_index_base         bsr_base)
{
    // Uncompressed CSR matrix on host
    std::vector<T> hcsr_val_A;

    // Generate uncompressed CSR matrix on host (or read from file)
    matrix_factory.init_csr(bsr_row_ptr, bsr_col_ind, hcsr_val_A, Mb, Nb, nnzb, bsr_base);

    bsr_val.resize(row_block_dim * col_block_dim * nnzb);
    rocsparse_int idx = 0;
    switch(direction)
    {
    case rocsparse_direction_column:
    {
        for(rocsparse_int i = 0; i < Mb; ++i)
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
        for(rocsparse_int i = 0; i < Mb; ++i)
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

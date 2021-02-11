/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse.hpp"
#include "rocsparse_matrix.hpp"

std::string rocsparse_exepath();

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_base
{
protected:
    rocsparse_matrix_factory_base(){};

public:
    virtual ~rocsparse_matrix_factory_base(){};

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
        = 0;

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
        = 0;

    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
        = 0;
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_random : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    bool   m_fullrank;
    bool   m_to_int;
    double m_percentage;

public:
    rocsparse_matrix_factory_random(bool   fullrank,
                                    bool   to_int     = false,
                                    double percentage = static_cast<double>(0))
        : m_fullrank(fullrank)
        , m_to_int(to_int)
        , m_percentage(percentage){};

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        // Compute non-zero entries of the matrix
        if(m_percentage > 0)
        {
            J s = ((N < 96) ? N : 96) * m_percentage;
            nnz = s * M;
            csr_row_ptr.resize(M + 1);
            csr_col_ind.resize(nnz);
            csr_val.resize(nnz);
            csr_row_ptr[0] = 0;
            nnz            = 0;
            for(J i = 0; i < M; ++i)
            {

                for(J j = i - s / 2; j < i + s / 2; ++j)
                {
                    if(j >= 0 && j < N)
                    {
                        csr_col_ind[nnz] = j;
                        if(this->m_to_int)
                        {
                            csr_val[nnz] = random_generator_exact<T>();
                        }
                        else
                        {
                            csr_val[nnz] = random_generator<T>();
                        }
                        ++nnz;
                    }
                }
                csr_row_ptr[i + 1] = nnz;
            }

            csr_col_ind.resize(nnz);
            csr_val.resize(nnz);

            if(base == rocsparse_index_base_one)
            {
                for(int i = 0; i <= M; ++i)
                {
                    csr_row_ptr[i] += 1;
                }
                for(int i = 0; i < nnz; ++i)
                {
                    csr_col_ind[i] += 1;
                }
            }
        }
        else
        {
            if(M < 32 && N < 32)
            {
                nnz = (M * N) / 4;
                if(this->m_fullrank)
                {
                    nnz = std::max(nnz, static_cast<I>(M));
                }
                nnz = std::max(nnz, static_cast<I>(M));
                nnz = std::min(nnz, static_cast<I>(M) * static_cast<I>(N));
            }
            else
            {
                nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;
            }

            // Sample random matrix
            std::vector<J> row_ind(nnz);
            // Sample COO matrix
            rocsparse_init_coo_matrix<J>(
                row_ind, csr_col_ind, csr_val, M, N, nnz, base, this->m_fullrank, this->m_to_int);

            // Convert to CSR
            host_coo_to_csr(M, row_ind, csr_row_ptr, base);
        }
    }

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        I nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        if(this->m_to_int)
        {
            for(I i = 0; i < nvalues; ++i)
            {
                bsr_val[i] = random_generator_exact<T>();
            }
        }
        else
        {
            for(I i = 0; i < nvalues; ++i)
            {
                bsr_val[i] = random_generator<T>();
            }
        }
    }

    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        // Compute non-zero entries of the matrix
        if(M < 32 && N < 32)
        {
            nnz = (M * N) / 4;
        }
        else
        {
            nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;
        }
        // Sample random matrix

        rocsparse_init_coo_matrix(
            coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, this->m_fullrank, this->m_to_int);
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_rocalution : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    std::string m_filename;
    bool        m_toint;

public:
    rocsparse_matrix_factory_rocalution(const char* filename, bool toint = false)
        : m_filename(filename)
        , m_toint(toint){};

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        //
        // Temporarily the file contains a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //

        I nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(I i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
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

    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
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

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_mtx : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    std::string m_filename;

public:
    rocsparse_matrix_factory_mtx(const char* filename)
        : m_filename(filename){};

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        //
        // Temporarily the file contains a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //
        I nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(I i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        rocsparse_init_csr_mtx(
            this->m_filename.c_str(), csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);
    }

    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        rocsparse_init_coo_mtx(
            this->m_filename.c_str(), coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_laplace2d : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    J m_dimx, m_dimy;

public:
    rocsparse_matrix_factory_laplace2d(J dimx, J dimy)
        : m_dimx(dimx)
        , m_dimy(dimy){};

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        //
        // Temporarily laplace2d generates a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //
        I nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(I i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        rocsparse_init_csr_laplace2d(
            csr_row_ptr, csr_col_ind, csr_val, this->m_dimx, this->m_dimy, M, N, nnz, base);
    }

    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        rocsparse_init_coo_laplace2d(
            coo_row_ind, coo_col_ind, coo_val, this->m_dimx, this->m_dimy, M, N, nnz, base);
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_laplace3d : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    J m_dimx, m_dimy, m_dimz;

public:
    rocsparse_matrix_factory_laplace3d(J dimx, J dimy, J dimz)
        : m_dimx(dimx)
        , m_dimy(dimy)
        , m_dimz(dimz){};

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        //
        // Temporarily laplace3d generates a CSR matrix.
        //
        this->init_csr(bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, base);

        //
        // Then temporarily skip the values.
        //
        I nvalues = nnzb * row_block_dim * col_block_dim;
        bsr_val.resize(nvalues);
        for(I i = 0; i < nvalues; ++i)
        {
            bsr_val[i] = random_generator<T>();
        }
    }

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
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

    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
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

struct rocsparse_matrix_utils
{

    template <typename T>
    static void init(host_dense_matrix<T>& that)
    {
        rocsparse_init(that.val, that.m, that.n, that.ld);
    }

    template <typename T>
    static void init_exact(host_dense_matrix<T>& that)
    {
        rocsparse_init_exact(that.val, that.m, that.n, that.ld);
    }

    template <typename T>
    static void compress(device_csr_matrix<T>&       c,
                         const device_csr_matrix<T>& that,
                         rocsparse_index_base        base)
    {
        c.define(that.m, that.n, 0, base);
        rocsparse_handle handle;
        rocsparse_create_handle(&handle);
        rocsparse_mat_descr descr;
        rocsparse_create_mat_descr(&descr);
        rocsparse_set_mat_index_base(descr, that.base);
        T                            tol   = static_cast<T>(0);
        rocsparse_int                nnz_c = 0;
        device_vector<rocsparse_int> dnnz_per_row(that.m);
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, that.m, descr, that.val, that.ptr, dnnz_per_row, &nnz_c, tol));
        c.define(that.m, that.n, nnz_c, base);
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                            that.m,
                                                            that.n,
                                                            descr,
                                                            that.val,
                                                            that.ptr,
                                                            that.ind,
                                                            that.nnz,
                                                            dnnz_per_row,
                                                            c.val,
                                                            c.ptr,
                                                            c.ind,
                                                            tol));
        rocsparse_destroy_mat_descr(descr);
        rocsparse_destroy_handle(handle);
    }

    template <typename T>
    static void convert(const device_csr_matrix<T>& that,
                        rocsparse_direction         dirb,
                        rocsparse_int               row_block_dim,
                        rocsparse_int               col_block_dim,
                        rocsparse_index_base        base,
                        device_gebsr_matrix<T>&     c)
    {

        rocsparse_int mb = (that.m + row_block_dim - 1) / row_block_dim;
        rocsparse_int nb = (that.n + col_block_dim - 1) / col_block_dim;

        rocsparse_int nnzb = 0;

        c.define(dirb, mb, nb, nnzb, row_block_dim, col_block_dim, base);

        rocsparse_handle handle;
        rocsparse_create_handle(&handle);

        rocsparse_mat_descr that_descr;
        rocsparse_create_mat_descr(&that_descr);
        rocsparse_set_mat_index_base(that_descr, that.base);

        rocsparse_mat_descr c_descr;
        rocsparse_create_mat_descr(&c_descr);
        rocsparse_set_mat_index_base(c_descr, base);

        // Convert CSR to GEBSR
        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_buffer_size<T>(handle,
                                                                 c.block_direction,
                                                                 that.m,
                                                                 that.n,
                                                                 that_descr,
                                                                 that.val,
                                                                 that.ptr,
                                                                 that.ind,
                                                                 c.row_block_dim,
                                                                 c.col_block_dim,
                                                                 &buffer_size));

        int* buffer = NULL;
        hipMalloc(&buffer, buffer_size);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                      c.block_direction,
                                                      that.m,
                                                      that.n,
                                                      that_descr,
                                                      that.ptr,
                                                      that.ind,
                                                      c_descr,
                                                      c.ptr,
                                                      c.row_block_dim,
                                                      c.col_block_dim,
                                                      &nnzb,
                                                      buffer));

        c.define(dirb, mb, nb, nnzb, row_block_dim, col_block_dim, base);

        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr<T>(handle,
                                                     c.block_direction,
                                                     that.m,
                                                     that.n,
                                                     that_descr,
                                                     that.val,
                                                     that.ptr,
                                                     that.ind,
                                                     c_descr,
                                                     c.val,
                                                     c.ptr,
                                                     c.ind,
                                                     c.row_block_dim,
                                                     c.col_block_dim,
                                                     buffer));

        hipFree(buffer);
        rocsparse_destroy_mat_descr(c_descr);
        rocsparse_destroy_mat_descr(that_descr);
        rocsparse_destroy_handle(handle);
    }

    template <typename T>
    static void convert(const device_csr_matrix<T>& that,
                        rocsparse_direction         dirb,
                        rocsparse_int               block_dim,
                        rocsparse_index_base        base,
                        device_gebsr_matrix<T>&     c)
    {
        rocsparse_int mb = (that.m + block_dim - 1) / block_dim;
        rocsparse_int nb = (that.n + block_dim - 1) / block_dim;

        rocsparse_int nnzb = 0;

        c.define(dirb, mb, nb, nnzb, block_dim, block_dim, base);

        {

            rocsparse_handle handle;
            rocsparse_create_handle(&handle);

            rocsparse_mat_descr that_descr;
            rocsparse_create_mat_descr(&that_descr);
            rocsparse_set_mat_index_base(that_descr, that.base);

            rocsparse_mat_descr c_descr;
            rocsparse_create_mat_descr(&c_descr);
            rocsparse_set_mat_index_base(c_descr, base);
            // Convert sample CSR matrix to bsr
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                        dirb,
                                                        that.m,
                                                        that.n,
                                                        that_descr,
                                                        that.ptr,
                                                        that.ind,
                                                        block_dim,
                                                        c_descr,
                                                        c.ptr,
                                                        &nnzb));

            c.define(dirb, mb, nb, nnzb, block_dim, block_dim, base);

            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                       dirb,
                                                       that.m,
                                                       that.n,
                                                       that_descr,
                                                       that.val,
                                                       that.ptr,
                                                       that.ind,
                                                       block_dim,
                                                       c_descr,
                                                       c.val,
                                                       c.ptr,
                                                       c.ind));
            rocsparse_destroy_mat_descr(c_descr);
            rocsparse_destroy_mat_descr(that_descr);
            rocsparse_destroy_handle(handle);
        }
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    rocsparse_direction  arg_dir;
    J                    arg_m;
    J                    arg_n;
    J                    arg_row_block_dim;
    J                    arg_col_block_dim;
    J                    arg_block_dim;
    rocsparse_index_base arg_base;

    rocsparse_matrix_factory_base<T, I, J>* m_instance;

public:
    virtual ~rocsparse_matrix_factory()
    {
        if(this->m_instance)
        {
            delete this->m_instance;
            this->m_instance = nullptr;
        }
    }

    rocsparse_matrix_factory(const Arguments&      arg,
                             rocsparse_matrix_init matrix,
                             bool                  to_int    = false,
                             bool                  full_rank = false,
                             bool                  noseed    = false)
        : arg_dir(arg.direction)
        , arg_m(arg.M)
        , arg_n(arg.N)
        , arg_row_block_dim(arg.row_block_dimA)
        , arg_col_block_dim(arg.col_block_dimA)
        , arg_block_dim(arg.block_dim)
        , arg_base(arg.baseA)
    {
        //
        // FORCE REINIT.
        //
        if(false == noseed)
        {
            rocsparse_seedrand();
        }

        switch(matrix)
        {
        case rocsparse_matrix_random:
        {
            double percentage = static_cast<double>(arg.percentage);
            this->m_instance
                = new rocsparse_matrix_factory_random<T, I, J>(full_rank, to_int, percentage);
            break;
        }

        case rocsparse_matrix_laplace_2d:
        {
            this->m_instance = new rocsparse_matrix_factory_laplace2d<T, I, J>(arg.dimx, arg.dimy);
            break;
        }

        case rocsparse_matrix_laplace_3d:
        {
            this->m_instance
                = new rocsparse_matrix_factory_laplace3d<T, I, J>(arg.dimx, arg.dimy, arg.dimz);
            break;
        }

        case rocsparse_matrix_file_rocalution:
        {
            std::string filename
                = arg.timing ? arg.filename
                             : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";
            this->m_instance
                = new rocsparse_matrix_factory_rocalution<T, I, J>(filename.c_str(), to_int);
            break;
        }

        case rocsparse_matrix_file_mtx:
        {
            std::string filename
                = arg.timing ? arg.filename
                             : rocsparse_exepath() + "../matrices/" + arg.filename + ".mtx";
            this->m_instance = new rocsparse_matrix_factory_mtx<T, I, J>(filename.c_str());
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

    rocsparse_matrix_factory(const Arguments& arg,
                             bool             to_int    = false,
                             bool             full_rank = false,
                             bool             noseed    = false)
        : rocsparse_matrix_factory(arg, arg.matrix, to_int, full_rank, noseed)
    {
    }

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        this->m_instance->init_csr(csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);
    }

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        this->m_instance->init_gebsr(
            bsr_row_ptr, bsr_col_ind, bsr_val, Mb, Nb, nnzb, row_block_dim, col_block_dim, base);
    }
    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        this->m_instance->init_coo(coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    }

    void init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n, rocsparse_index_base base)
    {
        that.base = base;
        that.m    = m;
        that.n    = n;
        this->m_instance->init_csr(
            that.ptr, that.ind, that.val, that.m, that.n, that.nnz, that.base);
        m = that.m;
        n = that.n;
    }

    void init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n)
    {
        this->init_csr(that, m, n, this->arg_base);
    }

    void init_csr(host_csr_matrix<T, I, J>& that)
    {
        that.base = this->arg_base;
        that.m    = this->arg_m;
        that.n    = this->arg_n;
        this->m_instance->init_csr(
            that.ptr, that.ind, that.val, that.m, that.n, that.nnz, that.base);
    }

    void init_bsr(host_gebsr_matrix<T, I, J>&   that,
                  device_gebsr_matrix<T, I, J>& that_on_device,
                  J&                            mb_,
                  J&                            nb_)
    {

        //
        // Initialize in case init_csr requires it as input.
        //
        J block_dim = this->arg_block_dim;

        J                        M    = mb_ * block_dim;
        J                        N    = nb_ * block_dim;
        rocsparse_index_base     base = this->arg_base;
        host_csr_matrix<T, I, J> hA_uncompressed;
        this->init_csr(hA_uncompressed, M, N, base);

        {
            device_csr_matrix<T, I, J> dA_uncompressed(hA_uncompressed);
            device_csr_matrix<T, I, J> dA_compressed;
            rocsparse_matrix_utils::compress(dA_compressed, dA_uncompressed, this->arg_base);
            rocsparse_matrix_utils::convert(
                dA_compressed, this->arg_dir, block_dim, base, that_on_device);
        }

        that(that_on_device);

        mb_ = that.mb;
        nb_ = that.nb;
    }

    void init_gebsr(host_gebsr_matrix<T, I, J>&   that,
                    device_gebsr_matrix<T, I, J>& that_on_device,
                    J&                            mb_,
                    J&                            nb_)
    {

        //
        // Initialize in case init_csr requires it as input.
        //
        J row_block_dim = this->arg_row_block_dim;
        J col_block_dim = this->arg_col_block_dim;

        J                        M    = mb_ * row_block_dim;
        J                        N    = nb_ * col_block_dim;
        rocsparse_index_base     base = this->arg_base;
        host_csr_matrix<T, I, J> hA_uncompressed;
        this->init_csr(hA_uncompressed, M, N, base);
#if 0
      {
	device_csr_matrix<T, I, J> dA_uncompressed(hA_uncompressed);
	device_csr_matrix<T, I, J> dA_compressed;
	rocsparse_matrix_utils::compress(dA_compressed, dA_uncompressed, this->arg_base);
	rocsparse_matrix_utils::convert(dA_compressed, this->arg_dir, row_block_dim, col_block_dim, base, that_on_device);
      }
#endif
        {
            device_csr_matrix<T, I, J> dA_uncompressed(hA_uncompressed);
            rocsparse_matrix_utils::convert(
                dA_uncompressed, this->arg_dir, row_block_dim, col_block_dim, base, that_on_device);
        }

        that(that_on_device);

        mb_ = that.mb;
        nb_ = that.nb;
    }

    void init_bsr(host_gebsr_matrix<T, I, J>& that, J& mb_, J& nb_)
    {
        device_gebsr_matrix<T, I, J> dB;
        init_bsr(that, dB, mb_, nb_);
    }

    void init_gebsr(host_gebsr_matrix<T, I, J>& that,
                    rocsparse_direction         block_dir_,
                    J&                          mb_,
                    J&                          nb_,
                    I&                          nnzb_,
                    J&                          row_block_dim_,
                    J&                          col_block_dim_,
                    rocsparse_index_base        base_)
    {
        that.block_direction = block_dir_;
        that.mb              = mb_;
        that.nb              = nb_;
        that.row_block_dim   = row_block_dim_;
        that.col_block_dim   = col_block_dim_;
        that.base            = base_;
        that.nnzb            = nnzb_;
        this->m_instance->init_gebsr(that.ptr,
                                     that.ind,
                                     that.val,
                                     that.mb,
                                     that.nb,
                                     that.nnzb,
                                     that.row_block_dim,
                                     that.col_block_dim,
                                     that.base);

        mb_            = that.mb;
        nb_            = that.nb;
        nnzb_          = that.nnzb;
        row_block_dim_ = that.row_block_dim;
        col_block_dim_ = that.col_block_dim;
    }

    void init_gebsr(host_gebsr_matrix<T, I, J>& that)
    {
        that.block_direction = this->arg_dir;
        that.mb              = this->arg_m;
        that.nb              = this->arg_n;
        that.row_block_dim   = this->arg_row_block_dim;
        that.col_block_dim   = this->arg_col_block_dim;
        that.base            = this->arg_base;
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

    void init_gebsr(host_gebsr_matrix<T, I, J>& that, J M, J N, J row_block_dim, J col_block_dim)
    {

        that.base          = this->arg_base;
        that.mb            = M;
        that.nb            = N;
        that.row_block_dim = row_block_dim;
        that.col_block_dim = col_block_dim;
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

    void init_gebsr_spezial(host_gebsr_matrix<T, I, J>& that, J& Mb, J& Nb)
    {
        I idx = 0;

        host_csr_matrix<T, I, J> hA;
        rocsparse_direction      direction     = this->arg_dir;
        rocsparse_index_base     base          = this->arg_base;
        J                        row_block_dim = this->arg_row_block_dim;
        J                        col_block_dim = this->arg_col_block_dim;
        this->init_csr(hA, Mb, Nb, base);

        that.define(direction, Mb, Nb, hA.nnz, row_block_dim, col_block_dim, base);

        switch(direction)
        {
        case rocsparse_direction_column:
        {
            for(J i = 0; i < Mb; ++i)
            {
                for(J r = 0; r < row_block_dim; ++r)
                {
                    for(I k = hA.ptr[i] - base; k < hA.ptr[i + 1] - base; ++k)
                    {
                        for(J c = 0; c < col_block_dim; ++c)
                        {
                            that.val[k * row_block_dim * col_block_dim + c * row_block_dim + r]
                                = static_cast<T>(++idx);
                        }
                    }
                }
            }
            break;
        }

        case rocsparse_direction_row:
        {
            for(J i = 0; i < Mb; ++i)
            {
                for(J r = 0; r < row_block_dim; ++r)
                {
                    for(I k = hA.ptr[i] - base; k < hA.ptr[i + 1] - base; ++k)
                    {
                        for(J c = 0; c < col_block_dim; ++c)
                        {
                            that.val[k * row_block_dim * col_block_dim + r * col_block_dim + c]
                                = static_cast<T>(++idx);
                        }
                    }
                }
            }
            break;
        }
        }

        that.ptr.transfer_from(hA.ptr);
        that.ind.transfer_from(hA.ind);
    }

    void init_coo(host_coo_matrix<T, I, J>& that)
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

/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_MATRIX_HPP
#define ROCSPARSE_MATRIX_HPP

#include "rocsparse_vector.hpp"

template <typename T>
struct device_gebsc_matrix;
template <typename T>
struct device_gebsr_matrix;
template <typename T>
struct device_gebsx_matrix;

template <typename T>
struct device_coo_matrix;
template <typename T>
struct device_csc_matrix;
template <typename T>
struct device_csr_matrix;
template <typename T>
struct device_csx_matrix;

template <typename T>
struct host_coo_matrix;
template <typename T>
struct host_csx_matrix;
template <typename T>
struct host_csr_matrix;
template <typename T>
struct host_csc_matrix;

template <typename T>
struct host_gebsx_matrix;
template <typename T>
struct host_gebsc_matrix;
template <typename T>
struct host_gebsr_matrix;

template <typename IMPL>
struct coo_matrix_traits;
template <typename IMPL>
struct csx_matrix_traits;
template <typename IMPL>
struct gebsx_matrix_traits;

template <typename T>
struct coo_matrix_traits<device_coo_matrix<T>>
{
    template <typename S>
    using array_t = device_vector<S>;
};

template <typename T>
struct csx_matrix_traits<device_csx_matrix<T>>
{
    template <typename S>
    using array_t = device_vector<S>;
};

template <typename T>
struct gebsx_matrix_traits<device_gebsx_matrix<T>>
{
    template <typename S>
    using array_t = device_vector<S>;
};

template <typename T>
struct coo_matrix_traits<host_coo_matrix<T>>
{
    template <typename S>
    using array_t = host_vector<S>;
};

template <typename T>
struct csx_matrix_traits<host_csx_matrix<T>>
{
    template <typename S>
    using array_t = host_vector<S>;
};

template <typename T>
struct gebsx_matrix_traits<host_gebsx_matrix<T>>
{
    template <typename S>
    using array_t = host_vector<S>;
};

template <typename T, typename IMPL>
struct coo_matrix
{
    template <typename S>
    using array_t = typename coo_matrix_traits<IMPL>::template array_t<S>;

    rocsparse_int          m{};
    rocsparse_int          n{};
    rocsparse_int          nnz{};
    rocsparse_index_base   base{};
    array_t<rocsparse_int> row_ind;
    array_t<rocsparse_int> col_ind;
    array_t<T>             val;
    coo_matrix();
    coo_matrix(rocsparse_int m_, rocsparse_int n_, rocsparse_int nnz_, rocsparse_index_base base_);
};

template <typename T>
struct device_coo_matrix : public coo_matrix<T, device_coo_matrix<T>>
{
    device_coo_matrix();
    device_coo_matrix(rocsparse_int        m_,
                      rocsparse_int        n_,
                      rocsparse_int        nnz_,
                      rocsparse_index_base base_);
};

template <typename T>
struct host_coo_matrix : public coo_matrix<T, host_coo_matrix<T>>
{
    host_coo_matrix();
    host_coo_matrix(rocsparse_int        m_,
                    rocsparse_int        n_,
                    rocsparse_int        nnz_,
                    rocsparse_index_base base_);
};

template <typename T, typename IMPL>
struct csx_matrix
{
    template <typename S>
    using array_t = typename csx_matrix_traits<IMPL>::template array_t<S>;

    rocsparse_direction    dir{};
    rocsparse_int          m{};
    rocsparse_int          n{};
    rocsparse_int          nnz{};
    rocsparse_index_base   base{};
    array_t<rocsparse_int> ptr;
    array_t<rocsparse_int> ind;
    array_t<T>             val;
    csx_matrix();
    csx_matrix(rocsparse_direction  dir_,
               rocsparse_int        m_,
               rocsparse_int        n_,
               rocsparse_int        nnz_,
               rocsparse_index_base base_);
};

template <typename T>
struct device_csx_matrix : public csx_matrix<T, device_csx_matrix<T>>
{
    device_csx_matrix();
    device_csx_matrix(rocsparse_direction  dir_,
                      rocsparse_int        m_,
                      rocsparse_int        n_,
                      rocsparse_int        nnz_,
                      rocsparse_index_base base_);
};

template <typename T>
struct host_csx_matrix : public csx_matrix<T, host_csx_matrix<T>>
{
    host_csx_matrix();
    host_csx_matrix(rocsparse_direction  dir_,
                    rocsparse_int        m_,
                    rocsparse_int        n_,
                    rocsparse_int        nnz_,
                    rocsparse_index_base base_);
};

template <typename T>
struct host_csr_matrix : public host_csx_matrix<T>
{
    host_csr_matrix();
    host_csr_matrix(rocsparse_int        m_,
                    rocsparse_int        n_,
                    rocsparse_int        nnz_,
                    rocsparse_index_base base_);
};

template <typename T>
struct host_csc_matrix : public host_csx_matrix<T>
{
    host_csc_matrix();
    host_csc_matrix(rocsparse_int        m_,
                    rocsparse_int        n_,
                    rocsparse_int        nnz_,
                    rocsparse_index_base base_);
};

template <typename T>
struct device_csc_matrix : public device_csx_matrix<T>
{
    device_csc_matrix();
    device_csc_matrix(rocsparse_int        m_,
                      rocsparse_int        n_,
                      rocsparse_int        nnz_,
                      rocsparse_index_base base_);
};

template <typename T, typename IMPL>
struct gebsx_matrix
{
    template <typename S>
    using array_t = typename gebsx_matrix_traits<IMPL>::template array_t<S>;
    rocsparse_direction    dir{};
    rocsparse_int          mb{};
    rocsparse_int          nb{};
    rocsparse_int          nnzb{};
    rocsparse_direction    block_direction{};
    rocsparse_int          row_block_dim{};
    rocsparse_int          col_block_dim{};
    rocsparse_index_base   base{};
    array_t<rocsparse_int> ptr{};
    array_t<rocsparse_int> ind{};
    array_t<T>             val{};
    gebsx_matrix();
    gebsx_matrix(rocsparse_direction  dir_,
                 rocsparse_direction  block_dir_,
                 rocsparse_int        mb_,
                 rocsparse_int        nb_,
                 rocsparse_int        nnzb_,
                 rocsparse_int        row_block_dim_,
                 rocsparse_int        col_block_dim_,
                 rocsparse_index_base base_);
};

template <typename T>
struct device_gebsx_matrix : public gebsx_matrix<T, device_gebsx_matrix<T>>
{
    device_gebsx_matrix();
    device_gebsx_matrix(rocsparse_direction  dir_,
                        rocsparse_direction  dirb_,
                        rocsparse_int        mb_,
                        rocsparse_int        nb_,
                        rocsparse_int        nnzb_,
                        rocsparse_int        row_block_dim_,
                        rocsparse_int        col_block_dim_,
                        rocsparse_index_base base_);
};

template <typename T>
struct host_gebsx_matrix : public gebsx_matrix<T, host_gebsx_matrix<T>>
{
    host_gebsx_matrix();
    host_gebsx_matrix(rocsparse_direction  dir_,
                      rocsparse_direction  dirb_,
                      rocsparse_int        mb_,
                      rocsparse_int        nb_,
                      rocsparse_int        nnzb_,
                      rocsparse_int        row_block_dim_,
                      rocsparse_int        col_block_dim_,
                      rocsparse_index_base base_);
};

template <typename T>
struct device_gebsr_matrix : public device_gebsx_matrix<T>
{
    device_gebsr_matrix();
    device_gebsr_matrix(const host_gebsr_matrix<T>& that_, bool transfer = true);
    device_gebsr_matrix(rocsparse_direction  dirb_,
                        rocsparse_int        mb_,
                        rocsparse_int        nb_,
                        rocsparse_int        nnzb_,
                        rocsparse_int        row_block_dim_,
                        rocsparse_int        col_block_dim_,
                        rocsparse_index_base base_);
    void transfer_from(const host_gebsr_matrix<T>& that);
};

template <typename T>
struct host_gebsr_matrix : public host_gebsx_matrix<T>
{
    host_gebsr_matrix();
    host_gebsr_matrix(const device_gebsr_matrix<T>& that_, bool transfer = true);
    host_gebsr_matrix(rocsparse_direction  dirb_,
                      rocsparse_int        mb_,
                      rocsparse_int        nb_,
                      rocsparse_int        nnzb_,
                      rocsparse_int        row_block_dim_,
                      rocsparse_int        col_block_dim_,
                      rocsparse_index_base base_);

    void transfer_from(const device_gebsr_matrix<T>& that);
};

template <typename T>
struct device_gebsc_matrix : public device_gebsx_matrix<T>
{
    device_gebsc_matrix();
    device_gebsc_matrix(const host_gebsc_matrix<T>& that_, bool transfer = true);
    device_gebsc_matrix(rocsparse_direction  dirb_,
                        rocsparse_int        mb_,
                        rocsparse_int        nb_,
                        rocsparse_int        nnzb_,
                        rocsparse_int        row_block_dim_,
                        rocsparse_int        col_block_dim_,
                        rocsparse_index_base base_);
    void transfer_from(const host_gebsc_matrix<T>& that);
};

template <typename T>
struct host_gebsc_matrix : public host_gebsx_matrix<T>
{
    host_gebsc_matrix();
    host_gebsc_matrix(const device_gebsc_matrix<T>& that_, bool transfer = true);
    host_gebsc_matrix(rocsparse_direction  dirb_,
                      rocsparse_int        mb_,
                      rocsparse_int        nb_,
                      rocsparse_int        nnzb_,
                      rocsparse_int        row_block_dim_,
                      rocsparse_int        col_block_dim_,
                      rocsparse_index_base base_);

    void transfer_from(const device_gebsc_matrix<T>& that);
    void unit_check(host_gebsc_matrix<T>& that, bool check_values = true);
};

//////////////////// IMPLEMENTATIONS

template <typename T>
device_gebsr_matrix<T>::device_gebsr_matrix(rocsparse_direction  dirb_,
                                            rocsparse_int        mb_,
                                            rocsparse_int        nb_,
                                            rocsparse_int        nnzb_,
                                            rocsparse_int        row_block_dim_,
                                            rocsparse_int        col_block_dim_,
                                            rocsparse_index_base base_)
    : device_gebsx_matrix<T>(
        rocsparse_direction_row, dirb_, mb_, nb_, nnzb_, row_block_dim_, col_block_dim_, base_){};

template <typename T>
device_gebsr_matrix<T>::device_gebsr_matrix(){};

template <typename T>
device_gebsr_matrix<T>::device_gebsr_matrix(const host_gebsr_matrix<T>& that_, bool transfer)
    : device_gebsx_matrix<T>(rocsparse_direction_row,
                             that_.block_direction,
                             that_.mb,
                             that_.nb,
                             that_.nnzb,
                             that_.row_block_dim,
                             that_.col_block_dim,
                             that_.base)
{
    if(transfer)
    {
        this->transfer_from(that_);
    }
};

template <typename T>
device_gebsx_matrix<T>::device_gebsx_matrix(){};
template <typename T>
device_gebsx_matrix<T>::device_gebsx_matrix(rocsparse_direction  dir_,
                                            rocsparse_direction  dirb_,
                                            rocsparse_int        mb_,
                                            rocsparse_int        nb_,
                                            rocsparse_int        nnzb_,
                                            rocsparse_int        row_block_dim_,
                                            rocsparse_int        col_block_dim_,
                                            rocsparse_index_base base_)
    : gebsx_matrix<T, device_gebsx_matrix<T>>(
        dir_, dirb_, mb_, nb_, nnzb_, row_block_dim_, col_block_dim_, base_)
{
}

template <typename T>
host_gebsx_matrix<T>::host_gebsx_matrix(){};

template <typename T>
host_gebsx_matrix<T>::host_gebsx_matrix(rocsparse_direction  dir_,
                                        rocsparse_direction  dirb_,
                                        rocsparse_int        mb_,
                                        rocsparse_int        nb_,
                                        rocsparse_int        nnzb_,
                                        rocsparse_int        row_block_dim_,
                                        rocsparse_int        col_block_dim_,
                                        rocsparse_index_base base_)
    : gebsx_matrix<T, host_gebsx_matrix<T>>(
        dir_, dirb_, mb_, nb_, nnzb_, row_block_dim_, col_block_dim_, base_)
{
}

template <typename T>
host_gebsr_matrix<T>::host_gebsr_matrix(){};

template <typename T>
host_gebsr_matrix<T>::host_gebsr_matrix(const device_gebsr_matrix<T>& that_, bool transfer)
    : host_gebsx_matrix<T>(rocsparse_direction_row,
                           that_.block_direction,
                           that_.mb,
                           that_.nb,
                           that_.nnzb,
                           that_.row_block_dim,
                           that_.col_block_dim,
                           that_.base)
{
    if(transfer)
    {
        this->transfer_from(that_);
    }
};

template <typename T>
host_gebsr_matrix<T>::host_gebsr_matrix(rocsparse_direction  dirb_,
                                        rocsparse_int        mb_,
                                        rocsparse_int        nb_,
                                        rocsparse_int        nnzb_,
                                        rocsparse_int        row_block_dim_,
                                        rocsparse_int        col_block_dim_,
                                        rocsparse_index_base base_)
    : host_gebsx_matrix<T>(
        rocsparse_direction_row, dirb_, mb_, nb_, nnzb_, row_block_dim_, col_block_dim_, base_)
{
}

template <typename T>
device_gebsc_matrix<T>::device_gebsc_matrix(){};

template <typename T>
device_gebsc_matrix<T>::device_gebsc_matrix(const host_gebsc_matrix<T>& that_, bool transfer)
    : device_gebsx_matrix<T>(rocsparse_direction_column,
                             that_.block_direction,
                             that_.mb,
                             that_.nb,
                             that_.nnzb,
                             that_.row_block_dim,
                             that_.col_block_dim,
                             that_.base)
{
    if(transfer)
    {
        this->transfer_from(that_);
    }
};

template <typename T>
device_gebsc_matrix<T>::device_gebsc_matrix(rocsparse_direction  dirb_,
                                            rocsparse_int        mb_,
                                            rocsparse_int        nb_,
                                            rocsparse_int        nnzb_,
                                            rocsparse_int        row_block_dim_,
                                            rocsparse_int        col_block_dim_,
                                            rocsparse_index_base base_)
    : device_gebsx_matrix<T>(
        rocsparse_direction_column, dirb_, mb_, nb_, nnzb_, row_block_dim_, col_block_dim_, base_)
{
}

template <typename T>
host_gebsc_matrix<T>::host_gebsc_matrix(){};

template <typename T>
host_gebsc_matrix<T>::host_gebsc_matrix(const device_gebsc_matrix<T>& that_, bool transfer)
    : host_gebsx_matrix<T>(rocsparse_direction_column,
                           that_.block_direction,
                           that_.mb,
                           that_.nb,
                           that_.nnzb,
                           that_.row_block_dim,
                           that_.col_block_dim,
                           that_.base)
{
    if(transfer)
    {
        this->transfer_from(that_);
    }
};

template <typename T>
host_gebsc_matrix<T>::host_gebsc_matrix(rocsparse_direction  dirb_,
                                        rocsparse_int        mb_,
                                        rocsparse_int        nb_,
                                        rocsparse_int        nnzb_,
                                        rocsparse_int        row_block_dim_,
                                        rocsparse_int        col_block_dim_,
                                        rocsparse_index_base base_)
    : host_gebsx_matrix<T>(rocsparse_direction_column,
                           dirb_,
                           mb_,
                           nb_,
                           nnzb_,
                           row_block_dim_,
                           col_block_dim_,
                           base_){};

template <typename T>
void device_gebsr_matrix<T>::transfer_from(const host_gebsr_matrix<T>& that)
{
    CHECK_HIP_ERROR((this->mb == that.mb && this->nb == that.nb && this->nnzb == that.nnzb
                     && this->dir == that.dir && this->block_direction == that.block_direction
                     && this->row_block_dim == that.row_block_dim
                     && this->col_block_dim == that.col_block_dim && this->base == that.base)
                        ? hipSuccess
                        : hipErrorInvalidValue);
    CHECK_HIP_ERROR(hipMemcpy(
        this->ptr, that.ptr, sizeof(rocsparse_int) * (that.mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(this->ind, that.ind, sizeof(rocsparse_int) * that.nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(this->val,
                              that.val,
                              sizeof(T) * that.nnzb * that.row_block_dim * that.col_block_dim,
                              hipMemcpyHostToDevice));
}

template <typename T>
void host_gebsr_matrix<T>::transfer_from(const device_gebsr_matrix<T>& that)
{
    CHECK_HIP_ERROR((this->mb == that.mb && this->nb == that.nb && this->nnzb == that.nnzb
                     && this->dir == that.dir && this->block_direction == that.block_direction
                     && this->row_block_dim == that.row_block_dim
                     && this->col_block_dim == that.col_block_dim && this->base == that.base)
                        ? hipSuccess
                        : hipErrorInvalidValue);
    CHECK_HIP_ERROR(hipMemcpy(
        this->ptr, that.ptr, sizeof(rocsparse_int) * (that.mb + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(this->ind, that.ind, sizeof(rocsparse_int) * that.nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(this->val,
                              that.val,
                              sizeof(T) * that.nnzb * that.row_block_dim * that.col_block_dim,
                              hipMemcpyDeviceToHost));
}

template <typename T>
void device_gebsc_matrix<T>::transfer_from(const host_gebsc_matrix<T>& that)
{
    CHECK_HIP_ERROR((this->mb == that.mb && this->nb == that.nb && this->nnzb == that.nnzb
                     && this->dir == that.dir && this->block_direction == that.block_direction
                     && this->row_block_dim == that.row_block_dim
                     && this->col_block_dim == that.col_block_dim && this->base == that.base)
                        ? hipSuccess
                        : hipErrorInvalidValue);
    CHECK_HIP_ERROR(hipMemcpy(
        this->ptr, that.ptr, sizeof(rocsparse_int) * (that.nb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(this->ind, that.ind, sizeof(rocsparse_int) * that.nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(this->val,
                              that.val,
                              sizeof(T) * that.nnzb * that.row_block_dim * that.col_block_dim,
                              hipMemcpyHostToDevice));
}

template <typename T>
void host_gebsc_matrix<T>::transfer_from(const device_gebsc_matrix<T>& that)
{
    CHECK_HIP_ERROR((this->mb == that.mb && this->nb == that.nb && this->nnzb == that.nnzb
                     && this->dir == that.dir && this->block_direction == that.block_direction
                     && this->row_block_dim == that.row_block_dim
                     && this->col_block_dim == that.col_block_dim && this->base == that.base)
                        ? hipSuccess
                        : hipErrorInvalidValue);
    CHECK_HIP_ERROR(hipMemcpy(
        this->ptr, that.ptr, sizeof(rocsparse_int) * (that.nb + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(this->ind, that.ind, sizeof(rocsparse_int) * that.nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(this->val,
                              that.val,
                              sizeof(T) * that.nnzb * that.row_block_dim * that.col_block_dim,
                              hipMemcpyDeviceToHost));
}

template <typename T>
void host_gebsc_matrix<T>::unit_check(host_gebsc_matrix<T>& that, bool check_values)
{
    {
        rocsparse_int dir1 = (rocsparse_int)this->dir;
        rocsparse_int dir2 = (rocsparse_int)that.dir;
        unit_check_general<rocsparse_int>(1, 1, 1, &dir1, &dir2);
    }

    {
        rocsparse_int dirb1 = (rocsparse_int)this->block_direction;
        rocsparse_int dirb2 = (rocsparse_int)that.block_direction;
        unit_check_general<rocsparse_int>(1, 1, 1, &dirb1, &dirb2);
    }

    unit_check_general<rocsparse_int>(1, 1, 1, &this->mb, &that.mb);

    unit_check_general<rocsparse_int>(1, 1, 1, &this->nb, &that.nb);

    unit_check_general<rocsparse_int>(1, 1, 1, &this->nnzb, &that.nnzb);

    unit_check_general<rocsparse_int>(1, 1, 1, &this->row_block_dim, &that.row_block_dim);

    unit_check_general<rocsparse_int>(1, 1, 1, &this->col_block_dim, &that.col_block_dim);

    unit_check_general<rocsparse_int>(1, this->nnzb, 1, this->ind, that.ind);

    unit_check_general<rocsparse_int>(1, this->nb + 1, 1, this->ptr, that.ptr);

    if(check_values)
    {
        unit_check_general<T>(
            1, this->nnzb * this->row_block_dim * this->col_block_dim, 1, this->val, that.val);
    }
};

template <typename T>
device_csc_matrix<T>::device_csc_matrix(){};
template <typename T>
device_csc_matrix<T>::device_csc_matrix(rocsparse_int        m_,
                                        rocsparse_int        n_,
                                        rocsparse_int        nnz_,
                                        rocsparse_index_base base_)
    : device_csx_matrix<T>(rocsparse_direction_column, m_, n_, nnz_, base_)
{
}

template <typename T>
host_csc_matrix<T>::host_csc_matrix(){};
template <typename T>
host_csc_matrix<T>::host_csc_matrix(rocsparse_int        m_,
                                    rocsparse_int        n_,
                                    rocsparse_int        nnz_,
                                    rocsparse_index_base base_)
    : host_csx_matrix<T>(rocsparse_direction_column, m_, n_, nnz_, base_)
{
}

template <typename T>
host_csr_matrix<T>::host_csr_matrix(){};
template <typename T>
host_csr_matrix<T>::host_csr_matrix(rocsparse_int        m_,
                                    rocsparse_int        n_,
                                    rocsparse_int        nnz_,
                                    rocsparse_index_base base_)
    : host_csx_matrix<T>(rocsparse_direction_row, m_, n_, nnz_, base_)
{
}

template <typename T>
host_csx_matrix<T>::host_csx_matrix(){};
template <typename T>
host_csx_matrix<T>::host_csx_matrix(rocsparse_direction  dir_,
                                    rocsparse_int        m_,
                                    rocsparse_int        n_,
                                    rocsparse_int        nnz_,
                                    rocsparse_index_base base_)
    : csx_matrix<T, host_csx_matrix<T>>(dir_, m_, n_, nnz_, base_){};
template <typename T>
struct device_csr_matrix : public device_csx_matrix<T>
{
    device_csr_matrix(){};
    device_csr_matrix(rocsparse_int        m_,
                      rocsparse_int        n_,
                      rocsparse_int        nnz_,
                      rocsparse_index_base base_)
        : device_csx_matrix<T>(rocsparse_direction_row, m_, n_, nnz_, base_)
    {
    }
};

template <typename T>
device_csx_matrix<T>::device_csx_matrix(){};

template <typename T>
device_csx_matrix<T>::device_csx_matrix(rocsparse_direction  dir_,
                                        rocsparse_int        m_,
                                        rocsparse_int        n_,
                                        rocsparse_int        nnz_,
                                        rocsparse_index_base base_)
    : csx_matrix<T, device_csx_matrix<T>>(dir_, m_, n_, nnz_, base_){};

template <typename T, typename IMPL>
coo_matrix<T, IMPL>::coo_matrix(){};

template <typename T, typename IMPL>
coo_matrix<T, IMPL>::coo_matrix(rocsparse_int        m_,
                                rocsparse_int        n_,
                                rocsparse_int        nnz_,
                                rocsparse_index_base base_)
    : m(m_)
    , n(n_)
    , nnz(nnz_)
    , base(base_)
    , row_ind(nnz_)
    , col_ind(nnz_)
    , val(nnz_){};

template <typename T>
host_coo_matrix<T>::host_coo_matrix(){};

template <typename T>
host_coo_matrix<T>::host_coo_matrix(rocsparse_int        m_,
                                    rocsparse_int        n_,
                                    rocsparse_int        nnz_,
                                    rocsparse_index_base base_)
    : coo_matrix<T, host_coo_matrix<T>>(m_, n_, nnz_, base_){};

template <typename T>
device_coo_matrix<T>::device_coo_matrix(){};

template <typename T>
device_coo_matrix<T>::device_coo_matrix(rocsparse_int        m_,
                                        rocsparse_int        n_,
                                        rocsparse_int        nnz_,
                                        rocsparse_index_base base_)
    : coo_matrix<T, device_coo_matrix<T>>(m_, n_, nnz_, base_){};

template <typename T, typename IMPL>
csx_matrix<T, IMPL>::csx_matrix(){};

template <typename T, typename IMPL>
csx_matrix<T, IMPL>::csx_matrix(rocsparse_direction  dir_,
                                rocsparse_int        m_,
                                rocsparse_int        n_,
                                rocsparse_int        nnz_,
                                rocsparse_index_base base_)
    : dir(dir_)
    , m(m_)
    , n(n_)
    , nnz(nnz_)
    , base(base_)
    , ptr((dir_) ? (m_ + 1) : (n_ + 1))
    , ind(nnz_)
    , val(nnz_){};

template <typename T, typename IMPL>
gebsx_matrix<T, IMPL>::gebsx_matrix(){};

template <typename T, typename IMPL>
gebsx_matrix<T, IMPL>::gebsx_matrix(rocsparse_direction  dir_,
                                    rocsparse_direction  block_dir_,
                                    rocsparse_int        mb_,
                                    rocsparse_int        nb_,
                                    rocsparse_int        nnzb_,
                                    rocsparse_int        row_block_dim_,
                                    rocsparse_int        col_block_dim_,
                                    rocsparse_index_base base_)
    : dir(dir_)
    , mb(mb_)
    , nb(nb_)
    , nnzb(nnzb_)
    , block_direction(block_dir_)
    , row_block_dim(row_block_dim_)
    , col_block_dim(col_block_dim_)
    , base(base_)
    , ptr((rocsparse_direction_row == dir_) ? (mb + 1) : (nb + 1))
    , ind(nnzb)
    , val(nnzb * row_block_dim * col_block_dim){};

#endif // ROCSPARSE_MATRIX_HPP

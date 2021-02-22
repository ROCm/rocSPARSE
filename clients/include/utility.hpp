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

/*! \file
 *  \brief utility.hpp provides common utilities
 */

#pragma once
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "rocsparse_matrix.hpp"
#include "rocsparse_test.hpp"

#include <hip/hip_runtime_api.h>
#include <vector>
/* ==================================================================================== */
// Return index type
template <typename I>
rocsparse_indextype get_indextype(void);

/* ==================================================================================== */
// Return data type
template <typename T>
rocsparse_datatype get_datatype(void);

/* ==================================================================================== */
/*! \brief  local handle which is automatically created and destroyed  */
class rocsparse_local_handle
{
    rocsparse_handle handle{};

public:
    rocsparse_local_handle()
    {
        rocsparse_create_handle(&this->handle);
    }
    ~rocsparse_local_handle()
    {
        rocsparse_destroy_handle(this->handle);
    }

    // Allow rocsparse_local_handle to be used anywhere rocsparse_handle is expected
    operator rocsparse_handle&()
    {
        return this->handle;
    }
    operator const rocsparse_handle&() const
    {
        return this->handle;
    }
};

/* ==================================================================================== */
/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class rocsparse_local_mat_descr
{
    rocsparse_mat_descr descr{};

public:
    rocsparse_local_mat_descr()
    {
        rocsparse_create_mat_descr(&this->descr);
    }

    ~rocsparse_local_mat_descr()
    {
        rocsparse_destroy_mat_descr(this->descr);
    }

    // Allow rocsparse_local_mat_descr to be used anywhere rocsparse_mat_descr is expected
    operator rocsparse_mat_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_mat_descr&() const
    {
        return this->descr;
    }
};

/* ==================================================================================== */
/*! \brief  local matrix info which is automatically created and destroyed  */
class rocsparse_local_mat_info
{
    rocsparse_mat_info info{};

public:
    rocsparse_local_mat_info()
    {
        rocsparse_create_mat_info(&this->info);
    }
    ~rocsparse_local_mat_info()
    {
        rocsparse_destroy_mat_info(this->info);
    }

    // Allow rocsparse_local_mat_info to be used anywhere rocsparse_mat_info is expected
    operator rocsparse_mat_info&()
    {
        return this->info;
    }
    operator const rocsparse_mat_info&() const
    {
        return this->info;
    }
};

/* ==================================================================================== */
/*! \brief  hyb matrix structure helper to access data for tests  */
struct test_hyb
{
    rocsparse_int           m;
    rocsparse_int           n;
    rocsparse_hyb_partition partition;
    rocsparse_int           ell_nnz;
    rocsparse_int           ell_width;
    rocsparse_int*          ell_col_ind;
    void*                   ell_val;
    rocsparse_int           coo_nnz;
    rocsparse_int*          coo_row_ind;
    rocsparse_int*          coo_col_ind;
    void*                   coo_val;
};

/* ==================================================================================== */
/*! \brief  local hyb matrix structure which is automatically created and destroyed  */
class rocsparse_local_hyb_mat
{
    rocsparse_hyb_mat hyb{};

public:
    rocsparse_local_hyb_mat()
    {
        rocsparse_create_hyb_mat(&this->hyb);
    }
    ~rocsparse_local_hyb_mat()
    {
        rocsparse_destroy_hyb_mat(this->hyb);
    }

    // Allow rocsparse_local_hyb_mat to be used anywhere rocsparse_hyb_mat is expected
    operator rocsparse_hyb_mat&()
    {
        return this->hyb;
    }
    operator const rocsparse_hyb_mat&() const
    {
        return this->hyb;
    }
};

/* ==================================================================================== */
/*! \brief  local dense vector structure which is automatically created and destroyed  */
class rocsparse_local_spvec
{
    rocsparse_spvec_descr descr{};

public:
    rocsparse_local_spvec(int64_t              size,
                          int64_t              nnz,
                          void*                indices,
                          void*                values,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        rocsparse_create_spvec_descr(
            &this->descr, size, nnz, indices, values, idx_type, idx_base, compute_type);
    }
    ~rocsparse_local_spvec()
    {
        if(this->descr != nullptr)
        {
            rocsparse_destroy_spvec_descr(this->descr);
        }
    }

    // Allow rocsparse_local_spvec to be used anywhere rocsparse_spvec_descr is expected
    operator rocsparse_spvec_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_spvec_descr&() const
    {
        return this->descr;
    }
};

/* ==================================================================================== */
/*! \brief  local sparse matrix structure which is automatically created and destroyed  */
class rocsparse_local_spmat
{
    rocsparse_spmat_descr descr{};

public:
    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          int64_t              nnz,
                          void*                coo_row_ind,
                          void*                coo_col_ind,
                          void*                coo_val,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        rocsparse_create_coo_descr(&this->descr,
                                   m,
                                   n,
                                   nnz,
                                   coo_row_ind,
                                   coo_col_ind,
                                   coo_val,
                                   idx_type,
                                   idx_base,
                                   compute_type);
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    rocsparse_local_spmat(coo_matrix<MODE, T, I>& h)
        : rocsparse_local_spmat(h.m,
                                h.n,
                                h.nnz,
                                h.row_ind,
                                h.col_ind,
                                h.val,
                                get_indextype<I>(),
                                h.base,
                                get_datatype<T>())
    {
    }

    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          int64_t              nnz,
                          void*                coo_ind,
                          void*                coo_val,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        rocsparse_create_coo_aos_descr(
            &this->descr, m, n, nnz, coo_ind, coo_val, idx_type, idx_base, compute_type);
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    rocsparse_local_spmat(coo_aos_matrix<MODE, T, I>& h)
        : rocsparse_local_spmat(
            h.m, h.n, h.nnz, h.ind, h.val, get_indextype<I>(), h.base, get_datatype<T>())
    {
    }

    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          int64_t              nnz,
                          void*                row_col_ptr,
                          void*                row_col_ind,
                          void*                val,
                          rocsparse_indextype  row_col_ptr_type,
                          rocsparse_indextype  row_col_ind_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type,
                          rocsparse_format     format)
    {

        if(format == rocsparse_format_csr)
        {
            rocsparse_create_csr_descr(&this->descr,
                                       m,
                                       n,
                                       nnz,
                                       row_col_ptr,
                                       row_col_ind,
                                       val,
                                       row_col_ptr_type,
                                       row_col_ind_type,
                                       idx_base,
                                       compute_type);
        }
        else if(format == rocsparse_format_csc)
        {
            rocsparse_create_csc_descr(&this->descr,
                                       m,
                                       n,
                                       nnz,
                                       row_col_ptr,
                                       row_col_ind,
                                       val,
                                       row_col_ptr_type,
                                       row_col_ind_type,
                                       idx_base,
                                       compute_type);
        }
    }

    template <memory_mode::value_t MODE,
              rocsparse_direction  direction_,
              typename T,
              typename I = rocsparse_int,
              typename J = rocsparse_int>
    rocsparse_local_spmat(csx_matrix<MODE, direction_, T, I, J>& h)
        : rocsparse_local_spmat(h.m,
                                h.n,
                                h.nnz,
                                h.ptr,
                                h.ind,
                                h.val,
                                get_indextype<I>(),
                                get_indextype<J>(),
                                h.base,
                                get_datatype<T>(),
                                rocsparse_format_csr)
    {
    }

    rocsparse_local_spmat(int64_t              m,
                          int64_t              n,
                          void*                ell_col_ind,
                          void*                ell_val,
                          int64_t              ell_width,
                          rocsparse_indextype  idx_type,
                          rocsparse_index_base idx_base,
                          rocsparse_datatype   compute_type)
    {
        rocsparse_create_ell_descr(
            &this->descr, m, n, ell_col_ind, ell_val, ell_width, idx_type, idx_base, compute_type);
    }

    template <memory_mode::value_t MODE, typename T, typename I = rocsparse_int>
    rocsparse_local_spmat(ell_matrix<MODE, T, I>& h)
        : rocsparse_local_spmat(
            h.m, h.n, h.ind, h.val, h.width, get_indextype<I>(), h.base, get_datatype<T>())
    {
    }

    ~rocsparse_local_spmat()
    {
        if(this->descr != nullptr)
            rocsparse_destroy_spmat_descr(this->descr);
    }

    // Allow rocsparse_local_spmat to be used anywhere rocsparse_spmat_descr is expected
    operator rocsparse_spmat_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_spmat_descr&() const
    {
        return this->descr;
    }
};

/* ==================================================================================== */
/*! \brief  local dense vector structure which is automatically created and destroyed  */
class rocsparse_local_dnvec
{
    rocsparse_dnvec_descr descr{};

public:
    rocsparse_local_dnvec(int64_t size, void* values, rocsparse_datatype compute_type)
    {
        rocsparse_create_dnvec_descr(&this->descr, size, values, compute_type);
    }

    template <memory_mode::value_t MODE, typename T>
    rocsparse_local_dnvec(dense_matrix<MODE, T>& h)
        : rocsparse_local_dnvec(h.m, h.val, get_datatype<T>())
    {
    }

    ~rocsparse_local_dnvec()
    {
        if(this->descr != nullptr)
            rocsparse_destroy_dnvec_descr(this->descr);
    }

    // Allow rocsparse_local_dnvec to be used anywhere rocsparse_dnvec_descr is expected
    operator rocsparse_dnvec_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_dnvec_descr&() const
    {
        return this->descr;
    }
};

/* ==================================================================================== */
/*! \brief  local dense matrix structure which is automatically created and destroyed  */
class rocsparse_local_dnmat
{
    rocsparse_dnmat_descr descr{};

public:
    rocsparse_local_dnmat(int64_t            rows,
                          int64_t            cols,
                          int64_t            ld,
                          void*              values,
                          rocsparse_datatype compute_type,
                          rocsparse_order    order)
    {
        rocsparse_create_dnmat_descr(&this->descr, rows, cols, ld, values, compute_type, order);
    }

    template <memory_mode::value_t MODE, typename T>
    rocsparse_local_dnmat(dense_matrix<MODE, T>& h, rocsparse_order order)
        : rocsparse_local_dnmat(h.m, h.n, h.ld, h.val, get_datatype<T>(), order)
    {
    }

    ~rocsparse_local_dnmat()
    {
        if(this->descr != nullptr)
            rocsparse_destroy_dnmat_descr(this->descr);
    }

    // Allow rocsparse_local_dnmat to be used anywhere rocsparse_dnmat_descr is expected
    operator rocsparse_dnmat_descr&()
    {
        return this->descr;
    }
    operator const rocsparse_dnmat_descr&() const
    {
        return this->descr;
    }
};

/* ==================================================================================== */
/*  timing: HIP only provides very limited timers function clock() and not general;
            rocsparse sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return
 *  wall time
 */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return
 *  wall time
 */
double get_time_us_sync(hipStream_t stream);

/* ==================================================================================== */
// Return path of this executable
std::string rocsparse_exepath();

#endif // UTILITY_HPP

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

static inline const float* get_boost_tol(const float* tol)
{
    return tol;
}

static inline const double* get_boost_tol(const double* tol)
{
    return tol;
}

static inline const float* get_boost_tol(const rocsparse_float_complex* tol)
{
    return reinterpret_cast<const float*>(tol);
}

static inline const double* get_boost_tol(const rocsparse_double_complex* tol)
{
    return reinterpret_cast<const double*>(tol);
}

//
// @brief Utility methods for matrices..
//
struct rocsparse_matrix_utils
{

    //
    // @brief Initialize a host dense matrix with random values.
    // @param[in] that Fill \p that matrix.
    //
    template <typename T>
    static void init(host_dense_matrix<T>& that)
    {
        switch(that.order)
        {
        case rocsparse_order_column:
        {
            rocsparse_init(that.val, that.m, that.n, that.ld);
            break;
        }

        case rocsparse_order_row:
        {
            //
            // Little trick but the resulting matrix is the transpose of the matrix obtained from rocsparse_order_column.
            // If this poses a problem, we need to refactor rocsparse_init.
            //
            rocsparse_init(that.val, that.n, that.m, that.ld);
            break;
        }
        }
    }

    //
    // @brief Initialize a host dense matrix with random integer values.
    // @param[in] that Fill \p that matrix.
    //
    template <typename T>
    static void init_exact(host_dense_matrix<T>& that)
    {
        switch(that.order)
        {
        case rocsparse_order_column:
        {
            rocsparse_init_exact(that.val, that.m, that.n, that.ld);
            break;
        }

        case rocsparse_order_row:
        {
            //
            // Little trick but the resulting matrix is the transpose of the matrix obtained from rocsparse_order_column.
            // If this poses a problem, we need to refactor rocsparse_init_exact.
            //
            rocsparse_init_exact(that.val, that.n, that.m, that.ld);
            break;
        }
        }
    }

    //
    // @brief Compress a \p device_csr_matrix
    // @param[out] result Define a \p device_csr_matrix resulting from the compression of \p that.
    // @param[in] that That matrix to compress.
    //
    template <typename T>
    static void compress(device_csr_matrix<T>&       result,
                         const device_csr_matrix<T>& that,
                         rocsparse_index_base        base)
    {
        result.define(that.m, that.n, 0, base);
        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
        rocsparse_mat_descr descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
        rocsparse_set_mat_index_base(descr, that.base);
        T                            tol   = static_cast<T>(0);
        rocsparse_int                nnz_c = 0;
        device_vector<rocsparse_int> dnnz_per_row(that.m);
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, that.m, descr, that.val, that.ptr, dnnz_per_row, &nnz_c, tol));
        result.define(that.m, that.n, nnz_c, base);
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                            that.m,
                                                            that.n,
                                                            descr,
                                                            that.val,
                                                            that.ptr,
                                                            that.ind,
                                                            that.nnz,
                                                            dnnz_per_row,
                                                            result.val,
                                                            result.ptr,
                                                            result.ind,
                                                            tol));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
    }

    //
    // @brief Convert a \p device_csr_matrix into a \p device_gebsr_matrix.
    // @param[out] result Define a \p device_csr_matrix resulting from the conversion of \p that.
    // @param[in] that That matrix to compress.
    //
    template <typename T>
    static void convert(const device_csr_matrix<T>& that,
                        rocsparse_direction         dirb,
                        rocsparse_int               row_block_dim,
                        rocsparse_int               col_block_dim,
                        rocsparse_index_base        base,
                        device_gebsr_matrix<T>&     result)
    {
        rocsparse_int mb = (that.m + row_block_dim - 1) / row_block_dim;
        rocsparse_int nb = (that.n + col_block_dim - 1) / col_block_dim;

        rocsparse_int nnzb = 0;

        result.define(dirb, mb, nb, nnzb, row_block_dim, col_block_dim, base);

        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));

        rocsparse_mat_descr that_descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&that_descr));
        rocsparse_set_mat_index_base(that_descr, that.base);

        rocsparse_mat_descr result_descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&result_descr));
        rocsparse_set_mat_index_base(result_descr, base);

        // Convert CSR to GEBSR
        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_buffer_size<T>(handle,
                                                                 result.block_direction,
                                                                 that.m,
                                                                 that.n,
                                                                 that_descr,
                                                                 that.val,
                                                                 that.ptr,
                                                                 that.ind,
                                                                 result.row_block_dim,
                                                                 result.col_block_dim,
                                                                 &buffer_size));

        int* buffer = NULL;
        hipMalloc(&buffer, buffer_size);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                      result.block_direction,
                                                      that.m,
                                                      that.n,
                                                      that_descr,
                                                      that.ptr,
                                                      that.ind,
                                                      result_descr,
                                                      result.ptr,
                                                      result.row_block_dim,
                                                      result.col_block_dim,
                                                      &nnzb,
                                                      buffer));

        result.define(dirb, mb, nb, nnzb, row_block_dim, col_block_dim, base);

        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr<T>(handle,
                                                     result.block_direction,
                                                     that.m,
                                                     that.n,
                                                     that_descr,
                                                     that.val,
                                                     that.ptr,
                                                     that.ind,
                                                     result_descr,
                                                     result.val,
                                                     result.ptr,
                                                     result.ind,
                                                     result.row_block_dim,
                                                     result.col_block_dim,
                                                     buffer));

        hipFree(buffer);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(result_descr));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(that_descr));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
    }

    //
    // @brief Convert a \p device_csr_matrix into a \p device_gebsr_matrix.
    // @param[out] result Define a \p device_gebsr_matrix resulting from the conversion of \p that.
    // @param[in] that That matrix to compress.
    //
    template <typename T>
    static void convert(const device_csr_matrix<T>& that,
                        rocsparse_direction         dirb,
                        rocsparse_int               block_dim,
                        rocsparse_index_base        base,
                        device_gebsr_matrix<T>&     c)
    {
        rocsparse_int mb   = (that.m + block_dim - 1) / block_dim;
        rocsparse_int nb   = (that.n + block_dim - 1) / block_dim;
        rocsparse_int nnzb = 0;

        c.define(dirb, mb, nb, nnzb, block_dim, block_dim, base);

        {

            rocsparse_handle handle;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));

            rocsparse_mat_descr that_descr;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&that_descr));
            rocsparse_set_mat_index_base(that_descr, that.base);

            rocsparse_mat_descr c_descr;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&c_descr));
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
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(c_descr));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(that_descr));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
        }
    }

    typedef enum
    {
        bsrilu0_analysis = 1,
        bsrilu0_solve    = 2,
        bsrilu0_all      = 3
    } bsrilu0_step;

    template <typename T>
    static void bsrilu0(rocsparse_mat_descr       that_descr,
                        device_gebsr_matrix<T>&   that,
                        rocsparse_mat_info        info,
                        rocsparse_analysis_policy analysis,
                        rocsparse_solve_policy    solve,
                        int                       boost,
                        T                         h_boost_val,
                        T                         h_boost_tol,
                        size_t*                   p_buffer_size,
                        void*                     buffer,
                        bsrilu0_step              step = bsrilu0_all)
    {

        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                                   that.block_direction,
                                                                   that.mb,
                                                                   that.nnzb,
                                                                   that_descr,
                                                                   that.val,
                                                                   that.ptr,
                                                                   that.ind,
                                                                   that.row_block_dim,
                                                                   info,
                                                                   p_buffer_size));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));

            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case bsrilu0_all:
        case bsrilu0_analysis:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_analysis<T>(handle,
                                                                that.block_direction,
                                                                that.mb,
                                                                that.nnzb,
                                                                that_descr,
                                                                that.val,
                                                                that.ptr,
                                                                that.ind,
                                                                that.row_block_dim,
                                                                info,
                                                                analysis,
                                                                solve,
                                                                buffer));
            if(step == bsrilu0_all)
            {
                step = bsrilu0_solve;
            }
            else
            {
                break;
            }
        }

        case bsrilu0_solve:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_numeric_boost<T>(
                handle, info, boost, get_boost_tol(&h_boost_tol), &h_boost_val));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0<T>(handle,
                                                       that.block_direction,
                                                       that.mb,
                                                       that.nnzb,
                                                       that_descr,
                                                       that.val,
                                                       that.ptr,
                                                       that.ind,
                                                       that.row_block_dim,
                                                       info,
                                                       solve,
                                                       buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            break;
        }
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
    }

    typedef enum
    {
        bsric0_analysis = 1,
        bsric0_solve    = 2,
        bsric0_all      = 3
    } bsric0_step;

    template <typename T>
    static void bsric0(rocsparse_mat_descr       that_descr,
                       device_gebsr_matrix<T>&   that,
                       rocsparse_mat_info        info,
                       rocsparse_analysis_policy analysis,
                       rocsparse_solve_policy    solve,
                       size_t*                   p_buffer_size,
                       void*                     buffer,
                       bsric0_step               step = bsric0_all)
    {
        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_buffer_size<T>(handle,
                                                                  that.block_direction,
                                                                  that.mb,
                                                                  that.nnzb,
                                                                  that_descr,
                                                                  that.val,
                                                                  that.ptr,
                                                                  that.ind,
                                                                  that.row_block_dim,
                                                                  info,
                                                                  p_buffer_size));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case bsric0_all:
        case bsric0_analysis:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsric0_analysis<T>(handle,
                                                               that.block_direction,
                                                               that.mb,
                                                               that.nnzb,
                                                               that_descr,
                                                               that.val,
                                                               that.ptr,
                                                               that.ind,
                                                               that.row_block_dim,
                                                               info,
                                                               analysis,
                                                               solve,
                                                               buffer));
            if(step == bsric0_all)
            {
                step = bsric0_solve;
            }
            else
            {
                break;
            }
        }

        case bsric0_solve:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsric0<T>(handle,
                                                      that.block_direction,
                                                      that.mb,
                                                      that.nnzb,
                                                      that_descr,
                                                      that.val,
                                                      that.ptr,
                                                      that.ind,
                                                      that.row_block_dim,
                                                      info,
                                                      solve,
                                                      buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsric0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            break;
        }
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
    }

    typedef enum
    {
        csric0_analysis = 1,
        csric0_solve    = 2,
        csric0_all      = 3
    } csric0_step;

    template <typename T>
    static void csric0(rocsparse_mat_descr       that_descr,
                       device_csr_matrix<T>&     that,
                       rocsparse_mat_info        info,
                       rocsparse_analysis_policy analysis,
                       rocsparse_solve_policy    solve,
                       size_t*                   p_buffer_size,
                       void*                     buffer,
                       csric0_step               step = csric0_all)
    {
        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(rocsparse_csric0_buffer_size<T>(handle,
                                                                  that.m,
                                                                  that.nnz,
                                                                  that_descr,
                                                                  that.val,
                                                                  that.ptr,
                                                                  that.ind,
                                                                  info,
                                                                  p_buffer_size));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case csric0_all:
        case csric0_analysis:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                               that.m,
                                                               that.nnz,
                                                               that_descr,
                                                               that.val,
                                                               that.ptr,
                                                               that.ind,
                                                               info,
                                                               analysis,
                                                               solve,
                                                               buffer));
            if(step == csric0_all)
            {
                step = csric0_solve;
            }
            else
            {
                break;
            }
        }

        case csric0_solve:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csric0<T>(handle,
                                                      that.m,
                                                      that.nnz,
                                                      that_descr,
                                                      that.val,
                                                      that.ptr,
                                                      that.ind,
                                                      info,
                                                      solve,
                                                      buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            break;
        }
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
    }

    typedef enum
    {
        csrilu0_analysis = 1,
        csrilu0_solve    = 2,
        csrilu0_all      = 3
    } csrilu0_step;

    template <typename T>
    static void csrilu0(rocsparse_mat_descr       that_descr,
                        device_csr_matrix<T>&     that,
                        rocsparse_mat_info        info,
                        rocsparse_analysis_policy analysis,
                        rocsparse_solve_policy    solve,
                        int                       boost,
                        T                         h_boost_val,
                        T                         h_boost_tol,
                        size_t*                   p_buffer_size,
                        void*                     buffer,
                        csrilu0_step              step = csrilu0_all)
    {
        if(!buffer)
        {
            rocsparse_handle handle;
            CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_buffer_size<T>(handle,
                                                                   that.m,
                                                                   that.nnz,
                                                                   that_descr,
                                                                   that.val,
                                                                   that.ptr,
                                                                   that.ind,
                                                                   info,
                                                                   p_buffer_size));
            CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
            return;
        }

        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        switch(step)
        {
        case csrilu0_all:
        case csrilu0_analysis:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_analysis<T>(handle,
                                                                that.m,
                                                                that.nnz,
                                                                that_descr,
                                                                that.val,
                                                                that.ptr,
                                                                that.ind,
                                                                info,
                                                                analysis,
                                                                solve,
                                                                buffer));
            if(step == csrilu0_all)
            {
                step = csrilu0_solve;
            }
            else
            {
                break;
            }
        }

        case csrilu0_solve:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_numeric_boost<T>(
                handle, info, boost, get_boost_tol(&h_boost_tol), &h_boost_val));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0<T>(handle,
                                                       that.m,
                                                       that.nnz,
                                                       that_descr,
                                                       that.val,
                                                       that.ptr,
                                                       that.ind,
                                                       info,
                                                       solve,
                                                       buffer));

            rocsparse_int hsolve_pivot_1[1];
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrilu0_zero_pivot(handle, info, hsolve_pivot_1),
                                    (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

            break;
        }
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
    }

    template <typename T, typename I, typename J>
    static rocsparse_status host_csrsym(const host_csr_matrix<T, I, J>& A,
                                        host_csr_matrix<T, I, J>&       symA)
    {
        const auto M    = A.m;
        const auto NNZ  = A.nnz;
        const auto base = A.base;

        if(M != A.n)
        {
            return rocsparse_status_invalid_value;
        }

        //
        // Compute transpose.
        //
        host_csr_matrix<T, I, J> trA(M, M, NNZ, base);
        {
            for(J i = 0; i <= M; ++i)
            {
                trA.ptr[i] = static_cast<I>(0);
            }

            for(J i = 0; i < M; ++i)
            {
                for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
                {
                    const J j = A.ind[k] - base;
                    trA.ptr[j + 1] += 1;
                }
            }

            for(J i = 2; i <= M; ++i)
            {
                trA.ptr[i] += trA.ptr[i - 1];
            }

            for(J i = 0; i < M; ++i)
            {
                for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
                {
                    const J j           = A.ind[k] - base;
                    trA.ind[trA.ptr[j]] = i;
                    trA.val[trA.ptr[j]] = A.val[k];
                    ++trA.ptr[j];
                }
            }

            for(J i = M; i > 0; --i)
            {
                trA.ptr[i] = trA.ptr[i - 1];
            }
            trA.ptr[0] = 0;

            if(rocsparse_index_base_one == base)
            {
                for(J i = 0; i <= M; ++i)
                {
                    trA.ptr[i] += base;
                }

                for(I i = 0; i < NNZ; ++i)
                {
                    trA.ind[i] += base;
                }
            }
        }
        //
        // Compute transpose done.
        //

        //
        // Compute (A + A^T) / 2
        //
        J* blank = (J*)calloc(M, sizeof(J));
        if(!blank)
        {
            return rocsparse_status_memory_error;
        }

        J* select = (J*)malloc(M * sizeof(J));
        if(!select)
        {
            return rocsparse_status_memory_error;
        }

        symA.define(M, M, 0, base);

        for(J i = 0; i <= M; ++i)
        {
            symA.ptr[i] = 0;
        }

        for(J i = 0; i < M; ++i)
        {
            //
            // Load row from A.
            //
            J select_n = 0;
            for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
            {
                J j = A.ind[k] - base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Load row from A^T
            //
            for(I k = trA.ptr[i] - trA.base; k < trA.ptr[i + 1] - trA.base; ++k)
            {
                J j = trA.ind[k] - trA.base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Reset blank.
            //
            for(J k = 0; k < select_n; ++k)
            {
                blank[select[k]] = 0;
            }

            symA.ptr[i + 1] = select_n;
        }

        for(J i = 2; i <= M; ++i)
        {
            symA.ptr[i] += symA.ptr[i - 1];
        }

        symA.define(M, M, symA.ptr[M], base);

        for(J i = 0; i < M; ++i)
        {
            //
            // Load row from A.
            //
            J select_n = 0;
            for(I k = A.ptr[i] - base; k < A.ptr[i + 1] - base; ++k)
            {
                J j = A.ind[k] - base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            //
            // Load row from A^T
            //
            for(I k = trA.ptr[i] - trA.base; k < trA.ptr[i + 1] - base; ++k)
            {
                J j = trA.ind[k] - trA.base;
                if(!blank[j])
                {
                    select[select_n] = j;
                    blank[j]         = ++select_n;
                }
            }

            std::sort(select, select + select_n);

            for(J k = 0; k < select_n; ++k)
            {
                blank[select[k]] = 0;
            }

            for(J k = 0; k < select_n; ++k)
            {
                symA.ind[symA.ptr[i] + k] = select[k];
            }
        }

        if(rocsparse_index_base_one == base)
        {
            for(J i = 0; i <= M; ++i)
            {
                symA.ptr[i] += base;
            }

            for(I i = 0; i < symA.nnz; ++i)
            {
                symA.ind[i] += base;
            }
        }

        free(select);
        free(blank);

        return rocsparse_status_success;
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_base
{
protected:
    rocsparse_matrix_factory_base(){};

public:
    virtual ~rocsparse_matrix_factory_base(){};

    // @brief Initialize a csr-sparse matrix.
    // @param[out] csr_row_ptr vector of offsets.
    // @param[out] csr_col_ind vector of column indices.
    // @param[out] csr_val vector of values.
    // @param[inout] M number of rows.
    // @param[inout] N number of columns.
    // @param[inout] nnz number of non-zeros.
    // @param[in] base base of indices.
    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
        = 0;

    // @brief Initialize a gebsr-sparse matrix.
    // @param[out]   bsr_row_ptr vector of offsets.
    // @param[out]   bsr_col_ind vector of column indices.
    // @param[out]   bsr_val vector of values.
    // @param[in]    dirb number of rows.
    // @param[inout] Mb number of rows.
    // @param[inout] Nb number of columns.
    // @param[inout] nnzb number of non-zeros.
    // @param[inout] row_block_dim row dimension of the blocks.
    // @param[inout] col_block_dim column dimension of the blocks.
    // @param[in] base base of indices.
    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            rocsparse_direction  dirb,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
        = 0;

    // @brief Initialize a coo-sparse matrix.
    // @param[out]   coo_row_ind vector of row indices.
    // @param[out]   coo_col_ind vector of column indices.
    // @param[out]   coo_val vector of values.
    // @param[inout] M number of rows.
    // @param[inout] N number of columns.
    // @param[inout] nnz number of non-zeros.
    // @param[in] base base of indices.
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
    bool                       m_fullrank;
    bool                       m_to_int;
    rocsparse_matrix_init_kind m_matrix_init_kind;

public:
    rocsparse_matrix_factory_random(bool                       fullrank,
                                    bool                       to_int = false,
                                    rocsparse_matrix_init_kind matrix_init_kind
                                    = rocsparse_matrix_init_kind_default)
        : m_fullrank(fullrank)
        , m_to_int(to_int)
        , m_matrix_init_kind(matrix_init_kind){};

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {

        switch(this->m_matrix_init_kind)
        {
        case rocsparse_matrix_init_kind_tunedavg:
        {
            rocsparse_int alpha = static_cast<rocsparse_int>(0);
            if(N >= 16384)
            {
                alpha = static_cast<rocsparse_int>(4);
            }
            else if(N >= 8192)
            {
                alpha = static_cast<rocsparse_int>(8);
            }
            else if(N >= 4096)
            {
                alpha = static_cast<rocsparse_int>(16);
            }
            else if(N >= 1024)
            {
                alpha = static_cast<rocsparse_int>(32);
            }
            else
            {
                alpha = static_cast<rocsparse_int>(64);
            }

            nnz = M * alpha;
            nnz = std::min(nnz, static_cast<I>(M) * static_cast<I>(N));

            // Sample random matrix
            std::vector<J> row_ind(nnz);
            // Sample COO matrix
            rocsparse_init_coo_matrix<J>(
                row_ind, csr_col_ind, csr_val, M, N, nnz, base, this->m_fullrank, this->m_to_int);

            // Convert to CSR
            host_coo_to_csr(M, row_ind, csr_row_ptr, base);
            break;
        }

        case rocsparse_matrix_init_kind_default:
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
            break;
        }
        }
    }

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            rocsparse_direction  dirb,
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
struct rocsparse_matrix_factory_rocalution;

template <typename T>
struct rocsparse_matrix_factory_rocalution<T, rocsparse_int, rocsparse_int>
    : public rocsparse_matrix_factory_base<T, rocsparse_int, rocsparse_int>
{
private:
    std::string m_filename;
    bool        m_toint;
    using I = rocsparse_int;
    using J = rocsparse_int;

public:
    rocsparse_matrix_factory_rocalution(const char* filename, bool toint = false)
        : m_filename(filename)
        , m_toint(toint){};

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            rocsparse_direction  dirb,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {

        //
        // Initialize in case init_csr requires it as input.
        //
        std::vector<I> csr_row_ptr;
        std::vector<J> csr_col_ind;
        std::vector<T> csr_val;
        J              M = Mb * row_block_dim;
        J              N = Nb * col_block_dim;

        host_csr_matrix<T, I, J> hA_uncompressed(M, N, 0, base);
        this->init_csr(hA_uncompressed.ptr,
                       hA_uncompressed.ind,
                       hA_uncompressed.val,
                       hA_uncompressed.m,
                       hA_uncompressed.n,
                       hA_uncompressed.nnz,
                       hA_uncompressed.base);

        device_gebsr_matrix<T, I, J> that_on_device;
        {
            device_csr_matrix<T, I, J> dA_uncompressed(hA_uncompressed);
            rocsparse_matrix_utils::convert(
                dA_uncompressed, dirb, row_block_dim, col_block_dim, base, that_on_device);
        }

        Mb            = that_on_device.mb;
        Nb            = that_on_device.nb;
        nnzb          = that_on_device.nnzb;
        row_block_dim = that_on_device.row_block_dim;
        col_block_dim = that_on_device.col_block_dim;
        that_on_device.ptr.transfer_to(bsr_row_ptr);
        that_on_device.ind.transfer_to(bsr_col_ind);
        that_on_device.val.transfer_to(bsr_val);
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

template <typename T, typename I, typename J>
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
                            rocsparse_direction  dirb,
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
                            rocsparse_direction  dirb,
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
                            rocsparse_direction  dirb,
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
                            rocsparse_direction  dirb,
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

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory_zero : public rocsparse_matrix_factory_base<T, I, J>
{
public:
    rocsparse_matrix_factory_zero(){};

    virtual void init_csr(std::vector<I>&      csr_row_ptr,
                          std::vector<J>&      csr_col_ind,
                          std::vector<T>&      csr_val,
                          J&                   M,
                          J&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        csr_row_ptr.resize(M + 1, static_cast<I>(base));
        csr_col_ind.resize(0);
        csr_val.resize(0);

        nnz = 0;
    }

    virtual void init_gebsr(std::vector<I>&      bsr_row_ptr,
                            std::vector<J>&      bsr_col_ind,
                            std::vector<T>&      bsr_val,
                            rocsparse_direction  dirb,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        bsr_row_ptr.resize(Mb + 1, static_cast<I>(base));
        bsr_col_ind.resize(0);
        bsr_val.resize(0);

        nnzb = 0;
    }

    virtual void init_coo(std::vector<I>&      coo_row_ind,
                          std::vector<I>&      coo_col_ind,
                          std::vector<T>&      coo_val,
                          I&                   M,
                          I&                   N,
                          I&                   nnz,
                          rocsparse_index_base base)
    {
        coo_row_ind.resize(0);
        coo_col_ind.resize(0);
        coo_val.resize(0);

        nnz = 0;
    }
};

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
struct rocsparse_matrix_factory : public rocsparse_matrix_factory_base<T, I, J>
{
private:
    const Arguments& m_arg;

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
        : m_arg(arg)
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
            rocsparse_matrix_init_kind matrix_init_kind = arg.matrix_init_kind;
            this->m_instance
                = new rocsparse_matrix_factory_random<T, I, J>(full_rank, to_int, matrix_init_kind);
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

        case rocsparse_matrix_zero:
        {
            this->m_instance = new rocsparse_matrix_factory_zero<T, I, J>();
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
                            rocsparse_direction  dirb,
                            J&                   Mb,
                            J&                   Nb,
                            I&                   nnzb,
                            J&                   row_block_dim,
                            J&                   col_block_dim,
                            rocsparse_index_base base)
    {
        this->m_instance->init_gebsr(bsr_row_ptr,
                                     bsr_col_ind,
                                     bsr_val,
                                     dirb,
                                     Mb,
                                     Nb,
                                     nnzb,
                                     row_block_dim,
                                     col_block_dim,
                                     base);
    }

    virtual void init_bsr(std::vector<I>&      bsr_row_ptr,
                          std::vector<J>&      bsr_col_ind,
                          std::vector<T>&      bsr_val,
                          rocsparse_direction  dirb,
                          J&                   Mb,
                          J&                   Nb,
                          I&                   nnzb,
                          J&                   block_dim,
                          rocsparse_index_base base)
    {
        this->m_instance->init_gebsr(
            bsr_row_ptr, bsr_col_ind, bsr_val, dirb, Mb, Nb, nnzb, block_dim, block_dim, base);
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

    // @brief Init host csr matrix.
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

    void init_csc(host_csc_matrix<T, I, J>& that, J& m, J& n, rocsparse_index_base base)
    {
        that.base = base;
        this->m_instance->init_csr(that.ptr, that.ind, that.val, n, m, that.nnz, that.base);
        that.m = m;
        that.n = n;
    }

    void init_csr(host_csr_matrix<T, I, J>& that)
    {
        that.base = this->m_arg.baseA;
        that.m    = this->m_arg.M;
        that.n    = this->m_arg.N;
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
        J                        block_dim = this->m_arg.block_dim;
        J                        M         = mb_ * block_dim;
        J                        N         = nb_ * block_dim;
        rocsparse_index_base     base      = this->m_arg.baseA;
        host_csr_matrix<T, I, J> hA_uncompressed;
        this->init_csr(hA_uncompressed, M, N, base);

        {
            device_csr_matrix<T, I, J> dA_uncompressed(hA_uncompressed);
            device_csr_matrix<T, I, J> dA_compressed;
            rocsparse_matrix_utils::compress(dA_compressed, dA_uncompressed, this->m_arg.baseA);
            rocsparse_matrix_utils::convert(
                dA_compressed, this->m_arg.direction, block_dim, base, that_on_device);
        }

        that(that_on_device);

        mb_ = that.mb;
        nb_ = that.nb;
    }

    void init_bsr(host_gebsr_matrix<T, I, J>& that, J& mb_, J& nb_)
    {
        device_gebsr_matrix<T, I, J> dB;
        this->init_bsr(that, dB, mb_, nb_);
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
                                     that.block_direction,
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
        that.block_direction = this->m_arg.direction;
        that.mb              = this->m_arg.M;
        that.nb              = this->m_arg.N;
        that.nnzb            = this->m_arg.nnz;
        that.row_block_dim   = this->m_arg.row_block_dimA;
        that.col_block_dim   = this->m_arg.col_block_dimA;
        that.base            = this->m_arg.baseA;
        this->m_instance->init_gebsr(that.ptr,
                                     that.ind,
                                     that.val,
                                     that.block_direction,
                                     that.mb,
                                     that.nb,
                                     that.nnzb,
                                     that.row_block_dim,
                                     that.col_block_dim,
                                     that.base);
    }

    void init_gebsr(
        host_gebsr_matrix<T, I, J>& that, J& mb, J& nb, J& row_block_dim, J& col_block_dim)
    {

        that.base          = this->m_arg.baseA;
        that.mb            = mb;
        that.nb            = nb;
        that.nnzb          = this->m_arg.nnz;
        that.row_block_dim = row_block_dim;
        that.col_block_dim = col_block_dim;

        this->m_instance->init_gebsr(that.ptr,
                                     that.ind,
                                     that.val,
                                     that.block_direction,
                                     that.mb,
                                     that.nb,
                                     that.nnzb,
                                     that.row_block_dim,
                                     that.col_block_dim,
                                     that.base);

        mb            = that.mb;
        nb            = that.nb;
        row_block_dim = that.row_block_dim;
        col_block_dim = that.col_block_dim;
    }

    void init_csr(host_csr_matrix<T, I, J>& that, J& m, J& n)
    {
        this->init_csr(that, m, n, this->m_arg.baseA);
    }

    void init_gebsr_spezial(host_gebsr_matrix<T, I, J>& that, J& Mb, J& Nb)
    {
        I idx = 0;

        host_csr_matrix<T, I, J> hA;
        rocsparse_direction      direction     = this->m_arg.direction;
        rocsparse_index_base     base          = this->m_arg.baseA;
        J                        row_block_dim = this->m_arg.row_block_dimA;
        J                        col_block_dim = this->m_arg.col_block_dimA;
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

    void init_coo(host_coo_matrix<T, I>& that)
    {
        that.base = this->arg.baseA;
        that.m    = this->arg.M;
        that.n    = this->arg.N;
        this->m_instance->init_coo(
            that.row_ind, that.col_ind, that.val, that.m, that.n, that.nnz, that.base);
    }

    void init_coo(host_coo_matrix<T, I>& that, I& M, I& N, rocsparse_index_base base)
    {
        that.base = base;
        that.m    = M;
        that.n    = N;
        this->m_instance->init_coo(
            that.row_ind, that.col_ind, that.val, that.m, that.n, that.nnz, that.base);
        M = that.m;
        N = that.n;
    }

    void init_coo_aos(host_coo_aos_matrix<T, I>& that, I& M, I& N, rocsparse_index_base base)
    {
        host_csr_matrix<T, I, I> hA;
        this->init_csr(hA, M, N, base);
        that.define(hA.m, hA.n, hA.nnz, hA.base);
        host_csr_to_coo_aos(hA.m, hA.nnz, hA.ptr, hA.ind, that.ind, hA.base);
        that.val.transfer_from(hA.val);
    }

    void init_ell(host_ell_matrix<T, I>& that, I& M, I& N, rocsparse_index_base base)
    {
        host_csr_matrix<T, I, I> hA;
        this->init_csr(hA, M, N, base);
        that.define(hA.m, hA.n, 0, hA.base);
        host_csr_to_ell(
            hA.m, hA.ptr, hA.ind, hA.val, that.ind, that.val, that.width, hA.base, that.base);
        that.nnz = that.width * that.m;
    }

    void init_hyb(
        rocsparse_hyb_mat hyb, I& M, I& N, I& nnz, rocsparse_index_base base, bool& conform)
    {
        conform                                = true;
        rocsparse_hyb_partition part           = this->m_arg.part;
        rocsparse_int           user_ell_width = this->m_arg.algo;

        host_csr_matrix<T, I, I> hA;
        this->init_csr(hA, M, N, base);
        nnz = hA.nnz;

        // ELL width limit
        rocsparse_int width_limit = 2 * (hA.nnz - 1) / M + 1;

        // Limit ELL user width
        if(part == rocsparse_hyb_partition_user)
        {
            user_ell_width *= (hA.nnz / M);
            user_ell_width = std::min(width_limit, user_ell_width);
        }

        if(part == rocsparse_hyb_partition_max)
        {
            // Compute max ELL width
            rocsparse_int ell_max_width = 0;
            for(rocsparse_int i = 0; i < M; ++i)
            {
                ell_max_width = std::max(hA.ptr[i + 1] - hA.ptr[i], ell_max_width);
            }

            if(ell_max_width > width_limit)
            {
                conform = false;
                return;
            }
        }

        device_csr_matrix<T, I, I> dA(hA);

        rocsparse_handle handle;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_handle(&handle));
        rocsparse_mat_descr descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
        // Set matrix index base
        CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

        // Convert CSR matrix to HYB
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(
            handle, M, N, descr, dA.val, dA.ptr, dA.ind, hyb, user_ell_width, part));

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));
        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_handle(handle));
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

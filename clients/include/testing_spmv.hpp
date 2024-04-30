/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "auto_testing_bad_arg.hpp"

template <rocsparse_format FORMAT, typename I, typename J, typename T>
struct testing_matrix_type_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csr, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csr_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csr_matrix<U, I, J>;
};

//
// TRAITS FOR CSC FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_csc, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_csc_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_csc_matrix<U, I, J>;
};

//
// TRAITS FOR BSR FORMAT.
//
template <typename I, typename J, typename T>
struct testing_matrix_type_traits<rocsparse_format_bsr, I, J, T>
{
    template <typename U>
    using host_sparse_matrix = host_gebsr_matrix<U, I, J>;
    template <typename U>
    using device_sparse_matrix = device_gebsr_matrix<U, I, J>;
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_matrix<U, I>;
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_coo_aos_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_coo_aos_matrix<U, I>;
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename T>
struct testing_matrix_type_traits<rocsparse_format_ell, I, I, T>
{
    template <typename U>
    using host_sparse_matrix = host_ell_matrix<U, I>;
    template <typename U>
    using device_sparse_matrix = device_ell_matrix<U, I>;
};

template <rocsparse_format FORMAT,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
struct testing_spmv_dispatch_traits;

//
// TRAITS FOR CSR FORMAT.
//
template <typename I, typename J, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_csr, I, J, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csr, I, J, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, J>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      J&                                 m,
                                      J&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_csr(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_csrmv<T, I, J, A, X, Y>(trans,
                                     hA.m,
                                     hA.n,
                                     hA.nnz,
                                     *h_alpha,
                                     hA.ptr,
                                     hA.ind,
                                     hA.val,
                                     hx,
                                     *h_beta,
                                     hy,
                                     hA.base,
                                     matrix_type,
                                     alg,
                                     false);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return csrmv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR CSC FORMAT.
//
template <typename I, typename J, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_csc, I, J, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_csc, I, J, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, J>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      J&                                 m,
                                      J&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_csc(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_cscmv<T, I, J, A, X, Y>(trans,
                                     hA.m,
                                     hA.n,
                                     hA.nnz,
                                     *h_alpha,
                                     hA.ptr,
                                     hA.ind,
                                     hA.val,
                                     hx,
                                     *h_beta,
                                     hy,
                                     hA.base,
                                     matrix_type,
                                     alg);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return cscmv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR BSR FORMAT.
//
template <typename I, typename J, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_bsr, I, J, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_bsr, I, J, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, J>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      J&                                 m,
                                      J&                                 n,
                                      rocsparse_index_base               base)
    {
        J block_dim = matrix_factory.m_arg.block_dim;
        matrix_factory.init_gebsr(hA, m, n, block_dim, block_dim, base);
        m *= block_dim;
        n *= block_dim;
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.mb * dA.row_block_dim,
                            display_key_t::N,
                            dA.nb * dA.col_block_dim,
                            display_key_t::nnz,
                            dA.nnzb * dA.row_block_dim * dA.col_block_dim,
                            display_key_t::bdim,
                            dA.row_block_dim,
                            display_key_t::bdir,
                            rocsparse_direction2string(dA.block_direction),
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_bsrmv<T, I, J, A, X, Y>(hA.block_direction,
                                     trans,
                                     hA.mb,
                                     hA.nb,
                                     hA.nnzb,
                                     *h_alpha,
                                     hA.ptr,
                                     hA.ind,
                                     hA.val,
                                     hA.row_block_dim,
                                     hx,
                                     *h_beta,
                                     hy,
                                     hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return bsrmv_gbyte_count<A, X, T>(hA.mb, hA.nb, hA.nnzb, hA.row_block_dim, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(
            hA.mb * hA.row_block_dim, hA.nnzb * hA.row_block_dim * hA.col_block_dim, nonzero_beta);
    }
};

//
// TRAITS FOR COO FORMAT.
//
template <typename I, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_coo, I, I, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo, I, I, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, I>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      I&                                 m,
                                      I&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_coo(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_coomv<T, I, A, X, Y>(trans,
                                  hA.m,
                                  hA.n,
                                  hA.nnz,
                                  *h_alpha,
                                  hA.row_ind,
                                  hA.col_ind,
                                  hA.val,
                                  hx,
                                  *h_beta,
                                  hy,
                                  hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return coomv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR COO AOS FORMAT.
//
template <typename I, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_coo_aos, I, I, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_coo_aos, I, I, A>;

    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, I>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      I&                                 m,
                                      I&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_coo_aos(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_coomv_aos<T, I, A, X, Y>(
            trans, hA.m, hA.n, hA.nnz, *h_alpha, hA.ind, hA.val, hx, *h_beta, hy, hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return coomv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }
};

//
// TRAITS FOR ELL FORMAT.
//
template <typename I, typename A, typename X, typename Y, typename T>
struct testing_spmv_dispatch_traits<rocsparse_format_ell, I, I, A, X, Y, T>
{
    using traits = testing_matrix_type_traits<rocsparse_format_ell, I, I, A>;
    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

    static void sparse_initialization(rocsparse_matrix_factory<A, I, I>& matrix_factory,
                                      host_sparse_matrix<A>&             hA,
                                      I&                                 m,
                                      I&                                 n,
                                      rocsparse_index_base               base)
    {
        matrix_factory.init_ell(hA, m, n, base);
    }

    template <typename... Ts>
    static void display_info(const Arguments&         arg,
                             display_key_t::key_t     trans,
                             const char*              trans_value,
                             device_sparse_matrix<A>& dA,
                             Ts&&... ts)
    {
        display_timing_info(trans,
                            trans_value,
                            display_key_t::M,
                            dA.m,
                            display_key_t::N,
                            dA.n,
                            display_key_t::nnz,
                            dA.nnz,
                            ts...);
    }

    static void host_calculation(rocsparse_operation    trans,
                                 T*                     h_alpha,
                                 host_sparse_matrix<A>& hA,
                                 X*                     hx,
                                 T*                     h_beta,
                                 Y*                     hy,
                                 rocsparse_spmv_alg     alg,
                                 rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general)
    {
        host_ellmv<T, I, A, X, Y>(
            trans, hA.m, hA.n, *h_alpha, hA.ind, hA.val, hA.width, hx, *h_beta, hy, hA.base);
    }

    static double byte_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return ellmv_gbyte_count<A, X, Y>(hA.m, hA.n, hA.nnz, nonzero_beta);
    }

    static double gflop_count(host_sparse_matrix<A>& hA, bool nonzero_beta)
    {
        return spmv_gflop_count(hA.m, hA.nnz, nonzero_beta);
    }
};

template <rocsparse_format FORMAT,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
struct testing_spmv_dispatch
{
private:
    using traits = testing_spmv_dispatch_traits<FORMAT, I, J, A, X, Y, T>;
    template <typename U>
    using host_sparse_matrix = typename traits::template host_sparse_matrix<U>;
    template <typename U>
    using device_sparse_matrix = typename traits::template device_sparse_matrix<U>;

public:
    static void testing_spmv_bad_arg(const Arguments& arg)
    {
        const T local_alpha = static_cast<T>(6);
        const T local_beta  = static_cast<T>(2);

        rocsparse_local_handle local_handle;

        rocsparse_handle     handle = local_handle;
        rocsparse_operation  trans  = rocsparse_operation_none;
        const void*          alpha  = (const void*)&local_alpha;
        const void*          beta   = (const void*)&local_beta;
        rocsparse_spmv_alg   alg    = rocsparse_spmv_alg_default;
        rocsparse_spmv_stage stage  = rocsparse_spmv_stage_compute;
        size_t               local_buffer_size;
        size_t*              buffer_size  = &local_buffer_size;
        void*                temp_buffer  = (void*)0x4;
        rocsparse_datatype   compute_type = get_datatype<T>();

#define PARAMS \
    handle, trans, alpha, mat, x, beta, y, compute_type, alg, stage, buffer_size, temp_buffer

        //
        // AUTOMATIC BAD ARGS.
        //
        {
            device_dense_matrix<X>  dx;
            device_dense_matrix<Y>  dy;
            device_sparse_matrix<A> dA;
            rocsparse_local_spmat   local_mat(dA);
            rocsparse_local_dnvec   local_x(dx);
            rocsparse_local_dnvec   local_y(dy);

            rocsparse_spmat_descr mat = local_mat;
            rocsparse_dnvec_descr x   = local_x;
            rocsparse_dnvec_descr y   = local_y;

            //
            // WITH 2 ARGUMENTS BEING SKIPPED DURING THE CHECK.
            //
            static const int nex   = 2;
            static const int ex[2] = {10, 11};
            select_bad_arg_analysis(rocsparse_spmv, nex, ex, PARAMS);

            buffer_size = nullptr;
            temp_buffer = nullptr;
            EXPECT_ROCSPARSE_STATUS(rocsparse_spmv(PARAMS), rocsparse_status_invalid_pointer);
        }

#undef PARAMS
    }

    static void testing_spmv(const Arguments& arg)
    {
        J                      M           = arg.M;
        J                      N           = arg.N;
        rocsparse_operation    trans       = arg.transA;
        rocsparse_index_base   base        = arg.baseA;
        rocsparse_spmv_alg     alg         = arg.spmv_alg;
        rocsparse_matrix_type  matrix_type = arg.matrix_type;
        rocsparse_fill_mode    uplo        = arg.uplo;
        rocsparse_storage_mode storage     = arg.storage;
        rocsparse_datatype     ttype       = get_datatype<T>();

        // Create rocsparse handle
        rocsparse_local_handle handle(arg);

        host_scalar<T> h_alpha(arg.get_alpha<T>());
        host_scalar<T> h_beta(arg.get_beta<T>());

        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);

#define PARAMS(alpha_, A_, x_, beta_, y_, stage) \
    handle, trans, alpha_, A_, x_, beta_, y_, ttype, alg, stage, &buffer_size, dbuffer

        //
        // INITIALIZATE THE SPARSE MATRIX
        //
        host_sparse_matrix<A> hA;
        {
            int dev;
            hipGetDevice(&dev);

            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, dev);

            const bool has_datafile = rocsparse_arguments_has_datafile(arg);
            bool       to_int       = false;
            to_int |= (prop.warpSize == 32);
            to_int |= (alg != rocsparse_spmv_alg_csr_stream);
            to_int |= (trans != rocsparse_operation_none && has_datafile);
            to_int |= (matrix_type == rocsparse_matrix_type_symmetric && has_datafile);
            static constexpr bool             full_rank = false;
            rocsparse_matrix_factory<A, I, J> matrix_factory(
                arg, arg.unit_check ? to_int : false, full_rank);
            traits::sparse_initialization(matrix_factory, hA, M, N, base);
        }

        if((matrix_type == rocsparse_matrix_type_symmetric && M != N)
           || (matrix_type == rocsparse_matrix_type_triangular && M != N))
        {
            return;
        }

        device_sparse_matrix<A> dA(hA);

        host_dense_matrix<X> hx((trans == rocsparse_operation_none) ? N : M, 1);
        rocsparse_matrix_utils::init_exact(hx);
        device_dense_matrix<X> dx(hx);

        host_dense_matrix<Y> hy((trans == rocsparse_operation_none) ? M : N, 1);
        rocsparse_matrix_utils::init_exact(hy);
        device_dense_matrix<Y> dy(hy);

        rocsparse_local_spmat matA(dA);
        rocsparse_local_dnvec x(dx);
        rocsparse_local_dnvec y(dy);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(
                matA, rocsparse_spmat_matrix_type, &matrix_type, sizeof(matrix_type)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spmat_set_attribute(matA, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_attribute(
                                    matA, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                                rocsparse_status_success);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Run buffer size
        void*  dbuffer     = nullptr;
        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmv(PARAMS(h_alpha, matA, x, h_beta, y, rocsparse_spmv_stage_buffer_size)));
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

        // Run preprocess
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spmv(PARAMS(h_alpha, matA, x, h_beta, y, rocsparse_spmv_stage_preprocess)));

        if(arg.unit_check)
        {
            // Run solve
            CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmv(
                PARAMS(h_alpha, matA, x, h_beta, y, rocsparse_spmv_stage_compute)));

            host_dense_matrix<Y> hy_copy(hy);
            traits::host_calculation(trans, h_alpha, hA, hx, h_beta, hy, alg, matrix_type);

            hy.near_check(dy);

            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("Y_pointer_mode_host", dy);
            }

            dy.transfer_from(hy_copy);
            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            CHECK_ROCSPARSE_ERROR(testing::rocsparse_spmv(
                PARAMS(d_alpha, matA, x, d_beta, y, rocsparse_spmv_stage_compute)));
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            hy.near_check(dy);
            if(ROCSPARSE_REPRODUCIBILITY)
            {
                rocsparse_reproducibility::save("Y_pointer_mode_device", dy);
            }
        }

        if(arg.timing)
        {
            const int number_cold_calls = 2;
            const int number_hot_calls  = arg.iters;

            // Warm up
            for(int iter = 0; iter < number_cold_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_spmv(
                    PARAMS(h_alpha, matA, x, h_beta, y, rocsparse_spmv_stage_compute)));
            }

            double gpu_time_used = get_time_us();

            // Performance run
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_spmv(
                    PARAMS(h_alpha, matA, x, h_beta, y, rocsparse_spmv_stage_compute)));
            }

            gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

            const double gflop_count = traits::gflop_count(hA, *h_beta != static_cast<T>(0));
            const double gbyte_count = traits::byte_count(hA, *h_beta != static_cast<T>(0));

            const double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
            const double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

            traits::display_info(arg,
                                 display_key_t::trans_A,
                                 rocsparse_operation2string(trans),
                                 dA,
                                 display_key_t::alpha,
                                 *h_alpha,
                                 display_key_t::beta,
                                 *h_beta,
                                 display_key_t::algorithm,
                                 rocsparse_spmvalg2string(alg),
                                 display_key_t::gflops,
                                 gpu_gflops,
                                 display_key_t::bandwidth,
                                 gpu_gbyte,
                                 display_key_t::time_ms,
                                 get_gpu_time_msec(gpu_time_used));
        }

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    }
};

/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "testing.hpp"
#include "utility.hpp"
#include <rocsparse.hpp>

template <typename T>
void testing_gebsrmm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    T h_alpha = static_cast<T>(1);
    T h_beta  = static_cast<T>(1);

    // Declaration.
    rocsparse_handle     handle        = local_handle;
    rocsparse_direction  dir           = rocsparse_direction_row;
    rocsparse_operation  trans_A       = rocsparse_operation_none;
    rocsparse_operation  trans_B       = rocsparse_operation_none;
    rocsparse_int        mb            = safe_size;
    rocsparse_int        n             = safe_size;
    rocsparse_int        kb            = safe_size;
    rocsparse_int        nnzb          = safe_size;
    const T*             alpha         = &h_alpha;
    rocsparse_mat_descr  descr         = local_descr;
    const T*             bsr_val       = (const T*)0x4;
    const rocsparse_int* bsr_row_ptr   = (const rocsparse_int*)0x4;
    const rocsparse_int* bsr_col_ind   = (const rocsparse_int*)0x4;
    rocsparse_int        row_block_dim = safe_size;
    rocsparse_int        col_block_dim = safe_size;
    const T*             B             = (const T*)0x4;
    rocsparse_int        ldb           = safe_size;
    const T*             beta          = &h_beta;
    T*                   C             = (T*)0x4;
    rocsparse_int        ldc           = safe_size;

#define PARAMS                                                                          \
    handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, descr, bsr_val, bsr_row_ptr, \
        bsr_col_ind, row_block_dim, col_block_dim, B, ldb, beta, C, ldc

    auto_testing_bad_arg(rocsparse_gebsrmm<T>, PARAMS);

    //
    // CHECK NOT IMPLEMENTED CASE
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric));
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_not_implemented);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    //
    // CHECK NOT IMPLEMENTED CASE
    //
    {
        auto tmp = trans_A;

        trans_A = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_not_implemented);

        trans_A = rocsparse_operation_conjugate_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_not_implemented);

        trans_A = tmp;
    }

    //
    // CHECK NOT IMPLEMENTED CASE
    //
    {
        trans_B = rocsparse_operation_conjugate_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_not_implemented);
    }

    //
    // Testing wrong leading dimensions.
    //
    {
        //
        // op(A) = A, op(B) = B
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_none;
        trans_B = rocsparse_operation_none;

        //  ldb < k
        ldb = kb * col_block_dim - 1;
        ldc = mb * row_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = kb * col_block_dim;
        ldc = mb * row_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);
    }

    {
        //
        // op(A) = A, op(B) = B^T
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_none;
        trans_B = rocsparse_operation_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = mb * row_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = n;
        ldc = mb * row_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

#ifdef ROCSPARSE_GEBSRMM_CONJUGATE_TRANSPOSE_B_SUPPORTED
        trans_B = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = mb * row_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < m
        ldb = n;
        ldc = mb * row_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif
    }

#ifdef ROCSPARSE_GEBSRMM_TRANSPOSE_A_SUPPORTED
    {
        //
        // op(A) = A^T, op(B) = B
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_none;

        //  ldb < m
        ldb = mb * row_block_dim - 1;
        ldc = kb * col_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = mb * row_block_dim;
        ldc = kb * col_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

#ifdef ROCSPARSE_GEBSRMM_CONJUGATE_TRANSPOSE_A_SUPPORTED
        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < m
        ldb = mb * row_block_dim - 1;
        ldc = kb * col_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = mb * row_block_dim;
        ldc = kb * col_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif
    }

    {
        //
        // op(A) = A^T, op(B) = B^T
        //
        mb      = 3;
        n       = 14;
        kb      = 32;
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = kb * col_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * col_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

#ifdef ROCSPARSE_GEBSRMM_CONJUGATE_TRANSPOSE_A_SUPPORTED
        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = kb * col_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * col_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif

#ifdef ROCSPARSE_GEBSRMM_CONJUGATE_TRANSPOSE_B_SUPPORTED
        trans_A = rocsparse_operation_transpose;
        trans_B = rocsparse_operation_conjugate_transpose;

        //  ldb < n
        ldb = n - 1;
        ldc = kb * col_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * col_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        trans_A = rocsparse_operation_conjugate_transpose;
        //  ldb < n
        ldb = n - 1;
        ldc = kb * col_block_dim;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);

        //  ldc < k
        ldb = n;
        ldc = kb * col_block_dim - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(PARAMS), rocsparse_status_invalid_size);
#endif
    }
#endif

#undef PARAMS

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 dir,
                                                 trans_A,
                                                 trans_B,
                                                 mb,
                                                 n,
                                                 kb,
                                                 nnzb,
                                                 alpha,
                                                 descr,
                                                 nullptr,
                                                 bsr_row_ptr,
                                                 nullptr,
                                                 row_block_dim,
                                                 col_block_dim,
                                                 B,
                                                 ldb,
                                                 beta,
                                                 C,
                                                 ldc),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_gebsrmm(const Arguments& arg)
{
    rocsparse_int        M             = arg.M;
    rocsparse_int        N             = arg.N;
    rocsparse_int        K             = arg.K;
    rocsparse_int        row_block_dim = arg.row_block_dimA;
    rocsparse_int        col_block_dim = arg.col_block_dimA;
    rocsparse_operation  transA        = arg.transA;
    rocsparse_operation  transB        = arg.transB;
    rocsparse_direction  direction     = arg.direction;
    rocsparse_index_base base          = arg.baseA;

    rocsparse_int Mb = -1;
    rocsparse_int Kb = -1;

    if(row_block_dim > 0 && col_block_dim > 0)
    {
        Mb = (M + row_block_dim - 1) / row_block_dim;
        Kb = (K + col_block_dim - 1) / col_block_dim;
    }

    host_scalar<T> h_alpha, h_beta;
    *h_alpha = arg.get_alpha<T>();
    *h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || N <= 0 || Kb <= 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        static const size_t safe_size    = 100;
        rocsparse_int*      dbsr_row_ptr = (rocsparse_int*)0x4;
        rocsparse_int*      dbsr_col_ind = (rocsparse_int*)0x4;
        T*                  dbsr_val     = (T*)0x4;
        T*                  dB           = (T*)0x4;
        T*                  dC           = (T*)0x4;

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_gebsrmm<T>(handle,
                                 direction,
                                 transA,
                                 transB,
                                 Mb,
                                 N,
                                 Kb,
                                 safe_size,
                                 h_alpha,
                                 descr,
                                 dbsr_val,
                                 dbsr_row_ptr,
                                 dbsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 dB,
                                 safe_size,
                                 h_beta,
                                 dC,
                                 safe_size),
            (Mb < 0 || N < 0 || Kb < 0 || row_block_dim <= 0 || col_block_dim <= 0)
                ? rocsparse_status_invalid_size
                : rocsparse_status_success);
        return;
    }

#define PARAMS(alpha, A, B, beta, C)                                                         \
    handle, A.block_direction, transA, transB, A.mb, C.n, A.nb, A.nnzb, alpha, descr, A.val, \
        A.ptr, A.ind, A.row_block_dim, A.col_block_dim, B, B.ld, beta, C, C.ld

    host_gebsr_matrix<T>        hA;
    rocsparse_matrix_factory<T> matrix_factory(arg);
    matrix_factory.init_gebsr_spezial(hA, Mb, Kb);

    M = hA.mb * hA.row_block_dim;
    K = hA.nb * hA.col_block_dim;

    // Allocate host memory for dense matrices
    host_dense_matrix<T> hC(M, N);
    rocsparse_matrix_utils::init(hC);

    host_dense_matrix<T> hB((transB == rocsparse_operation_none) ? K : N,
                            (transB == rocsparse_operation_none) ? N : K);
    rocsparse_matrix_utils::init(hB);

    device_gebsr_matrix<T> dA(hA);
    device_dense_matrix<T> dB(hB), dC(hC);

    if(arg.unit_check)
    {
        //
        // Pointer mode host
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gebsrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));

        //
        // Compute on host
        //
        {
            host_dense_matrix<T> hC_copy(hC);
            host_gebsrmm<T>(PARAMS(h_alpha, hA, hB, h_beta, hC));
            hC.near_check(dC);
            dC = hC_copy;
        }

        // Copy data from CPU to device
        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gebsrmm<T>(PARAMS(d_alpha, dA, dB, d_beta, dC)));
        hC.near_check(dC);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = gebsrmm_gflop_count(dC.n,
                                                 dA.nnzb,
                                                 dA.row_block_dim,
                                                 dA.col_block_dim,
                                                 dC.m * dC.n,
                                                 *h_beta != static_cast<T>(0));

        double gbyte_count = gebsrmm_gbyte_count<T>(dA.mb,
                                                    dA.nnzb,
                                                    dA.row_block_dim,
                                                    dA.col_block_dim,
                                                    dB.m * dB.n,
                                                    dC.m * dC.n,
                                                    *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "K",
                            K,
                            "dir",
                            dA.block_direction,
                            "transA",
                            transA,
                            "transB",
                            transB,
                            "nnzb",
                            dA.nnzb,
                            "row_block_dim",
                            dA.row_block_dim,
                            "col_block_dim",
                            dA.col_block_dim,
                            "nnz_B",
                            dB.m * dB.n,
                            "nnz_C",
                            dC.m * dC.n,
                            "alpha",
                            *h_alpha,
                            "beta",
                            *h_beta,
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            gpu_time_used / 1e3);
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_gebsrmm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsrmm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_gebsrmm_extra(const Arguments& arg) {}

/* ************************************************************************
* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "auto_testing_bad_arg.hpp"
#include <tuple>

template <typename I, typename T>
void testing_spmm_batched_coo_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    I                    m           = safe_size;
    I                    n           = safe_size;
    I                    k           = safe_size;
    I                    nnz         = safe_size;
    void*                coo_val     = (void*)0x4;
    void*                coo_row_ind = (void*)0x4;
    void*                coo_col_ind = (void*)0x4;
    void*                B           = (void*)0x4;
    void*                C           = (void*)0x4;
    size_t*              buffer_size = (size_t*)0x4;
    void*                temp_buffer = (void*)0x4;
    rocsparse_operation  trans_A     = rocsparse_operation_none;
    rocsparse_operation  trans_B     = rocsparse_operation_none;
    rocsparse_index_base base        = rocsparse_index_base_zero;
    rocsparse_order      order       = rocsparse_order_column;
    rocsparse_spmm_alg   alg         = rocsparse_spmm_alg_default;
    rocsparse_spmm_stage stage       = rocsparse_spmm_stage_auto;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    T alpha = static_cast<T>(1.0);
    T beta  = static_cast<T>(0.0);

    // SpMM structures
    rocsparse_local_spmat local_mat_A(
        m, k, nnz, coo_row_ind, coo_col_ind, coo_val, itype, base, ttype);
    rocsparse_local_dnmat local_mat_B(k, n, k, B, ttype, order);
    rocsparse_local_dnmat local_mat_C(m, n, m, C, ttype, order);

    rocsparse_spmat_descr mat_A = local_mat_A;
    rocsparse_dnmat_descr mat_B = local_mat_B;
    rocsparse_dnmat_descr mat_C = local_mat_C;

#define PARAMS                                                                                    \
    handle, trans_A, trans_B, &alpha, mat_A, mat_B, &beta, mat_C, ttype, alg, stage, buffer_size, \
        temp_buffer

    int     batch_count_A;
    int     batch_count_B;
    int     batch_count_C;
    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;

    // C_i = A * B_i
    batch_count_A  = 1;
    batch_count_B  = 10;
    batch_count_C  = 5;
    batch_stride_A = 0;
    batch_stride_B = k * n;
    batch_stride_C = m * n;
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(PARAMS), rocsparse_status_invalid_value);

    // C_i = A_i * B
    batch_count_A  = 10;
    batch_count_B  = 1;
    batch_count_C  = 5;
    batch_stride_A = nnz;
    batch_stride_B = 0;
    batch_stride_C = m * n;
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(PARAMS), rocsparse_status_invalid_value);

    // C_i = A_i * B_i
    batch_count_A  = 10;
    batch_count_B  = 10;
    batch_count_C  = 5;
    batch_stride_A = nnz;
    batch_stride_B = k * n;
    batch_stride_C = m * n;
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_strided_batch(mat_A, batch_count_A, batch_stride_A),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_B, batch_count_B, batch_stride_B),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_dnmat_set_strided_batch(mat_C, batch_count_C, batch_stride_C),
                            rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(PARAMS), rocsparse_status_invalid_value);
#undef PARAMS
}

template <typename I, typename T>
void testing_spmm_batched_coo(const Arguments& arg)
{
    I                    M       = arg.M;
    I                    N       = arg.N;
    I                    K       = arg.K;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_operation  trans_B = arg.transB;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_spmm_alg   alg     = arg.spmm_alg;
    rocsparse_order      order   = arg.order;

    I batch_count_A = arg.batch_count_A;
    I batch_count_B = arg.batch_count_B;
    I batch_count_C = arg.batch_count_C;

    T halpha = arg.get_alpha<T>();
    T hbeta  = arg.get_beta<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    if(M <= 0 || N <= 0 || K <= 0)
    {
        return;
    }

    bool Ci_A_Bi  = (batch_count_A == 1 && batch_count_B == batch_count_C);
    bool Ci_Ai_B  = (batch_count_B == 1 && batch_count_A == batch_count_C);
    bool Ci_Ai_Bi = (batch_count_A == batch_count_C && batch_count_A == batch_count_B);

    if(!Ci_A_Bi && !Ci_Ai_B && !Ci_Ai_Bi)
    {
        return;
    }

    // Allocate host memory for matrix
    rocsparse_matrix_factory<T, I> matrix_factory(arg);

    // Allocate host memory for matrix
    host_vector<I> hcoo_row_ind_temp;
    host_vector<I> hcoo_col_ind_temp;
    host_vector<T> hcoo_val_temp;

    I nnz_A;
    matrix_factory.init_coo(hcoo_row_ind_temp,
                            hcoo_col_ind_temp,
                            hcoo_val_temp,
                            (trans_A == rocsparse_operation_none) ? M : K,
                            (trans_A == rocsparse_operation_none) ? K : M,
                            nnz_A,
                            base);

    // Some matrix properties
    I A_m = (trans_A == rocsparse_operation_none) ? M : K;
    I A_n = (trans_A == rocsparse_operation_none) ? K : M;
    I B_m = (trans_B == rocsparse_operation_none) ? K : N;
    I B_n = (trans_B == rocsparse_operation_none) ? N : K;
    I C_m = M;
    I C_n = N;

    I ldb = (order == rocsparse_order_column)
                ? ((trans_B == rocsparse_operation_none) ? (2 * K) : (2 * N))
                : ((trans_B == rocsparse_operation_none) ? (2 * N) : (2 * K));
    I ldc = (order == rocsparse_order_column) ? (2 * M) : (2 * N);

    I nrowB = (order == rocsparse_order_column) ? ldb : B_m;
    I ncolB = (order == rocsparse_order_column) ? B_n : ldb;
    I nrowC = (order == rocsparse_order_column) ? ldc : C_m;
    I ncolC = (order == rocsparse_order_column) ? C_n : ldc;

    I nnz_B = nrowB * ncolB;
    I nnz_C = nrowC * ncolC;

    I batch_stride_A = (batch_count_A > 1) ? nnz_A : 0;
    I batch_stride_B = (batch_count_B > 1) ? nnz_B : 0;
    I batch_stride_C = (batch_count_C > 1) ? nnz_C : 0;

    // Allocate host memory for all batches of A matrix
    host_vector<I> hcoo_row_ind(batch_count_A * nnz_A);
    host_vector<I> hcoo_col_ind(batch_count_A * nnz_A);
    host_vector<T> hcoo_val(batch_count_A * nnz_A);

    for(I i = 0; i < batch_count_A; i++)
    {
        for(size_t j = 0; j < nnz_A; j++)
        {
            hcoo_row_ind[nnz_A * i + j] = hcoo_row_ind_temp[j];
            hcoo_col_ind[nnz_A * i + j] = hcoo_col_ind_temp[j];
            hcoo_val[nnz_A * i + j]     = hcoo_val_temp[j];
        }
    }

    // Allocate host memory for vectors
    host_vector<T> hB(batch_count_B * nnz_B);
    host_vector<T> hC_1(batch_count_C * nnz_C);
    host_vector<T> hC_2(batch_count_C * nnz_C);
    host_vector<T> hC_gold(batch_count_C * nnz_C);

    // Initialize data on CPU
    rocsparse_init<T>(hB, batch_count_B * nnz_B, 1, 1);
    rocsparse_init<T>(hC_1, batch_count_C * nnz_C, 1, 1);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // Allocate device memory
    device_vector<I> dcoo_row_ind(hcoo_row_ind);
    device_vector<I> dcoo_col_ind(hcoo_col_ind);
    device_vector<T> dcoo_val(hcoo_val);
    device_vector<T> dB(hB);
    device_vector<T> dC_1(hC_1);
    device_vector<T> dC_2(hC_2);
    device_vector<T> dalpha(1);
    device_vector<T> dbeta(1);

    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, &hbeta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(
        A_m, A_n, nnz_A, dcoo_row_ind, dcoo_col_ind, dcoo_val, itype, base, ttype);
    rocsparse_local_dnmat B(B_m, B_n, ldb, dB, ttype, order);
    rocsparse_local_dnmat C1(C_m, C_n, ldc, dC_1, ttype, order);
    rocsparse_local_dnmat C2(C_m, C_n, ldc, dC_2, ttype, order);

    CHECK_ROCSPARSE_ERROR(rocsparse_coo_set_strided_batch(A, batch_count_A, batch_stride_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(B, batch_count_B, batch_stride_B));
    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(C1, batch_count_C, batch_stride_C));
    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(C2, batch_count_C, batch_stride_C));

    // Query SpMM buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         A,
                                         B,
                                         &hbeta,
                                         C1,
                                         ttype,
                                         alg,
                                         rocsparse_spmm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                         trans_A,
                                         trans_B,
                                         &halpha,
                                         A,
                                         B,
                                         &hbeta,
                                         C1,
                                         ttype,
                                         alg,
                                         rocsparse_spmm_stage_preprocess,
                                         &buffer_size,
                                         dbuffer));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                             trans_A,
                                             trans_B,
                                             &halpha,
                                             A,
                                             B,
                                             &hbeta,
                                             C1,
                                             ttype,
                                             alg,
                                             rocsparse_spmm_stage_compute,
                                             &buffer_size,
                                             dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                             trans_A,
                                             trans_B,
                                             dalpha,
                                             A,
                                             B,
                                             dbeta,
                                             C2,
                                             ttype,
                                             alg,
                                             rocsparse_spmm_stage_compute,
                                             &buffer_size,
                                             dbuffer));

        // Copy output to host
        hC_1.transfer_from(dC_1);
        hC_2.transfer_from(dC_2);

        // CPU coomm_batched
        host_coomm_batched<T, I>(A_m,
                                 N,
                                 A_n,
                                 nnz_A,
                                 batch_count_A,
                                 batch_stride_A,
                                 trans_A,
                                 trans_B,
                                 halpha,
                                 hcoo_row_ind.data(),
                                 hcoo_col_ind.data(),
                                 hcoo_val.data(),
                                 hB.data(),
                                 ldb,
                                 batch_count_B,
                                 batch_stride_B,
                                 hbeta,
                                 hC_gold.data(),
                                 ldc,
                                 batch_count_C,
                                 batch_stride_C,
                                 order,
                                 base);

        hC_gold.near_check(hC_1);
        hC_gold.near_check(hC_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 &halpha,
                                                 A,
                                                 B,
                                                 &hbeta,
                                                 C1,
                                                 ttype,
                                                 alg,
                                                 rocsparse_spmm_stage_compute,
                                                 &buffer_size,
                                                 dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 &halpha,
                                                 A,
                                                 B,
                                                 &hbeta,
                                                 C1,
                                                 ttype,
                                                 alg,
                                                 rocsparse_spmm_stage_compute,
                                                 &buffer_size,
                                                 dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count
            = batch_count_C
              * spmm_gflop_count(N, nnz_A, (I)C_m * (I)C_n, hbeta != static_cast<T>(0));
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = coomm_batched_gbyte_count<T>(A_m,
                                                          nnz_A,
                                                          (I)B_m * (I)B_n,
                                                          (I)C_m * (I)C_n,
                                                          batch_count_A,
                                                          batch_count_B,
                                                          batch_count_C,
                                                          hbeta != static_cast<T>(0));
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "K",
                            K,
                            "nnz_A",
                            nnz_A,
                            "batch_count_A",
                            batch_count_A,
                            "batch_count_B",
                            batch_count_B,
                            "batch_count_C",
                            batch_count_C,
                            "alpha",
                            halpha,
                            "beta",
                            hbeta,
                            "Algorithm",
                            rocsparse_spmmalg2string(alg),
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                       \
    template void testing_spmm_batched_coo_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmm_batched_coo<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

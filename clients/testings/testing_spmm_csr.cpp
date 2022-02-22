/* ************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

template <typename I, typename J, typename T>
void testing_spmm_csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    J                    m           = safe_size;
    J                    n           = safe_size;
    J                    k           = safe_size;
    J                    ncol_B      = safe_size;
    I                    nnz         = safe_size;
    const T*             alpha       = (const T*)0x4;
    const T*             beta        = (const T*)0x4;
    void*                csr_val     = (void*)0x4;
    void*                csr_row_ptr = (void*)0x4;
    void*                csr_col_ind = (void*)0x4;
    void*                B           = (void*)0x4;
    void*                C           = (void*)0x4;
    rocsparse_operation  trans_A     = rocsparse_operation_none;
    rocsparse_operation  trans_B     = rocsparse_operation_none;
    rocsparse_index_base base        = rocsparse_index_base_zero;
    rocsparse_order      order       = rocsparse_order_column;
    rocsparse_spmm_alg   alg         = rocsparse_spmm_alg_default;
    rocsparse_spmm_stage stage       = rocsparse_spmm_stage_auto;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // SpMM structures
    rocsparse_local_spmat local_mat_A(m,
                                      n,
                                      nnz,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      csr_val,
                                      itype,
                                      jtype,
                                      base,
                                      ttype,
                                      rocsparse_format_csr);
    rocsparse_local_dnmat local_mat_B(k, ncol_B, k, B, ttype, order);
    rocsparse_local_dnmat local_mat_C(m, n, m, C, ttype, order);

    rocsparse_spmat_descr mat_A = local_mat_A;
    rocsparse_dnmat_descr mat_B = local_mat_B;
    rocsparse_dnmat_descr mat_C = local_mat_C;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {11, 12};

#define PARAMS                                                                                  \
    handle, trans_A, trans_B, alpha, mat_A, mat_B, beta, mat_C, ttype, alg, stage, buffer_size, \
        temp_buffer
    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_spmm, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS

    EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(handle,
                                           trans_A,
                                           trans_B,
                                           alpha,
                                           mat_A,
                                           mat_B,
                                           beta,
                                           mat_C,
                                           ttype,
                                           alg,
                                           stage,
                                           nullptr,
                                           nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename I, typename J, typename T>
void testing_spmm_csr(const Arguments& arg)
{
    J                    M       = arg.M;
    J                    N       = arg.N;
    J                    K       = arg.K;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_operation  trans_B = arg.transB;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_spmm_alg   alg     = arg.spmm_alg;
    rocsparse_order      order   = arg.order;

    T halpha = arg.get_alpha<T>();
    T hbeta  = arg.get_beta<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || K <= 0)
    {
        // M == N == 0 means nnz can only be 0, too
        static const I safe_size = 100;

        // Allocate memory on device
        device_vector<I> dcsr_row_ptr(safe_size);
        device_vector<J> dcsr_col_ind(safe_size);
        device_vector<T> dcsr_val(safe_size);
        device_vector<T> dB(safe_size);
        device_vector<T> dC(safe_size);

        // Check SpMM when structures can be created
        if(M == 0 && N == 0 && K == 0)
        {
            // Pointer mode
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            J A_m = 0;
            J A_n = 0;
            J B_m = 0;
            J B_n = 0;
            J C_m = 0;
            J C_n = 0;

            J ldb = 0;
            J ldc = 0;

            // Check structures
            I nnz_A = 0;

            rocsparse_local_spmat A(A_m,
                                    A_n,
                                    nnz_A,
                                    dcsr_row_ptr,
                                    dcsr_col_ind,
                                    dcsr_val,
                                    itype,
                                    jtype,
                                    base,
                                    ttype,
                                    rocsparse_format_csr);
            rocsparse_local_dnmat B(B_m, B_n, ldb, dB, ttype, order);
            rocsparse_local_dnmat C(C_m, C_n, ldc, dC, ttype, order);

            size_t buffer_size;
            EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(handle,
                                                   trans_A,
                                                   trans_B,
                                                   &halpha,
                                                   A,
                                                   B,
                                                   &hbeta,
                                                   C,
                                                   ttype,
                                                   alg,
                                                   rocsparse_spmm_stage_buffer_size,
                                                   &buffer_size,
                                                   nullptr),
                                    rocsparse_status_success);

            void* dbuffer;
            CHECK_HIP_ERROR(hipMalloc(&dbuffer, safe_size));

            EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(handle,
                                                   trans_A,
                                                   trans_B,
                                                   &halpha,
                                                   A,
                                                   B,
                                                   &hbeta,
                                                   C,
                                                   ttype,
                                                   alg,
                                                   rocsparse_spmm_stage_preprocess,
                                                   &buffer_size,
                                                   dbuffer),
                                    rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(rocsparse_spmm(handle,
                                                   trans_A,
                                                   trans_B,
                                                   &halpha,
                                                   A,
                                                   B,
                                                   &hbeta,
                                                   C,
                                                   ttype,
                                                   alg,
                                                   rocsparse_spmm_stage_compute,
                                                   &buffer_size,
                                                   dbuffer),
                                    rocsparse_status_success);

            CHECK_HIP_ERROR(hipFree(dbuffer));
        }

        return;
    }

    // Allocate host memory for matrix
    host_vector<I> hcsr_row_ptr;
    host_vector<J> hcsr_col_ind;
    host_vector<T> hcsr_val;

    // Allocate host memory for matrix
    rocsparse_matrix_factory<T, I, J> matrix_factory(arg);

    I nnz_A;
    matrix_factory.init_csr(hcsr_row_ptr,
                            hcsr_col_ind,
                            hcsr_val,
                            (trans_A == rocsparse_operation_none) ? M : K,
                            (trans_A == rocsparse_operation_none) ? K : M,
                            nnz_A,
                            base);

    // Some matrix properties
    J A_m = (trans_A == rocsparse_operation_none) ? M : K;
    J A_n = (trans_A == rocsparse_operation_none) ? K : M;
    J B_m = (trans_B == rocsparse_operation_none) ? K : N;
    J B_n = (trans_B == rocsparse_operation_none) ? N : K;
    J C_m = M;
    J C_n = N;

    J ldb = (order == rocsparse_order_column)
                ? ((trans_B == rocsparse_operation_none) ? (2 * K) : (2 * N))
                : ((trans_B == rocsparse_operation_none) ? (2 * N) : (2 * K));
    J ldc = (order == rocsparse_order_column) ? (2 * M) : (2 * N);

    J nrowB = (order == rocsparse_order_column) ? ldb : B_m;
    J ncolB = (order == rocsparse_order_column) ? B_n : ldb;
    J nrowC = (order == rocsparse_order_column) ? ldc : C_m;
    J ncolC = (order == rocsparse_order_column) ? C_n : ldc;

    I nnz_B = nrowB * ncolB;
    I nnz_C = nrowC * ncolC;

    // Allocate host memory for vectors
    host_vector<T> hB(nnz_B);
    host_vector<T> hC_1(nnz_C);
    host_vector<T> hC_2(nnz_C);
    host_vector<T> hC_gold(nnz_C);

    // Initialize data on CPU
    rocsparse_init<T>(hB, nnz_B, 1, 1);
    rocsparse_init<T>(hC_1, nnz_C, 1, 1);

    hC_2    = hC_1;
    hC_gold = hC_1;

    // Allocate device memory
    device_vector<I> dcsr_row_ptr(A_m + 1);
    device_vector<J> dcsr_col_ind(nnz_A);
    device_vector<T> dcsr_val(nnz_A);
    device_vector<T> dB(nnz_B);
    device_vector<T> dC_1(nnz_C);
    device_vector<T> dC_2(nnz_C);
    device_vector<T> dalpha(1);
    device_vector<T> dbeta(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(I) * (A_m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(J) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1, sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2, sizeof(T) * nnz_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbeta, &hbeta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(A_m,
                            A_n,
                            nnz_A,
                            dcsr_row_ptr,
                            dcsr_col_ind,
                            dcsr_val,
                            itype,
                            jtype,
                            base,
                            ttype,
                            rocsparse_format_csr);
    rocsparse_local_dnmat B(B_m, B_n, ldb, dB, ttype, order);
    rocsparse_local_dnmat C1(C_m, C_n, ldc, dC_1, ttype, order);
    rocsparse_local_dnmat C2(C_m, C_n, ldc, dC_2, ttype, order);

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
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

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
        // SpMM

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
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        // CPU csrmm
        host_csrmm<T, I, J>(A_m,
                            N,
                            A_n,
                            trans_A,
                            trans_B,
                            halpha,
                            hcsr_row_ptr,
                            hcsr_col_ind,
                            hcsr_val,
                            hB,
                            ldb,
                            hbeta,
                            hC_gold,
                            ldc,
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
            = spmm_gflop_count(N, nnz_A, (I)C_m * (I)C_n, hbeta != static_cast<T>(0));
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = csrmm_gbyte_count<T>(
            A_m, nnz_A, (I)B_m * (I)B_n, (I)C_m * (I)C_n, hbeta != static_cast<T>(0));
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "K",
                            K,
                            "nnz_A",
                            nnz_A,
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

    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                               \
    template void testing_spmm_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmm_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

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

#include "testing.hpp"

#include "auto_testing_bad_arg.hpp"
template <typename T>
void testing_csrmm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Local decalrations.

    rocsparse_handle     handle      = local_handle;
    rocsparse_operation  trans_A     = rocsparse_operation_none;
    rocsparse_operation  trans_B     = rocsparse_operation_none;
    rocsparse_int        m           = safe_size;
    rocsparse_int        n           = safe_size;
    rocsparse_int        k           = safe_size;
    rocsparse_int        nnz         = safe_size;
    const T              alpha       = static_cast<T>(2);
    rocsparse_mat_descr  descr       = local_descr;
    const T*             csr_val     = (const T*)0x4;
    const rocsparse_int* csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int* csr_col_ind = (const rocsparse_int*)0x4;
    const T*             B           = (const T*)0x4;
    rocsparse_int        ldb         = safe_size;
    const T              beta        = static_cast<T>(2);
    T*                   C           = (T*)0x4;
    rocsparse_int        ldc         = safe_size;

#define PARAMS                                                                                   \
    handle, trans_A, trans_B, m, n, k, nnz, &alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, \
        ldb, &beta, C, ldc

    auto_testing_bad_arg(rocsparse_csrmm<T>, PARAMS);

    {
        auto tmp = trans_A;
        trans_A  = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_not_implemented);
        trans_A = tmp;
    }

    {
        auto tmp = trans_B;
        trans_B  = rocsparse_operation_conjugate_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_not_implemented);
        trans_B = tmp;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(PARAMS), rocsparse_status_not_implemented);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));
#undef PARAMS
}

template <typename T>
void testing_csrmm(const Arguments& arg)
{
    rocsparse_int        M      = arg.M;
    rocsparse_int        N      = arg.N;
    rocsparse_int        K      = arg.K;
    rocsparse_operation  transA = arg.transA;
    rocsparse_operation  transB = arg.transB;
    rocsparse_index_base base   = arg.baseA;

    host_scalar<T> h_alpha, h_beta;

    *h_alpha.val = arg.get_alpha<T>();
    *h_beta.val  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || K <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        rocsparse_int* dcsr_row_ptr = (rocsparse_int*)0x4;
        rocsparse_int* dcsr_col_ind = (rocsparse_int*)0x4;
        T*             dcsr_val     = (T*)0x4;
        T*             dB           = (T*)0x4;
        T*             dC           = (T*)0x4;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmm<T>(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   safe_size,
                                                   h_alpha.val,
                                                   descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   dB,
                                                   safe_size,
                                                   h_beta.val,
                                                   dC,
                                                   safe_size),
                                (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                                          : rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrix
    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA,
                            (transA == rocsparse_operation_none ? M : K),
                            (transA == rocsparse_operation_none ? K : M));

    // Some matrix properties
    rocsparse_int B_m
        = (transB == rocsparse_operation_none) ? (transA == rocsparse_operation_none ? K : M) : N;
    rocsparse_int B_n = (transB == rocsparse_operation_none ? N : K);
    rocsparse_int C_m = (transA == rocsparse_operation_none) ? M : K;
    rocsparse_int C_n = N;

    host_dense_matrix<T> hB(B_m, B_n), hC(C_m, C_n);

    rocsparse_matrix_utils::init(hB);
    rocsparse_matrix_utils::init(hC);

    device_csr_matrix<T>   dA(hA);
    device_dense_matrix<T> dC(hC);
    device_dense_matrix<T> dB(hB);

    // Copy data from CPU to device
#define PARAMS(alpha_, A_, B_, beta_, C_)                                                       \
    handle, transA, transB, M, N, K, A_.nnz, alpha_.val, descr, A_.val, A_.ptr, A_.ind, B_.val, \
        B_.ld, beta_.val, C_.val, C_.ld

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));

        { // Copy data from CPU to device
            host_dense_matrix<T> hC_copy(hC);

            // CPU csrmm
            host_csrmm(hC.m,
                       hC.n,
                       transB,
                       *h_alpha.val,
                       hA.ptr,
                       hA.ind,
                       hA.val,
                       hB.val,
                       hB.ld,
                       *h_beta.val,
                       hC.val,
                       hC.ld,
                       rocsparse_order_column,
                       base);

            hC.unit_check(dC);
            dC.transfer_from(hC_copy);
        }

        device_scalar<T> d_alpha(h_alpha);
        device_scalar<T> d_beta(h_beta);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmm<T>(PARAMS(d_alpha, dA, dB, d_beta, dC)));
        hC.unit_check(dC);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmm<T>(PARAMS(h_alpha, dA, dB, h_beta, dC)));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = csrmm_gflop_count<rocsparse_int, rocsparse_int>(
            N, dA.nnz, dC.m * dC.n, *h_beta.val != static_cast<T>(0));
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);
        double gbyte_count = csrmm_gbyte_count<T, rocsparse_int, rocsparse_int>(
            M, dA.nnz, dB.m * dB.n, dC.m * dC.n, *h_beta.val != static_cast<T>(0));
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "K",
                            K,
                            "transA",
                            rocsparse_operation2string(transA),
                            "transB",
                            rocsparse_operation2string(transB),
                            "nnz_A",
                            dA.nnz,
                            "nnz_B",
                            dB.m * dB.n,
                            "nnz_C",
                            dC.m * dC.n,
                            "alpha",
                            *h_alpha.val,
                            "beta",
                            *h_beta.val,
                            "GFlop/s",
                            gpu_gflops,
                            "GB/s",
                            gpu_gbyte,
                            "msec",
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            (arg.unit_check ? "yes" : "no"));
    }

#undef PARAMS
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_csrmm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrmm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

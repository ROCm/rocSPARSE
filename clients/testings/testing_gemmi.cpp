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
#include "testing.hpp"

template <typename T>
void testing_gemmi_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;
    T h_beta  = 0.1;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create rocsparse mat descriptor
    rocsparse_local_mat_descr descr;

    // rocsparse operations
    rocsparse_operation trans_A = rocsparse_operation_none;
    rocsparse_operation trans_B = rocsparse_operation_transpose;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);
    device_vector<T>             dA(safe_size);
    device_vector<T>             dC(safe_size);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dA || !dC)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_gemmi()
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(nullptr,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               nullptr,
                                               dA,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               nullptr,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               nullptr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               descr,
                                               nullptr,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               nullptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               nullptr,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               nullptr,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               nullptr,
                                               safe_size),
                            rocsparse_status_invalid_pointer);

    // Test invalid sizes
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               -1,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               nullptr,
                                               nullptr,
                                               safe_size,
                                               descr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               safe_size),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               -1,
                                               safe_size,
                                               safe_size,
                                               nullptr,
                                               nullptr,
                                               safe_size,
                                               descr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               safe_size),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               -1,
                                               safe_size,
                                               nullptr,
                                               nullptr,
                                               safe_size,
                                               descr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               safe_size),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               -1,
                                               nullptr,
                                               nullptr,
                                               safe_size,
                                               descr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               safe_size),
                            rocsparse_status_invalid_size);

    // Test invalid leading dimensions
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               -1,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               safe_size),
                            rocsparse_status_invalid_value);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                               trans_A,
                                               trans_B,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               dA,
                                               safe_size,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               &h_beta,
                                               dC,
                                               -1),
                            rocsparse_status_invalid_value);
}

template <typename T>
void testing_gemmi(const Arguments& arg)
{
    rocsparse_int               M      = arg.M;
    rocsparse_int               N      = arg.N;
    rocsparse_int               K      = arg.K;
    rocsparse_operation         transA = arg.transA;
    rocsparse_operation         transB = arg.transB;
    rocsparse_index_base        base   = arg.baseA;
    rocsparse_matrix_factory<T> matrix_factory(arg);
    T                           h_alpha = arg.get_alpha<T>();
    T                           h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create rocsparse mat descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || K < 0)
    {
        static const size_t safe_size = 100;

        EXPECT_ROCSPARSE_STATUS(rocsparse_gemmi<T>(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   safe_size,
                                                   nullptr,
                                                   nullptr,
                                                   safe_size,
                                                   descr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   safe_size),
                                (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                                          : rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrix B
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    // Sample matrix B
    rocsparse_int nnz_B;
    matrix_factory.init_csr(hcsr_row_ptr,
                            hcsr_col_ind,
                            hcsr_val,
                            (transB == rocsparse_operation_none) ? K : N,
                            (transB == rocsparse_operation_none) ? N : K,
                            nnz_B,
                            base);

    rocsparse_int nrow_B = (transB == rocsparse_operation_none) ? K : N;

    rocsparse_int lda = std::max(1, M);
    rocsparse_int ldc = std::max(1, M);

    rocsparse_int nnz_A = lda * K;
    rocsparse_int nnz_C = ldc * N;

    // Allocate host memory for matrix A and C
    host_vector<T> hA(nnz_A);
    host_vector<T> hC_1(nnz_C);
    host_vector<T> hC_2(nnz_C);
    host_vector<T> hC_gold(nnz_C);

    // Sample matrix A
    rocsparse_init<T>(hA, M, K, lda);

    // Sample matrix C
    rocsparse_init<T>(hC_gold, M, N, ldc);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(nrow_B + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz_B);
    device_vector<T>             dcsr_val(nnz_B);
    device_vector<T>             dA(nnz_A);
    device_vector<T>             dC_1(nnz_C);
    device_vector<T>             dC_2(nnz_C);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dA || !dC_1 || !dC_2 || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (nrow_B + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_gold, sizeof(T) * nnz_C, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dC_2, dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_gemmi<T>(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 nnz_B,
                                                 &h_alpha,
                                                 dA,
                                                 lda,
                                                 descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 &h_beta,
                                                 dC_1,
                                                 ldc));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_gemmi<T>(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 nnz_B,
                                                 d_alpha,
                                                 dA,
                                                 lda,
                                                 descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 d_beta,
                                                 dC_2,
                                                 ldc));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        // CPU gemmi
        host_gemmi<T>(M,
                      N,
                      transA,
                      transB,
                      h_alpha,
                      hA,
                      lda,
                      hcsr_row_ptr,
                      hcsr_col_ind,
                      hcsr_val,
                      h_beta,
                      hC_gold,
                      ldc,
                      base);

        unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
        unit_check_general<T>(M, N, ldc, hC_gold, hC_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gemmi<T>(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     nnz_B,
                                                     &h_alpha,
                                                     dA,
                                                     lda,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     &h_beta,
                                                     dC_1,
                                                     ldc));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gemmi<T>(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     nnz_B,
                                                     &h_alpha,
                                                     dA,
                                                     lda,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     &h_beta,
                                                     dC_1,
                                                     ldc));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops = csrmm_gflop_count<T>(M, nnz_B, nnz_C, h_beta != static_cast<T>(0))
                            / gpu_time_used * 1e6;
        double gpu_gbyte
            = csrmm_gbyte_count<T>(nrow_B, nnz_B, nnz_A, nnz_C, h_beta != static_cast<T>(0))
              / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "K"
                  << std::setw(12) << "transA" << std::setw(12) << "transB" << std::setw(12)
                  << "nnz_A" << std::setw(12) << "nnz_B" << std::setw(12) << "nnz_C"
                  << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << K << std::setw(12)
                  << rocsparse_operation2string(transA) << std::setw(12)
                  << rocsparse_operation2string(transB) << std::setw(12) << nnz_A << std::setw(12)
                  << nnz_B << std::setw(12) << nnz_C << std::setw(12) << h_alpha << std::setw(12)
                  << h_beta << std::setw(12) << gpu_gflops << std::setw(12) << gpu_gbyte
                  << std::setw(12) << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls
                  << std::setw(12) << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_gemmi_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gemmi<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

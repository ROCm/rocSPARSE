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

#include "testing.hpp"

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_csrgeam_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_descr_A;
    rocsparse_local_mat_descr local_descr_B;
    rocsparse_local_mat_descr local_descr_C;

    rocsparse_handle          handle        = local_handle;
    rocsparse_int             m             = safe_size;
    rocsparse_int             n             = safe_size;
    const T*                  alpha         = (const T*)0x4;
    const rocsparse_mat_descr descr_A       = local_descr_A;
    rocsparse_int             nnz_A         = safe_size;
    const T*                  csr_val_A     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;
    const T*                  beta          = (const T*)0x4;
    const rocsparse_mat_descr descr_B       = local_descr_B;
    rocsparse_int             nnz_B         = safe_size;
    const T*                  csr_val_B     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr_B = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind_B = (const rocsparse_int*)0x4;
    const rocsparse_mat_descr descr_C       = local_descr_C;
    T*                        csr_val_C     = (T*)0x4;
    rocsparse_int*            csr_row_ptr_C = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind_C = (rocsparse_int*)0x4;
    rocsparse_int*            nnz_C         = (rocsparse_int*)0x4;

#define PARAMS_NNZ                                                                             \
    handle, m, n, descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, \
        csr_col_ind_B, descr_C, csr_row_ptr_C, nnz_C

#define PARAMS                                                                                   \
    handle, m, n, alpha, descr_A, nnz_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, beta, descr_B, \
        nnz_B, csr_val_B, csr_row_ptr_B, csr_col_ind_B, descr_C, csr_val_C, csr_row_ptr_C,       \
        csr_col_ind_C

    auto_testing_bad_arg(rocsparse_csrgeam_nnz, PARAMS_NNZ);
    auto_testing_bad_arg(rocsparse_csrgeam<T>, PARAMS);

#undef PARAMS
#undef PARAMS_NNZ
}

template <typename T>
void testing_csrgeam(const Arguments& arg)
{
    rocsparse_int                      M         = arg.M;
    rocsparse_int                      N         = arg.N;
    rocsparse_index_base               baseA     = arg.baseA;
    rocsparse_index_base               baseB     = arg.baseB;
    rocsparse_index_base               baseC     = arg.baseC;
    static constexpr bool              full_rank = false;
    rocsparse_matrix_factory<T>        matrix_factory(arg, arg.timing ? false : true, full_rank);
    rocsparse_matrix_factory_random<T> matrix_factory_random(full_rank);
    T                                  h_alpha = arg.get_alpha<T>();
    T                                  h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrC, baseC));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr_A(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_A(safe_size);
        device_vector<T>             dcsr_val_A(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_B(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_B(safe_size);
        device_vector<T>             dcsr_val_B(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_C(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_C(safe_size);
        device_vector<T>             dcsr_val_C(safe_size);

        if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
           || !dcsr_val_B || !dcsr_row_ptr_C || !dcsr_col_ind_C || !dcsr_val_C)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int nnz_C;

        rocsparse_status status_1 = rocsparse_csrgeam_nnz(handle,
                                                          M,
                                                          N,
                                                          descrA,
                                                          safe_size,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          descrB,
                                                          safe_size,
                                                          dcsr_row_ptr_B,
                                                          dcsr_col_ind_B,
                                                          descrC,
                                                          dcsr_row_ptr_C,
                                                          &nnz_C);
        rocsparse_status status_2 = rocsparse_csrgeam<T>(handle,
                                                         M,
                                                         N,
                                                         &h_alpha,
                                                         descrA,
                                                         safe_size,
                                                         dcsr_val_A,
                                                         dcsr_row_ptr_A,
                                                         dcsr_col_ind_A,
                                                         &h_beta,
                                                         descrB,
                                                         safe_size,
                                                         dcsr_val_B,
                                                         dcsr_row_ptr_B,
                                                         dcsr_col_ind_B,
                                                         descrC,
                                                         dcsr_val_C,
                                                         dcsr_row_ptr_C,
                                                         dcsr_col_ind_C);

        // alpha == nullptr && beta != nullptr
        EXPECT_ROCSPARSE_STATUS(
            status_1, (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            status_2, (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrices
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;
    host_vector<rocsparse_int> hcsr_row_ptr_B;
    host_vector<rocsparse_int> hcsr_col_ind_B;
    host_vector<T>             hcsr_val_B;

    // Sample matrix
    rocsparse_int nnz_A = 4;
    rocsparse_int nnz_B = 4;
    rocsparse_int hnnz_C_gold;
    rocsparse_int hnnz_C_1;
    rocsparse_int hnnz_C_2;

    // Sample A
    matrix_factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, N, nnz_A, baseA);

    // Sample B
    matrix_factory_random.init_csr(hcsr_row_ptr_B, hcsr_col_ind_B, hcsr_val_B, M, N, nnz_B, baseB);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz_A);
    device_vector<T>             dcsr_val_A(nnz_A);
    device_vector<rocsparse_int> dcsr_row_ptr_B(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_B(nnz_B);
    device_vector<T>             dcsr_val_B(nnz_B);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);
    device_vector<rocsparse_int> dcsr_row_ptr_C_1(M + 1);
    device_vector<rocsparse_int> dcsr_row_ptr_C_2(M + 1);
    device_vector<rocsparse_int> dnnz_C_2(1);

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
       || !dcsr_val_B || !d_alpha || !d_beta || !dcsr_row_ptr_C_1 || !dcsr_row_ptr_C_2 || !dnnz_C_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_B, hcsr_row_ptr_B, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_B, hcsr_col_ind_B, sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_B, hcsr_val_B, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Obtain nnz of C

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam_nnz(handle,
                                                    M,
                                                    N,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &hnnz_C_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam_nnz(handle,
                                                    M,
                                                    N,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrC,
                                                    dcsr_row_ptr_C_2,
                                                    dnnz_C_2));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_row_ptr_C_1(M + 1);
        host_vector<rocsparse_int> hcsr_row_ptr_C_2(M + 1);
        CHECK_HIP_ERROR(
            hipMemcpy(&hnnz_C_2, dnnz_C_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_1,
                                  dcsr_row_ptr_C_1,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_2,
                                  dcsr_row_ptr_C_2,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));

        // CPU csrgemm_nnz
        host_vector<rocsparse_int> hcsr_row_ptr_C_gold(M + 1);
        host_csrgeam_nnz<T>(M,
                            N,
                            h_alpha,
                            hcsr_row_ptr_A,
                            hcsr_col_ind_A,
                            h_beta,
                            hcsr_row_ptr_B,
                            hcsr_col_ind_B,
                            hcsr_row_ptr_C_gold,
                            &hnnz_C_gold,
                            baseA,
                            baseB,
                            baseC);

        // Check nnz of C

        unit_check_scalar(hnnz_C_gold, hnnz_C_1);
        unit_check_scalar(hnnz_C_gold, hnnz_C_2);

        // Check row pointers of C

        unit_check_segments<rocsparse_int>(M + 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_1);
        unit_check_segments<rocsparse_int>(M + 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_2);

        // Allocate device memory for C
        device_vector<rocsparse_int> dcsr_col_ind_C_1(hnnz_C_1);
        device_vector<rocsparse_int> dcsr_col_ind_C_2(hnnz_C_2);
        device_vector<T>             dcsr_val_C_1(hnnz_C_1);
        device_vector<T>             dcsr_val_C_2(hnnz_C_2);

        if(!dcsr_col_ind_C_1 || !dcsr_col_ind_C_2 || !dcsr_val_C_1 || !dcsr_val_C_2)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Perform matrix matrix multiplication

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam<T>(handle,
                                                   M,
                                                   N,
                                                   &h_alpha,
                                                   descrA,
                                                   nnz_A,
                                                   dcsr_val_A,
                                                   dcsr_row_ptr_A,
                                                   dcsr_col_ind_A,
                                                   &h_beta,
                                                   descrB,
                                                   nnz_B,
                                                   dcsr_val_B,
                                                   dcsr_row_ptr_B,
                                                   dcsr_col_ind_B,
                                                   descrC,
                                                   dcsr_val_C_1,
                                                   dcsr_row_ptr_C_1,
                                                   dcsr_col_ind_C_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam<T>(handle,
                                                   M,
                                                   N,
                                                   d_alpha,
                                                   descrA,
                                                   nnz_A,
                                                   dcsr_val_A,
                                                   dcsr_row_ptr_A,
                                                   dcsr_col_ind_A,
                                                   d_beta,
                                                   descrB,
                                                   nnz_B,
                                                   dcsr_val_B,
                                                   dcsr_row_ptr_B,
                                                   dcsr_col_ind_B,
                                                   descrC,
                                                   dcsr_val_C_2,
                                                   dcsr_row_ptr_C_2,
                                                   dcsr_col_ind_C_2));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_col_ind_C_1(hnnz_C_1);
        host_vector<rocsparse_int> hcsr_col_ind_C_2(hnnz_C_2);
        host_vector<T>             hcsr_val_C_1(hnnz_C_1);
        host_vector<T>             hcsr_val_C_2(hnnz_C_2);

        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_1,
                                  dcsr_col_ind_C_1,
                                  sizeof(rocsparse_int) * hnnz_C_1,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_2,
                                  dcsr_col_ind_C_2,
                                  sizeof(rocsparse_int) * hnnz_C_2,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_1, dcsr_val_C_1, sizeof(T) * hnnz_C_1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_2, dcsr_val_C_2, sizeof(T) * hnnz_C_2, hipMemcpyDeviceToHost));

        // CPU csrgemm
        host_vector<rocsparse_int> hcsr_col_ind_C_gold(hnnz_C_gold);
        host_vector<T>             hcsr_val_C_gold(hnnz_C_gold);
        host_csrgeam<T>(M,
                        N,
                        h_alpha,
                        hcsr_row_ptr_A,
                        hcsr_col_ind_A,
                        hcsr_val_A,
                        h_beta,
                        hcsr_row_ptr_B,
                        hcsr_col_ind_B,
                        hcsr_val_B,
                        hcsr_row_ptr_C_gold,
                        hcsr_col_ind_C_gold,
                        hcsr_val_C_gold,
                        baseA,
                        baseB,
                        baseC);

        // Check C
        unit_check_segments<rocsparse_int>(hnnz_C_gold, hcsr_col_ind_C_gold, hcsr_col_ind_C_1);
        unit_check_segments<rocsparse_int>(hnnz_C_gold, hcsr_col_ind_C_gold, hcsr_col_ind_C_2);

        near_check_segments<T>(hnnz_C_gold, hcsr_val_C_gold, hcsr_val_C_1);
        near_check_segments<T>(hnnz_C_gold, hcsr_val_C_gold, hcsr_val_C_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        rocsparse_int nnz_C;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam_nnz(handle,
                                                        M,
                                                        N,
                                                        descrA,
                                                        nnz_A,
                                                        dcsr_row_ptr_A,
                                                        dcsr_col_ind_A,
                                                        descrB,
                                                        nnz_B,
                                                        dcsr_row_ptr_B,
                                                        dcsr_col_ind_B,
                                                        descrC,
                                                        dcsr_row_ptr_C_1,
                                                        &nnz_C));

            device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
            device_vector<T>             dcsr_val_C(nnz_C);

            if(!dcsr_col_ind_C || !dcsr_val_C)
            {
                CHECK_HIP_ERROR(hipErrorOutOfMemory);
                return;
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam<T>(handle,
                                                       M,
                                                       N,
                                                       &h_alpha,
                                                       descrA,
                                                       nnz_A,
                                                       dcsr_val_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       &h_beta,
                                                       descrB,
                                                       nnz_B,
                                                       dcsr_val_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       descrC,
                                                       dcsr_val_C,
                                                       dcsr_row_ptr_C_1,
                                                       dcsr_col_ind_C));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam_nnz(handle,
                                                    M,
                                                    N,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &nnz_C));

        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
        device_vector<T>             dcsr_val_C(nnz_C);

        if(!dcsr_col_ind_C || !dcsr_val_C)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrgeam<T>(handle,
                                                       M,
                                                       N,
                                                       &h_alpha,
                                                       descrA,
                                                       nnz_A,
                                                       dcsr_val_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       &h_beta,
                                                       descrB,
                                                       nnz_B,
                                                       dcsr_val_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       descrC,
                                                       dcsr_val_C,
                                                       dcsr_row_ptr_C_1,
                                                       dcsr_col_ind_C));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gpu_gflops = csrgeam_gflop_count<T>(nnz_A, nnz_B, nnz_C, &h_alpha, &h_beta)
                            / gpu_solve_time_used * 1e6;
        double gpu_gbyte = csrgeam_gbyte_count<T>(M, nnz_A, nnz_B, nnz_C, &h_alpha, &h_beta)
                           / gpu_solve_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz_A"
                  << std::setw(12) << "nnz_B" << std::setw(12) << "nnz_C" << std::setw(12)
                  << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
                  << std::setw(12) << "GB/s" << std::setw(16) << "nnz msec" << std::setw(16)
                  << "gemm msec" << std::setw(12) << "iter" << std::setw(12) << "verified"
                  << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz_A
                  << std::setw(12) << nnz_B << std::setw(12) << nnz_C << std::setw(12) << h_alpha
                  << std::setw(12) << h_beta << std::setw(12) << gpu_gflops << std::setw(12)
                  << gpu_gbyte << std::setw(16) << gpu_analysis_time_used / 1e3 << std::setw(16)
                  << gpu_solve_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csrgeam_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrgeam<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

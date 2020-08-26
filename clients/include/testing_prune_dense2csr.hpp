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

#pragma once
#ifndef TESTING_PRUNE_DENSE2CSR_HPP
#define TESTING_PRUNE_DENSE2CSR_HPP

#include <rocsparse.hpp>

#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"
#include "utility.hpp"

template <typename T>
void testing_prune_dense2csr_bad_arg(const Arguments& arg)
{
    static constexpr size_t safe_size = 100;

    static constexpr rocsparse_int M                      = 10;
    static constexpr rocsparse_int N                      = 10;
    static constexpr rocsparse_int LDA                    = M;
    static constexpr T             threshold              = static_cast<T>(1);
    static rocsparse_int           nnz_total_dev_host_ptr = 100;
    static size_t                  buffer_size            = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);
    device_vector<T>             dA(safe_size);
    device_vector<T>             dtemp_buffer(safe_size);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dA || !dtemp_buffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Test rocsparse_prune_dense2csr_buffer_size
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_buffer_size<T>(nullptr,
                                                                     M,
                                                                     N,
                                                                     dA,
                                                                     LDA,
                                                                     &threshold,
                                                                     descr,
                                                                     dcsr_val,
                                                                     dcsr_row_ptr,
                                                                     dcsr_col_ind,
                                                                     &buffer_size),
                            rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_buffer_size<T>(handle,
                                                                     M,
                                                                     N,
                                                                     dA,
                                                                     LDA,
                                                                     &threshold,
                                                                     descr,
                                                                     dcsr_val,
                                                                     dcsr_row_ptr,
                                                                     dcsr_col_ind,
                                                                     nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_prune_dense2csr_nnz
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(nullptr,
                                                             M,
                                                             N,
                                                             dA,
                                                             LDA,
                                                             &threshold,
                                                             descr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             -1,
                                                             N,
                                                             dA,
                                                             LDA,
                                                             &threshold,
                                                             descr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             M,
                                                             -1,
                                                             dA,
                                                             LDA,
                                                             &threshold,
                                                             descr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             M,
                                                             N,
                                                             dA,
                                                             -1,
                                                             &threshold,
                                                             descr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             M,
                                                             N,
                                                             (const T*)nullptr,
                                                             LDA,
                                                             &threshold,
                                                             descr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             M,
                                                             N,
                                                             dA,
                                                             LDA,
                                                             (const T*)nullptr,
                                                             descr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             M,
                                                             N,
                                                             dA,
                                                             LDA,
                                                             &threshold,
                                                             nullptr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             M,
                                                             N,
                                                             dA,
                                                             LDA,
                                                             &threshold,
                                                             descr,
                                                             nullptr,
                                                             &nnz_total_dev_host_ptr,
                                                             dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_prune_dense2csr_nnz<T>(
            handle, M, N, dA, LDA, &threshold, descr, dcsr_row_ptr, nullptr, dtemp_buffer),
        rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                             M,
                                                             N,
                                                             dA,
                                                             LDA,
                                                             &threshold,
                                                             descr,
                                                             dcsr_row_ptr,
                                                             &nnz_total_dev_host_ptr,
                                                             nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_prune_dense2csr
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(nullptr,
                                                         M,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         -1,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         -1,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         dA,
                                                         -1,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         (const T*)nullptr,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         (const T*)nullptr,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         nullptr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         (T*)nullptr,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         nullptr,
                                                         dcsr_col_ind,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         nullptr,
                                                         dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr<T>(handle,
                                                         M,
                                                         N,
                                                         dA,
                                                         LDA,
                                                         &threshold,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_prune_dense2csr(const Arguments& arg)
{
    rocsparse_int        M         = arg.M;
    rocsparse_int        N         = arg.N;
    rocsparse_int        LDA       = arg.denseld;
    rocsparse_index_base base      = arg.baseA;
    T                    threshold = static_cast<T>(arg.alpha);

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || LDA < M)
    {
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_prune_dense2csr<T>(
                handle, M, N, nullptr, LDA, nullptr, descr, nullptr, nullptr, nullptr, nullptr),
            (M < 0 || N < 0 || LDA < M) ? rocsparse_status_invalid_size : rocsparse_status_success);

        return;
    }

    // Allocate host memory
    host_vector<T>             h_A(LDA * N);
    host_vector<rocsparse_int> h_nnz_total_dev_host_ptr(1);

    // Allocate device memory
    device_vector<T>             d_A(LDA * N);
    device_vector<rocsparse_int> d_nnz_total_dev_host_ptr(1);
    device_vector<rocsparse_int> d_csr_row_ptr(M + 1);
    if(!d_A || !d_nnz_total_dev_host_ptr || !d_csr_row_ptr)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Initialize a random matrix.
    rocsparse_seedrand();

    // Initialize the entire allocated memory.
    for(rocsparse_int i = 0; i < LDA; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            h_A[j * LDA + i] = -1;
        }
    }

    // Random initialization of the matrix.
    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            h_A[j * LDA + i] = random_generator_normal<T>();
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(T) * LDA * N, hipMemcpyHostToDevice));

    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_buffer_size<T>(
        handle, M, N, d_A, LDA, &threshold, descr, nullptr, d_csr_row_ptr, nullptr, &buffer_size));

    T* d_temp_buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_temp_buffer, buffer_size));

    if(!d_temp_buffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    T* d_threshold = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_threshold, sizeof(T)));

    if(!d_threshold)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_HIP_ERROR(hipMemcpy(d_threshold, &threshold, sizeof(T), hipMemcpyHostToDevice));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                           M,
                                                           N,
                                                           d_A,
                                                           LDA,
                                                           &threshold,
                                                           descr,
                                                           d_csr_row_ptr,
                                                           h_nnz_total_dev_host_ptr,
                                                           d_temp_buffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_nnz<T>(handle,
                                                           M,
                                                           N,
                                                           d_A,
                                                           LDA,
                                                           d_threshold,
                                                           descr,
                                                           d_csr_row_ptr,
                                                           d_nnz_total_dev_host_ptr,
                                                           d_temp_buffer));

    device_vector<rocsparse_int> d_csr_col_ind(h_nnz_total_dev_host_ptr[0]);
    device_vector<T>             d_csr_val(h_nnz_total_dev_host_ptr[0]);

    if(arg.unit_check)
    {
        host_vector<rocsparse_int> h_nnz_total_copied_from_device(1);
        CHECK_HIP_ERROR(hipMemcpy(h_nnz_total_copied_from_device,
                                  d_nnz_total_dev_host_ptr,
                                  sizeof(rocsparse_int),
                                  hipMemcpyDeviceToHost));

        unit_check_general<rocsparse_int>(
            1, 1, 1, h_nnz_total_dev_host_ptr, h_nnz_total_copied_from_device);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr<T>(handle,
                                                           M,
                                                           N,
                                                           d_A,
                                                           LDA,
                                                           &threshold,
                                                           descr,
                                                           d_csr_val,
                                                           d_csr_row_ptr,
                                                           d_csr_col_ind,
                                                           d_temp_buffer));

        host_vector<rocsparse_int> h_csr_row_ptr(M + 1);
        host_vector<rocsparse_int> h_csr_col_ind(h_nnz_total_dev_host_ptr[0]);
        host_vector<T>             h_csr_val(h_nnz_total_dev_host_ptr[0]);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            h_csr_row_ptr, d_csr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(h_csr_col_ind,
                                  d_csr_col_ind,
                                  sizeof(rocsparse_int) * h_nnz_total_dev_host_ptr[0],
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_csr_val, d_csr_val, sizeof(T) * h_nnz_total_dev_host_ptr[0], hipMemcpyDeviceToHost));

        // call host and check results
        host_vector<rocsparse_int> h_csr_row_ptr_cpu;
        host_vector<rocsparse_int> h_csr_col_ind_cpu;
        host_vector<T>             h_csr_val_cpu;
        host_vector<rocsparse_int> h_nnz_cpu(1);

        host_prune_dense2csr(M,
                             N,
                             h_A,
                             LDA,
                             base,
                             threshold,
                             h_nnz_cpu[0],
                             h_csr_val_cpu,
                             h_csr_row_ptr_cpu,
                             h_csr_col_ind_cpu);

        unit_check_general<rocsparse_int>(1, 1, 1, h_nnz_cpu, h_nnz_total_dev_host_ptr);
        unit_check_general<rocsparse_int>(1, (M + 1), 1, h_csr_row_ptr_cpu, h_csr_row_ptr);
        unit_check_general<rocsparse_int>(
            1, h_nnz_total_dev_host_ptr[0], 1, h_csr_col_ind_cpu, h_csr_col_ind);
        unit_check_general<T>(1, h_nnz_total_dev_host_ptr[0], 1, h_csr_val_cpu, h_csr_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr<T>(handle,
                                                               M,
                                                               N,
                                                               d_A,
                                                               LDA,
                                                               &threshold,
                                                               descr,
                                                               d_csr_val,
                                                               d_csr_row_ptr,
                                                               d_csr_col_ind,
                                                               d_temp_buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr<T>(handle,
                                                               M,
                                                               N,
                                                               d_A,
                                                               LDA,
                                                               &threshold,
                                                               descr,
                                                               d_csr_val,
                                                               d_csr_row_ptr,
                                                               d_csr_col_ind,
                                                               d_temp_buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = prune_dense2csr_gbyte_count<T>(M, N, h_nnz_total_dev_host_ptr[0])
                           / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::setw(12) << "iter"
                  << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12)
                  << h_nnz_total_dev_host_ptr[0] << std::setw(12) << gpu_gbyte << std::setw(12)
                  << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(d_temp_buffer));
    CHECK_HIP_ERROR(hipFree(d_threshold));
}

#endif // TESTING_PRUNE_DENSE2CSR_HPP

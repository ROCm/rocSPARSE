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
void testing_prune_dense2csr_by_percentage_bad_arg(const Arguments& arg)
{
    static constexpr size_t safe_size = 100;

    static constexpr rocsparse_int M                      = 10;
    static constexpr rocsparse_int N                      = 10;
    static constexpr rocsparse_int LDA                    = M;
    static constexpr T             percentage             = static_cast<T>(0);
    static rocsparse_int           nnz_total_dev_host_ptr = 100;
    static size_t                  buffer_size            = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

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

    // Test rocsparse_prune_dense2csr_by_percentage_buffer_size
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage_buffer_size<T>(nullptr,
                                                                                   M,
                                                                                   N,
                                                                                   dA,
                                                                                   LDA,
                                                                                   percentage,
                                                                                   descr,
                                                                                   dcsr_val,
                                                                                   dcsr_row_ptr,
                                                                                   dcsr_col_ind,
                                                                                   info,
                                                                                   &buffer_size),
                            rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage_buffer_size<T>(handle,
                                                                                   M,
                                                                                   N,
                                                                                   dA,
                                                                                   LDA,
                                                                                   percentage,
                                                                                   descr,
                                                                                   dcsr_val,
                                                                                   dcsr_row_ptr,
                                                                                   dcsr_col_ind,
                                                                                   info,
                                                                                   nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_prune_dense2csr_nnz_by_percentage
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(nullptr,
                                                                           M,
                                                                           N,
                                                                           dA,
                                                                           LDA,
                                                                           percentage,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           dtemp_buffer),
                            rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           -1,
                                                                           N,
                                                                           dA,
                                                                           LDA,
                                                                           percentage,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           -1,
                                                                           dA,
                                                                           LDA,
                                                                           percentage,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           dA,
                                                                           -1,
                                                                           percentage,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           dA,
                                                                           LDA,
                                                                           -1,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           &buffer_size),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           dA,
                                                                           LDA,
                                                                           101,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           &buffer_size),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           (const T*)nullptr,
                                                                           LDA,
                                                                           percentage,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           dA,
                                                                           LDA,
                                                                           percentage,
                                                                           nullptr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           dA,
                                                                           LDA,
                                                                           percentage,
                                                                           descr,
                                                                           nullptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_prune_dense2csr_nnz_by_percentage<T>(
            handle, M, N, dA, LDA, percentage, descr, dcsr_row_ptr, nullptr, info, dtemp_buffer),
        rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                           M,
                                                                           N,
                                                                           dA,
                                                                           LDA,
                                                                           percentage,
                                                                           descr,
                                                                           dcsr_row_ptr,
                                                                           &nnz_total_dev_host_ptr,
                                                                           info,
                                                                           nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_prune_dense2csr_by_percentage
    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(nullptr,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_handle);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       -1,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       -1,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       -1,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       -1,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       101,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       (const T*)nullptr,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       nullptr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       (T*)nullptr,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       nullptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       nullptr,
                                                                       info,
                                                                       dtemp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                       M,
                                                                       N,
                                                                       dA,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       dcsr_val,
                                                                       dcsr_row_ptr,
                                                                       dcsr_col_ind,
                                                                       info,
                                                                       nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_prune_dense2csr_by_percentage(const Arguments& arg)
{
    rocsparse_int        M          = arg.M;
    rocsparse_int        N          = arg.N;
    rocsparse_int        LDA        = arg.denseld;
    rocsparse_index_base base       = arg.baseA;
    T                    percentage = static_cast<T>(arg.percentage);

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || LDA < M || percentage < 0.0 || percentage > 100.0)
    {
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                       M,
                                                       N,
                                                       nullptr,
                                                       LDA,
                                                       percentage,
                                                       descr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       info,
                                                       nullptr),
            (M < 0 || N < 0 || LDA < M || percentage < 0.0 || percentage > 100.0)
                ? rocsparse_status_invalid_size
                : rocsparse_status_success);

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
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < LDA; ++i)
        {
            h_A[j * LDA + i] = -1;
        }
    }

    // Random initialization of the matrix.
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            h_A[j * LDA + i] = random_generator_normal<T>();
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(T) * LDA * N, hipMemcpyHostToDevice));

    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_by_percentage_buffer_size<T>(handle,
                                                                                 M,
                                                                                 N,
                                                                                 d_A,
                                                                                 LDA,
                                                                                 percentage,
                                                                                 descr,
                                                                                 nullptr,
                                                                                 d_csr_row_ptr,
                                                                                 nullptr,
                                                                                 info,
                                                                                 &buffer_size));

    T* d_temp_buffer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_temp_buffer, buffer_size));

    if(!d_temp_buffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                         M,
                                                                         N,
                                                                         d_A,
                                                                         LDA,
                                                                         percentage,
                                                                         descr,
                                                                         d_csr_row_ptr,
                                                                         h_nnz_total_dev_host_ptr,
                                                                         info,
                                                                         d_temp_buffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_nnz_by_percentage<T>(handle,
                                                                         M,
                                                                         N,
                                                                         d_A,
                                                                         LDA,
                                                                         percentage,
                                                                         descr,
                                                                         d_csr_row_ptr,
                                                                         d_nnz_total_dev_host_ptr,
                                                                         info,
                                                                         d_temp_buffer));

    device_vector<rocsparse_int> d_csr_col_ind(h_nnz_total_dev_host_ptr[0]);
    device_vector<T>             d_csr_val(h_nnz_total_dev_host_ptr[0]);

    if(!d_csr_col_ind || !d_csr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

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
        CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                         M,
                                                                         N,
                                                                         d_A,
                                                                         LDA,
                                                                         percentage,
                                                                         descr,
                                                                         d_csr_val,
                                                                         d_csr_row_ptr,
                                                                         d_csr_col_ind,
                                                                         info,
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

        host_prune_dense2csr_by_percentage(M,
                                           N,
                                           h_A,
                                           LDA,
                                           base,
                                           percentage,
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

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                             M,
                                                                             N,
                                                                             d_A,
                                                                             LDA,
                                                                             percentage,
                                                                             descr,
                                                                             d_csr_val,
                                                                             d_csr_row_ptr,
                                                                             d_csr_col_ind,
                                                                             info,
                                                                             d_temp_buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_prune_dense2csr_by_percentage<T>(handle,
                                                                             M,
                                                                             N,
                                                                             d_A,
                                                                             LDA,
                                                                             percentage,
                                                                             descr,
                                                                             d_csr_val,
                                                                             d_csr_row_ptr,
                                                                             d_csr_col_ind,
                                                                             info,
                                                                             d_temp_buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte
            = prune_dense2csr_by_percentage_gbyte_count<T>(M, N, h_nnz_total_dev_host_ptr[0])
              / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "percentage" << std::setw(12) << "GB/s" << std::setw(12)
                  << "msec" << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12)
                  << h_nnz_total_dev_host_ptr[0] << std::setw(12) << percentage << std::setw(12)
                  << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3 << std::setw(12)
                  << number_hot_calls << std::setw(12) << (arg.unit_check ? "yes" : "no")
                  << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(d_temp_buffer));
}

#define INSTANTIATE(TYPE)                                                                    \
    template void testing_prune_dense2csr_by_percentage_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_prune_dense2csr_by_percentage<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
// INSTANTIATE(rocsparse_float_complex);
// INSTANTIATE(rocsparse_double_complex);

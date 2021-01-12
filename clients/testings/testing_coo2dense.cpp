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

template <typename T>
void testing_coo2dense_bad_arg(const Arguments& arg)
{

    static constexpr size_t        safe_size = 100;
    static constexpr rocsparse_int M         = 10;
    static constexpr rocsparse_int N         = 10;
    static constexpr rocsparse_int NNZ       = 10;
    static constexpr rocsparse_int LD        = M;
    rocsparse_local_handle         handle;

    device_vector<T>             d_dense_val(safe_size);
    device_vector<rocsparse_int> d_coo_row_ind(2);
    device_vector<rocsparse_int> d_coo_col_ind(2);
    device_vector<T>             d_coo_val(2);

    if(!d_dense_val || !d_coo_row_ind || !d_coo_col_ind || !d_coo_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_mat_descr descr = nullptr;
    rocsparse_create_mat_descr(&descr);

    // Testing invalid handle.
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coo2dense(
            nullptr, 0, 0, 0, nullptr, (const T*)nullptr, nullptr, nullptr, (T*)nullptr, 0),
        rocsparse_status_invalid_handle);

    // Testing invalid pointers.
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                N,
                                                NNZ,
                                                nullptr,
                                                (const T*)d_coo_val,
                                                d_coo_row_ind,
                                                d_coo_col_ind,
                                                (T*)d_dense_val,
                                                LD),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                N,
                                                NNZ,
                                                descr,
                                                (const T*)nullptr,
                                                d_coo_row_ind,
                                                d_coo_col_ind,
                                                (T*)d_dense_val,
                                                LD),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                N,
                                                NNZ,
                                                descr,
                                                (const T*)d_coo_val,
                                                nullptr,
                                                d_coo_col_ind,
                                                (T*)d_dense_val,
                                                LD),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                N,
                                                NNZ,
                                                descr,
                                                (const T*)d_coo_val,
                                                d_coo_row_ind,
                                                nullptr,
                                                (T*)d_dense_val,
                                                LD),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                N,
                                                NNZ,
                                                descr,
                                                (const T*)d_coo_val,
                                                d_coo_row_ind,
                                                d_coo_col_ind,
                                                (T*)nullptr,
                                                LD),
                            rocsparse_status_invalid_pointer);

    // Testing invalid size on M
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                -1,
                                                N,
                                                NNZ,
                                                descr,
                                                (const T*)d_coo_val,
                                                d_coo_row_ind,
                                                d_coo_col_ind,
                                                (T*)d_dense_val,
                                                LD),
                            rocsparse_status_invalid_size);
    // Testing invalid size on N
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                -1,
                                                NNZ,
                                                descr,
                                                (const T*)d_coo_val,
                                                d_coo_row_ind,
                                                d_coo_col_ind,
                                                (T*)d_dense_val,
                                                LD),
                            rocsparse_status_invalid_size);
    // Testing invalid size on NNZ
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                N,
                                                -1,
                                                descr,
                                                (const T*)d_coo_val,
                                                d_coo_row_ind,
                                                d_coo_col_ind,
                                                (T*)d_dense_val,
                                                LD),
                            rocsparse_status_invalid_size);
    // Testing invalid size on LD
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo2dense(handle,
                                                M,
                                                -1,
                                                NNZ,
                                                descr,
                                                (const T*)d_coo_val,
                                                d_coo_row_ind,
                                                d_coo_col_ind,
                                                (T*)d_dense_val,
                                                M - 1),
                            rocsparse_status_invalid_size);
    rocsparse_destroy_mat_descr(descr);
}

template <typename T>
void testing_coo2dense(const Arguments& arg)
{
    rocsparse_int          M  = arg.M;
    rocsparse_int          N  = arg.N;
    rocsparse_int          LD = arg.denseld;
    rocsparse_local_handle handle;
    rocsparse_mat_descr    descr;
    rocsparse_create_mat_descr(&descr);
    rocsparse_set_mat_index_base(descr, arg.baseA);

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || LD < M)
    {
        rocsparse_status expected_status = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_size;

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_coo2dense(
                handle, M, N, 100, descr, (const T*)nullptr, nullptr, nullptr, (T*)nullptr, LD),
            expected_status);
        return;
    }

    // Allocate memory.
    host_vector<T>   h_dense_val(LD * N);
    device_vector<T> d_dense_val(LD * N);

    host_vector<rocsparse_int>   h_nnzPerRow(M);
    device_vector<rocsparse_int> d_nnzPerRow(M);
    if(!d_nnzPerRow || !d_dense_val || !h_nnzPerRow || !h_dense_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_seedrand();

    // Random initialization of the matrix.
    for(rocsparse_int i = 0; i < LD; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            h_dense_val[j * LD + i] = -2;
        }
    }

    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            h_dense_val[j * LD + i] = random_generator<T>(0, 9);
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(d_dense_val, h_dense_val, sizeof(T) * LD * N, hipMemcpyHostToDevice));

    rocsparse_int nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_nnz(handle,
                                        rocsparse_direction_row,
                                        M,
                                        N,
                                        descr,
                                        (const T*)d_dense_val,
                                        LD,
                                        d_nnzPerRow,
                                        &nnz));

    // Transfer.
    CHECK_HIP_ERROR(
        hipMemcpy(h_nnzPerRow, d_nnzPerRow, sizeof(rocsparse_int) * M, hipMemcpyDeviceToHost));

    host_vector<rocsparse_int> h_coo_row_ind(std::max(nnz, 1));
    host_vector<T>             h_coo_val(std::max(nnz, 1));
    host_vector<rocsparse_int> h_coo_col_ind(std::max(nnz, 1));
    if(!h_coo_row_ind || !h_coo_val || !h_coo_col_ind)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    device_vector<rocsparse_int> d_coo_row_ind(nnz);
    device_vector<T>             d_coo_val(nnz);
    device_vector<rocsparse_int> d_coo_col_ind(nnz);
    if(!d_coo_row_ind || !d_coo_val || !d_coo_col_ind)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Convert the dense matrix to a compressed sparse matrix.
    CHECK_ROCSPARSE_ERROR(rocsparse_dense2coo<T>(handle,
                                                 M,
                                                 N,
                                                 descr,
                                                 d_dense_val,
                                                 LD,
                                                 d_nnzPerRow,
                                                 d_coo_val,
                                                 d_coo_row_ind,
                                                 d_coo_col_ind));

    // Copy on host.
    CHECK_HIP_ERROR(
        hipMemcpy(h_coo_val, d_coo_val, sizeof(T) * std::max(nnz, 1), hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(h_coo_row_ind,
                              d_coo_row_ind,
                              sizeof(rocsparse_int) * std::max(nnz, 1),
                              hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(h_coo_col_ind,
                              d_coo_col_ind,
                              sizeof(rocsparse_int) * std::max(nnz, 1),
                              hipMemcpyDeviceToHost));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_coo2dense(handle,
                                                  M,
                                                  N,
                                                  nnz,
                                                  descr,
                                                  (const T*)d_coo_val,
                                                  d_coo_row_ind,
                                                  d_coo_col_ind,
                                                  (T*)d_dense_val,
                                                  LD));

        host_vector<T> gpu_dense_val(LD * N);
        CHECK_HIP_ERROR(
            hipMemcpy(gpu_dense_val, d_dense_val, sizeof(T) * LD * N, hipMemcpyDeviceToHost));

        host_vector<T> cpu_dense_val = h_dense_val;
        host_coo_to_dense(M,
                          N,
                          nnz,
                          rocsparse_get_mat_index_base(descr),
                          h_coo_val,
                          h_coo_row_ind,
                          h_coo_col_ind,
                          cpu_dense_val,
                          LD,
                          rocsparse_order_column);

        unit_check_general(M, N, LD, (T*)h_dense_val, (T*)cpu_dense_val);
        unit_check_general(M, N, LD, (T*)h_dense_val, (T*)gpu_dense_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm-up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {

            CHECK_ROCSPARSE_ERROR(rocsparse_coo2dense(handle,
                                                      M,
                                                      N,
                                                      nnz,
                                                      descr,
                                                      (const T*)d_coo_val,
                                                      d_coo_row_ind,
                                                      d_coo_col_ind,
                                                      (T*)d_dense_val,
                                                      LD));
        }

        double gpu_time_used = get_time_us();
        {
            // Performance run
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_coo2dense(handle,
                                                          M,
                                                          N,
                                                          nnz,
                                                          descr,
                                                          (const T*)d_coo_val,
                                                          d_coo_row_ind,
                                                          d_coo_col_ind,
                                                          (T*)d_dense_val,
                                                          LD));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = coo2dense_gbyte_count<rocsparse_int, T>(M, N, nnz) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);
        // clang-format off
        std::cout
	  << std::setw(20) << "M"
	  << std::setw(20) << "N"
	  << std::setw(20) << "LD"
	  << std::setw(20) << "nnz"
	  << std::setw(20) << "GB/s"
	  << std::setw(20) << "msec"
	  << std::setw(20) << "iter"
	  << std::setw(20) << "verified"
	  << std::endl;

        std::cout
	  << std::setw(20) << M
	  << std::setw(20) << N
	  << std::setw(20) << LD
	  << std::setw(20) << nnz
	  << std::setw(20) << gpu_gbyte
	  << std::setw(20) << gpu_time_used / 1e3
	  << std::setw(20) << number_hot_calls
	  << std::setw(20) << (arg.unit_check ? "yes" : "no")
	  << std::endl;
        // clang-format on
    }

    // Destroy the matrix descriptor.
    rocsparse_destroy_mat_descr(descr);
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_coo2dense_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_coo2dense<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

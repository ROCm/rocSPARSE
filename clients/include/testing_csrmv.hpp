/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef TESTING_CSRMV_HPP
#define TESTING_CSRMV_HPP

#include <rocsparse.hpp>

#include "flops.hpp"
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
void testing_csrmv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;
    T h_beta  = 0.1;

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
    device_vector<T>             dx(safe_size);
    device_vector<T>             dy(safe_size);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_csrmv_analysis()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(nullptr,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        nullptr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        nullptr,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        nullptr,
                                                        dcsr_col_ind,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        nullptr,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrmv()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(nullptr,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               nullptr,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               nullptr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               nullptr,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dcsr_val,
                                               nullptr,
                                               dcsr_col_ind,
                                               info,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               nullptr,
                                               info,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               nullptr,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               dx,
                                               nullptr,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               info,
                                               dx,
                                               &h_beta,
                                               nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrmv_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(nullptr, info), rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_csrmv(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         dim_x     = arg.dimx;
    rocsparse_int         dim_y     = arg.dimy;
    rocsparse_int         dim_z     = arg.dimz;
    rocsparse_operation   trans     = arg.transA;
    rocsparse_index_base  base      = arg.baseA;
    rocsparse_matrix_init mat       = arg.matrix;
    uint32_t              adaptive  = arg.algo;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info_ptr;

    // Differentiate between algorithm 0 (csrmv without analysis step) and
    //                       algorithm 1 (csrmv with analysis step)
    rocsparse_mat_info info = adaptive ? info_ptr : nullptr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<T>             dx(safe_size);
        device_vector<T>             dy(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // If adaptive, perform analysis step
        if(adaptive)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                                trans,
                                                                M,
                                                                N,
                                                                safe_size,
                                                                descr,
                                                                dcsr_val,
                                                                dcsr_row_ptr,
                                                                dcsr_col_ind,
                                                                info),
                                    (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                     : rocsparse_status_success);
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                                   trans,
                                                   M,
                                                   N,
                                                   safe_size,
                                                   &h_alpha,
                                                   descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   info,
                                                   dx,
                                                   &h_beta,
                                                   dy),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        // If adaptive, clear data
        if(adaptive)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(handle, info), rocsparse_status_success);
        }

        return;
    }

    // Allocate host memory for matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int nnz;
    rocsparse_init_csr_matrix(hcsr_row_ptr,
                              hcsr_col_ind,
                              hcsr_val,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz,
                              base,
                              mat,
                              filename.c_str(),
                              arg.timing ? false : adaptive,
                              full_rank);

    // Allocate host memory for vectors
    host_vector<T> hx(N);
    host_vector<T> hy_1(M);
    host_vector<T> hy_2(M);
    host_vector<T> hy_gold(M);

    // Initialize data on CPU
    rocsparse_init<T>(hx, 1, N, 1);
    rocsparse_init<T>(hy_1, 1, M, 1);
    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);
    device_vector<T>             dx(N);
    device_vector<T>             dy_1(M);
    device_vector<T>             dy_2(M);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy_1 || !dy_2 || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));

    // If adaptive, run analysis step
    if(adaptive)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(
            handle, trans, M, N, nnz, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info));
    }

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                 trans,
                                                 M,
                                                 N,
                                                 nnz,
                                                 &h_alpha,
                                                 descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 info,
                                                 dx,
                                                 &h_beta,
                                                 dy_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                 trans,
                                                 M,
                                                 N,
                                                 nnz,
                                                 d_alpha,
                                                 descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 info,
                                                 dx,
                                                 d_beta,
                                                 dy_2));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU csrmv
        host_csrmv<T>(M,
                      nnz,
                      h_alpha,
                      hcsr_row_ptr,
                      hcsr_col_ind,
                      hcsr_val,
                      hx,
                      h_beta,
                      hy_gold,
                      base,
                      adaptive);

        near_check_general<T>(1, M, 1, hy_gold, hy_1);
        near_check_general<T>(1, M, 1, hy_gold, hy_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                     trans,
                                                     M,
                                                     N,
                                                     nnz,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     &h_beta,
                                                     dy_1));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                     trans,
                                                     M,
                                                     N,
                                                     nnz,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     &h_beta,
                                                     dy_1));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops
            = spmv_gflop_count<T>(M, nnz, h_beta != static_cast<T>(0)) / gpu_time_used * 1e6;
        double gpu_gbyte
            = csrmv_gbyte_count<T>(M, N, nnz, h_beta != static_cast<T>(0)) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "Algorithm" << std::setw(12) << "GFlop/s" << std::setw(12) << "GB/s"
                  << std::setw(12) << "msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz
                  << std::setw(12) << h_alpha << std::setw(12) << h_beta << std::setw(12)
                  << (adaptive ? "adaptive" : "stream") << std::setw(12) << gpu_gflops
                  << std::setw(12) << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // If adaptive, clear analysis data
    if(adaptive)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    }
}

#endif // TESTING_CSRMV_HPP

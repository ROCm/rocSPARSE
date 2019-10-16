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
#ifndef TESTING_HYBMV_HPP
#define TESTING_HYBMV_HPP

#include <rocsparse.hpp>

#include "flops.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"
#include "utility.hpp"

template <typename T>
void testing_hybmv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;
    T h_beta  = 0.1;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create hyb matrix
    rocsparse_local_hyb_mat hyb;

    // Allocate memory on device
    device_vector<T> dx(safe_size);
    device_vector<T> dy(safe_size);

    if(!dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_hybmv()
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hybmv<T>(
            nullptr, rocsparse_operation_none, &h_alpha, descr, hyb, dx, &h_beta, dy),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hybmv<T>(handle, rocsparse_operation_none, nullptr, descr, hyb, dx, &h_beta, dy),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hybmv<T>(
            handle, rocsparse_operation_none, &h_alpha, nullptr, hyb, dx, &h_beta, dy),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hybmv<T>(
            handle, rocsparse_operation_none, &h_alpha, descr, nullptr, dx, &h_beta, dy),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hybmv<T>(
            handle, rocsparse_operation_none, &h_alpha, descr, hyb, nullptr, &h_beta, dy),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hybmv<T>(handle, rocsparse_operation_none, &h_alpha, descr, hyb, dx, nullptr, dy),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hybmv<T>(
            handle, rocsparse_operation_none, &h_alpha, descr, hyb, dx, &h_beta, nullptr),
        rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_hybmv(const Arguments& arg)
{
    rocsparse_int           M              = arg.M;
    rocsparse_int           N              = arg.N;
    rocsparse_int           K              = arg.K;
    rocsparse_int           dim_x          = arg.dimx;
    rocsparse_int           dim_y          = arg.dimy;
    rocsparse_int           dim_z          = arg.dimz;
    rocsparse_operation     trans          = arg.transA;
    rocsparse_index_base    base           = arg.baseA;
    rocsparse_matrix_init   mat            = arg.matrix;
    rocsparse_hyb_partition part           = arg.part;
    rocsparse_int           user_ell_width = arg.algo;
    bool                    full_rank      = false;
    std::string             filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create hyb matrix
    rocsparse_local_hyb_mat hyb;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);

        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_hybmv<T>(handle, trans, &h_alpha, descr, hyb, dx, &h_beta, dy));

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
                              arg.timing ? false : true,
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

    // ELL width limit
    rocsparse_int width_limit = 2 * (nnz - 1) / M + 1;

    // Limit ELL user width
    if(part == rocsparse_hyb_partition_user)
    {
        user_ell_width *= (nnz / M);
        user_ell_width = std::min(width_limit, user_ell_width);
    }

    // Convert CSR matrix to HYB
    rocsparse_status status = rocsparse_csr2hyb<T>(
        handle, M, N, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, hyb, user_ell_width, part);

    if(part == rocsparse_hyb_partition_max)
    {
        // Compute max ELL width
        rocsparse_int ell_max_width = 0;
        for(rocsparse_int i = 0; i < M; ++i)
        {
            ell_max_width = std::max(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i], ell_max_width);
        }

        if(ell_max_width > width_limit)
        {
            return;
        }
    }

    CHECK_ROCSPARSE_ERROR(status);

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_hybmv<T>(handle, trans, &h_alpha, descr, hyb, dx, &h_beta, dy_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_hybmv<T>(handle, trans, d_alpha, descr, hyb, dx, d_beta, dy_2));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU hybmv
        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        rocsparse_int ell_width = dhyb->ell_width;
        rocsparse_int ell_nnz   = dhyb->ell_nnz;
        rocsparse_int coo_nnz   = dhyb->coo_nnz;

        host_vector<rocsparse_int> hhyb_ell_col_ind(ell_nnz);
        host_vector<T>             hhyb_ell_val(ell_nnz);
        host_vector<rocsparse_int> hhyb_coo_row_ind(coo_nnz);
        host_vector<rocsparse_int> hhyb_coo_col_ind(coo_nnz);
        host_vector<T>             hhyb_coo_val(coo_nnz);

        if(ell_nnz > 0)
        {
            CHECK_HIP_ERROR(hipMemcpy(hhyb_ell_col_ind,
                                      dhyb->ell_col_ind,
                                      sizeof(rocsparse_int) * ell_nnz,
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(
                hipMemcpy(hhyb_ell_val, dhyb->ell_val, sizeof(T) * ell_nnz, hipMemcpyDeviceToHost));
        }

        if(coo_nnz > 0)
        {
            CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_row_ind,
                                      dhyb->coo_row_ind,
                                      sizeof(rocsparse_int) * coo_nnz,
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_col_ind,
                                      dhyb->coo_col_ind,
                                      sizeof(rocsparse_int) * coo_nnz,
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(
                hipMemcpy(hhyb_coo_val, dhyb->coo_val, sizeof(T) * coo_nnz, hipMemcpyDeviceToHost));
        }

        host_hybmv<T>(M,
                      N,
                      h_alpha,
                      ell_nnz,
                      hhyb_ell_col_ind,
                      hhyb_ell_val,
                      ell_width,
                      coo_nnz,
                      hhyb_coo_row_ind,
                      hhyb_coo_col_ind,
                      hhyb_coo_val,
                      hx,
                      h_beta,
                      hy_gold,
                      base);

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
            CHECK_ROCSPARSE_ERROR(
                rocsparse_hybmv<T>(handle, trans, &h_alpha, descr, hyb, dx, &h_beta, dy_1));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_hybmv<T>(handle, trans, &h_alpha, descr, hyb, dx, &h_beta, dy_1));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops
            = spmv_gflop_count<T>(M, nnz, h_beta != static_cast<T>(0)) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "partition";
        if(part == rocsparse_hyb_partition_user)
        {
            std::cout << std::setw(12) << "width";
        }
        std::cout << std::setw(12) << "GFlop/s" << std::setw(12) << "msec" << std::setw(12)
                  << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz
                  << std::setw(12) << h_alpha << std::setw(12) << h_beta << std::setw(12)
                  << rocsparse_partition2string(part);
        if(part == rocsparse_hyb_partition_user)
        {
            std::cout << std::setw(12) << user_ell_width;
        }
        std::cout << std::setw(12) << gpu_gflops << std::setw(12) << gpu_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#endif // TESTING_HYBMV_HPP

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
#ifndef TESTING_CSRSV_HPP
#define TESTING_CSRSV_HPP

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
void testing_csrsv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;

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
    device_vector<T>             dbuffer(safe_size);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy || !dbuffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_csrsv_buffer_size()
    size_t buffer_size;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(nullptr,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dcsr_val,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(handle,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           nullptr,
                                                           dcsr_val,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(handle,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           nullptr,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(handle,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dcsr_val,
                                                           nullptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(handle,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dcsr_val,
                                                           dcsr_row_ptr,
                                                           nullptr,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(handle,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dcsr_val,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           nullptr,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(handle,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dcsr_val,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrsv_analysis()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(nullptr,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        nullptr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        nullptr,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        nullptr,
                                                        dcsr_col_ind,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        nullptr,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        nullptr,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrsv_solve()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(nullptr,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     nullptr,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     nullptr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     nullptr,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     nullptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     nullptr,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     nullptr,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     nullptr,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
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
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrsv_zero_pivot()
    rocsparse_int position;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(nullptr, descr, info, &position),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, nullptr, &position),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrsv_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(nullptr, descr, info),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(handle, nullptr, info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(handle, descr, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_csrsv(const Arguments& arg)
{
    rocsparse_int             M         = arg.M;
    rocsparse_int             N         = arg.N;
    rocsparse_int             K         = arg.K;
    rocsparse_int             dim_x     = arg.dimx;
    rocsparse_int             dim_y     = arg.dimy;
    rocsparse_int             dim_z     = arg.dimz;
    rocsparse_operation       trans     = arg.transA;
    rocsparse_diag_type       diag      = arg.diag;
    rocsparse_fill_mode       uplo      = arg.uplo;
    rocsparse_analysis_policy apol      = arg.apol;
    rocsparse_solve_policy    spol      = arg.spol;
    rocsparse_index_base      base      = arg.baseA;
    rocsparse_matrix_init     mat       = arg.matrix;
    bool                      full_rank = true;
    std::string               filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_alpha = arg.get_alpha<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix diag type
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr, diag));

    // Set matrix fill mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr, uplo));

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0)
    {
        static const size_t safe_size = 100;
        size_t              buffer_size;
        rocsparse_int       pivot;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<T>             dx(safe_size);
        device_vector<T>             dy(safe_size);
        device_vector<T>             dbuffer(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy || !dbuffer)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_buffer_size<T>(handle,
                                                               trans,
                                                               M,
                                                               safe_size,
                                                               descr,
                                                               dcsr_val,
                                                               dcsr_row_ptr,
                                                               dcsr_col_ind,
                                                               info,
                                                               &buffer_size),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_analysis<T>(handle,
                                                            trans,
                                                            M,
                                                            safe_size,
                                                            descr,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind,
                                                            info,
                                                            apol,
                                                            spol,
                                                            dbuffer),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_solve<T>(handle,
                                                         trans,
                                                         M,
                                                         safe_size,
                                                         &h_alpha,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         info,
                                                         dx,
                                                         dy,
                                                         spol,
                                                         dbuffer),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, &pivot),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_clear(handle, descr, info),
                                rocsparse_status_success);

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
                              false,
                              full_rank);

    // Allocate host memory for vectors
    host_vector<T>             hx(N);
    host_vector<T>             hy_1(M);
    host_vector<T>             hy_2(M);
    host_vector<T>             hy_gold(M);
    host_vector<rocsparse_int> h_analysis_pivot_1(1);
    host_vector<rocsparse_int> h_analysis_pivot_2(1);
    host_vector<rocsparse_int> h_analysis_pivot_gold(1);
    host_vector<rocsparse_int> h_solve_pivot_1(1);
    host_vector<rocsparse_int> h_solve_pivot_2(1);
    host_vector<rocsparse_int> h_solve_pivot_gold(1);

    // Initialize data on CPU
    rocsparse_init<T>(hx, 1, N, 1);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);
    device_vector<T>             dx(N);
    device_vector<T>             dy_1(M);
    device_vector<T>             dy_2(M);
    device_vector<T>             d_alpha(1);
    device_vector<rocsparse_int> d_analysis_pivot_2(1);
    device_vector<rocsparse_int> d_solve_pivot_2(1);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy_1 || !dy_2 || !d_alpha
       || !d_analysis_pivot_2 || !d_solve_pivot_2)
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

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size<T>(
        handle, trans, M, nnz, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, info, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        rocsparse_status status_analysis_1;
        rocsparse_status status_analysis_2;
        rocsparse_status status_solve_1;
        rocsparse_status status_solve_2;

        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // Perform analysis step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(handle,
                                                          trans,
                                                          M,
                                                          nnz,
                                                          descr,
                                                          dcsr_val,
                                                          dcsr_row_ptr,
                                                          dcsr_col_ind,
                                                          info,
                                                          apol,
                                                          spol,
                                                          dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, h_analysis_pivot_1),
                                (h_analysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(handle,
                                                          trans,
                                                          M,
                                                          nnz,
                                                          descr,
                                                          dcsr_val,
                                                          dcsr_row_ptr,
                                                          dcsr_col_ind,
                                                          info,
                                                          apol,
                                                          spol,
                                                          dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, d_analysis_pivot_2),
                                (h_analysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

        // Perform solve step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                       trans,
                                                       M,
                                                       nnz,
                                                       &h_alpha,
                                                       descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       info,
                                                       dx,
                                                       dy_1,
                                                       spol,
                                                       dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, h_solve_pivot_1),
                                (h_solve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                       trans,
                                                       M,
                                                       nnz,
                                                       d_alpha,
                                                       descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       info,
                                                       dx,
                                                       dy_2,
                                                       spol,
                                                       dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrsv_zero_pivot(handle, descr, info, d_solve_pivot_2),
                                (h_solve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(h_analysis_pivot_2, d_analysis_pivot_2, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(h_solve_pivot_2, d_solve_pivot_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU csrsv
        host_csrsv<T>(M,
                      h_alpha,
                      hcsr_row_ptr,
                      hcsr_col_ind,
                      hcsr_val,
                      hx,
                      hy_gold,
                      diag,
                      uplo,
                      base,
                      h_analysis_pivot_gold,
                      h_solve_pivot_gold);

        // Check pivots
        unit_check_general<rocsparse_int>(1, 1, 1, h_analysis_pivot_gold, h_analysis_pivot_1);
        unit_check_general<rocsparse_int>(1, 1, 1, h_analysis_pivot_gold, h_analysis_pivot_2);
        unit_check_general<rocsparse_int>(1, 1, 1, h_solve_pivot_gold, h_solve_pivot_1);
        unit_check_general<rocsparse_int>(1, 1, 1, h_solve_pivot_gold, h_solve_pivot_2);

        // Check solution vector if no pivot has been found
        if(h_analysis_pivot_gold[0] == -1 && h_solve_pivot_gold[0] == -1)
        {
            near_check_general<T>(1, M, 1, hy_gold, hy_1);
            near_check_general<T>(1, M, 1, hy_gold, hy_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(handle,
                                                              trans,
                                                              M,
                                                              nnz,
                                                              descr,
                                                              dcsr_val,
                                                              dcsr_row_ptr,
                                                              dcsr_col_ind,
                                                              info,
                                                              apol,
                                                              spol,
                                                              dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                           trans,
                                                           M,
                                                           nnz,
                                                           &h_alpha,
                                                           descr,
                                                           dcsr_val,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           dx,
                                                           dy_1,
                                                           spol,
                                                           dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_clear(handle, descr, info));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis<T>(handle,
                                                          trans,
                                                          M,
                                                          nnz,
                                                          descr,
                                                          dcsr_val,
                                                          dcsr_row_ptr,
                                                          dcsr_col_ind,
                                                          info,
                                                          apol,
                                                          spol,
                                                          dbuffer));

        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve<T>(handle,
                                                           trans,
                                                           M,
                                                           nnz,
                                                           &h_alpha,
                                                           descr,
                                                           dcsr_val,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           dx,
                                                           dy_1,
                                                           spol,
                                                           dbuffer));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gpu_gflops = csrsv_gflop_count<T>(M, nnz, diag) / gpu_solve_time_used * 1e6;
        double gpu_gbyte  = csrsv_gbyte_count<T>(M, nnz) / gpu_solve_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "nnz" << std::setw(12) << "alpha"
                  << std::setw(12) << "pivot" << std::setw(16) << "operation" << std::setw(12)
                  << "diag_type" << std::setw(12) << "fill_mode" << std::setw(16)
                  << "analysis_policy" << std::setw(16) << "solve_policy" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(16) << "analysis_msec"
                  << std::setw(16) << "solve_msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << nnz << std::setw(12) << h_alpha
                  << std::setw(12) << std::min(h_analysis_pivot_gold[0], h_solve_pivot_gold[0])
                  << std::setw(16) << rocsparse_operation2string(trans) << std::setw(12)
                  << rocsparse_diagtype2string(diag) << std::setw(12)
                  << rocsparse_fillmode2string(uplo) << std::setw(16)
                  << rocsparse_analysis2string(apol) << std::setw(16)
                  << rocsparse_solve2string(spol) << std::setw(12) << gpu_gflops << std::setw(12)
                  << gpu_gbyte << std::setw(16) << gpu_analysis_time_used / 1e3 << std::setw(16)
                  << gpu_solve_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Clear csrsv meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_clear(handle, descr, info));

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#endif // TESTING_CSRSV_HPP

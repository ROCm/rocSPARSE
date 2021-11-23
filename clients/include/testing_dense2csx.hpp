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

#pragma once
#ifndef TESTING_DENSE2CSX_HPP
#define TESTING_DENSE2CSX_HPP

#include <rocsparse.hpp>

#include "auto_testing_bad_arg.hpp"
#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"
#include "utility.hpp"
#include <chrono>
#include <thread>

template <rocsparse_direction DIRA, typename T, typename FUNC>
void testing_dense2csx_bad_arg(const Arguments& arg, FUNC& dense2csx)
{

    static constexpr size_t        safe_size = 100;
    static constexpr rocsparse_int M         = 10;
    static constexpr rocsparse_int N         = 10;
    static constexpr rocsparse_int LD        = M;
    rocsparse_local_handle         handle;

    device_vector<T>             d_dense_val(safe_size);
    device_vector<rocsparse_int> d_nnz_per_row_columns(2);
    device_vector<rocsparse_int> d_csx_row_col_ptr(2);
    device_vector<rocsparse_int> d_csx_col_row_ind(2);
    device_vector<T>             d_csx_val(2);

    if(!d_dense_val || !d_nnz_per_row_columns || !d_csx_row_col_ptr || !d_csx_col_row_ind
       || !d_csx_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_local_mat_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, rocsparse_index_base_zero));

    //
    // Testing invalid handle.
    //
    EXPECT_ROCSPARSE_STATUS(
        dense2csx(nullptr, 0, 0, nullptr, nullptr, 0, nullptr, (T*)nullptr, nullptr, nullptr),
        rocsparse_status_invalid_handle);

    //
    // Testing invalid pointers.
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      nullptr,
                                      d_dense_val,
                                      LD,
                                      d_nnz_per_row_columns,
                                      (T*)d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descr,
                                      nullptr,
                                      LD,
                                      d_nnz_per_row_columns,
                                      (T*)d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descr,
                                      d_dense_val,
                                      LD,
                                      nullptr,
                                      (T*)d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descr,
                                      d_dense_val,
                                      LD,
                                      d_nnz_per_row_columns,
                                      (T*)nullptr,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descr,
                                      d_dense_val,
                                      LD,
                                      d_nnz_per_row_columns,
                                      (T*)d_csx_val,
                                      nullptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descr,
                                      d_dense_val,
                                      LD,
                                      d_nnz_per_row_columns,
                                      (T*)d_csx_val,
                                      d_csx_row_col_ptr,
                                      nullptr),
                            rocsparse_status_invalid_pointer);

    //
    // Testing invalid size on M
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      -1,
                                      N,
                                      descr,
                                      d_dense_val,
                                      LD,
                                      d_nnz_per_row_columns,
                                      (T*)d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_size);
    //
    // Testing invalid size on N
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      -1,
                                      descr,
                                      d_dense_val,
                                      LD,
                                      d_nnz_per_row_columns,
                                      (T*)d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_size);
    //
    // Testing invalid size on LD
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descr,
                                      d_dense_val,
                                      M - 1,
                                      d_nnz_per_row_columns,
                                      (T*)d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind),
                            rocsparse_status_invalid_size);
}

template <rocsparse_direction DIRA, typename T, typename FUNC>
void testing_dense2csx(const Arguments& arg, FUNC& dense2csx)

{
    static constexpr rocsparse_direction direction = DIRA;

    rocsparse_int        M      = arg.M;
    rocsparse_int        N      = arg.N;
    rocsparse_int        LD     = arg.denseld;
    rocsparse_index_base baseA  = arg.baseA;
    rocsparse_int        DIMDIR = (rocsparse_direction_row == DIRA) ? M : N;

    rocsparse_local_handle handle;

    rocsparse_local_mat_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, baseA));
    //
    // Argument sanity check before allocating invalid memory
    //
    if(M <= 0 || N <= 0 || LD < M)
    {
        rocsparse_status expected_status = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_size;

        EXPECT_ROCSPARSE_STATUS(
            dense2csx(handle, M, N, descr, nullptr, LD, nullptr, (T*)nullptr, nullptr, nullptr),
            expected_status);
        return;
    }

    //
    // Allocate memory.
    //
    host_vector<T>   h_dense_val(LD * N);
    device_vector<T> d_dense_val(LD * N);

    host_vector<rocsparse_int>   h_nnz_per_row_columns(DIMDIR);
    device_vector<rocsparse_int> d_nnz_per_row_columns(DIMDIR);
    if(!d_nnz_per_row_columns || !d_dense_val || !h_nnz_per_row_columns || !h_dense_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //
    // Initialize the dense matrix.
    //

    {
        //
        // Initialize the random generator.
        //
        rocsparse_seedrand();

        //
        // Initialize the entire allocated memory.
        //
        for(rocsparse_int i = 0; i < LD; ++i)
        {
            for(rocsparse_int j = 0; j < N; ++j)
            {
                h_dense_val[j * LD + i] = -1;
            }
        }

        //
        // Random initialization of the matrix.
        //
        for(rocsparse_int i = 0; i < M; ++i)
        {
            for(rocsparse_int j = 0; j < N; ++j)
            {
                h_dense_val[j * LD + i] = random_generator<T>(0, 4);
            }
        }
    }

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(d_dense_val, h_dense_val, sizeof(T) * LD * N, hipMemcpyHostToDevice));

    rocsparse_int nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_nnz<T>(
        handle, direction, M, N, descr, d_dense_val, LD, d_nnz_per_row_columns, &nnz));

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(h_nnz_per_row_columns,
                              d_nnz_per_row_columns,
                              sizeof(rocsparse_int) * DIMDIR,
                              hipMemcpyDeviceToHost));

    device_vector<rocsparse_int> d_csx_row_col_ptr(DIMDIR + 1);
    device_vector<T>             d_csx_val(nnz);
    device_vector<rocsparse_int> d_csx_col_row_ind(nnz);
    if(!d_csx_row_col_ptr || !d_csx_val || !d_csx_col_row_ind)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    host_vector<rocsparse_int> cpu_csx_row_col_ptr(DIMDIR + 1);
    host_vector<T>             cpu_csx_val(nnz);
    host_vector<rocsparse_int> cpu_csx_col_row_ind(nnz);
    if(!cpu_csx_row_col_ptr || !cpu_csx_val || !cpu_csx_col_row_ind)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    if(arg.unit_check)
    {

        //
        // Compute the reference host first.
        //
        host_dense2csx<DIRA, T>(M,
                                N,
                                rocsparse_get_mat_index_base(descr),
                                h_dense_val,
                                LD,
                                rocsparse_order_column,
                                (const rocsparse_int*)h_nnz_per_row_columns,
                                (T*)cpu_csx_val,
                                (rocsparse_int*)cpu_csx_row_col_ptr,
                                (rocsparse_int*)cpu_csx_col_row_ind);

        CHECK_ROCSPARSE_ERROR(dense2csx(handle,
                                        M,
                                        N,
                                        descr,
                                        d_dense_val,
                                        LD,
                                        d_nnz_per_row_columns,
                                        (T*)d_csx_val,
                                        (rocsparse_int*)d_csx_row_col_ptr,
                                        (rocsparse_int*)d_csx_col_row_ind));

        void* buffer
            = malloc(std::max(sizeof(T), sizeof(rocsparse_int)) * std::max(DIMDIR + 1, nnz));
        //
        // Transfer and check results.
        //
        CHECK_HIP_ERROR(hipMemcpy(buffer, d_csx_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        unit_check_segments(nnz, (T*)cpu_csx_val, (T*)buffer);
        CHECK_HIP_ERROR(hipMemcpy(
            buffer, d_csx_col_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        unit_check_segments(nnz, (rocsparse_int*)cpu_csx_col_row_ind, (rocsparse_int*)buffer);

        CHECK_HIP_ERROR(hipMemcpy(buffer,
                                  d_csx_row_col_ptr,
                                  sizeof(rocsparse_int) * (DIMDIR + 1),
                                  hipMemcpyDeviceToHost));

        unit_check_segments(
            (DIMDIR + 1), (rocsparse_int*)cpu_csx_row_col_ptr, (rocsparse_int*)buffer);

        free(buffer);
        buffer = nullptr;
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        //
        // Warm-up
        //

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(dense2csx(handle,
                                            M,
                                            N,
                                            descr,
                                            d_dense_val,
                                            LD,
                                            d_nnz_per_row_columns,
                                            (T*)d_csx_val,
                                            d_csx_row_col_ptr,
                                            d_csx_col_row_ind));
        }

        double gpu_time_used = get_time_us();
        {
            //
            // Performance run
            //
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(dense2csx(handle,
                                                M,
                                                N,
                                                descr,
                                                d_dense_val,
                                                LD,
                                                d_nnz_per_row_columns,
                                                (T*)d_csx_val,
                                                d_csx_row_col_ptr,
                                                d_csx_col_row_ind));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = dense2csx_gbyte_count<DIRA, T>(M, N, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "LD",
                            LD,
                            "nnz",
                            nnz,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            arg.unit_check ? "yes" : "no");
    }
}

#endif // TESTING_DENSE2CSX_HPP

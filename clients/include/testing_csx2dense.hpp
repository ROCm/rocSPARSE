/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_CSX2DENSE_HPP
#define TESTING_CSX2DENSE_HPP

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
void testing_csx2dense_bad_arg(const Arguments& arg, FUNC& csx2dense)
{

    static constexpr size_t        safe_size = 100;
    static constexpr rocsparse_int M         = 10;
    static constexpr rocsparse_int N         = 10;
    static constexpr rocsparse_int LD        = M;
    rocsparse_local_handle         handle;
    rocsparse_local_mat_descr      descr;

    device_vector<T>             d_dense_val(safe_size);
    device_vector<rocsparse_int> d_csx_row_col_ptr(2);
    device_vector<rocsparse_int> d_csx_col_row_ind(2);
    device_vector<T>             d_csx_val(2);

    if(!d_dense_val || !d_csx_row_col_ptr || !d_csx_col_row_ind || !d_csx_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //
    // Testing invalid handle.
    //
    EXPECT_ROCSPARSE_STATUS(
        csx2dense(nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr, (T*)nullptr, 0),
        rocsparse_status_invalid_handle);

    //
    // Testing invalid pointers.
    //
    EXPECT_ROCSPARSE_STATUS(csx2dense(handle,
                                      M,
                                      N,
                                      nullptr,
                                      d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind,
                                      (T*)d_dense_val,
                                      LD),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(csx2dense(handle,
                                      M,
                                      N,
                                      descr,
                                      nullptr,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind,
                                      (T*)d_dense_val,
                                      LD),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        csx2dense(handle, M, N, descr, d_csx_val, nullptr, d_csx_col_row_ind, (T*)d_dense_val, LD),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        csx2dense(handle, M, N, descr, d_csx_val, d_csx_row_col_ptr, nullptr, (T*)d_dense_val, LD),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        csx2dense(
            handle, M, N, descr, d_csx_val, d_csx_row_col_ptr, d_csx_col_row_ind, (T*)nullptr, LD),
        rocsparse_status_invalid_pointer);

    //
    // Testing invalid size on M
    //
    EXPECT_ROCSPARSE_STATUS(csx2dense(handle,
                                      -1,
                                      N,
                                      descr,
                                      d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind,
                                      (T*)d_dense_val,
                                      LD),
                            rocsparse_status_invalid_size);
    //
    // Testing invalid size on N
    //
    EXPECT_ROCSPARSE_STATUS(csx2dense(handle,
                                      M,
                                      -1,
                                      descr,
                                      d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind,
                                      (T*)d_dense_val,
                                      LD),
                            rocsparse_status_invalid_size);
    //
    // Testing invalid size on LD
    //
    EXPECT_ROCSPARSE_STATUS(csx2dense(handle,
                                      M,
                                      -1,
                                      descr,
                                      d_csx_val,
                                      d_csx_row_col_ptr,
                                      d_csx_col_row_ind,
                                      (T*)d_dense_val,
                                      M - 1),
                            rocsparse_status_invalid_size);
}

template <rocsparse_direction DIRA, typename T, typename FUNC1, typename FUNC2>
void testing_csx2dense(const Arguments& arg, FUNC1& csx2dense, FUNC2& dense2csx)

{
    rocsparse_int             M      = arg.M;
    rocsparse_int             N      = arg.N;
    rocsparse_int             LD     = arg.denseld;
    rocsparse_int             DIMDIR = (rocsparse_direction_row == DIRA) ? M : N;
    rocsparse_local_handle    handle;
    rocsparse_local_mat_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, arg.baseA));

    //
    // Argument sanity check before allocating invalid memory
    //
    if(M <= 0 || N <= 0 || LD < M)
    {
        rocsparse_status expected_status = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_size;

        EXPECT_ROCSPARSE_STATUS(
            csx2dense(handle, M, N, descr, nullptr, nullptr, nullptr, (T*)nullptr, LD),
            expected_status);
        return;
    }

    //
    // Allocate memory.
    //
    host_vector<T>   h_dense_val(LD * N);
    host_vector<T>   h_dense_val_ref(LD * N);
    device_vector<T> d_dense_val(LD * N);

    host_vector<rocsparse_int>   h_nnzPerRowColumn(DIMDIR);
    device_vector<rocsparse_int> d_nnzPerRowColumn(DIMDIR);
    if(!d_nnzPerRowColumn || !d_dense_val || !h_nnzPerRowColumn || !h_dense_val)
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
                h_dense_val_ref[j * LD + i] = -1;
            }
        }
        for(rocsparse_int i = 0; i < LD; ++i)
        {
            for(rocsparse_int j = 0; j < N; ++j)
            {
                h_dense_val[j * LD + i] = -2;
            }
        }

        //
        // Random initialization of the matrix.
        //
        for(rocsparse_int i = 0; i < M; ++i)
        {
            for(rocsparse_int j = 0; j < N; ++j)
            {
                h_dense_val_ref[j * LD + i] = random_generator<T>(0, 4) == 0 ? 1 : 0;
            }
        }
    }

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(
        hipMemcpy(d_dense_val, h_dense_val_ref, sizeof(T) * LD * N, hipMemcpyHostToDevice));

    rocsparse_int nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_nnz<T>(handle, DIRA, M, N, descr, d_dense_val, LD, d_nnzPerRowColumn, &nnz));

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(h_nnzPerRowColumn,
                              d_nnzPerRowColumn,
                              sizeof(rocsparse_int) * DIMDIR,
                              hipMemcpyDeviceToHost));

    device_vector<rocsparse_int> d_csx_row_col_ptr(DIMDIR + 1);
    device_vector<T>             d_csx_val(std::max(nnz, 1));
    device_vector<rocsparse_int> d_csx_col_row_ind(std::max(nnz, 1));
    if(!d_csx_row_col_ptr || !d_csx_val || !d_csx_col_row_ind)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    host_vector<rocsparse_int> cpu_csx_row_col_ptr(DIMDIR + 1);
    host_vector<T>             cpu_csx_val(std::max(nnz, 1));
    host_vector<rocsparse_int> cpu_csx_col_row_ind(std::max(nnz, 1));
    if(!cpu_csx_row_col_ptr || !cpu_csx_val || !cpu_csx_col_row_ind)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //
    // Convert the dense matrix to a compressed sparse matrix.
    //
    CHECK_ROCSPARSE_ERROR(dense2csx(handle,
                                    M,
                                    N,
                                    descr,
                                    d_dense_val,
                                    LD,
                                    d_nnzPerRowColumn,
                                    d_csx_val,
                                    d_csx_row_col_ptr,
                                    d_csx_col_row_ind));

    //
    // Copy on host.
    //
    CHECK_HIP_ERROR(
        hipMemcpy(cpu_csx_val, d_csx_val, sizeof(T) * std::max(nnz, 1), hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(cpu_csx_row_col_ptr,
                              d_csx_row_col_ptr,
                              sizeof(rocsparse_int) * (DIMDIR + 1),
                              hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(cpu_csx_col_row_ind,
                              d_csx_col_row_ind,
                              sizeof(rocsparse_int) * std::max(nnz, 1),
                              hipMemcpyDeviceToHost));

    if(arg.unit_check)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(d_dense_val, h_dense_val, sizeof(T) * LD * N, hipMemcpyHostToDevice));

        host_csx2dense<DIRA, T>(M,
                                N,
                                rocsparse_get_mat_index_base(descr),
                                rocsparse_order_column,
                                cpu_csx_val,
                                (const rocsparse_int*)cpu_csx_row_col_ptr,
                                (const rocsparse_int*)cpu_csx_col_row_ind,
                                (T*)h_dense_val,
                                LD);

        CHECK_ROCSPARSE_ERROR(csx2dense(handle,
                                        M,
                                        N,
                                        descr,
                                        d_csx_val,
                                        d_csx_row_col_ptr,
                                        d_csx_col_row_ind,
                                        (T*)d_dense_val,
                                        LD));

        void* buffer;
        rocsparse_hipHostMalloc(&buffer, sizeof(T) * LD * N);
        CHECK_HIP_ERROR(hipMemcpy(buffer, d_dense_val, sizeof(T) * LD * N, hipMemcpyDeviceToHost));
        unit_check_general(M, N, (T*)h_dense_val, LD, (T*)buffer, LD);
        unit_check_general(M, N, (T*)h_dense_val, LD, (T*)h_dense_val_ref, LD);
        rocsparse_hipHostFree(buffer);
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

            CHECK_ROCSPARSE_ERROR(csx2dense(handle,
                                            M,
                                            N,
                                            descr,
                                            d_csx_val,
                                            d_csx_row_col_ptr,
                                            d_csx_col_row_ind,
                                            (T*)d_dense_val,
                                            LD));
        }

        double gpu_time_used = get_time_us();
        {
            //
            // Performance run
            //
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(csx2dense(handle,
                                                M,
                                                N,
                                                descr,
                                                d_csx_val,
                                                d_csx_row_col_ptr,
                                                d_csx_col_row_ind,
                                                (T*)d_dense_val,
                                                LD));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csx2dense_gbyte_count<DIRA, T>(M, N, nnz);
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
                            get_gpu_time_msec(gpu_time_used));
    }
}

#endif // TESTING_CSX2DENSE_HPP

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
#ifndef TESTING_DENSE2CSX_HPP
#define TESTING_DENSE2CSX_HPP

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
#include <chrono>
#include <thread>

template <rocsparse_direction DIRA, typename T, typename FUNC>
void testing_dense2csx_bad_arg(const Arguments& arg, FUNC& dense2csx)
{

    static constexpr size_t              safe_size = 100;
    static constexpr rocsparse_int       M         = 10;
    static constexpr rocsparse_int       N         = 10;
    static constexpr rocsparse_int       LD        = M;
    static constexpr rocsparse_direction dirA      = DIRA;
    rocsparse_local_handle               handle;

    device_vector<T>             d_A(safe_size);
    device_vector<rocsparse_int> d_nnzPerRowColumn(2);
    device_vector<rocsparse_int> d_csxRowColPtrA(2);
    device_vector<rocsparse_int> d_csxColRowIndA(2);
    device_vector<T>             d_csxValA(2);

    if(!d_A || !d_nnzPerRowColumn || !d_csxRowColPtrA || !d_csxColRowIndA || !d_csxValA)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_local_mat_descr descrA;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, rocsparse_index_base_zero));

    //
    // Testing invalid handle.
    //
    EXPECT_ROCSPARSE_STATUS(
        dense2csx(
            nullptr, 0, 0, nullptr, (const T*)nullptr, 0, nullptr, (T*)nullptr, nullptr, nullptr),
        rocsparse_status_invalid_handle);

    //
    // Testing invalid pointers.
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      nullptr,
                                      (const T*)d_A,
                                      LD,
                                      d_nnzPerRowColumn,
                                      (T*)d_csxValA,
                                      d_csxRowColPtrA,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descrA,
                                      (const T*)nullptr,
                                      LD,
                                      d_nnzPerRowColumn,
                                      (T*)d_csxValA,
                                      d_csxRowColPtrA,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descrA,
                                      (const T*)d_A,
                                      LD,
                                      nullptr,
                                      (T*)d_csxValA,
                                      d_csxRowColPtrA,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descrA,
                                      (const T*)d_A,
                                      LD,
                                      d_nnzPerRowColumn,
                                      (T*)nullptr,
                                      d_csxRowColPtrA,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descrA,
                                      (const T*)d_A,
                                      LD,
                                      d_nnzPerRowColumn,
                                      (T*)d_csxValA,
                                      nullptr,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descrA,
                                      (const T*)d_A,
                                      LD,
                                      d_nnzPerRowColumn,
                                      (T*)d_csxValA,
                                      d_csxRowColPtrA,
                                      nullptr),
                            rocsparse_status_invalid_pointer);

    //
    // Testing invalid size on M
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      -1,
                                      N,
                                      descrA,
                                      (const T*)d_A,
                                      LD,
                                      d_nnzPerRowColumn,
                                      (T*)d_csxValA,
                                      d_csxRowColPtrA,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_size);
    //
    // Testing invalid size on N
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      -1,
                                      descrA,
                                      (const T*)d_A,
                                      LD,
                                      d_nnzPerRowColumn,
                                      (T*)d_csxValA,
                                      d_csxRowColPtrA,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_size);
    //
    // Testing invalid size on LD
    //
    EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                      M,
                                      N,
                                      descrA,
                                      (const T*)d_A,
                                      M - 1,
                                      d_nnzPerRowColumn,
                                      (T*)d_csxValA,
                                      d_csxRowColPtrA,
                                      d_csxColRowIndA),
                            rocsparse_status_invalid_size);
}

template <rocsparse_direction DIRA, typename T, typename FUNC>
void testing_dense2csx(const Arguments& arg, FUNC& dense2csx)

{
    static constexpr rocsparse_direction dirA = DIRA;

    rocsparse_int        M      = arg.M;
    rocsparse_int        N      = arg.N;
    rocsparse_int        LD     = arg.denseld;
    rocsparse_index_base baseA  = arg.baseA;
    rocsparse_int        DIMDIR = (rocsparse_direction_row == DIRA) ? M : N;

    rocsparse_local_handle handle;

    rocsparse_local_mat_descr descrA;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));

    //
    // Argument sanity check before allocating invalid memory
    //
    if(M <= 0 || N <= 0 || LD < M)
    {
        rocsparse_status expected_status = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_size;

        EXPECT_ROCSPARSE_STATUS(dense2csx(handle,
                                          M,
                                          N,
                                          descrA,
                                          (const T*)nullptr,
                                          LD,
                                          nullptr,
                                          (T*)nullptr,
                                          nullptr,
                                          nullptr),
                                expected_status);
        return;
    }

    //
    // Allocate memory.
    //
    host_vector<T>   h_A(LD * N);
    device_vector<T> d_A(LD * N);

    host_vector<rocsparse_int>   h_nnzPerRowColumn(DIMDIR);
    device_vector<rocsparse_int> d_nnzPerRowColumn(DIMDIR);
    if(!d_nnzPerRowColumn || !d_A || !h_nnzPerRowColumn || !h_A)
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
                h_A[j * LD + i] = -1;
            }
        }

        //
        // Random initialization of the matrix.
        //
        for(rocsparse_int i = 0; i < M; ++i)
        {
            for(rocsparse_int j = 0; j < N; ++j)
            {
                h_A[j * LD + i] = random_generator<T>(0, 4);
            }
        }
    }

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(T) * LD * N, hipMemcpyHostToDevice));

    rocsparse_int nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_nnz(handle, dirA, M, N, descrA, (const T*)d_A, LD, d_nnzPerRowColumn, &nnz));

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(h_nnzPerRowColumn,
                              d_nnzPerRowColumn,
                              sizeof(rocsparse_int) * DIMDIR,
                              hipMemcpyDeviceToHost));

    device_vector<rocsparse_int> d_csxRowColPtrA(DIMDIR + 1);
    device_vector<T>             d_csxValA(nnz);
    device_vector<rocsparse_int> d_csxColRowIndA(nnz);
    if(!d_csxRowColPtrA || !d_csxValA || !d_csxColRowIndA)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    host_vector<rocsparse_int> cpu_csxRowColPtrA(DIMDIR + 1);
    host_vector<T>             cpu_csxValA(nnz);
    host_vector<rocsparse_int> cpu_csxColRowIndA(nnz);
    if(!cpu_csxRowColPtrA || !cpu_csxValA || !cpu_csxColRowIndA)
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
                                rocsparse_get_mat_index_base(descrA),
                                (const T*)h_A,
                                LD,
                                (const rocsparse_int*)h_nnzPerRowColumn,
                                (T*)cpu_csxValA,
                                cpu_csxRowColPtrA,
                                cpu_csxColRowIndA);

        CHECK_ROCSPARSE_ERROR(dense2csx(handle,
                                        M,
                                        N,
                                        descrA,
                                        (const T*)d_A,
                                        LD,
                                        d_nnzPerRowColumn,
                                        (T*)d_csxValA,
                                        (rocsparse_int*)d_csxRowColPtrA,
                                        (rocsparse_int*)d_csxColRowIndA));

        void* buffer
            = malloc(std::max(sizeof(T), sizeof(rocsparse_int)) * std::max(DIMDIR + 1, nnz));
        //
        // Transfer and check results.
        //
        CHECK_HIP_ERROR(hipMemcpy(buffer, d_csxValA, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        unit_check_general(1, nnz, 1, (T*)cpu_csxValA, (T*)buffer);
        CHECK_HIP_ERROR(
            hipMemcpy(buffer, d_csxColRowIndA, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        unit_check_general(1, nnz, 1, (rocsparse_int*)cpu_csxColRowIndA, (rocsparse_int*)buffer);

        CHECK_HIP_ERROR(hipMemcpy(
            buffer, d_csxRowColPtrA, sizeof(rocsparse_int) * (DIMDIR + 1), hipMemcpyDeviceToHost));

        unit_check_general(
            1, (DIMDIR + 1), 1, (rocsparse_int*)cpu_csxRowColPtrA, (rocsparse_int*)buffer);

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
                                            descrA,
                                            (const T*)d_A,
                                            LD,
                                            d_nnzPerRowColumn,
                                            (T*)d_csxValA,
                                            d_csxRowColPtrA,
                                            d_csxColRowIndA));
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
                                                descrA,
                                                (const T*)d_A,
                                                LD,
                                                d_nnzPerRowColumn,
                                                (T*)d_csxValA,
                                                d_csxRowColPtrA,
                                                d_csxColRowIndA));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = dense2csx_gbyte_count<DIRA, T>(M, N, nnz) / gpu_time_used * 1e6;

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
}

#endif // TESTING_DENSE2CSX_HPP

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
#ifndef TESTING_NNZ_HPP
#define TESTING_NNZ_HPP

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
void testing_nnz_bad_arg(const Arguments& arg)
{

    static constexpr size_t        safe_size = 100;
    static constexpr rocsparse_int M         = 10;
    static constexpr rocsparse_int N         = 10;
    static constexpr rocsparse_int LD        = M;

    rocsparse_direction    dirA = rocsparse_direction_row;
    rocsparse_local_handle handle;

    device_vector<T> d_A(safe_size);

    device_vector<rocsparse_int> d_nnzPerRowColumn(safe_size), d_nnzTotalDevHostPtr(safe_size);

    if(!d_A || !d_nnzPerRowColumn || !d_nnzTotalDevHostPtr)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_mat_descr descrA = nullptr;
    rocsparse_create_mat_descr(&descrA);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    //
    // Testing invalid handle.
    //
    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(nullptr,
                                          dirA,
                                          M,
                                          N,
                                          descrA,
                                          (const T*)d_A,
                                          LD,
                                          d_nnzPerRowColumn,
                                          d_nnzTotalDevHostPtr),
                            rocsparse_status_invalid_handle);

    //
    // Testing invalid pointers.
    //
    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                          dirA,
                                          M,
                                          N,
                                          nullptr,
                                          (const T*)d_A,
                                          LD,
                                          d_nnzPerRowColumn,
                                          d_nnzTotalDevHostPtr),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                          dirA,
                                          M,
                                          N,
                                          nullptr,
                                          (const T*)d_A,
                                          LD,
                                          d_nnzPerRowColumn,
                                          d_nnzTotalDevHostPtr),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                          dirA,
                                          M,
                                          N,
                                          descrA,
                                          (const T*)nullptr,
                                          LD,
                                          d_nnzPerRowColumn,
                                          d_nnzTotalDevHostPtr),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_nnz(handle, dirA, M, N, descrA, (const T*)d_A, LD, nullptr, d_nnzTotalDevHostPtr),
        rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_nnz(handle, dirA, M, N, descrA, (const T*)d_A, LD, d_nnzPerRowColumn, nullptr),
        rocsparse_status_invalid_pointer);

    //
    // Testing invalid direction
    //
    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                          (rocsparse_direction)77,
                                          -1,
                                          -1,
                                          descrA,
                                          (const T*)nullptr,
                                          -1,
                                          nullptr,
                                          nullptr),
                            rocsparse_status_invalid_value);

    //
    // Testing invalid size on M
    //
    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                          dirA,
                                          -1,
                                          N,
                                          descrA,
                                          (const T*)d_A,
                                          LD,
                                          d_nnzPerRowColumn,
                                          d_nnzTotalDevHostPtr),
                            rocsparse_status_invalid_size);

    //
    // Testing invalid size on N
    //
    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                          dirA,
                                          M,
                                          -1,
                                          descrA,
                                          (const T*)d_A,
                                          LD,
                                          d_nnzPerRowColumn,
                                          d_nnzTotalDevHostPtr),
                            rocsparse_status_invalid_size);

    //
    // Testing invalid size on LD
    //
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_nnz(
            handle, dirA, M, N, descrA, (const T*)d_A, 0, d_nnzPerRowColumn, d_nnzTotalDevHostPtr),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                          dirA,
                                          M,
                                          N,
                                          descrA,
                                          (const T*)d_A,
                                          M - 1,
                                          d_nnzPerRowColumn,
                                          d_nnzTotalDevHostPtr),
                            rocsparse_status_invalid_size);

    rocsparse_destroy_mat_descr(descrA);
}

template <typename T>
void testing_nnz(const Arguments& arg)
{
    rocsparse_int       M    = arg.M;
    rocsparse_int       N    = arg.N;
    rocsparse_direction dirA = arg.direction;
    rocsparse_int       LD   = arg.denseld;

    rocsparse_local_handle handle;
    rocsparse_mat_descr    descrA;
    rocsparse_create_mat_descr(&descrA);

    //
    // Argument sanity check before allocating invalid memory
    //
    if(M <= 0 || N <= 0 || LD < M)
    {
        rocsparse_status expected_status = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_size;

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_nnz(handle, dirA, M, N, descrA, (const T*)nullptr, LD, nullptr, nullptr),
            expected_status);

        if(rocsparse_status_success == expected_status)
        {
            rocsparse_int h_nnz = 77;
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_nnz(handle, dirA, M, N, descrA, (const T*)nullptr, LD, nullptr, nullptr),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_nnz(handle, dirA, M, N, descrA, (const T*)nullptr, LD, nullptr, &h_nnz),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(0 == h_nnz ? rocsparse_status_success
                                               : rocsparse_status_internal_error,
                                    rocsparse_status_success);

            h_nnz = 139;
            device_vector<rocsparse_int> d_nnz(1);
            CHECK_HIP_ERROR(hipMemcpy(
                (rocsparse_int*)d_nnz, &h_nnz, sizeof(rocsparse_int) * 1, hipMemcpyHostToDevice));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_nnz(handle, dirA, M, N, descrA, (const T*)nullptr, LD, nullptr, nullptr),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(rocsparse_nnz(handle,
                                                  dirA,
                                                  M,
                                                  N,
                                                  descrA,
                                                  (const T*)nullptr,
                                                  LD,
                                                  nullptr,
                                                  (rocsparse_int*)d_nnz),
                                    rocsparse_status_success);

            CHECK_HIP_ERROR(hipMemcpy(
                &h_nnz, (rocsparse_int*)d_nnz, sizeof(rocsparse_int) * 1, hipMemcpyDeviceToHost));

            EXPECT_ROCSPARSE_STATUS(0 == h_nnz ? rocsparse_status_success
                                               : rocsparse_status_internal_error,
                                    rocsparse_status_success);
        }

        return;
    }

    //
    // Create the dense matrix.
    //
    rocsparse_int MN = (dirA == rocsparse_direction_row) ? M : N;

    host_vector<T>             h_A(LD * N);
    host_vector<rocsparse_int> h_nnzPerRowColumn(MN);
    host_vector<rocsparse_int> hd_nnzPerRowColumn(MN);
    host_vector<rocsparse_int> h_nnzTotalDevHostPtr(1);
    host_vector<rocsparse_int> hd_nnzTotalDevHostPtr(1);

    // Allocate device memory
    device_vector<T>             d_A(LD * N);
    device_vector<rocsparse_int> d_nnzPerRowColumn(MN);
    device_vector<rocsparse_int> d_nnzTotalDevHostPtr(1);
    if(!h_nnzPerRowColumn || !d_nnzPerRowColumn || !d_A)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //
    // Initialize a random matrix.
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

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(T) * LD * N, hipMemcpyHostToDevice));

    //
    // Unit check.
    //
    if(arg.unit_check)
    {
        //
        // Compute the reference host first.
        //
        host_nnz<T>(dirA, M, N, descrA, (const T*)h_A, LD, h_nnzPerRowColumn, h_nnzTotalDevHostPtr);

        //
        // Pointer mode device for nnz and call.
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz(handle,
                                            dirA,
                                            M,
                                            N,
                                            descrA,
                                            (const T*)d_A,
                                            LD,
                                            d_nnzPerRowColumn,
                                            d_nnzTotalDevHostPtr));

        //
        // Transfer.
        //
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzPerRowColumn,
                                  d_nnzPerRowColumn,
                                  sizeof(rocsparse_int) * MN,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzTotalDevHostPtr,
                                  d_nnzTotalDevHostPtr,
                                  sizeof(rocsparse_int) * 1,
                                  hipMemcpyDeviceToHost));

        //
        // Check results.
        //
        unit_check_general<rocsparse_int>(1, MN, 1, hd_nnzPerRowColumn, h_nnzPerRowColumn);
        unit_check_general<rocsparse_int>(1, 1, 1, hd_nnzTotalDevHostPtr, h_nnzTotalDevHostPtr);

        //
        // Pointer mode host for nnz and call.
        //
        rocsparse_int dh_nnz;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz(
            handle, dirA, M, N, descrA, (const T*)d_A, LD, d_nnzPerRowColumn, &dh_nnz));

        //
        // Transfer.
        //
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzPerRowColumn,
                                  d_nnzPerRowColumn,
                                  sizeof(rocsparse_int) * MN,
                                  hipMemcpyDeviceToHost));

        //
        // Check results.
        //
        unit_check_general<rocsparse_int>(1, MN, 1, hd_nnzPerRowColumn, h_nnzPerRowColumn);
        unit_check_general<rocsparse_int>(1, 1, 1, &dh_nnz, h_nnzTotalDevHostPtr);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        //
        // Warm-up
        //
        rocsparse_int h_nnz;
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_nnz(
                handle, dirA, M, N, descrA, (const T*)d_A, LD, d_nnzPerRowColumn, &h_nnz));
        }

        double gpu_time_used = get_time_us();
        {
            //
            // Performance run
            //
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_nnz(
                    handle, dirA, M, N, descrA, (const T*)d_A, LD, d_nnzPerRowColumn, &h_nnz));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double
	  gpu_gbyte = nnz_gbyte_count<T>(M, N, dirA) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);
        // clang-format off
        std::cout
            << std::setw(20) << "M" 
            << std::setw(20) << "N" 
            << std::setw(20) << "LD"
	    << std::setw(20) << "nnz"
	    << std::setw(20) << "dir"
            << std::setw(20) << "GB/s"
	    << std::setw(20) << "msec"
	    << std::setw(20) << "iter"
            << std::setw(20) << "verified"
	    << std::endl;

        std::cout
            << std::setw(20) << M 
            << std::setw(20) << N 
            << std::setw(20) << LD
	    << std::setw(20) << h_nnz
	    << std::setw(20) << rocsparse_direction2string(dirA)
	    << std::setw(20) << gpu_gbyte
	    << std::setw(20) << gpu_time_used / 1e3
	    << std::setw(20) << number_hot_calls
	    << std::setw(20) << (arg.unit_check ? "yes" : "no")
	    << std::endl;
        // clang-format on
    }

    //
    // Destroy the matrix descriptor.
    //
    rocsparse_destroy_mat_descr(descrA);
}

#endif // TESTING_NNZ_HPP

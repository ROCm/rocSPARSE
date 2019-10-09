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
#ifndef TESTING_GTHR_HPP
#define TESTING_GTHR_HPP

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
void testing_gthr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<rocsparse_int> dx_ind(safe_size);
    device_vector<T>             dx_val(safe_size);
    device_vector<T>             dy(safe_size);

    if(!dx_ind || !dx_val || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_gthr()
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_gthr<T>(nullptr, safe_size, dy, dx_val, dx_ind, rocsparse_index_base_zero),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_gthr<T>(handle, safe_size, nullptr, dx_val, dx_ind, rocsparse_index_base_zero),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_gthr<T>(handle, safe_size, dy, nullptr, dx_ind, rocsparse_index_base_zero),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_gthr<T>(handle, safe_size, dy, dx_val, nullptr, rocsparse_index_base_zero),
        rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_gthr(const Arguments& arg)
{
    rocsparse_int        M    = arg.M;
    rocsparse_int        nnz  = arg.nnz;
    rocsparse_index_base base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dx_ind(safe_size);
        device_vector<T>             dx_val(safe_size);
        device_vector<T>             dy(safe_size);

        if(!dx_ind || !dx_val || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(rocsparse_gthr<T>(handle, nnz, dy, dx_val, dx_ind, base),
                                nnz < 0 ? rocsparse_status_invalid_size : rocsparse_status_success);

        return;
    }

    // Allocate host memory
    host_vector<rocsparse_int> hx_ind(nnz);
    host_vector<T>             hx_val_1(nnz);
    host_vector<T>             hx_val_2(nnz);
    host_vector<T>             hx_val_gold(nnz);
    host_vector<T>             hy(M);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, 1, M);
    rocsparse_init<T>(hy, 1, M, 1);

    // Allocate device memory
    device_vector<rocsparse_int> dx_ind(nnz);
    device_vector<T>             dx_val_1(nnz);
    device_vector<T>             dx_val_2(nnz);
    device_vector<T>             dy(M);

    if(!dx_ind || !dx_val_1 || !dx_val_2 || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * M, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_gthr<T>(handle, nnz, dy, dx_val_1, dx_ind, base));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_gthr<T>(handle, nnz, dy, dx_val_2, dx_ind, base));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hx_val_1, dx_val_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hx_val_2, dx_val_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // CPU gthr
        host_gthr<T>(nnz, hy, hx_val_gold, hx_ind, base);

        unit_check_general<T>(1, nnz, 1, hx_val_gold, hx_val_1);
        unit_check_general<T>(1, nnz, 1, hx_val_gold, hx_val_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gthr<T>(handle, nnz, dy, dx_val_1, dx_ind, base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gthr<T>(handle, nnz, dy, dx_val_1, dx_ind, base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = gthr_gbyte_count<T>(nnz) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "nnz" << std::setw(12) << "GB/s" << std::setw(12) << "usec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << nnz << std::setw(12) << gpu_gbyte << std::setw(12)
                  << gpu_time_used << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#endif // TESTING_GTHR_HPP

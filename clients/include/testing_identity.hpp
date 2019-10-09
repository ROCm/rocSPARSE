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
#ifndef TESTING_IDENTITY_HPP
#define TESTING_IDENTITY_HPP

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
void testing_identity_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<rocsparse_int> dp(safe_size);

    if(!dp)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_create_identity_permutation()
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_identity_permutation(nullptr, safe_size, dp),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_identity_permutation(handle, safe_size, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_identity(const Arguments& arg)
{
    rocsparse_int N = arg.N;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dp(safe_size);

        if(!dp)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_create_identity_permutation(handle, N, dp),
                                (N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        return;
    }

    // Allocate host memory
    host_vector<rocsparse_int> hp(N);
    host_vector<rocsparse_int> hp_gold(N);

    // Allocate device memory
    device_vector<rocsparse_int> dp(N);

    if(!dp)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, N, dp));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hp, dp, sizeof(rocsparse_int) * N, hipMemcpyDeviceToHost));

        // CPU identity
        for(rocsparse_int i = 0; i < N; ++i)
        {
            hp_gold[i] = i;
        }

        unit_check_general<rocsparse_int>(1, N, 1, hp_gold, hp);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, N, dp));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, N, dp));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = identity_gbyte_count<T>(N) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "N" << std::setw(12) << "GB/s" << std::setw(12) << "usec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;
        std::cout << std::setw(12) << N << std::setw(12) << gpu_gbyte << std::setw(12)
                  << gpu_time_used << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#endif // TESTING_IDENTITY_HPP

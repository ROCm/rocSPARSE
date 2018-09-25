/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <rocsparse.h>
#include <algorithm>

using namespace rocsparse;
using namespace rocsparse_test;

void testing_identity_bad_arg(void)
{
    rocsparse_int n         = 100;
    rocsparse_int safe_size = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    auto p_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

    rocsparse_int* p = (rocsparse_int*)p_managed.get();

    if(!p)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for (p == nullptr)
    {
        rocsparse_int* p_null = nullptr;

        status = rocsparse_create_identity_permutation(handle, n, p_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: p is nullptr");
    }

    // Testing for(handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_create_identity_permutation(handle_null, n, p);
        verify_rocsparse_status_invalid_handle(status);
    }
}

rocsparse_status testing_identity(Arguments argus)
{
    rocsparse_int n         = argus.N;
    rocsparse_int safe_size = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(n <= 0)
    {
        auto p_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

        rocsparse_int* p = (rocsparse_int*)p_managed.get();

        if(!p)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error, "!p");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_create_identity_permutation(handle, n, p);

        if(n < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: n < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "n >= 0");
        }

        return rocsparse_status_success;
    }

    // Host structures
    std::vector<rocsparse_int> hp(n);
    std::vector<rocsparse_int> hp_gold(n);

    // create_identity_permutation on host
    for(rocsparse_int i = 0; i < n; ++i)
    {
        hp_gold[i] = i;
    }

    // Allocate memory on the device
    auto dp_managed = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * n), device_free};

    rocsparse_int* dp = (rocsparse_int*)dp_managed.get();

    if(!dp)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!p");
        return rocsparse_status_memory_error;
    }

    if(argus.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, n, dp));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hp.data(), dp, sizeof(rocsparse_int) * n, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, n, 1, hp_gold.data(), hp.data());
    }

    if(argus.timing)
    {
        rocsparse_int number_cold_calls = 2;
        rocsparse_int number_hot_calls  = argus.iters;

        for(rocsparse_int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_create_identity_permutation(handle, n, dp);
        }

        double gpu_time_used = get_time_us();

        for(rocsparse_int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_create_identity_permutation(handle, n, dp);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        double bandwidth = sizeof(rocsparse_int) * n / gpu_time_used / 1e6;

        printf("n\t\tGB/s\tmsec\n");
        printf("%8d\t%0.2lf\t%0.2lf\n", n, bandwidth, gpu_time_used);
    }
    return rocsparse_status_success;
}

#endif // TESTING_IDENTITY_HPP

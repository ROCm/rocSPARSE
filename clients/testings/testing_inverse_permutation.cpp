/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

template <typename T>
void testing_inverse_permutation_bad_arg(const Arguments& arg)
{
    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    rocsparse_int*       p    = (rocsparse_int*)0x4;
    rocsparse_int*       q    = (rocsparse_int*)0x4;
    rocsparse_int        n    = 1;
    rocsparse_index_base base = rocsparse_index_base_zero;

    EXPECT_ROCSPARSE_STATUS(rocsparse_inverse_permutation(nullptr, n, p, q, base),
                            rocsparse_status_invalid_handle);
    n = -1;
    EXPECT_ROCSPARSE_STATUS(rocsparse_inverse_permutation(handle, n, p, q, base),
                            rocsparse_status_invalid_size);
    n = 1;
    EXPECT_ROCSPARSE_STATUS(rocsparse_inverse_permutation(handle, n, p, nullptr, base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_inverse_permutation(handle, n, nullptr, q, base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_inverse_permutation(handle, n, p, q, (rocsparse_index_base)-1),
        rocsparse_status_invalid_value);
}

template <typename T>
void testing_inverse_permutation(const Arguments& arg)
{
    rocsparse_int        N    = arg.N;
    rocsparse_index_base base = arg.baseA;
    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(N == 0)
    {
        EXPECT_ROCSPARSE_STATUS(rocsparse_inverse_permutation(handle, N, nullptr, nullptr, base),
                                rocsparse_status_success);
        return;
    }

    // Allocate host memory
    host_dense_vector<rocsparse_int> hp(N);

    for(rocsparse_int i = 0; i < N; ++i)
    {
        hp[i] = i + base;
    }

    for(rocsparse_int k = 0; k < std::min(20, N / 2); ++k)
    {
        rocsparse_int i = random_generator_exact<rocsparse_int>(0, N - 1);
        rocsparse_int j = random_generator_exact<rocsparse_int>(0, N - 1);
        if(i != j)
        {
            std::swap(hp[i], hp[j]);
        }
    }
    device_dense_vector<rocsparse_int> dp(hp);
    device_dense_vector<rocsparse_int> dq(N);

    if(arg.unit_check)
    {
        host_dense_vector<rocsparse_int> hq(N);

        for(rocsparse_int i = 0; i < N; ++i)
        {
            hq[hp[i] - base] = i + base;
        }
        CHECK_ROCSPARSE_ERROR(rocsparse_inverse_permutation(handle, N, dp, dq, base));
        hq.unit_check(dq);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_inverse_permutation(handle, N, dp, dq, base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_inverse_permutation(handle, N, dp, dq, base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = inverse_permutation_gbyte_count<T>(N);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info("N",
                            N,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                          \
    template void testing_inverse_permutation_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_inverse_permutation<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_inverse_permutation_extra(const Arguments& arg) {}

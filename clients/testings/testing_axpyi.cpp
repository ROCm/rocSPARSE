/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_axpyi_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle = local_handle;
    rocsparse_int        nnz    = safe_size;
    const T*             alpha  = &h_alpha;
    const T*             x_val  = static_cast<const T*>((void*)0x4);
    const rocsparse_int* x_ind  = (const rocsparse_int*)0x4;
    T*                   y      = static_cast<T*>((void*)0x4);
    rocsparse_index_base base   = rocsparse_index_base_zero;

#define PARAMS handle, nnz, alpha, x_val, x_ind, y, base
    auto_testing_bad_arg(rocsparse_axpyi<T>, PARAMS);
#undef PARAMS
}

template <typename T>
void testing_axpyi(const Arguments& arg)
{
    rocsparse_int        M    = arg.M;
    rocsparse_int        nnz  = arg.nnz;
    rocsparse_index_base base = arg.baseA;

    T h_alpha = arg.get_alpha<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

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
        EXPECT_ROCSPARSE_STATUS(rocsparse_axpyi<T>(handle, nnz, &h_alpha, dx_val, dx_ind, dy, base),
                                nnz < 0 ? rocsparse_status_invalid_size : rocsparse_status_success);

        return;
    }

    // Allocate host memory
    host_vector<rocsparse_int> hx_ind(nnz);
    host_vector<T>             hx_val(nnz);
    host_vector<T>             hy_1(M);
    host_vector<T>             hy_2(M);
    host_vector<T>             hy_gold(M);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, 1, M);
    rocsparse_init<T>(hx_val, 1, nnz, 1);
    rocsparse_init<T>(hy_1, 1, M, 1);
    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<rocsparse_int> dx_ind(nnz);
    device_vector<T>             dx_val(nnz);
    device_vector<T>             dy_1(M);
    device_vector<T>             dy_2(M);
    device_vector<T>             d_alpha(1);

    if(!dx_ind || !dx_val || !dy_1 || !dy_2 || !d_alpha)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_axpyi<T>(handle, nnz, &h_alpha, dx_val, dx_ind, dy_1, base));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_axpyi<T>(handle, nnz, d_alpha, dx_val, dx_ind, dy_2, base));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU axpyi
        host_axpby<rocsparse_int, T>(M, nnz, h_alpha, hx_val, hx_ind, 1.0, hy_gold, base);

        unit_check_segments<T>(M, hy_gold, hy_1);
        unit_check_segments<T>(M, hy_gold, hy_2);
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
                rocsparse_axpyi<T>(handle, nnz, &h_alpha, dx_val, dx_ind, dy_1, base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_axpyi<T>(handle, nnz, &h_alpha, dx_val, dx_ind, dy_1, base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = axpyi_gflop_count(nnz);
        double gbyte_count = axpby_gbyte_count<T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info("size",
                            M,
                            "nnz",
                            nnz,
                            "alpha",
                            h_alpha,
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_axpyi_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_axpyi<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_axpyi_extra(const Arguments& arg) {}

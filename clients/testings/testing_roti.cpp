/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_roti_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle   = local_handle;
    rocsparse_int          nnz      = 2;
    T*                     x_val    = (T*)0x4;
    const rocsparse_int*   x_ind    = (const rocsparse_int*)0x4;
    T*                     y        = (T*)0x4;
    const T*               c        = (const T*)0x4;
    const T*               s        = (const T*)0x4;
    rocsparse_index_base   idx_base = rocsparse_index_base_zero;
    bad_arg_analysis(rocsparse_roti<T>, handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

template <typename T>
void testing_roti(const Arguments& arg)
{
    rocsparse_int        M    = arg.M;
    rocsparse_int        nnz  = arg.nnz;
    rocsparse_index_base base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory
    host_vector<rocsparse_int> hx_ind(nnz);
    host_vector<T>             hx_val_1(nnz);
    host_vector<T>             hx_val_2(nnz);
    host_vector<T>             hx_val_gold(nnz);
    host_vector<T>             hy_1(M);
    host_vector<T>             hy_2(M);
    host_vector<T>             hy_gold(M);
    host_vector<T>             hc(1);
    host_vector<T>             hs(1);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, M + base);
    rocsparse_init<T>(hx_val_1, 1, nnz, 1);
    rocsparse_init<T>(hy_1, 1, M, 1);
    rocsparse_init<T>(hc, 1, 1, 1);
    rocsparse_init<T>(hs, 1, 1, 1);
    hx_val_2    = hx_val_1;
    hx_val_gold = hx_val_1;
    hy_2        = hy_1;
    hy_gold     = hy_1;

    // Allocate device memory
    device_vector<rocsparse_int> dx_ind(nnz);
    device_vector<T>             dx_val_1(nnz);
    device_vector<T>             dx_val_2(nnz);
    device_vector<T>             dy_1(M);
    device_vector<T>             dy_2(M);
    device_vector<T>             dc(1);
    device_vector<T>             ds(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val_1, hx_val_1, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dx_val_2, hx_val_2, sizeof(T) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_roti<T>(handle, nnz, dx_val_1, dx_ind, dy_1, &hc[0], &hs[0], base));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_roti<T>(handle, nnz, dx_val_2, dx_ind, dy_2, dc, ds, base));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hx_val_1, dx_val_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hx_val_2, dx_val_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU roti
        host_roti<rocsparse_int, T>(nnz, hx_val_gold, hx_ind, hy_gold, hc, hs, base);

        hx_val_gold.unit_check(hx_val_1);
        hx_val_gold.unit_check(hx_val_2);
        hy_gold.unit_check(hy_1);
        hy_gold.unit_check(hy_2);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("X pointer mode host",
                                            hx_val_1,
                                            "Y pointer mode host",
                                            hy_1,
                                            "X pointer mode device",
                                            hx_val_2,
                                            "Y pointer mode device",
                                            hy_2);
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
            CHECK_ROCSPARSE_ERROR(
                rocsparse_roti<T>(handle, nnz, dx_val_1, dx_ind, dy_1, &hc[0], &hs[0], base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_roti<T>(handle, nnz, dx_val_1, dx_ind, dy_1, &hc[0], &hs[0], base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = roti_gflop_count<rocsparse_int>(nnz);
        double gbyte_count = roti_gbyte_count<T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info(display_key_t::nnz,
                            nnz,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                           \
    template void testing_roti_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_roti<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_roti_extra(const Arguments& arg) {}

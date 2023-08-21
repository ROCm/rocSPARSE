/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
template <typename I, typename T>
void testing_axpby_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle      local_handle;
    rocsparse_handle            handle = local_handle;
    rocsparse_const_spvec_descr x      = (rocsparse_const_spvec_descr)0x4;
    rocsparse_dnvec_descr       y      = (rocsparse_dnvec_descr)0x4;
    const void*                 alpha  = (const void*)0x4;
    const void*                 beta   = (const void*)0x4;
    bad_arg_analysis(rocsparse_axpby, handle, alpha, x, beta, y);
}

template <typename I, typename T>
void testing_axpby(const Arguments& arg)
{
    I size = arg.M;
    I nnz  = arg.nnz;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocsparse_index_base base = arg.baseA;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Argument sanity check before allocating invalid memory
    if(size <= 0 || nnz <= 0)
    {
        // Allocate memory on device
        device_vector<T> dy(size > 0 ? size : 100);

        if(!dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Check structures
        rocsparse_local_spvec x(size, nnz, nullptr, nullptr, itype, base, ttype);
        rocsparse_local_dnvec y(size, dy, ttype);

        // Check Scatter when structures were created
        if(size >= 0 && nnz >= 0)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_axpby(handle, &h_alpha, x, &h_beta, y),
                                    rocsparse_status_success);
        }

        return;
    }

    // Allocate host memory for matrix
    host_vector<I> hx_ind(nnz);
    host_vector<T> hx_val(nnz);
    host_vector<T> hy_1(size);
    host_vector<T> hy_2(size);
    host_vector<T> hy_gold(size);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, size + base);
    rocsparse_init<T>(hx_val, 1, nnz, 1);
    rocsparse_init<T>(hy_1, 1, size, 1);
    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<I> dx_ind(nnz);
    device_vector<T> dx_val(nnz);
    device_vector<T> dy_1(size);
    device_vector<T> dy_2(size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    if(!dx_ind || !dx_val || !dy_1 || !dy_2 || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, dy_1, sizeof(T) * size, hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, itype, base, ttype);
    rocsparse_local_dnvec y1(size, dy_1, ttype);
    rocsparse_local_dnvec y2(size, dy_2, ttype);

    if(arg.unit_check)
    {
        // axpby - host pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_axpby(handle, &h_alpha, x, &h_beta, y1));

        // axpby - device pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_axpby(handle, d_alpha, x, d_beta, y2));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * size, hipMemcpyDeviceToHost));

        // CPU axpby
        host_axpby<I, T>(size, nnz, h_alpha, hx_val, hx_ind, h_beta, hy_gold, base);

        hy_gold.unit_check(hy_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_axpby(handle, &h_alpha, x, &h_beta, y1));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_axpby(handle, &h_alpha, x, &h_beta, y1));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = axpby_gflop_count(nnz);
        double gbyte_count = axpby_gbyte_count<T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info("size",
                            size,
                            "nnz",
                            nnz,
                            "alpha",
                            h_alpha,
                            "beta",
                            h_beta,
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, TTYPE)                                            \
    template void testing_axpby_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_axpby<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_axpby_extra(const Arguments& arg) {}

/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_scatter_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle      local_handle;
    rocsparse_handle            handle = local_handle;
    rocsparse_const_spvec_descr x      = (rocsparse_const_spvec_descr)0x4;
    rocsparse_dnvec_descr       y      = (rocsparse_dnvec_descr)0x4;
    bad_arg_analysis(rocsparse_scatter, handle, x, y);
}

template <typename I, typename T>
void testing_scatter(const Arguments& arg)
{
    I size = arg.M;
    I nnz  = arg.nnz;

    rocsparse_index_base base = arg.baseA;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory for matrix
    host_vector<I> hx_ind(nnz);
    host_vector<T> hx_val(nnz);
    host_vector<T> hy(size);
    host_vector<T> hy_gold(size);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, size + base);
    rocsparse_init<T>(hx_val, 1, nnz, 1);
    rocsparse_init<T>(hy, 1, size, 1);
    hy_gold = hy;

    // Allocate device memory
    device_vector<I> dx_ind(nnz);
    device_vector<T> dx_val(nnz);
    device_vector<T> dy(size);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size, hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, itype, base, ttype);
    rocsparse_local_dnvec y(size, dy, ttype);

    if(arg.unit_check)
    {
        // Scatter
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_scatter(handle, x, y));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy, dy, sizeof(T) * size, hipMemcpyDeviceToHost));

        // CPU scatter
        host_sctr<I, T>(nnz, hx_val, hx_ind, hy_gold, base);

        hy_gold.unit_check(hy);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y", hy);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_scatter(handle, x, y));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_scatter(handle, x, y));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = sctr_gbyte_count<T>(nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info(display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, TTYPE)                                              \
    template void testing_scatter_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_scatter<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int8_t);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int8_t);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

void testing_scatter_extra(const Arguments& arg) {}

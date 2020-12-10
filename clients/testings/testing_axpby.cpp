/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
    I size = 100;
    I nnz  = 100;

    T alpha = (T)2;
    T beta  = (T)3;

    rocsparse_index_base base = rocsparse_index_base_zero;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<I> dx_ind(nnz);
    device_vector<T> dx_val(nnz);
    device_vector<T> dy(size);

    if(!dx_ind || !dx_val || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Structures
    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, itype, base, ttype);
    rocsparse_local_dnvec y(size, dy, ttype);

    EXPECT_ROCSPARSE_STATUS(rocsparse_axpby(nullptr, &alpha, x, &beta, y),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_axpby(handle, nullptr, x, &beta, y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_axpby(handle, &alpha, nullptr, &beta, y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_axpby(handle, &alpha, x, nullptr, y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_axpby(handle, &alpha, x, &beta, nullptr),
                            rocsparse_status_invalid_pointer);
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
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(size <= 0 || nnz <= 0)
    {
        // Allocate memory on device
        device_vector<T> dy(100);

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
    rocsparse_init_index(hx_ind, nnz, 1, size);
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
        CHECK_ROCSPARSE_ERROR(rocsparse_axpby(handle, &h_alpha, x, &h_beta, y1));

        // axpby - device pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_axpby(handle, d_alpha, x, d_beta, y2));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * size, hipMemcpyDeviceToHost));

        // CPU axpby
        host_axpby<I, T>(nnz, h_alpha, hx_val, hx_ind, h_beta, hy_gold, base);

        unit_check_general<T>(1, size, 1, hy_gold, hy_1);
        unit_check_general<T>(1, size, 1, hy_gold, hy_2);
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

        double gpu_gflops = axpby_gflop_count(nnz) / gpu_time_used * 1e6;
        double gpu_gbyte  = axpby_gbyte_count<I, T>(nnz) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "size" << std::setw(12) << "nnz" << std::setw(16) << "alpha"
                  << std::setw(12) << "beta" << std::setw(12) << "GFlop/s" << std::setw(12)
                  << "GB/s" << std::setw(12) << "usec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << size << std::setw(12) << nnz << std::setw(16) << h_alpha
                  << std::setw(12) << h_beta << std::setw(12) << gpu_gflops << std::setw(12)
                  << gpu_gbyte << std::setw(12) << gpu_time_used << std::setw(12)
                  << number_hot_calls << std::setw(12) << (arg.unit_check ? "yes" : "no")
                  << std::endl;
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

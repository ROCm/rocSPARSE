/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
void testing_spvv_bad_arg(const Arguments& arg)
{
    I size = 100;
    I nnz  = 100;

    T result;

    rocsparse_operation  trans = rocsparse_operation_none;
    rocsparse_index_base base  = rocsparse_index_base_zero;

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

    // Test SpVV with invalid buffer
    size_t buffer_size;

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(nullptr, trans, x, y, &result, ttype, &buffer_size, nullptr),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(handle, trans, nullptr, y, &result, ttype, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(handle, trans, x, nullptr, &result, ttype, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(handle, trans, x, y, nullptr, ttype, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spvv(handle, trans, x, y, &result, ttype, nullptr, nullptr),
                            rocsparse_status_invalid_pointer);

    // Test SpVV with valid buffer
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, 100));

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(nullptr, trans, x, y, &result, ttype, &buffer_size, dbuffer),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(handle, trans, nullptr, y, &result, ttype, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(handle, trans, x, nullptr, &result, ttype, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(handle, trans, x, y, nullptr, ttype, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);

    CHECK_HIP_ERROR(hipFree(dbuffer));
}

template <typename I, typename T>
void testing_spvv(const Arguments& arg)
{
    I size = arg.M;
    I nnz  = arg.nnz;

    size_t buffer_size;
    void*  temp_buffer;

    rocsparse_operation  trans = arg.transA;
    rocsparse_index_base base  = arg.baseA;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
        T h_result;

        // Allocate memory on device
        device_vector<T> dy(100);

        if(!dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Valid descriptors can only be created for nnz >= 0
        if(nnz < 0)
        {
            return;
        }

        rocsparse_local_spvec x(size, nnz, nullptr, nullptr, itype, base, ttype);
        rocsparse_local_dnvec y(size, dy, ttype);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Obtain buffer size
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spvv(handle, trans, x, y, &h_result, ttype, &buffer_size, nullptr),
            rocsparse_status_success);

        CHECK_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

        // SpVV
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spvv(handle, trans, x, y, &h_result, ttype, &buffer_size, temp_buffer),
            rocsparse_status_success);

        CHECK_HIP_ERROR(hipFree(temp_buffer));

        return;
    }

    // Allocate host memory
    host_vector<I> hx_ind(nnz);
    host_vector<T> hx_val(nnz);
    host_vector<T> hy(size);
    host_vector<T> hdot_1(1);
    host_vector<T> hdot_2(1);
    host_vector<T> hdot_gold(1);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, 1, size);
    rocsparse_init_alternating_sign<T>(hx_val, 1, nnz, 1);
    rocsparse_init_exact<T>(hy, 1, size, 1);

    // Allocate device memory
    device_vector<I> dx_ind(nnz);
    device_vector<T> dx_val(nnz);
    device_vector<T> dy(size);
    device_vector<T> ddot_2(1);

    if(!dx_ind || !dx_val || !dy || !ddot_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size, hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, itype, base, ttype);
    rocsparse_local_dnvec y(size, dy, ttype);

    // Obtain buffer size
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spvv(handle, trans, x, y, &hdot_1[0], ttype, &buffer_size, nullptr));
    CHECK_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spvv(handle, trans, x, y, &hdot_1[0], ttype, &buffer_size, temp_buffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_spvv(handle, trans, x, y, ddot_2, ttype, &buffer_size, temp_buffer));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hdot_2, ddot_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU SpVV
        if(trans == rocsparse_operation_none)
        {
            host_doti<I, T>(nnz, hx_val, hx_ind, hy, hdot_gold, base);
        }
        else
        {
            host_dotci<I, T>(nnz, hx_val, hx_ind, hy, hdot_gold, base);
        }

        unit_check_general<T>(1, 1, 1, hdot_gold, hdot_1);
        unit_check_general<T>(1, 1, 1, hdot_gold, hdot_2);
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
                rocsparse_spvv(handle, trans, x, y, &hdot_1[0], ttype, &buffer_size, temp_buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spvv(handle, trans, x, y, &hdot_1[0], ttype, &buffer_size, temp_buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops = doti_gflop_count(nnz) / gpu_time_used * 1e6;
        double gpu_gbyte  = doti_gbyte_count<T>(nnz) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "nnz" << std::setw(12) << "GFlop/s" << std::setw(12) << "GB/s"
                  << std::setw(12) << "usec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << nnz << std::setw(12) << gpu_gflops << std::setw(12)
                  << gpu_gbyte << std::setw(12) << gpu_time_used << std::setw(12)
                  << number_hot_calls << std::setw(12) << (arg.unit_check ? "yes" : "no")
                  << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(temp_buffer));
}

#define INSTANTIATE(ITYPE, TTYPE)                                           \
    template void testing_spvv_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spvv<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

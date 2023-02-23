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

template <typename I, typename X, typename Y, typename T>
void testing_spvv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle = local_handle;
    I                    size   = safe_size;
    I                    nnz    = safe_size;
    void*                x_ind  = (void*)0x4;
    void*                x_val  = (void*)0x4;
    void*                y      = (void*)0x4;
    void*                result = (void*)0x4;
    rocsparse_operation  trans  = rocsparse_operation_none;
    rocsparse_index_base base   = rocsparse_index_base_zero;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Structures
    rocsparse_local_spvec local_vec_x(size, nnz, x_ind, x_val, itype, base, ttype);
    rocsparse_local_dnvec local_vec_y(size, y, ttype);

    rocsparse_spvec_descr vec_x = local_vec_x;
    rocsparse_dnvec_descr vec_y = local_vec_y;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {6, 7};

#define PARAMS handle, trans, vec_x, vec_y, result, ttype, buffer_size, temp_buffer
    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_spvv, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_spvv, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_spvv, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_spvv, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spvv(handle, trans, vec_x, vec_y, result, ttype, nullptr, nullptr),
        rocsparse_status_invalid_pointer);
}

template <typename I, typename X, typename Y, typename T>
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
    rocsparse_datatype  xtype = get_datatype<X>();
    rocsparse_datatype  ytype = get_datatype<Y>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Grab stream used by handle
    hipStream_t stream = handle.get_stream();

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
        T h_result;

        // Allocate memory on device
        device_vector<Y> dy(100);

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

        rocsparse_local_spvec x(size, nnz, nullptr, nullptr, itype, base, xtype);
        rocsparse_local_dnvec y(size, dy, ytype);

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Obtain buffer size
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spvv(handle, trans, x, y, &h_result, ttype, &buffer_size, nullptr),
            rocsparse_status_success);

        CHECK_HIP_ERROR(rocsparse_hipMalloc(&temp_buffer, buffer_size));

        // SpVV
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spvv(handle, trans, x, y, &h_result, ttype, &buffer_size, temp_buffer),
            rocsparse_status_success);

        CHECK_HIP_ERROR(rocsparse_hipFree(temp_buffer));

        return;
    }

    // Allocate host memory
    host_vector<I> hx_ind(nnz);
    host_vector<X> hx_val(nnz);
    host_vector<Y> hy(size);
    host_vector<T> hdot_1(1);
    host_vector<T> hdot_2(1);
    host_vector<T> hdot_gold(1);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, size + base);
    rocsparse_init_alternating_sign<X>(hx_val, 1, nnz, 1);
    rocsparse_init_exact<Y>(hy, 1, size, 1);

    // Allocate device memory
    device_vector<I> dx_ind(nnz);
    device_vector<X> dx_val(nnz);
    device_vector<Y> dy(size);
    device_vector<T> ddot_2(1);

    if(!dx_ind || !dx_val || !dy || !ddot_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(X) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(Y) * size, hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, itype, base, xtype);
    rocsparse_local_dnvec y(size, dy, ytype);

    // Obtain buffer size
    CHECK_ROCSPARSE_ERROR(
        rocsparse_spvv(handle, trans, x, y, &hdot_1[0], ttype, &buffer_size, nullptr));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&temp_buffer, buffer_size));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_spvv(
            handle, trans, x, y, &hdot_1[0], ttype, &buffer_size, temp_buffer));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_spvv(handle, trans, x, y, ddot_2, ttype, &buffer_size, temp_buffer));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hdot_2, ddot_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU SpVV
        if(trans == rocsparse_operation_none)
        {
            host_doti<I, X, Y, T>(nnz, hx_val, hx_ind, hy, hdot_gold, base);
        }
        else
        {
            host_dotci<I, X, Y, T>(nnz, hx_val, hx_ind, hy, hdot_gold, base);
        }

        hdot_gold.unit_check(hdot_1);
        hdot_gold.unit_check(hdot_2);
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
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_spvv(handle, trans, x, y, &hdot_1[0], ttype, &buffer_size, temp_buffer));
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = doti_gflop_count(nnz);
        double gbyte_count = doti_gbyte_count<X, Y>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info("nnz",
                            nnz,
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(temp_buffer));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                         \
    template void testing_spvv_bad_arg<ITYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spvv<ITYPE, TTYPE, TTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

#define INSTANTIATE_MIXED(ITYPE, XTYPE, YTYPE, TTYPE)                                     \
    template void testing_spvv_bad_arg<ITYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spvv<ITYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE_MIXED(int32_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int64_t, int8_t, int8_t, int32_t);
INSTANTIATE_MIXED(int32_t, int8_t, int8_t, float);
INSTANTIATE_MIXED(int64_t, int8_t, int8_t, float);

void testing_spvv_extra(const Arguments& arg) {}

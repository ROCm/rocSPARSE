/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_csr2csc_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    rocsparse_int        m           = safe_size;
    rocsparse_int        n           = safe_size;
    rocsparse_int        nnz         = safe_size;
    const T*             csr_val     = (const T*)0x4;
    const rocsparse_int* csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int* csr_col_ind = (const rocsparse_int*)0x4;
    T*                   csc_val     = (T*)0x4;
    rocsparse_int*       csc_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*       csc_col_ind = (rocsparse_int*)0x4;
    rocsparse_action     action      = rocsparse_action_numeric;
    rocsparse_index_base base        = rocsparse_index_base_zero;
    size_t*              buffer_size = (size_t*)0x4;
    void*                temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE handle, m, n, nnz, csr_row_ptr, csr_col_ind, action, buffer_size
#define PARAMS                                                                               \
    handle, m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, csc_val, csc_row_ptr, csc_col_ind, \
        action, base, temp_buffer
    auto_testing_bad_arg(rocsparse_csr2csc_buffer_size, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_csr2csc<T>, PARAMS);
#undef PARAMS
#undef PARAMS_BUFFER_SIZE
}

template <typename T>
void testing_csr2csc(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M      = arg.M;
    rocsparse_int               N      = arg.N;
    rocsparse_index_base        base   = arg.baseA;
    rocsparse_action            action = arg.action;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<rocsparse_int> dcsc_row_ind(safe_size);
        device_vector<rocsparse_int> dcsc_col_ptr(safe_size);
        device_vector<T>             dcsc_val(safe_size);
        device_vector<T>             dbuffer(safe_size);

        size_t buffer_size;

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_csr2csc_buffer_size(
                handle, M, N, safe_size, dcsr_row_ptr, dcsr_col_ind, action, &buffer_size),
            (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csc<T>(handle,
                                                     M,
                                                     N,
                                                     safe_size,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     dcsc_val,
                                                     dcsc_row_ind,
                                                     dcsc_col_ptr,
                                                     action,
                                                     base,
                                                     dbuffer),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        return;
    }

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz, base);

    // Allocate host memory for CSC matrix
    host_vector<rocsparse_int> hcsc_row_ind(nnz);
    host_vector<rocsparse_int> hcsc_col_ptr(N + 1);
    host_vector<T>             hcsc_val(nnz);
    host_vector<rocsparse_int> hcsc_row_ind_gold;
    host_vector<rocsparse_int> hcsc_col_ptr_gold;
    host_vector<T>             hcsc_val_gold;

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);
    device_vector<rocsparse_int> dcsc_row_ind(nnz);
    device_vector<rocsparse_int> dcsc_col_ptr(N + 1);
    device_vector<T>             dcsc_val(nnz);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size(
        handle, M, N, nnz, dcsr_row_ptr, dcsr_col_ind, action, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2csc<T>(handle,
                                                   M,
                                                   N,
                                                   nnz,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   dcsc_val,
                                                   dcsc_row_ind,
                                                   dcsc_col_ptr,
                                                   action,
                                                   base,
                                                   dbuffer));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsc_row_ind, dcsc_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsc_col_ptr, dcsc_col_ptr, sizeof(rocsparse_int) * (N + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsc_val, dcsc_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // CPU csr2csc
        host_csr_to_csc(M,
                        N,
                        nnz,
                        hcsr_row_ptr.data(),
                        hcsr_col_ind.data(),
                        hcsr_val.data(),
                        hcsc_row_ind_gold,
                        hcsc_col_ptr_gold,
                        hcsc_val_gold,
                        action,
                        base);

        hcsc_row_ind_gold.unit_check(hcsc_row_ind);
        hcsc_col_ptr_gold.unit_check(hcsc_col_ptr);

        if(action == rocsparse_action_numeric)
        {
            hcsc_val_gold.unit_check(hcsc_val);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2csc<T>(handle,
                                                       M,
                                                       N,
                                                       nnz,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       dcsc_val,
                                                       dcsc_row_ind,
                                                       dcsc_col_ptr,
                                                       action,
                                                       base,
                                                       dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2csc<T>(handle,
                                                       M,
                                                       N,
                                                       nnz,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       dcsc_val,
                                                       dcsc_row_ind,
                                                       dcsc_col_ptr,
                                                       action,
                                                       base,
                                                       dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2csc_gbyte_count<T>(M, N, nnz, action);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "nnz",
                            nnz,
                            "action",
                            rocsparse_action2string(action),
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2csc_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2csc<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

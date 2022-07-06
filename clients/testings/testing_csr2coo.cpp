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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_csr2coo_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_csr_descr;
    rocsparse_local_mat_descr local_bsr_descr;

    rocsparse_handle     handle      = local_handle;
    const rocsparse_int* csr_row_ptr = (const rocsparse_int*)0x4;
    rocsparse_int        nnz         = safe_size;
    rocsparse_int        m           = safe_size;
    rocsparse_int*       coo_row_ind = (rocsparse_int*)0x4;
    rocsparse_index_base base        = rocsparse_index_base_zero;

#define PARAMS handle, csr_row_ptr, nnz, m, coo_row_ind, base
    auto_testing_bad_arg(rocsparse_csr2coo, PARAMS);
#undef PARAMS
}

template <typename T>
void testing_csr2coo(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M    = arg.M;
    rocsparse_int               N    = arg.N;
    rocsparse_index_base        base = arg.baseA;
    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcoo_row_ind(safe_size);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2coo(handle, dcsr_row_ptr, 0, M, dcoo_row_ind, base),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        return;
    }

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz, base);

    // Allocate host memory for COO matrix
    host_vector<rocsparse_int> hcoo_row_ind(nnz);
    host_vector<rocsparse_int> hcoo_row_ind_gold(nnz);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcoo_row_ind(nnz);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2coo(handle, dcsr_row_ptr, nnz, M, dcoo_row_ind, base));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_row_ind, dcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));

        // CPU csr2coo
        host_csr_to_coo(M, nnz, hcsr_row_ptr, hcoo_row_ind_gold, base);

        hcoo_row_ind_gold.unit_check(hcoo_row_ind);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_csr2coo(handle, dcsr_row_ptr, nnz, M, dcoo_row_ind, base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_csr2coo(handle, dcsr_row_ptr, nnz, M, dcoo_row_ind, base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2coo_gbyte_count<T>(M, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "nnz",
                            nnz,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            gpu_time_used / 1e3);
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2coo_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2coo<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

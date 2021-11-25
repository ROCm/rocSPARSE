/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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
void testing_coo2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    const rocsparse_int* coo_row_ind = (const rocsparse_int*)0x4;
    rocsparse_int        nnz         = safe_size;
    rocsparse_int        m           = safe_size;
    rocsparse_int*       csr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_index_base base        = rocsparse_index_base_zero;

#define PARAMS handle, coo_row_ind, nnz, m, csr_row_ptr, base
    auto_testing_bad_arg(rocsparse_coo2csr, PARAMS);
#undef PARAMS
}

template <typename T>
void testing_coo2csr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);

    rocsparse_int        M    = arg.M;
    rocsparse_int        N    = arg.N;
    rocsparse_index_base base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcoo_row_ind(safe_size);

        rocsparse_int nnz = (M > 0 && N > 0) ? 0 : -1;

        EXPECT_ROCSPARSE_STATUS(rocsparse_coo2csr(handle, dcoo_row_ind, nnz, M, dcsr_row_ptr, base),
                                (M < 0 || nnz < 0) ? rocsparse_status_invalid_size
                                                   : rocsparse_status_success);

        return;
    }

    // Allocate host memory for COO matrix
    host_vector<rocsparse_int> hcoo_row_ind;
    host_vector<rocsparse_int> hcoo_col_ind;
    host_vector<T>             hcoo_val;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_coo(hcoo_row_ind, hcoo_col_ind, hcoo_val, M, N, nnz, base);

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr(M + 1);
    host_vector<rocsparse_int> hcsr_row_ptr_gold(M + 1);

    // Allocate device memory
    device_vector<rocsparse_int> dcoo_row_ind(nnz);
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_coo2csr(handle, dcoo_row_ind, nnz, M, dcsr_row_ptr, base));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));

        // CPU coo2csr
        host_coo_to_csr(M, nnz, hcoo_row_ind.data(), hcsr_row_ptr_gold, base);
        hcsr_row_ptr_gold.unit_check(hcsr_row_ptr);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_coo2csr(handle, dcoo_row_ind, nnz, M, dcsr_row_ptr, base));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_coo2csr(handle, dcoo_row_ind, nnz, M, dcsr_row_ptr, base));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = coo2csr_gbyte_count<T>(M, nnz);
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
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            (arg.unit_check ? "yes" : "no"));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_coo2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_coo2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

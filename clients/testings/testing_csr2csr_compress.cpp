/*! \file */
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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_csr2csr_compress_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;
    static const T      safe_tol  = static_cast<T>(0);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr_A;

    rocsparse_handle          handle        = local_handle;
    rocsparse_int             m             = safe_size;
    rocsparse_int             n             = safe_size;
    const rocsparse_mat_descr descr_A       = local_descr_A;
    const T*                  csr_val_A     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr_A = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind_A = (const rocsparse_int*)0x4;
    rocsparse_int             nnz_A         = safe_size;
    rocsparse_int*            nnz_per_row   = (rocsparse_int*)0x4;
    T*                        csr_val_C     = (T*)0x4;
    rocsparse_int*            csr_row_ptr_C = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind_C = (rocsparse_int*)0x4;
    rocsparse_int*            nnz_C         = (rocsparse_int*)0x4;

    int       nargs_to_exclude_nnz   = 2;
    const int args_to_exclude_nnz[2] = {3, 7};

    int       nargs_to_exclude   = 1;
    const int args_to_exclude[1] = {12};
    T         tol                = safe_tol;

#define PARAMS_NNZ handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol
#define PARAMS                                                                                     \
    handle, m, n, descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, nnz_A, nnz_per_row, csr_val_C, \
        csr_row_ptr_C, csr_col_ind_C, tol
    auto_testing_bad_arg(
        rocsparse_nnz_compress<T>, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
    auto_testing_bad_arg(rocsparse_csr2csr_compress<T>, nargs_to_exclude, args_to_exclude, PARAMS);
#undef PARAMS
#undef PARAMS_NNZ
}

template <typename T>
void testing_csr2csr_compress(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M    = arg.M;
    rocsparse_int               N    = arg.N;
    rocsparse_index_base        base = arg.baseA;

    T tol = arg.get_alpha<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr descr_A;

    rocsparse_set_mat_index_base(descr_A, base);

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || std::real(tol) < std::real(static_cast<T>(0)))
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr_A(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_A(safe_size);
        device_vector<T>             dcsr_val_A(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_C(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_C(safe_size);
        device_vector<T>             dcsr_val_C(safe_size);
        device_vector<rocsparse_int> dnnz_per_row(safe_size);

        rocsparse_status status = rocsparse_status_success;
        if(M < 0 || N < 0)
        {
            status = rocsparse_status_invalid_size;
        }
        else if(std::real(tol) < std::real(static_cast<T>(0)))
        {
            status = rocsparse_status_invalid_value;
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                              M,
                                                              N,
                                                              descr_A,
                                                              dcsr_val_A,
                                                              dcsr_row_ptr_A,
                                                              dcsr_col_ind_A,
                                                              safe_size,
                                                              dnnz_per_row,
                                                              dcsr_val_C,
                                                              dcsr_row_ptr_C,
                                                              dcsr_col_ind_C,
                                                              tol),
                                status);

        return;
    }

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;
    host_vector<rocsparse_int> hcsr_row_ptr_C_gold;
    host_vector<rocsparse_int> hcsr_col_ind_C_gold;
    host_vector<T>             hcsr_val_C_gold;

    // Sample matrix
    rocsparse_int nnz_A;
    matrix_factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, N, nnz_A, base);

    // Allocate host memory for nnz_per_row array
    host_vector<rocsparse_int> hnnz_per_row(M);

    // Allocate host memory for compressed CSR row pointer array
    host_vector<rocsparse_int> hcsr_row_ptr_C(M + 1);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz_A);
    device_vector<T>             dcsr_val_A(nnz_A);
    device_vector<rocsparse_int> dcsr_row_ptr_C(M + 1);
    device_vector<rocsparse_int> dnnz_per_row(M);

    // Copy data from CPU to device
    dcsr_row_ptr_A.transfer_from(hcsr_row_ptr_A);
    dcsr_col_ind_A.transfer_from(hcsr_col_ind_A);
    dcsr_val_A.transfer_from(hcsr_val_A);

    if(arg.unit_check)
    {
        // Obtain compressed CSR nnz twice, first using host pointer for nnz and second using device pointer
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hnnz_C;
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, M, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &hnnz_C, tol));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        device_vector<rocsparse_int> dnnz_C(1);
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, M, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, dnnz_C, tol));

        rocsparse_int hnnz_C_copied_from_device = 0;
        CHECK_HIP_ERROR(hipMemcpy(
            &hnnz_C_copied_from_device, dnnz_C, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // Confirm that nnz is the same regardless of whether we use host or device pointers
        unit_check_scalar(hnnz_C, hnnz_C_copied_from_device);

        // Allocate device memory for compressed CSR col indices and values array
        device_vector<rocsparse_int> dcsr_col_ind_C(hnnz_C);
        device_vector<T>             dcsr_val_C(hnnz_C);
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                            M,
                                                            N,
                                                            descr_A,
                                                            dcsr_val_A,
                                                            dcsr_row_ptr_A,
                                                            dcsr_col_ind_A,
                                                            nnz_A,
                                                            dnnz_per_row,
                                                            dcsr_val_C,
                                                            dcsr_row_ptr_C,
                                                            dcsr_col_ind_C,
                                                            tol));

        // Allocate host memory for compressed CSR col indices and values array
        host_vector<rocsparse_int> hcsr_col_ind_C(hnnz_C);
        host_vector<T>             hcsr_val_C(hnnz_C);

        hcsr_row_ptr_C.transfer_from(dcsr_row_ptr_C);
        hcsr_col_ind_C.transfer_from(dcsr_col_ind_C);
        hcsr_val_C.transfer_from(dcsr_val_C);

        // CPU csr2csr_compress
        host_csr_to_csr_compress<T>(M,
                                    N,
                                    nnz_A,
                                    hcsr_row_ptr_A,
                                    hcsr_col_ind_A,
                                    hcsr_val_A,
                                    hcsr_row_ptr_C_gold,
                                    hcsr_col_ind_C_gold,
                                    hcsr_val_C_gold,
                                    base,
                                    tol);

        hcsr_row_ptr_C_gold.unit_check(hcsr_row_ptr_C);
        hcsr_col_ind_C_gold.unit_check(hcsr_col_ind_C);
        hcsr_val_C_gold.unit_check(hcsr_val_C);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int nnz_C;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
                handle, M, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &nnz_C, tol));

            // Allocate device memory for compressed CSR col indices and values array
            device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
            device_vector<T>             dcsr_val_C(nnz_C);

            CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                                M,
                                                                N,
                                                                descr_A,
                                                                dcsr_val_A,
                                                                dcsr_row_ptr_A,
                                                                dcsr_col_ind_A,
                                                                nnz_A,
                                                                dnnz_per_row,
                                                                dcsr_val_C,
                                                                dcsr_row_ptr_C,
                                                                dcsr_col_ind_C,
                                                                tol));
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, M, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &nnz_C, tol));

        // Allocate device memory for compressed CSR col indices and values array
        device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
        device_vector<T>             dcsr_val_C(nnz_C);

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2csr_compress<T>(handle,
                                                                M,
                                                                N,
                                                                descr_A,
                                                                dcsr_val_A,
                                                                dcsr_row_ptr_A,
                                                                dcsr_col_ind_A,
                                                                nnz_A,
                                                                dnnz_per_row,
                                                                dcsr_val_C,
                                                                dcsr_row_ptr_C,
                                                                dcsr_col_ind_C,
                                                                tol));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csr2csr_compress_gbyte_count<T>(M, nnz_A, nnz_C);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "nnz_A",
                            nnz_A,
                            "nnz_C",
                            nnz_C,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            arg.unit_check ? "yes" : "no");
    }
}

#define INSTANTIATE(TYPE)                                                       \
    template void testing_csr2csr_compress_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2csr_compress<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

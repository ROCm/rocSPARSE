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

#include "utility.hpp"
#include <rocsparse.hpp>

#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"

template <typename T>
void testing_csr2csr_compress_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;
    static const T      safe_tol  = static_cast<T>(0);

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr descr_A;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr_A(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind_A(safe_size);
    device_vector<T>             dcsr_val_A(safe_size);
    device_vector<rocsparse_int> dcsr_row_ptr_C(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind_C(safe_size);
    device_vector<T>             dcsr_val_C(safe_size);
    device_vector<rocsparse_int> dnnz_per_row(safe_size);

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_C || !dcsr_col_ind_C
       || !dcsr_val_C || !dnnz_per_row)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_csr2csr_compress()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(nullptr,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          dcsr_val_A,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          nullptr,
                                                          dcsr_val_A,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          nullptr,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          dcsr_val_A,
                                                          nullptr,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          dcsr_val_A,
                                                          dcsr_row_ptr_A,
                                                          nullptr,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          dcsr_val_A,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          nullptr,
                                                          dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          dcsr_val_A,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          nullptr,
                                                          dcsr_row_ptr_C,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          dcsr_val_A,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          dcsr_val_C,
                                                          nullptr,
                                                          dcsr_col_ind_C,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                          safe_size,
                                                          safe_size,
                                                          descr_A,
                                                          dcsr_val_A,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          safe_size,
                                                          dnnz_per_row,
                                                          dcsr_val_C,
                                                          dcsr_row_ptr_C,
                                                          nullptr,
                                                          safe_tol),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_csr2csr_compress(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         dim_x     = arg.dimx;
    rocsparse_int         dim_y     = arg.dimy;
    rocsparse_int         dim_z     = arg.dimz;
    rocsparse_index_base  base      = arg.baseA;
    rocsparse_matrix_init mat       = arg.matrix;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

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

        if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_C || !dcsr_col_ind_C
           || !dcsr_val_C || !dnnz_per_row)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

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

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int nnz_A;
    rocsparse_init_csr_matrix(hcsr_row_ptr_A,
                              hcsr_col_ind_A,
                              hcsr_val_A,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz_A,
                              base,
                              mat,
                              filename.c_str(),
                              false,
                              full_rank);

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

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_C || !dnnz_per_row)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr_A, hcsr_row_ptr_A, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind_A, hcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(T) * nnz_A, hipMemcpyHostToDevice));

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
        unit_check_general<rocsparse_int>(1, 1, 1, &hnnz_C, &hnnz_C_copied_from_device);

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

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C,
                                  dcsr_row_ptr_C,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind_C, dcsr_col_ind_C, sizeof(rocsparse_int) * hnnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C, dcsr_val_C, sizeof(T) * hnnz_C, hipMemcpyDeviceToHost));

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

        unit_check_general<rocsparse_int>(1, M + 1, 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C);
        unit_check_general<rocsparse_int>(1, hnnz_C, 1, hcsr_col_ind_C_gold, hcsr_col_ind_C);
        unit_check_general<T>(1, hnnz_C, 1, hcsr_val_C_gold, hcsr_val_C);
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

        double gpu_gbyte = csr2csr_compress_gbyte_count<T>(M, nnz_A, nnz_C) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz_A"
                  << std::setw(12) << "nnz_C" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz_A
                  << std::setw(12) << nnz_C << std::setw(12) << gpu_gbyte << std::setw(12)
                  << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#define INSTANTIATE(TYPE)                                                       \
    template void testing_csr2csr_compress_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2csr_compress<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

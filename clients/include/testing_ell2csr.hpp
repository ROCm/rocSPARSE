/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef TESTING_ELL2CSR_HPP
#define TESTING_ELL2CSR_HPP

#include <rocsparse.hpp>

#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"
#include "utility.hpp"

template <typename T>
void testing_ell2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor for CSR matrix
    rocsparse_local_mat_descr descrA;

    // Create matrix descriptor for ELL matrix
    rocsparse_local_mat_descr descrB;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);
    device_vector<rocsparse_int> dell_col_ind(safe_size);
    device_vector<T>             dell_val(safe_size);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dell_col_ind || !dell_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_ell2csr_width()
    rocsparse_int csr_nnz;
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr_nnz(nullptr,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dell_col_ind,
                                                  descrB,
                                                  dcsr_row_ptr,
                                                  &csr_nnz),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr_nnz(handle,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  safe_size,
                                                  dell_col_ind,
                                                  descrB,
                                                  dcsr_row_ptr,
                                                  &csr_nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr_nnz(handle,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  nullptr,
                                                  descrB,
                                                  dcsr_row_ptr,
                                                  &csr_nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr_nnz(handle,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dell_col_ind,
                                                  nullptr,
                                                  dcsr_row_ptr,
                                                  &csr_nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr_nnz(handle,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dell_col_ind,
                                                  descrB,
                                                  nullptr,
                                                  &csr_nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr_nnz(handle,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dell_col_ind,
                                                  descrB,
                                                  dcsr_row_ptr,
                                                  nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_ell2csr()
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(nullptr,
                                                 safe_size,
                                                 safe_size,
                                                 descrA,
                                                 safe_size,
                                                 dell_val,
                                                 dell_col_ind,
                                                 descrB,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 safe_size,
                                                 dell_val,
                                                 dell_col_ind,
                                                 descrB,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 descrA,
                                                 safe_size,
                                                 nullptr,
                                                 dell_col_ind,
                                                 descrB,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 descrA,
                                                 safe_size,
                                                 dell_val,
                                                 nullptr,
                                                 descrB,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 descrA,
                                                 safe_size,
                                                 dell_val,
                                                 dell_col_ind,
                                                 nullptr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 descrA,
                                                 safe_size,
                                                 dell_val,
                                                 dell_col_ind,
                                                 descrB,
                                                 nullptr,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 descrA,
                                                 safe_size,
                                                 dell_val,
                                                 dell_col_ind,
                                                 descrB,
                                                 dcsr_val,
                                                 nullptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 descrA,
                                                 safe_size,
                                                 dell_val,
                                                 dell_col_ind,
                                                 descrB,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_ell2csr(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         dim_x     = arg.dimx;
    rocsparse_int         dim_y     = arg.dimy;
    rocsparse_int         dim_z     = arg.dimz;
    rocsparse_index_base  baseA     = arg.baseA;
    rocsparse_index_base  baseB     = arg.baseB;
    rocsparse_matrix_init mat       = arg.matrix;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor for ELL matrix
    rocsparse_local_mat_descr descrA;

    // Create matrix descriptor for CSR matrix
    rocsparse_local_mat_descr descrB;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;
        size_t              ptr_size  = std::max(safe_size, static_cast<size_t>(M + 1));

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(ptr_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<rocsparse_int> dell_col_ind(safe_size);
        device_vector<T>             dell_val(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dell_col_ind || !dell_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        rocsparse_int csr_nnz;
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_ell2csr_nnz(
                handle, M, N, descrA, safe_size, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz),
            (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                     M,
                                                     N,
                                                     descrA,
                                                     safe_size,
                                                     dell_val,
                                                     dell_col_ind,
                                                     descrB,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;
    host_vector<rocsparse_int> hcsr_row_ptr_gold;
    host_vector<rocsparse_int> hcsr_col_ind_gold;
    host_vector<T>             hcsr_val_gold;

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int csr_nnz_gold;
    rocsparse_init_csr_matrix(hcsr_row_ptr_gold,
                              hcsr_col_ind_gold,
                              hcsr_val_gold,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              csr_nnz_gold,
                              baseB,
                              mat,
                              filename.c_str(),
                              false,
                              full_rank);

    // Convert to ELL
    host_vector<rocsparse_int> hell_col_ind;
    host_vector<T>             hell_val;

    rocsparse_int ell_width = 0;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        ell_width = std::max(hcsr_row_ptr_gold[i + 1] - hcsr_row_ptr_gold[i], ell_width);
    }

    rocsparse_int ell_nnz = ell_width * M;

    host_csr_to_ell(M,
                    hcsr_row_ptr_gold,
                    hcsr_col_ind_gold,
                    hcsr_val_gold,
                    hell_col_ind,
                    hell_val,
                    ell_width,
                    baseB,
                    baseA);

    hcsr_row_ptr_gold.clear();
    hcsr_col_ind_gold.clear();
    hcsr_val_gold.clear();

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dell_col_ind(ell_nnz);
    device_vector<T>             dell_val(ell_nnz);

    if(!dell_col_ind || !dell_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dell_col_ind, hell_col_ind, sizeof(rocsparse_int) * ell_nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dell_val, hell_val, sizeof(T) * ell_nnz, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Obtain CSR nnz
        rocsparse_int csr_nnz;
        CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz(
            handle, M, N, descrA, ell_width, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz));

        // Allocate device memory
        device_vector<rocsparse_int> dcsr_col_ind(csr_nnz);
        device_vector<T>             dcsr_val(csr_nnz);

        // Perform CSR conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr<T>(handle,
                                                   M,
                                                   N,
                                                   descrA,
                                                   ell_width,
                                                   dell_val,
                                                   dell_col_ind,
                                                   descrB,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind));

        // Copy output to host
        hcsr_row_ptr.resize(M + 1);
        hcsr_col_ind.resize(csr_nnz);
        hcsr_val.resize(csr_nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind, dcsr_col_ind, sizeof(rocsparse_int) * csr_nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val, dcsr_val, sizeof(T) * csr_nnz, hipMemcpyDeviceToHost));

        // CPU ell2csr
        rocsparse_int csr_nnz_gold;
        host_ell_to_csr<T>(M,
                           N,
                           hell_col_ind,
                           hell_val,
                           ell_width,
                           hcsr_row_ptr_gold,
                           hcsr_col_ind_gold,
                           hcsr_val_gold,
                           csr_nnz_gold,
                           baseA,
                           baseB);

        unit_check_general<rocsparse_int>(1, 1, 1, &csr_nnz_gold, &csr_nnz);
        unit_check_general<rocsparse_int>(1, M + 1, 1, hcsr_row_ptr_gold, hcsr_row_ptr);
        unit_check_general<rocsparse_int>(1, csr_nnz, 1, hcsr_col_ind_gold, hcsr_col_ind);
        unit_check_general<T>(1, csr_nnz, 1, hcsr_val_gold, hcsr_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        rocsparse_int csr_nnz;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz(
                handle, M, N, descrA, ell_width, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz));

            device_vector<rocsparse_int> dcsr_col_ind(csr_nnz);
            device_vector<T>             dcsr_val(csr_nnz);

            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr<T>(handle,
                                                       M,
                                                       N,
                                                       descrA,
                                                       ell_width,
                                                       dell_val,
                                                       dell_col_ind,
                                                       descrB,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz(
                handle, M, N, descrA, ell_width, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz));

            device_vector<rocsparse_int> dcsr_col_ind(csr_nnz);
            device_vector<T>             dcsr_val(csr_nnz);

            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr<T>(handle,
                                                       M,
                                                       N,
                                                       descrA,
                                                       ell_width,
                                                       dell_val,
                                                       dell_col_ind,
                                                       descrB,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = ell2csr_gbyte_count<T>(M, csr_nnz, ell_nnz) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "CSR nnz"
                  << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::setw(12) << "iter"
                  << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << csr_nnz
                  << std::setw(12) << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#endif // TESTING_ELL2CSR_HPP

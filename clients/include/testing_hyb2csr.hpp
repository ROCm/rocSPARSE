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

#pragma once
#ifndef TESTING_HYB2CSR_HPP
#define TESTING_HYB2CSR_HPP

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
void testing_hyb2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create HYB structure
    rocsparse_local_hyb_mat hyb;
    rocsparse_hyb_mat       ptr  = hyb;
    test_hyb*               dhyb = reinterpret_cast<test_hyb*>(ptr);

    dhyb->m       = safe_size;
    dhyb->n       = safe_size;
    dhyb->ell_nnz = safe_size;
    dhyb->coo_nnz = safe_size;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);
    device_vector<char>          dbuffer(safe_size);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbuffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_hyb2csr_buffer_size()
    size_t buffer_size;
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr_buffer_size(nullptr, descr, hyb, dcsr_row_ptr, &buffer_size),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr_buffer_size(handle, nullptr, hyb, dcsr_row_ptr, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr_buffer_size(handle, descr, nullptr, dcsr_row_ptr, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr_buffer_size(handle, descr, hyb, nullptr, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr_buffer_size(handle, descr, hyb, dcsr_row_ptr, nullptr),
        rocsparse_status_invalid_pointer);

    // Test rocsparse_hyb2csr()
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr<T>(nullptr, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr<T>(handle, nullptr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr<T>(handle, descr, nullptr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr<T>(handle, descr, hyb, nullptr, dcsr_row_ptr, dcsr_col_ind, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr<T>(handle, descr, hyb, dcsr_val, nullptr, dcsr_col_ind, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr<T>(handle, descr, hyb, dcsr_val, dcsr_row_ptr, nullptr, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_hyb2csr<T>(handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, nullptr),
        rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_hyb2csr(const Arguments& arg)
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

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Create HYB structure
    rocsparse_local_hyb_mat hyb;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Initialize pseudo HYB matrix
        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        dhyb->m       = M;
        dhyb->n       = N;
        dhyb->ell_nnz = safe_size;
        dhyb->coo_nnz = safe_size;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<char>          dbuffer(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbuffer)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        size_t buffer_size;

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_hyb2csr_buffer_size(handle, descr, hyb, dcsr_row_ptr, &buffer_size),
            (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_hyb2csr<T>(handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer),
            (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        return;
    }

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_gold;
    host_vector<rocsparse_int> hcsr_col_ind_gold;
    host_vector<T>             hcsr_val_gold;

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int nnz;
    rocsparse_init_csr_matrix(hcsr_row_ptr_gold,
                              hcsr_col_ind_gold,
                              hcsr_val_gold,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz,
                              base,
                              mat,
                              filename.c_str(),
                              false,
                              full_rank);

    // Allocate device memory and convert to HYB
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr_gold, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind_gold, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val_gold, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Convert CSR to HYB
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(handle,
                                               M,
                                               N,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               hyb,
                                               0,
                                               rocsparse_hyb_partition_auto));

    // Set all CSR arrays to zero
    CHECK_HIP_ERROR(hipMemset(dcsr_row_ptr, 0, sizeof(rocsparse_int) * (M + 1)));
    CHECK_HIP_ERROR(hipMemset(dcsr_col_ind, 0, sizeof(rocsparse_int) * nnz));
    CHECK_HIP_ERROR(hipMemset(dcsr_val, 0, sizeof(T) * nnz));

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_hyb2csr_buffer_size(handle, descr, hyb, dcsr_row_ptr, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_hyb2csr<T>(
            handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_row_ptr(M + 1);
        host_vector<rocsparse_int> hcsr_col_ind(nnz);
        host_vector<T>             hcsr_val(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind, dcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val, dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        unit_check_general<rocsparse_int>(1, M + 1, 1, hcsr_row_ptr_gold, hcsr_row_ptr);
        unit_check_general<rocsparse_int>(1, nnz, 1, hcsr_col_ind_gold, hcsr_col_ind);
        unit_check_general<T>(1, nnz, 1, hcsr_val_gold, hcsr_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_hyb2csr<T>(
                handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_hyb2csr<T>(
                handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        // Initialize pseudo HYB matrix
        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        double gpu_gbyte
            = hyb2csr_gbyte_count<T>(M, nnz, dhyb->ell_nnz, dhyb->coo_nnz) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::setw(12) << "iter"
                  << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz
                  << std::setw(12) << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#endif // TESTING_HYB2CSR_HPP

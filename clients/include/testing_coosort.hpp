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
#ifndef TESTING_COOSORT_HPP
#define TESTING_COOSORT_HPP

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
void testing_coosort_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<rocsparse_int> dcoo_row_ind(safe_size);
    device_vector<rocsparse_int> dcoo_col_ind(safe_size);
    device_vector<rocsparse_int> dbuffer(safe_size);

    if(!dcoo_row_ind || !dcoo_col_ind || !dbuffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_coosort_buffer_size()
    size_t buffer_size;
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_buffer_size(
            nullptr, safe_size, safe_size, safe_size, dcoo_row_ind, dcoo_col_ind, &buffer_size),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_buffer_size(
            handle, safe_size, safe_size, safe_size, nullptr, dcoo_col_ind, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_buffer_size(
            handle, safe_size, safe_size, safe_size, dcoo_row_ind, nullptr, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_buffer_size(
            handle, safe_size, safe_size, safe_size, dcoo_row_ind, dcoo_col_ind, nullptr),
        rocsparse_status_invalid_pointer);

    // Test rocsparse_coosort_by_row()
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_row(
            nullptr, safe_size, safe_size, safe_size, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_row(
            handle, safe_size, safe_size, safe_size, nullptr, dcoo_col_ind, nullptr, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_row(
            handle, safe_size, safe_size, safe_size, dcoo_row_ind, nullptr, nullptr, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_row(
            handle, safe_size, safe_size, safe_size, dcoo_row_ind, dcoo_col_ind, nullptr, nullptr),
        rocsparse_status_invalid_pointer);

    // Test rocsparse_coosort_by_column()
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_column(
            nullptr, safe_size, safe_size, safe_size, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_column(
            handle, safe_size, safe_size, safe_size, nullptr, dcoo_col_ind, nullptr, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_column(
            handle, safe_size, safe_size, safe_size, dcoo_row_ind, nullptr, nullptr, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_coosort_by_column(
            handle, safe_size, safe_size, safe_size, dcoo_row_ind, dcoo_col_ind, nullptr, nullptr),
        rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_coosort(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         dim_x     = arg.dimx;
    rocsparse_int         dim_y     = arg.dimy;
    rocsparse_int         dim_z     = arg.dimz;
    bool                  permute   = arg.algo;
    bool                  by_row    = arg.transA == rocsparse_operation_none;
    rocsparse_matrix_init mat       = arg.matrix;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcoo_row_ind(safe_size);
        device_vector<rocsparse_int> dcoo_col_ind(safe_size);
        device_vector<rocsparse_int> dbuffer(safe_size);

        if(!dcoo_row_ind || !dcoo_col_ind || !dbuffer)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        size_t buffer_size;
        EXPECT_ROCSPARSE_STATUS(rocsparse_coosort_buffer_size(
                                    handle, M, N, 0, dcoo_row_ind, dcoo_col_ind, &buffer_size),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        if(by_row)
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_coosort_by_row(
                    handle, M, N, 0, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer),
                (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        }
        else
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_coosort_by_column(
                    handle, M, N, 0, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer),
                (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        }

        return;
    }

    // Allocate host memory for COO matrix
    host_vector<rocsparse_int> hcoo_row_ind;
    host_vector<rocsparse_int> hcoo_col_ind;
    host_vector<T>             hcoo_val;
    host_vector<rocsparse_int> hcoo_row_ind_gold;
    host_vector<rocsparse_int> hcoo_col_ind_gold;
    host_vector<T>             hcoo_val_gold;

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int nnz;
    rocsparse_init_coo_matrix(hcoo_row_ind,
                              hcoo_col_ind,
                              hcoo_val,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz,
                              rocsparse_index_base_zero,
                              mat,
                              filename.c_str(),
                              false,
                              full_rank);

    // Unsort COO matrix
    host_vector<rocsparse_int> hperm(nnz);
    hcoo_row_ind_gold = hcoo_row_ind;
    hcoo_col_ind_gold = hcoo_col_ind;
    hcoo_val_gold     = hcoo_val;

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        rocsparse_int rng = rand() % nnz;

        std::swap(hcoo_row_ind[i], hcoo_row_ind[rng]);
        std::swap(hcoo_col_ind[i], hcoo_col_ind[rng]);
        std::swap(hcoo_val[i], hcoo_val[rng]);
    }

    // Allocate device memory
    device_vector<rocsparse_int> dcoo_row_ind(nnz);
    device_vector<rocsparse_int> dcoo_col_ind(nnz);
    device_vector<T>             dcoo_val(nnz);
    device_vector<rocsparse_int> dperm(nnz);

    if(!dcoo_row_ind || !dcoo_col_ind || !dcoo_val || !dperm)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_col_ind, hcoo_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcoo_val, hcoo_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_coosort_buffer_size(handle, M, N, nnz, dcoo_row_ind, dcoo_col_ind, &buffer_size));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Create permutation vector
        CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnz, dperm));

        // Sort COO matrix
        if(by_row)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_row(
                handle, M, N, nnz, dcoo_row_ind, dcoo_col_ind, permute ? dperm : nullptr, dbuffer));
        }
        else
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_column(
                handle, M, N, nnz, dcoo_row_ind, dcoo_col_ind, permute ? dperm : nullptr, dbuffer));

            // Sort host COO structure by column
            host_coosort_by_column<T>(M, nnz, hcoo_row_ind_gold, hcoo_col_ind_gold, hcoo_val_gold);
        }

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_row_ind, dcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_col_ind, dcoo_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));

        unit_check_general<rocsparse_int>(1, nnz, 1, hcoo_row_ind_gold, hcoo_row_ind);
        unit_check_general<rocsparse_int>(1, nnz, 1, hcoo_col_ind_gold, hcoo_col_ind);

        // Permute, copy and check values, if requested
        if(permute)
        {
            device_vector<T> dcoo_val_sorted(nnz);

            CHECK_ROCSPARSE_ERROR(rocsparse_gthr<T>(
                handle, nnz, dcoo_val, dcoo_val_sorted, dperm, rocsparse_index_base_zero));
            CHECK_HIP_ERROR(
                hipMemcpy(hcoo_val, dcoo_val_sorted, sizeof(T) * nnz, hipMemcpyDeviceToHost));

            unit_check_general<T>(1, nnz, 1, hcoo_val_gold, hcoo_val);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            if(by_row)
            {
                rocsparse_coosort_by_row(handle,
                                         M,
                                         N,
                                         nnz,
                                         dcoo_row_ind,
                                         dcoo_col_ind,
                                         permute ? dperm : nullptr,
                                         dbuffer);
            }
            else
            {
                rocsparse_coosort_by_column(handle,
                                            M,
                                            N,
                                            nnz,
                                            dcoo_row_ind,
                                            dcoo_col_ind,
                                            permute ? dperm : nullptr,
                                            dbuffer);
            }
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            if(by_row)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_row(handle,
                                                               M,
                                                               N,
                                                               nnz,
                                                               dcoo_row_ind,
                                                               dcoo_col_ind,
                                                               permute ? dperm : nullptr,
                                                               dbuffer));
            }
            else
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_column(handle,
                                                                  M,
                                                                  N,
                                                                  nnz,
                                                                  dcoo_row_ind,
                                                                  dcoo_col_ind,
                                                                  permute ? dperm : nullptr,
                                                                  dbuffer));
            }
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = coosort_gbyte_count<T>(nnz, permute) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "permute" << std::setw(12) << "dir" << std::setw(12) << "GB/s"
                  << std::setw(12) << "msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz
                  << std::setw(12) << (permute ? "yes" : "no") << std::setw(12)
                  << (by_row ? "row" : "column") << std::setw(12) << gpu_gbyte << std::setw(12)
                  << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Clear buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#endif // TESTING_COOSORT_HPP

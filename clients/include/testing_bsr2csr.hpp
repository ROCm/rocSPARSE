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
#ifndef TESTING_BSR2CSR_HPP
#define TESTING_BSR2CSR_HPP

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
void testing_bsr2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dbsr_col_ind(safe_size);
    device_vector<T>             dbsr_val(safe_size);
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_mat_descr bsr_descr = nullptr;
    rocsparse_create_mat_descr(&bsr_descr);

    rocsparse_mat_descr csr_descr = nullptr;
    rocsparse_create_mat_descr(&csr_descr);

    // Test invalid handle
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(nullptr,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_handle);

    // Test invalid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 nullptr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);

    // Test invalid pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 nullptr,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 nullptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 nullptr,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 nullptr,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 nullptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 nullptr),
                            rocsparse_status_invalid_pointer);

    // Test invalid direction
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 (rocsparse_direction)2,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_value);

    // Test invalid Mb
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 -1,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_size);

    // Test invalid Nb
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 -1,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_size);

    // Test invalid block dimension
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 0,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind),
                            rocsparse_status_invalid_size);
}

template <typename T>
void testing_bsr2csr(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         dim_x     = arg.dimx;
    rocsparse_int         dim_y     = arg.dimy;
    rocsparse_int         dim_z     = arg.dimz;
    rocsparse_index_base  bsr_base  = arg.baseA;
    rocsparse_index_base  csr_base  = arg.baseB;
    rocsparse_matrix_init mat       = arg.matrix;
    rocsparse_direction   direction = arg.direction;
    rocsparse_int         block_dim = arg.block_dim;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    rocsparse_int Mb = -1;
    rocsparse_int Nb = -1;
    if(block_dim > 0)
    {
        Mb = (M + block_dim - 1) / block_dim;
        Nb = (N + block_dim - 1) / block_dim;
    }

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_mat_descr bsr_descr = nullptr;
    rocsparse_create_mat_descr(&bsr_descr);

    rocsparse_mat_descr csr_descr = nullptr;
    rocsparse_create_mat_descr(&csr_descr);

    rocsparse_set_mat_index_base(bsr_descr, bsr_base);
    rocsparse_set_mat_index_base(csr_descr, csr_base);

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || Nb <= 0 || block_dim <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);

        if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind
           || !dcsr_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(handle,
                                                     direction,
                                                     Mb,
                                                     Nb,
                                                     bsr_descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     block_dim,
                                                     csr_descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind),
                                (Mb < 0 || Nb < 0 || block_dim <= 0) ? rocsparse_status_invalid_size
                                                                     : rocsparse_status_success);

        return;
    }

    //Allocate host memory for BSR matrix
    host_vector<rocsparse_int> hbsr_row_ptr;
    host_vector<rocsparse_int> hbsr_col_ind;
    host_vector<T>             hbsr_val;

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int nnzb = 0;
    rocsparse_init_bsr_matrix(hbsr_row_ptr,
                              hbsr_col_ind,
                              hbsr_val,
                              direction,
                              Mb,
                              Nb,
                              block_dim,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnzb,
                              bsr_base,
                              mat,
                              filename.c_str(),
                              false,
                              full_rank);

    // Mb and Nb can be modified by rocsparse_init_bsr_matrix
    M = Mb * block_dim;
    N = Nb * block_dim;

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr(M + 1);
    host_vector<rocsparse_int> hcsr_col_ind(nnzb * block_dim * block_dim);
    host_vector<T>             hcsr_val(nnzb * block_dim * block_dim);
    host_vector<rocsparse_int> hcsr_row_ptr_gold;
    host_vector<rocsparse_int> hcsr_col_ind_gold;
    host_vector<T>             hcsr_val_gold;

    // Allocate device memory
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val(nnzb * block_dim * block_dim);

    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnzb * block_dim * block_dim);
    device_vector<T>             dcsr_val(nnzb * block_dim * block_dim);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_val, hbsr_val, sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_bsr2csr<T>(handle,
                                                   direction,
                                                   Mb,
                                                   Nb,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   block_dim,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind,
                                  dcsr_col_ind,
                                  sizeof(rocsparse_int) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_val, dcsr_val, sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyDeviceToHost));

        // CPU bsr2csr
        host_bsr_to_csr<T>(direction,
                           Mb,
                           Nb,
                           block_dim,
                           bsr_base,
                           hbsr_row_ptr,
                           hbsr_col_ind,
                           hbsr_val,
                           csr_base,
                           hcsr_row_ptr_gold,
                           hcsr_col_ind_gold,
                           hcsr_val_gold);

        unit_check_general<rocsparse_int>(1, M + 1, 1, hcsr_row_ptr_gold, hcsr_row_ptr);
        unit_check_general<rocsparse_int>(
            1, nnzb * block_dim * block_dim, 1, hcsr_col_ind_gold, hcsr_col_ind);
        unit_check_general<T>(1, nnzb * block_dim * block_dim, 1, hcsr_val_gold, hcsr_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsr2csr<T>(handle,
                                                       direction,
                                                       Mb,
                                                       Nb,
                                                       bsr_descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       block_dim,
                                                       csr_descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsr2csr<T>(handle,
                                                       direction,
                                                       Mb,
                                                       Nb,
                                                       bsr_descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       block_dim,
                                                       csr_descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = bsr2csr_gbyte_count<T>(Mb, block_dim, nnzb) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "Mb"
                  << std::setw(12) << "Nb" << std::setw(12) << "blockdim" << std::setw(12) << "nnzb"
                  << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::setw(12) << "iter"
                  << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << Mb
                  << std::setw(12) << Nb << std::setw(12) << block_dim << std::setw(12) << nnzb
                  << std::setw(12) << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#endif // TESTING_BSR2CSR_HPP
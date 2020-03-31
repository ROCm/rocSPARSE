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
#ifndef TESTING_CSR2BSR_HPP
#define TESTING_CSR2BSR_HPP

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
void testing_csr2bsr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind(safe_size);
    device_vector<T>             dcsr_val(safe_size);
    device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dbsr_col_ind(safe_size);
    device_vector<T>             dbsr_val(safe_size);

    rocsparse_int hbsr_nnzb;

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbsr_row_ptr || !dbsr_col_ind || !dbsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Test rocsparse_csr2bsr_nnz()

    // Test invalid handle
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(nullptr,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_handle);

    // Test invalid pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  nullptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  nullptr,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  nullptr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  nullptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  nullptr),
                            rocsparse_status_invalid_pointer);

    // Test invalid direction
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  (rocsparse_direction)2,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_value);

    // Test invalid sizes
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  -1,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  -1,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  safe_size,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                  rocsparse_direction_row,
                                                  safe_size,
                                                  safe_size,
                                                  csr_descr,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  -1,
                                                  bsr_descr,
                                                  dbsr_row_ptr,
                                                  &hbsr_nnzb),
                            rocsparse_status_invalid_size);

    // Test rocsparse_csr2bsr()

    // Test invalid handle
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(nullptr,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_handle);

    // Test invalid pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 nullptr,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 nullptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 nullptr,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 nullptr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 nullptr,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 nullptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 nullptr),
                            rocsparse_status_invalid_pointer);

    // Test invalid direction
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 (rocsparse_direction)2,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_value);

    // Test invalid sizes
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 -1,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 -1,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 safe_size,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 csr_descr,
                                                 dcsr_val,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 -1,
                                                 bsr_descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind),
                            rocsparse_status_invalid_size);
}

template <typename T>
void testing_csr2bsr(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         dim_x     = arg.dimx;
    rocsparse_int         dim_y     = arg.dimy;
    rocsparse_int         dim_z     = arg.dimz;
    rocsparse_index_base  csr_base  = arg.baseA;
    rocsparse_index_base  bsr_base  = arg.baseB;
    rocsparse_matrix_init mat       = arg.matrix;
    rocsparse_direction   direction = arg.direction;
    rocsparse_int         block_dim = arg.block_dim;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    rocsparse_set_mat_index_base(csr_descr, csr_base);
    rocsparse_set_mat_index_base(bsr_descr, bsr_base);

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || block_dim <= 0)
    {
        static const size_t safe_size = 100;
        rocsparse_int       hbsr_nnzb;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbsr_row_ptr || !dbsr_col_ind
           || !dbsr_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(handle,
                                                      rocsparse_direction_row,
                                                      M,
                                                      N,
                                                      csr_descr,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      block_dim,
                                                      bsr_descr,
                                                      dbsr_row_ptr,
                                                      &hbsr_nnzb),
                                (M < 0 || N < 0 || block_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(handle,
                                                     direction,
                                                     M,
                                                     N,
                                                     csr_descr,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind,
                                                     block_dim,
                                                     bsr_descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind),
                                (M < 0 || N < 0 || block_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);

        return;
    }

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    // Allocate host memory for BSR CPU solution matrix
    host_vector<rocsparse_int> hbsr_row_ptr_gold;
    host_vector<rocsparse_int> hbsr_col_ind_gold;
    host_vector<T>             hbsr_val_gold;
    rocsparse_int              hbsr_nnzb_gold;

    rocsparse_seedrand();

    // Sample CSR matrix
    rocsparse_int nnz;
    rocsparse_init_csr_matrix(hcsr_row_ptr,
                              hcsr_col_ind,
                              hcsr_val,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz,
                              csr_base,
                              mat,
                              filename.c_str(),
                              false,
                              full_rank);

    // M and N can be modified in rocsparse_init_csr_matrix
    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    // Allocate device memory for CSR matrix
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);

    // Allocate host memory for BSR row ptr array
    host_vector<rocsparse_int> hbsr_row_ptr(Mb + 1);

    // Allocate device memory for BSR row ptr array
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dbsr_row_ptr)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy CSR data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Obtain BSR nnzb twice, first using host pointer for nnzb and second using device pointer
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hbsr_nnzb;
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                    direction,
                                                    M,
                                                    N,
                                                    csr_descr,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    dbsr_row_ptr,
                                                    &hbsr_nnzb));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        device_vector<rocsparse_int> dbsr_nnzb(1);
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                    direction,
                                                    M,
                                                    N,
                                                    csr_descr,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    block_dim,
                                                    bsr_descr,
                                                    dbsr_row_ptr,
                                                    dbsr_nnzb));

        rocsparse_int hbsr_nnzb_copied_from_device = 0;
        CHECK_HIP_ERROR(hipMemcpy(&hbsr_nnzb_copied_from_device,
                                  dbsr_nnzb,
                                  sizeof(rocsparse_int),
                                  hipMemcpyDeviceToHost));

        // Confirm that nnzb is the same regardless of whether we use host or device pointers
        unit_check_general<rocsparse_int>(1, 1, 1, &hbsr_nnzb, &hbsr_nnzb_copied_from_device);

        // Allocate device memory for BSR col indices and values array
        device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
        device_vector<T>             dbsr_val(hbsr_nnzb * block_dim * block_dim);

        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                   direction,
                                                   M,
                                                   N,
                                                   csr_descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   block_dim,
                                                   bsr_descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind));

        // Allocate host memory for BSR col indices and values array and cpu solution arrays
        host_vector<rocsparse_int> hbsr_col_ind(hbsr_nnzb);
        host_vector<T>             hbsr_val(hbsr_nnzb * block_dim * block_dim);

        // Copy BSR matrix output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr, dbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind, dbsr_col_ind, sizeof(rocsparse_int) * hbsr_nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val,
                                  dbsr_val,
                                  sizeof(T) * hbsr_nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));

        // CPU csr2bsr
        host_csr_to_bsr<T>(direction,
                           M,
                           N,
                           block_dim,
                           hbsr_nnzb_gold,
                           csr_base,
                           hcsr_row_ptr,
                           hcsr_col_ind,
                           hcsr_val,
                           bsr_base,
                           hbsr_row_ptr_gold,
                           hbsr_col_ind_gold,
                           hbsr_val_gold);

        unit_check_general<rocsparse_int>(1, 1, 1, &hbsr_nnzb_gold, &hbsr_nnzb);
        unit_check_general<rocsparse_int>(1, Mb + 1, 1, hbsr_row_ptr_gold, hbsr_row_ptr);
        unit_check_general<rocsparse_int>(1, hbsr_nnzb, 1, hbsr_col_ind_gold, hbsr_col_ind);
        unit_check_general<T>(1, hbsr_nnzb * block_dim * block_dim, 1, hbsr_val_gold, hbsr_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hbsr_nnzb;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                        direction,
                                                        M,
                                                        N,
                                                        csr_descr,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        block_dim,
                                                        bsr_descr,
                                                        dbsr_row_ptr,
                                                        &hbsr_nnzb));

            // Allocate device memory for BSR col indices and values array
            device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
            device_vector<T>             dbsr_val(hbsr_nnzb * block_dim * block_dim);

            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                       direction,
                                                       M,
                                                       N,
                                                       csr_descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       block_dim,
                                                       bsr_descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                        direction,
                                                        M,
                                                        N,
                                                        csr_descr,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        block_dim,
                                                        bsr_descr,
                                                        dbsr_row_ptr,
                                                        &hbsr_nnzb));

            // Allocate device memory for BSR col indices and values array
            device_vector<rocsparse_int> dbsr_col_ind(hbsr_nnzb);
            device_vector<T>             dbsr_val(hbsr_nnzb * block_dim * block_dim);

            CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                       direction,
                                                       M,
                                                       N,
                                                       csr_descr,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind,
                                                       block_dim,
                                                       bsr_descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte
            = csr2bsr_gbyte_count<T>(M, Mb, nnz, hbsr_nnzb, block_dim) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "Mb"
                  << std::setw(12) << "Nb" << std::setw(12) << "blockdim" << std::setw(12) << "nnzb"
                  << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::setw(12) << "iter"
                  << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << Mb
                  << std::setw(12) << Nb << std::setw(12) << block_dim << std::setw(12) << hbsr_nnzb
                  << std::setw(12) << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#endif // TESTING_CSR2BSR_HPP

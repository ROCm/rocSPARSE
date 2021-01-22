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
#include "utility.hpp"
#include <rocsparse.hpp>

template <typename T>
void testing_gebsrmm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;
    T h_beta  = 0.1;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Allocate memory on device
    device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dbsr_col_ind(safe_size);
    device_vector<T>             dbsr_val(safe_size);
    device_vector<T>             dB(safe_size);
    device_vector<T>             dC(safe_size);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dB || !dC)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test invalid handle
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(nullptr,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_handle);

    // Test invalid pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 nullptr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 nullptr,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 nullptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 nullptr,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 nullptr,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 nullptr,
                                                 safe_size),
                            rocsparse_status_invalid_pointer);

    // Test invalid size
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 -1,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 -1,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 -1,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 -1,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 0,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_size);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 0,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_invalid_size);

    // Test not implemented
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_transpose,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_conjugate_transpose,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_not_implemented);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmm<T>(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_conjugate_transpose,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 dB,
                                                 safe_size,
                                                 &h_beta,
                                                 dC,
                                                 safe_size),
                            rocsparse_status_not_implemented);
}

template <typename T>
void testing_gebsrmm(const Arguments& arg)
{
    rocsparse_int        M             = arg.M;
    rocsparse_int        N             = arg.N;
    rocsparse_int        K             = arg.K;
    rocsparse_int        row_block_dim = arg.row_block_dimA;
    rocsparse_int        col_block_dim = arg.col_block_dimA;
    rocsparse_operation  transA        = arg.transA;
    rocsparse_operation  transB        = arg.transB;
    rocsparse_direction  direction     = arg.direction;
    rocsparse_index_base base          = arg.baseA;

    rocsparse_int Mb = -1;
    rocsparse_int Kb = -1;

    if(row_block_dim > 0 && col_block_dim > 0)
    {
        Mb = (M + row_block_dim - 1) / row_block_dim;
        Kb = (K + col_block_dim - 1) / col_block_dim;
    }

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    h_beta = 0;
    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || N <= 0 || Kb <= 0 || row_block_dim <= 0 || col_block_dim <= 0)
    {
        static const size_t safe_size = 100;
        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);
        device_vector<T>             dB(safe_size);
        device_vector<T>             dC(safe_size);

        if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_gebsrmm<T>(handle,
                                 direction,
                                 transA,
                                 transB,
                                 Mb,
                                 N,
                                 Kb,
                                 safe_size,
                                 &h_alpha,
                                 descr,
                                 dbsr_val,
                                 dbsr_row_ptr,
                                 dbsr_col_ind,
                                 row_block_dim,
                                 col_block_dim,
                                 dB,
                                 safe_size,
                                 &h_beta,
                                 dC,
                                 safe_size),
            (Mb < 0 || N < 0 || Kb < 0 || row_block_dim <= 0 || col_block_dim <= 0)
                ? rocsparse_status_invalid_size
                : rocsparse_status_success);
        return;
    }

    // Allocate host memory for original CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_orig;
    host_vector<rocsparse_int> hcsr_col_ind_orig;
    host_vector<T>             hcsr_val_orig;

    // Allocate host memory for output BSR matrix
    host_vector<rocsparse_int> hbsr_row_ptr;
    host_vector<rocsparse_int> hbsr_col_ind;
    host_vector<T>             hbsr_val;

    rocsparse_matrix_factory<T> matrix_factory(arg);

    rocsparse_int nnzb = 0;
    rocsparse_init_gebsr_matrix_from_csr(matrix_factory,
                                         hbsr_row_ptr,
                                         hbsr_col_ind,
                                         hbsr_val,
                                         direction,
                                         Mb,
                                         Kb,
                                         row_block_dim,
                                         col_block_dim,
                                         nnzb,
                                         base);

    // M and K and Mb and Kb can be modified by rocsparse_init_csr_and_bsr_matrix
    M = Mb * row_block_dim;
    K = Kb * col_block_dim;

    // Some matrix properties
    rocsparse_int ldb = (transB == rocsparse_operation_none) ? K : N;
    rocsparse_int ldc = M;

    rocsparse_int ncol_B = (transB == rocsparse_operation_none ? N : K);
    rocsparse_int nnz_B  = ldb * ncol_B;
    rocsparse_int nnz_C  = ldc * N;

    // Allocate host memory for dense matrices
    host_vector<T> hB(nnz_B);
    host_vector<T> hC_1(nnz_C);
    host_vector<T> hC_2(nnz_C);
    host_vector<T> hC_gold(nnz_C);

    // Initialize data on CPU
    rocsparse_init<T>(hB, ldb, ncol_B, ldb);
    rocsparse_init<T>(hC_1, ldc, N, ldc);
    hC_2    = hC_1;
    hC_gold = hC_1;

    // Allocate device memory
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);
    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val(nnzb * row_block_dim * col_block_dim);
    device_vector<T>             dB(nnz_B);
    device_vector<T>             dC_1(nnz_C);
    device_vector<T>             dC_2(nnz_C);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dB || !dC_1 || !dC_2 || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr, sizeof(rocsparse_int) * (Mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val,
                              hbsr_val,
                              sizeof(T) * nnzb * row_block_dim * col_block_dim,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1, sizeof(T) * nnz_C, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2, sizeof(T) * nnz_C, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmm<T>(handle,
                                                   direction,
                                                   transA,
                                                   transB,
                                                   Mb,
                                                   N,
                                                   Kb,
                                                   nnzb,
                                                   &h_alpha,
                                                   descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   row_block_dim,
                                                   col_block_dim,
                                                   dB,
                                                   ldb,
                                                   &h_beta,
                                                   dC_1,
                                                   ldc));
        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmm<T>(handle,
                                                   direction,
                                                   transA,
                                                   transB,
                                                   Mb,
                                                   N,
                                                   Kb,
                                                   nnzb,
                                                   d_alpha,
                                                   descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   row_block_dim,
                                                   col_block_dim,
                                                   dB,
                                                   ldb,
                                                   d_beta,
                                                   dC_2,
                                                   ldc));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hC_1, dC_1, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2, dC_2, sizeof(T) * nnz_C, hipMemcpyDeviceToHost));

        // CPU gebsrmm
        host_gebsrmm<T>(Mb,
                        N,
                        Kb,
                        row_block_dim,
                        col_block_dim,
                        direction,
                        transA,
                        transB,
                        h_alpha,
                        hbsr_row_ptr,
                        hbsr_col_ind,
                        hbsr_val,
                        hB,
                        ldb,
                        h_beta,
                        hC_gold,
                        ldc,
                        base);

        near_check_general<T>(ldc, N, ldc, hC_gold, hC_1);
        near_check_general<T>(ldc, N, ldc, hC_gold, hC_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmm<T>(handle,
                                                       direction,
                                                       transA,
                                                       transB,
                                                       Mb,
                                                       N,
                                                       Kb,
                                                       nnzb,
                                                       &h_alpha,
                                                       descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       row_block_dim,
                                                       col_block_dim,
                                                       dB,
                                                       ldb,
                                                       &h_beta,
                                                       dC_1,
                                                       ldc));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmm<T>(handle,
                                                       direction,
                                                       transA,
                                                       transB,
                                                       Mb,
                                                       N,
                                                       Kb,
                                                       nnzb,
                                                       &h_alpha,
                                                       descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       row_block_dim,
                                                       col_block_dim,
                                                       dB,
                                                       ldb,
                                                       &h_beta,
                                                       dC_1,
                                                       ldc));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops
            = gebsrmm_gflop_count(
                  N, nnzb, row_block_dim, col_block_dim, nnz_C, h_beta != static_cast<T>(0))
              / gpu_time_used * 1e6;
        double gpu_gbyte
            = gebsrmm_gbyte_count<T>(
                  Mb, nnzb, row_block_dim, col_block_dim, nnz_B, nnz_C, h_beta != static_cast<T>(0))
              / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "K"
                  << std::setw(12) << "dir" << std::setw(12) << "transA" << std::setw(12)
                  << "transB" << std::setw(12) << "nnzb" << std::setw(12) << "row_block_dim"
                  << std::setw(12) << "col_block_dim" << std::setw(12) << "nnz_B" << std::setw(12)
                  << "nnz_C" << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << K << std::setw(12)
                  << rocsparse_direction2string(direction) << std::setw(12)
                  << rocsparse_operation2string(transA) << std::setw(12)
                  << rocsparse_operation2string(transB) << std::setw(12) << nnzb << std::setw(12)
                  << row_block_dim << std::setw(12) << col_block_dim << std::setw(12) << nnz_B
                  << std::setw(12) << nnz_C << std::setw(12) << h_alpha << std::setw(12) << h_beta
                  << std::setw(12) << gpu_gflops << std::setw(12) << gpu_gbyte << std::setw(12)
                  << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_gebsrmm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsrmm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

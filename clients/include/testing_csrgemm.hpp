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
#ifndef TESTING_CSRGEMM_HPP
#define TESTING_CSRGEMM_HPP

#include <rocsparse.hpp>

#include "flops.hpp"
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
void testing_csrgemm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    T h_alpha = 0.6;
    T h_beta  = 0.2;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;
    rocsparse_local_mat_descr descrD;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Allocate memory on device
    device_vector<rocsparse_int> dcsr_row_ptr_A(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind_A(safe_size);
    device_vector<T>             dcsr_val_A(safe_size);
    device_vector<rocsparse_int> dcsr_row_ptr_B(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind_B(safe_size);
    device_vector<T>             dcsr_val_B(safe_size);
    device_vector<rocsparse_int> dcsr_row_ptr_C(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind_C(safe_size);
    device_vector<T>             dcsr_val_C(safe_size);
    device_vector<rocsparse_int> dcsr_row_ptr_D(safe_size);
    device_vector<rocsparse_int> dcsr_col_ind_D(safe_size);
    device_vector<T>             dcsr_val_D(safe_size);
    device_vector<T>             dbuffer(safe_size);

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
       || !dcsr_val_B || !dcsr_row_ptr_C || !dcsr_col_ind_C || !dcsr_val_C || !dcsr_row_ptr_D
       || !dcsr_col_ind_D || !dcsr_val_D || !dbuffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

    size_t        buffer_size;
    rocsparse_int nnz_C;

    // ###############################################
    // Scenario 1: alpha == nullptr && beta == nullptr
    // ###############################################

    // Test rocsparse_csrgemm_buffer_size()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(nullptr,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm_nnz()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(nullptr,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  nullptr,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  nullptr,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  nullptr,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(nullptr,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 nullptr,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 nullptr,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 nullptr,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 nullptr,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 nullptr),
                            rocsparse_status_invalid_pointer);

    // ###############################################
    // Scenario 2: alpha != nullptr && beta == nullptr
    // ###############################################

    // Test rocsparse_csrgemm_buffer_size()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(nullptr,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             nullptr,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             nullptr,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             nullptr,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             nullptr,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             nullptr,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             &h_alpha,
                                                             descrA,
                                                             safe_size,
                                                             dcsr_row_ptr_A,
                                                             dcsr_col_ind_A,
                                                             descrB,
                                                             safe_size,
                                                             dcsr_row_ptr_B,
                                                             dcsr_col_ind_B,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm_nnz()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(nullptr,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  nullptr,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  nullptr,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  nullptr,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  nullptr,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  nullptr,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  nullptr,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  nullptr,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  descrA,
                                                  safe_size,
                                                  dcsr_row_ptr_A,
                                                  dcsr_col_ind_A,
                                                  descrB,
                                                  safe_size,
                                                  dcsr_row_ptr_B,
                                                  dcsr_col_ind_B,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(nullptr,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 nullptr,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 nullptr,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 nullptr,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 nullptr,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 nullptr,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 nullptr,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 nullptr,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 nullptr,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 nullptr,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 nullptr,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 nullptr,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 &h_alpha,
                                                 descrA,
                                                 safe_size,
                                                 dcsr_val_A,
                                                 dcsr_row_ptr_A,
                                                 dcsr_col_ind_A,
                                                 descrB,
                                                 safe_size,
                                                 dcsr_val_B,
                                                 dcsr_row_ptr_B,
                                                 dcsr_col_ind_B,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 nullptr),
                            rocsparse_status_invalid_pointer);

    // ###############################################
    // Scenario 3: alpha == nullptr && beta != nullptr
    // ###############################################

    // Test rocsparse_csrgemm_buffer_size()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(nullptr,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             dcsr_col_ind_D,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             &h_beta,
                                                             nullptr,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             dcsr_col_ind_D,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             nullptr,
                                                             dcsr_col_ind_D,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             nullptr,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             dcsr_col_ind_D,
                                                             nullptr,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle,
                                                             rocsparse_operation_none,
                                                             rocsparse_operation_none,
                                                             safe_size,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             &h_beta,
                                                             descrD,
                                                             safe_size,
                                                             dcsr_row_ptr_D,
                                                             dcsr_col_ind_D,
                                                             info,
                                                             nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm_nnz()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(nullptr,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  nullptr,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  nullptr,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  nullptr,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  nullptr,
                                                  &nnz_C,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  nullptr,
                                                  info,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  nullptr,
                                                  dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle,
                                                  rocsparse_operation_none,
                                                  rocsparse_operation_none,
                                                  safe_size,
                                                  safe_size,
                                                  safe_size,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  descrD,
                                                  safe_size,
                                                  dcsr_row_ptr_D,
                                                  dcsr_col_ind_D,
                                                  descrC,
                                                  dcsr_row_ptr_C,
                                                  &nnz_C,
                                                  info,
                                                  nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(nullptr,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 nullptr,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 nullptr,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 nullptr,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 nullptr,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 nullptr,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 nullptr,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 nullptr,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 nullptr,
                                                 info,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 nullptr,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 &h_beta,
                                                 descrD,
                                                 safe_size,
                                                 dcsr_val_D,
                                                 dcsr_row_ptr_D,
                                                 dcsr_col_ind_D,
                                                 descrC,
                                                 dcsr_val_C,
                                                 dcsr_row_ptr_C,
                                                 dcsr_col_ind_C,
                                                 info,
                                                 nullptr),
                            rocsparse_status_invalid_pointer);

    // ###############################################
    // Scenario 4: alpha != nullptr && beta != nullptr
    // ###############################################

    // NYI

    // Test rocsparse_csrgemm_buffer_size()
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(nullptr, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_handle);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, nullptr, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, nullptr, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, nullptr, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, nullptr, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, nullptr, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, nullptr, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, nullptr, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, nullptr, dcsr_col_ind_D, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, nullptr, info, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, nullptr, &buffer_size), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_buffer_size<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, info, nullptr), rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm_nnz()
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(nullptr, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_handle);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, nullptr, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, nullptr, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, nullptr, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, nullptr, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, nullptr, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, nullptr, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, nullptr, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, nullptr, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, nullptr, descrC, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, nullptr, dcsr_row_ptr_C, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, nullptr, &nnz_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, nullptr, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, nullptr, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, descrA, safe_size, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_row_ptr_B, dcsr_col_ind_B, descrD, safe_size, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_row_ptr_C, &nnz_C, info, nullptr), rocsparse_status_invalid_pointer);

    // Test rocsparse_csrgemm()
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(nullptr, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_handle);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, nullptr, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, nullptr, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, nullptr, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, nullptr, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, nullptr, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, nullptr, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, nullptr, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, nullptr, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, nullptr, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, nullptr, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, nullptr, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, nullptr, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, nullptr, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, nullptr, dcsr_row_ptr_C, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, nullptr, dcsr_col_ind_C, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, nullptr, info, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, nullptr, dbuffer), rocsparse_status_invalid_pointer);
    //    EXPECT_ROCSPARSE_STATUS(rocsparse_csrgemm<T>(handle, rocsparse_operation_none, rocsparse_operation_none, safe_size, safe_size, safe_size, &h_alpha, descrA, safe_size, dcsr_val_A, dcsr_row_ptr_A, dcsr_col_ind_A, descrB, safe_size, dcsr_val_B, dcsr_row_ptr_B, dcsr_col_ind_B, &h_beta, descrD, safe_size, dcsr_val_D, dcsr_row_ptr_D, dcsr_col_ind_D, descrC, dcsr_val_C, dcsr_row_ptr_C, dcsr_col_ind_C, info, nullptr), rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_csrgemm(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         dim_x     = arg.dimx;
    rocsparse_int         dim_y     = arg.dimy;
    rocsparse_int         dim_z     = arg.dimz;
    rocsparse_operation   transA    = arg.transA;
    rocsparse_operation   transB    = arg.transB;
    rocsparse_index_base  baseA     = arg.baseA;
    rocsparse_index_base  baseB     = arg.baseB;
    rocsparse_index_base  baseC     = arg.baseC;
    rocsparse_index_base  baseD     = arg.baseD;
    rocsparse_matrix_init mat       = arg.matrix;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    T* halpha_ptr = nullptr;
    T* hbeta_ptr  = nullptr;
    T* dalpha_ptr = nullptr;
    T* dbeta_ptr  = nullptr;

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

    // alpha == -99 means test for alpha == nullptr
    // beta  == -99 means test for beta == nullptr
    int scenario;
    if(h_alpha == static_cast<T>(-99) && h_beta == static_cast<T>(-99))
    {
        scenario = 1;
    }
    else if(h_alpha != static_cast<T>(-99) && h_beta == static_cast<T>(-99))
    {
        scenario   = 2;
        halpha_ptr = &h_alpha;
    }
    else if(h_alpha == static_cast<T>(-99) && h_beta != static_cast<T>(-99))
    {
        scenario  = 3;
        hbeta_ptr = &h_beta;
    }
    else if(h_alpha != static_cast<T>(-99) && h_beta != static_cast<T>(-99))
    {
        scenario   = 4;
        halpha_ptr = &h_alpha;
        hbeta_ptr  = &h_beta;
    }

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;
    rocsparse_local_mat_descr descrD;

    // Create matrix info for C
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrC, baseC));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrD, baseD));

    // Argument sanity check before allocating invalid memory
    if((scenario == 2 && (M <= 0 || N <= 0 || K <= 0)) || (scenario == 3 && (M <= 0 || N <= 0))
       || (scenario == 4 && (M <= 0 || N <= 0 || K <= 0)))
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr_A(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_A(safe_size);
        device_vector<T>             dcsr_val_A(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_B(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_B(safe_size);
        device_vector<T>             dcsr_val_B(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_C(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_C(safe_size);
        device_vector<T>             dcsr_val_C(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_D(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_D(safe_size);
        device_vector<T>             dcsr_val_D(safe_size);
        device_vector<T>             dbuffer(safe_size);

        if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
           || !dcsr_val_B || !dcsr_row_ptr_C || !dcsr_col_ind_C || !dcsr_val_C || !dcsr_row_ptr_D
           || !dcsr_col_ind_D || !dcsr_val_D || !dbuffer)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        size_t        buffer_size;
        rocsparse_int nnz_C;

        rocsparse_status status_1 = rocsparse_csrgemm_buffer_size<T>(handle,
                                                                     transA,
                                                                     transB,
                                                                     M,
                                                                     N,
                                                                     K,
                                                                     halpha_ptr,
                                                                     descrA,
                                                                     safe_size,
                                                                     dcsr_row_ptr_A,
                                                                     dcsr_col_ind_A,
                                                                     descrB,
                                                                     safe_size,
                                                                     dcsr_row_ptr_B,
                                                                     dcsr_col_ind_B,
                                                                     hbeta_ptr,
                                                                     descrD,
                                                                     safe_size,
                                                                     dcsr_row_ptr_D,
                                                                     dcsr_col_ind_D,
                                                                     info,
                                                                     &buffer_size);
        rocsparse_status status_2 = rocsparse_csrgemm_nnz(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          descrA,
                                                          safe_size,
                                                          dcsr_row_ptr_A,
                                                          dcsr_col_ind_A,
                                                          descrB,
                                                          safe_size,
                                                          dcsr_row_ptr_B,
                                                          dcsr_col_ind_B,
                                                          descrD,
                                                          safe_size,
                                                          dcsr_row_ptr_D,
                                                          dcsr_col_ind_D,
                                                          descrC,
                                                          dcsr_row_ptr_C,
                                                          &nnz_C,
                                                          info,
                                                          dbuffer);
        rocsparse_status status_3 = rocsparse_csrgemm<T>(handle,
                                                         transA,
                                                         transB,
                                                         M,
                                                         N,
                                                         K,
                                                         halpha_ptr,
                                                         descrA,
                                                         safe_size,
                                                         dcsr_val_A,
                                                         dcsr_row_ptr_A,
                                                         dcsr_col_ind_A,
                                                         descrB,
                                                         safe_size,
                                                         dcsr_val_B,
                                                         dcsr_row_ptr_B,
                                                         dcsr_col_ind_B,
                                                         hbeta_ptr,
                                                         descrD,
                                                         safe_size,
                                                         dcsr_val_D,
                                                         dcsr_row_ptr_D,
                                                         dcsr_col_ind_D,
                                                         descrC,
                                                         dcsr_val_C,
                                                         dcsr_row_ptr_C,
                                                         dcsr_col_ind_C,
                                                         info,
                                                         dbuffer);

        // scenario 4 is NYI, thus we skip it
        if(scenario == 2)
        {
            // alpha != nullptr && beta == nullptr
            EXPECT_ROCSPARSE_STATUS(status_1,
                                    (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                                              : rocsparse_status_success);
            EXPECT_ROCSPARSE_STATUS(status_2,
                                    (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                                              : rocsparse_status_success);
            EXPECT_ROCSPARSE_STATUS(status_3,
                                    (M < 0 || N < 0 || K < 0) ? rocsparse_status_invalid_size
                                                              : rocsparse_status_success);
        }
        else if(scenario == 3)
        {
            // alpha == nullptr && beta != nullptr
            EXPECT_ROCSPARSE_STATUS(status_1,
                                    (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                     : rocsparse_status_success);
            EXPECT_ROCSPARSE_STATUS(status_2,
                                    (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                     : rocsparse_status_success);
            EXPECT_ROCSPARSE_STATUS(status_3,
                                    (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                     : rocsparse_status_success);
        }

        return;
    }

    // Allocate host memory for matrices
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;
    host_vector<rocsparse_int> hcsr_row_ptr_B;
    host_vector<rocsparse_int> hcsr_col_ind_B;
    host_vector<T>             hcsr_val_B;
    host_vector<rocsparse_int> hcsr_row_ptr_D;
    host_vector<rocsparse_int> hcsr_col_ind_D;
    host_vector<T>             hcsr_val_D;

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int nnz_A = 4;
    rocsparse_int nnz_B = 4;
    rocsparse_int hnnz_C_gold;
    rocsparse_int hnnz_C_1;
    rocsparse_int hnnz_C_2;
    rocsparse_int nnz_D = 4;

    if(scenario == 2)
    {
        // alpha != nullptr && beta == nullptr
        rocsparse_init_csr_matrix(hcsr_row_ptr_A,
                                  hcsr_col_ind_A,
                                  hcsr_val_A,
                                  M,
                                  K,
                                  N,
                                  dim_x,
                                  dim_y,
                                  dim_z,
                                  nnz_A,
                                  baseA,
                                  mat,
                                  filename.c_str(),
                                  arg.timing ? false : true,
                                  full_rank);
        rocsparse_init_csr_matrix(hcsr_row_ptr_B,
                                  hcsr_col_ind_B,
                                  hcsr_val_B,
                                  K,
                                  N,
                                  M,
                                  dim_x,
                                  dim_y,
                                  dim_z,
                                  nnz_B,
                                  baseB,
                                  rocsparse_matrix_random,
                                  filename.c_str(),
                                  arg.timing ? false : true,
                                  full_rank);
    }
    else if(scenario == 3)
    {
        // alpha == nullptr && beta != nullptr
        rocsparse_init_csr_matrix(hcsr_row_ptr_D,
                                  hcsr_col_ind_D,
                                  hcsr_val_D,
                                  M,
                                  N,
                                  K,
                                  dim_x,
                                  dim_y,
                                  dim_z,
                                  nnz_D,
                                  baseD,
                                  mat,
                                  filename.c_str(),
                                  arg.timing ? false : true,
                                  full_rank);
    }

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz_A);
    device_vector<T>             dcsr_val_A(nnz_A);
    device_vector<rocsparse_int> dcsr_row_ptr_B(K + 1);
    device_vector<rocsparse_int> dcsr_col_ind_B(nnz_B);
    device_vector<T>             dcsr_val_B(nnz_B);
    device_vector<rocsparse_int> dcsr_row_ptr_D(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_D(nnz_D);
    device_vector<T>             dcsr_val_D(nnz_D);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);
    device_vector<rocsparse_int> dcsr_row_ptr_C_1(M + 1);
    device_vector<rocsparse_int> dcsr_row_ptr_C_2(M + 1);
    device_vector<rocsparse_int> dnnz_C_2(1);

    if(!dcsr_row_ptr_A || !dcsr_col_ind_A || !dcsr_val_A || !dcsr_row_ptr_B || !dcsr_col_ind_B
       || !dcsr_val_B || !dcsr_row_ptr_D || !dcsr_col_ind_D || !dcsr_val_D || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    if(scenario == 2)
    {
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_A,
                                  hcsr_row_ptr_A,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_A, hcsr_col_ind_A, sizeof(rocsparse_int) * nnz_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_A, hcsr_val_A, sizeof(T) * nnz_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_B,
                                  hcsr_row_ptr_B,
                                  sizeof(rocsparse_int) * (K + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_B, hcsr_col_ind_B, sizeof(rocsparse_int) * nnz_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_B, hcsr_val_B, sizeof(T) * nnz_B, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        dalpha_ptr = d_alpha;
    }
    else if(scenario == 3)
    {
        CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_D,
                                  hcsr_row_ptr_D,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dcsr_col_ind_D, hcsr_col_ind_D, sizeof(rocsparse_int) * nnz_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dcsr_val_D, hcsr_val_D, sizeof(T) * nnz_D, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
        dbeta_ptr = d_beta;
    }

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_buffer_size<T>(handle,
                                                           transA,
                                                           transB,
                                                           M,
                                                           N,
                                                           K,
                                                           halpha_ptr,
                                                           descrA,
                                                           nnz_A,
                                                           dcsr_row_ptr_A,
                                                           dcsr_col_ind_A,
                                                           descrB,
                                                           nnz_B,
                                                           dcsr_row_ptr_B,
                                                           dcsr_col_ind_B,
                                                           hbeta_ptr,
                                                           descrD,
                                                           nnz_D,
                                                           dcsr_row_ptr_D,
                                                           dcsr_col_ind_D,
                                                           info,
                                                           &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Obtain nnz of C

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrD,
                                                    nnz_D,
                                                    dcsr_row_ptr_D,
                                                    dcsr_col_ind_D,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &hnnz_C_1,
                                                    info,
                                                    dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrD,
                                                    nnz_D,
                                                    dcsr_row_ptr_D,
                                                    dcsr_col_ind_D,
                                                    descrC,
                                                    dcsr_row_ptr_C_2,
                                                    dnnz_C_2,
                                                    info,
                                                    dbuffer));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_row_ptr_C_1(M + 1);
        host_vector<rocsparse_int> hcsr_row_ptr_C_2(M + 1);
        CHECK_HIP_ERROR(
            hipMemcpy(&hnnz_C_2, dnnz_C_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_1,
                                  dcsr_row_ptr_C_1,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr_C_2,
                                  dcsr_row_ptr_C_2,
                                  sizeof(rocsparse_int) * (M + 1),
                                  hipMemcpyDeviceToHost));

        // CPU csrgemm_nnz
        host_vector<rocsparse_int> hcsr_row_ptr_C_gold(M + 1);
        host_csrgemm_nnz<T>(M,
                            N,
                            K,
                            halpha_ptr,
                            hcsr_row_ptr_A,
                            hcsr_col_ind_A,
                            hcsr_row_ptr_B,
                            hcsr_col_ind_B,
                            hbeta_ptr,
                            hcsr_row_ptr_D,
                            hcsr_col_ind_D,
                            hcsr_row_ptr_C_gold,
                            &hnnz_C_gold,
                            baseA,
                            baseB,
                            baseC,
                            baseD);

        // Check nnz of C
        unit_check_general(1, 1, 1, &hnnz_C_gold, &hnnz_C_1);
        unit_check_general(1, 1, 1, &hnnz_C_gold, &hnnz_C_2);

        // Check row pointers of C
        unit_check_general<rocsparse_int>(1, M + 1, 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_1);
        unit_check_general<rocsparse_int>(1, M + 1, 1, hcsr_row_ptr_C_gold, hcsr_row_ptr_C_2);

        // Allocate device memory for C
        device_vector<rocsparse_int> dcsr_col_ind_C_1(hnnz_C_1);
        device_vector<rocsparse_int> dcsr_col_ind_C_2(hnnz_C_2);
        device_vector<T>             dcsr_val_C_1(hnnz_C_1);
        device_vector<T>             dcsr_val_C_2(hnnz_C_2);

        // Perform matrix matrix multiplication

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   halpha_ptr,
                                                   descrA,
                                                   nnz_A,
                                                   dcsr_val_A,
                                                   dcsr_row_ptr_A,
                                                   dcsr_col_ind_A,
                                                   descrB,
                                                   nnz_B,
                                                   dcsr_val_B,
                                                   dcsr_row_ptr_B,
                                                   dcsr_col_ind_B,
                                                   hbeta_ptr,
                                                   descrD,
                                                   nnz_D,
                                                   dcsr_val_D,
                                                   dcsr_row_ptr_D,
                                                   dcsr_col_ind_D,
                                                   descrC,
                                                   dcsr_val_C_1,
                                                   dcsr_row_ptr_C_1,
                                                   dcsr_col_ind_C_1,
                                                   info,
                                                   dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   dalpha_ptr,
                                                   descrA,
                                                   nnz_A,
                                                   dcsr_val_A,
                                                   dcsr_row_ptr_A,
                                                   dcsr_col_ind_A,
                                                   descrB,
                                                   nnz_B,
                                                   dcsr_val_B,
                                                   dcsr_row_ptr_B,
                                                   dcsr_col_ind_B,
                                                   dbeta_ptr,
                                                   descrD,
                                                   nnz_D,
                                                   dcsr_val_D,
                                                   dcsr_row_ptr_D,
                                                   dcsr_col_ind_D,
                                                   descrC,
                                                   dcsr_val_C_2,
                                                   dcsr_row_ptr_C_2,
                                                   dcsr_col_ind_C_2,
                                                   info,
                                                   dbuffer));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_col_ind_C_1(hnnz_C_1);
        host_vector<rocsparse_int> hcsr_col_ind_C_2(hnnz_C_2);
        host_vector<T>             hcsr_val_C_1(hnnz_C_1);
        host_vector<T>             hcsr_val_C_2(hnnz_C_2);

        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_1,
                                  dcsr_col_ind_C_1,
                                  sizeof(rocsparse_int) * hnnz_C_1,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_C_2,
                                  dcsr_col_ind_C_2,
                                  sizeof(rocsparse_int) * hnnz_C_2,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_1, dcsr_val_C_1, sizeof(T) * hnnz_C_1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val_C_2, dcsr_val_C_2, sizeof(T) * hnnz_C_2, hipMemcpyDeviceToHost));

        // CPU csrgemm
        host_vector<rocsparse_int> hcsr_col_ind_C_gold(hnnz_C_gold);
        host_vector<T>             hcsr_val_C_gold(hnnz_C_gold);
        host_csrgemm<T>(M,
                        N,
                        K,
                        halpha_ptr,
                        hcsr_row_ptr_A,
                        hcsr_col_ind_A,
                        hcsr_val_A,
                        hcsr_row_ptr_B,
                        hcsr_col_ind_B,
                        hcsr_val_B,
                        hbeta_ptr,
                        hcsr_row_ptr_D,
                        hcsr_col_ind_D,
                        hcsr_val_D,
                        hcsr_row_ptr_C_gold,
                        hcsr_col_ind_C_gold,
                        hcsr_val_C_gold,
                        baseA,
                        baseB,
                        baseC,
                        baseD);

        // Check C
        unit_check_general<rocsparse_int>(1, hnnz_C_gold, 1, hcsr_col_ind_C_gold, hcsr_col_ind_C_1);
        unit_check_general<rocsparse_int>(1, hnnz_C_gold, 1, hcsr_col_ind_C_gold, hcsr_col_ind_C_2);
        near_check_general<T>(1, hnnz_C_gold, 1, hcsr_val_C_gold, hcsr_val_C_1);
        near_check_general<T>(1, hnnz_C_gold, 1, hcsr_val_C_gold, hcsr_val_C_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        descrA,
                                                        nnz_A,
                                                        dcsr_row_ptr_A,
                                                        dcsr_col_ind_A,
                                                        descrB,
                                                        nnz_B,
                                                        dcsr_row_ptr_B,
                                                        dcsr_col_ind_B,
                                                        descrD,
                                                        nnz_D,
                                                        dcsr_row_ptr_D,
                                                        dcsr_col_ind_D,
                                                        descrC,
                                                        dcsr_row_ptr_C_1,
                                                        &hnnz_C_1,
                                                        info,
                                                        dbuffer));

            device_vector<rocsparse_int> dcsr_col_ind_C(hnnz_C_1);
            device_vector<T>             dcsr_val_C(hnnz_C_1);

            CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       halpha_ptr,
                                                       descrA,
                                                       nnz_A,
                                                       dcsr_val_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       descrB,
                                                       nnz_B,
                                                       dcsr_val_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       hbeta_ptr,
                                                       descrD,
                                                       nnz_D,
                                                       dcsr_val_D,
                                                       dcsr_row_ptr_D,
                                                       dcsr_col_ind_D,
                                                       descrC,
                                                       dcsr_val_C,
                                                       dcsr_row_ptr_C_1,
                                                       dcsr_col_ind_C,
                                                       info,
                                                       dbuffer));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz(handle,
                                                    transA,
                                                    transB,
                                                    M,
                                                    N,
                                                    K,
                                                    descrA,
                                                    nnz_A,
                                                    dcsr_row_ptr_A,
                                                    dcsr_col_ind_A,
                                                    descrB,
                                                    nnz_B,
                                                    dcsr_row_ptr_B,
                                                    dcsr_col_ind_B,
                                                    descrD,
                                                    nnz_D,
                                                    dcsr_row_ptr_D,
                                                    dcsr_col_ind_D,
                                                    descrC,
                                                    dcsr_row_ptr_C_1,
                                                    &hnnz_C_1,
                                                    info,
                                                    dbuffer));

        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        device_vector<rocsparse_int> dcsr_col_ind_C(hnnz_C_1);
        device_vector<T>             dcsr_val_C(hnnz_C_1);

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrgemm<T>(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       halpha_ptr,
                                                       descrA,
                                                       nnz_A,
                                                       dcsr_val_A,
                                                       dcsr_row_ptr_A,
                                                       dcsr_col_ind_A,
                                                       descrB,
                                                       nnz_B,
                                                       dcsr_val_B,
                                                       dcsr_row_ptr_B,
                                                       dcsr_col_ind_B,
                                                       hbeta_ptr,
                                                       descrD,
                                                       nnz_D,
                                                       dcsr_val_D,
                                                       dcsr_row_ptr_D,
                                                       dcsr_col_ind_D,
                                                       descrC,
                                                       dcsr_val_C,
                                                       dcsr_row_ptr_C_1,
                                                       dcsr_col_ind_C,
                                                       info,
                                                       dbuffer));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gpu_gflops = csrgemm_gflop_count<T>(M,
                                                   halpha_ptr,
                                                   hcsr_row_ptr_A,
                                                   hcsr_col_ind_A,
                                                   hcsr_row_ptr_B,
                                                   hbeta_ptr,
                                                   hcsr_row_ptr_D,
                                                   baseA)
                            / gpu_solve_time_used * 1e6;
        double gpu_gbyte
            = csrgemm_gbyte_count<T>(M, N, K, nnz_A, nnz_B, hnnz_C_1, nnz_D, halpha_ptr, hbeta_ptr)
              / gpu_solve_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "opA" << std::setw(12) << "opB" << std::setw(12) << "M"
                  << std::setw(12) << "N" << std::setw(12) << "K" << std::setw(12) << "nnz_A"
                  << std::setw(12) << "nnz_B" << std::setw(12) << "nnz_C" << std::setw(12)
                  << "nnz_D" << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(16) << "nnz msec"
                  << std::setw(16) << "gemm msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << rocsparse_operation2string(transA) << std::setw(12)
                  << rocsparse_operation2string(transB) << std::setw(12) << M << std::setw(12) << N
                  << std::setw(12) << K << std::setw(12) << nnz_A << std::setw(12) << nnz_B
                  << std::setw(12) << hnnz_C_1 << std::setw(12) << nnz_D;
        if(scenario == 2 || scenario == 4)
        {
            std::cout << std::setw(12) << h_alpha;
        }
        else
        {
            std::cout << std::setw(12) << "null";
        }
        if(scenario == 3 || scenario == 4)
        {
            std::cout << std::setw(12) << h_beta;
        }
        else
        {
            std::cout << std::setw(12) << "null";
        }
        std::cout << std::setw(12) << gpu_gflops << std::setw(12) << gpu_gbyte << std::setw(16)
                  << gpu_analysis_time_used / 1e3 << std::setw(16) << gpu_solve_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#endif // TESTING_CSRGEMM_HPP

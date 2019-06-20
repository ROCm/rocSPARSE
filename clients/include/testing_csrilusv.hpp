/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#ifndef TESTING_CSRILUSV_HPP
#define TESTING_CSRILUSV_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <rocsparse.h>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
rocsparse_status testing_csrilusv(Arguments argus)
{
    rocsparse_index_base      idx_base = argus.idx_base;
    rocsparse_analysis_policy analysis = argus.analysis;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle               handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr_M(new descr_struct);
    rocsparse_mat_descr           descr_M = test_descr_M->descr;

    std::unique_ptr<mat_info_struct> unique_ptr_mat_info(new mat_info_struct);
    rocsparse_mat_info               info = unique_ptr_mat_info->info;

    // Initialize the matrix descriptor
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_M, idx_base));

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr;
    std::vector<rocsparse_int> hcsr_col_ind;
    std::vector<T>             hcsr_val;

    // Initial Data on CPU
    rocsparse_int m;
    rocsparse_int n;
    rocsparse_int nnz;

    if(read_bin_matrix(
           argus.filename.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base)
       != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", argus.filename.c_str());
        return rocsparse_status_internal_error;
    }

    // Allocate memory on device
    auto dptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto d_position_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int)), device_free};

    rocsparse_int* dptr       = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol       = (rocsparse_int*)dcol_managed.get();
    T*             dval       = (T*)dval_managed.get();
    rocsparse_int* d_position = (rocsparse_int*)d_position_managed.get();

    if(!dval || !dptr || !dcol || !d_position)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dval || !dptr || !dcol || !d_position");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain csrilu0 buffer size
    size_t size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_csrilu0_buffer_size(handle, m, nnz, descr_M, dval, dptr, dcol, info, &size));

    // Allocate buffer on the device
    auto dbuffer_managed = rocsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dbuffer");
        return rocsparse_status_memory_error;
    }

    // csrilu0 analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_analysis(handle,
                                                     m,
                                                     nnz,
                                                     descr_M,
                                                     dval,
                                                     dptr,
                                                     dcol,
                                                     info,
                                                     analysis,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer));

    // Compute incomplete LU factorization
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0(
        handle, m, nnz, descr_M, dval, dptr, dcol, info, rocsparse_solve_policy_auto, dbuffer));

    // Check for zero pivot
    rocsparse_int    hposition_1, hposition_2;
    rocsparse_status pivot_status_1, pivot_status_2;

    // Host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    pivot_status_1 = rocsparse_csrilu0_zero_pivot(handle, info, &hposition_1);

    // device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    pivot_status_2 = rocsparse_csrilu0_zero_pivot(handle, info, d_position);

    // Copy output to CPU
    std::vector<T> iluresult(nnz);
    CHECK_HIP_ERROR(hipMemcpy(iluresult.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(&hposition_2, d_position, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

    // Compute host reference csrilu0
    rocsparse_int position_gold
        = csrilu0(m, hcsr_row_ptr.data(), hcsr_col_ind.data(), hcsr_val.data(), idx_base);

    // Check zero pivot results
    unit_check_general(1, 1, 1, &position_gold, &hposition_1);
    unit_check_general(1, 1, 1, &position_gold, &hposition_2);

    // If zero pivot was found, do not go further
    if(hposition_1 != -1)
    {
        verify_rocsparse_status_zero_pivot(pivot_status_1, "expected rocsparse_status_zero_pivot");
        return rocsparse_status_success;
    }

    if(hposition_2 != -1)
    {
        verify_rocsparse_status_zero_pivot(pivot_status_2, "expected rocsparse_status_zero_pivot");
        return rocsparse_status_success;
    }

    // Check csrilu0 factorization
    unit_check_general(1, nnz, 1, hcsr_val.data(), iluresult.data());

    // Create matrix descriptors for csrsv
    std::unique_ptr<descr_struct> test_descr_L(new descr_struct);
    rocsparse_mat_descr           descr_L = test_descr_L->descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_L, idx_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit));

    std::unique_ptr<descr_struct> test_descr_U(new descr_struct);
    rocsparse_mat_descr           descr_U = test_descr_U->descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_U, idx_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit));

    // Obtain csrsv buffer sizes
    size_t size_lower, size_upper;
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size(
        handle, rocsparse_operation_none, m, nnz, descr_L, dval, dptr, dcol, info, &size_lower));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_buffer_size(
        handle, rocsparse_operation_none, m, nnz, descr_U, dval, dptr, dcol, info, &size_upper));

    // Sizes should match with csrilu0
    unit_check_general(1, 1, 1, &size, &size_lower);
    unit_check_general(1, 1, 1, &size, &size_upper);

    // csrsv analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis(handle,
                                                   rocsparse_operation_none,
                                                   m,
                                                   nnz,
                                                   descr_L,
                                                   dval,
                                                   dptr,
                                                   dcol,
                                                   info,
                                                   analysis,
                                                   rocsparse_solve_policy_auto,
                                                   dbuffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_analysis(handle,
                                                   rocsparse_operation_none,
                                                   m,
                                                   nnz,
                                                   descr_U,
                                                   dval,
                                                   dptr,
                                                   dcol,
                                                   info,
                                                   analysis,
                                                   rocsparse_solve_policy_auto,
                                                   dbuffer));

    // Initialize some more structures required for Lz = x
    T h_alpha = static_cast<T>(1);

    std::vector<T> hx(m, static_cast<T>(1));
    std::vector<T> hy_gold(m);
    std::vector<T> hz_gold(m);

    // Allocate device memory
    auto dx_managed      = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_1_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dz_1_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dz_2_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    T* dx      = (T*)dx_managed.get();
    T* dy_1    = (T*)dy_1_managed.get();
    T* dy_2    = (T*)dy_2_managed.get();
    T* dz_1    = (T*)dz_1_managed.get();
    T* dz_2    = (T*)dz_2_managed.get();
    T* d_alpha = (T*)d_alpha_managed.get();

    if(!dx || !dy_1 || !dy_2 || !dz_1 || !dz_2 || !d_alpha)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dx || !dy_1 || !dy_2 || !dz_1 || "
                                        "!dz_2 || !d_alpha");
        return rocsparse_status_memory_error;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Solve Lz = x

    // host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve(handle,
                                                rocsparse_operation_none,
                                                m,
                                                nnz,
                                                &h_alpha,
                                                descr_L,
                                                dval,
                                                dptr,
                                                dcol,
                                                info,
                                                dx,
                                                dz_1,
                                                rocsparse_solve_policy_auto,
                                                dbuffer));

    // Check for zero pivot
    pivot_status_1 = rocsparse_csrsv_zero_pivot(handle, descr_L, info, &hposition_1);

    // device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve(handle,
                                                rocsparse_operation_none,
                                                m,
                                                nnz,
                                                d_alpha,
                                                descr_L,
                                                dval,
                                                dptr,
                                                dcol,
                                                info,
                                                dx,
                                                dz_2,
                                                rocsparse_solve_policy_auto,
                                                dbuffer));

    // Check for zero pivot
    pivot_status_2 = rocsparse_csrsv_zero_pivot(handle, descr_L, info, d_position);

    // Host csrsv
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    position_gold = lsolve(m,
                           hcsr_row_ptr.data(),
                           hcsr_col_ind.data(),
                           hcsr_val.data(),
                           h_alpha,
                           hx.data(),
                           hz_gold.data(),
                           idx_base,
                           rocsparse_diag_type_unit,
                           prop.warpSize);

    // Check zero pivot results
    unit_check_general(1, 1, 1, &position_gold, &hposition_1);
    unit_check_general(1, 1, 1, &position_gold, &hposition_2);

    // If zero pivot was found, do not go further
    if(hposition_1 != -1)
    {
        verify_rocsparse_status_zero_pivot(pivot_status_1, "expected rocsparse_status_zero_pivot");
        return rocsparse_status_success;
    }

    if(hposition_2 != -1)
    {
        verify_rocsparse_status_zero_pivot(pivot_status_2, "expected rocsparse_status_zero_pivot");
        return rocsparse_status_success;
    }

    // Copy output from device to CPU
    std::vector<T> hz_1(m);
    std::vector<T> hz_2(m);

    CHECK_HIP_ERROR(hipMemcpy(hz_1.data(), dz_1, sizeof(T) * m, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hz_2.data(), dz_2, sizeof(T) * m, hipMemcpyDeviceToHost));

    // Check z
    unit_check_general(1, m, 1, hz_gold.data(), hz_1.data());
    unit_check_general(1, m, 1, hz_gold.data(), hz_2.data());

    // Solve Uy = z

    // host pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve(handle,
                                                rocsparse_operation_none,
                                                m,
                                                nnz,
                                                &h_alpha,
                                                descr_U,
                                                dval,
                                                dptr,
                                                dcol,
                                                info,
                                                dz_1,
                                                dy_1,
                                                rocsparse_solve_policy_auto,
                                                dbuffer));

    // Check for zero pivot
    pivot_status_1 = rocsparse_csrsv_zero_pivot(handle, descr_U, info, &hposition_1);

    // device pointer mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_csrsv_solve(handle,
                                                rocsparse_operation_none,
                                                m,
                                                nnz,
                                                d_alpha,
                                                descr_U,
                                                dval,
                                                dptr,
                                                dcol,
                                                info,
                                                dz_2,
                                                dy_2,
                                                rocsparse_solve_policy_auto,
                                                dbuffer));

    // Check for zero pivot
    pivot_status_2 = rocsparse_csrsv_zero_pivot(handle, descr_U, info, d_position);

    // Host csrsv
    position_gold = usolve(m,
                           hcsr_row_ptr.data(),
                           hcsr_col_ind.data(),
                           hcsr_val.data(),
                           h_alpha,
                           hz_gold.data(),
                           hy_gold.data(),
                           idx_base,
                           rocsparse_diag_type_non_unit,
                           prop.warpSize);

    // Check zero pivot results
    unit_check_general(1, 1, 1, &position_gold, &hposition_1);
    unit_check_general(1, 1, 1, &position_gold, &hposition_2);

    // If zero pivot was found, do not go further
    if(hposition_1 != -1)
    {
        verify_rocsparse_status_zero_pivot(pivot_status_1, "expected rocsparse_status_zero_pivot");
        return rocsparse_status_success;
    }

    if(hposition_2 != -1)
    {
        verify_rocsparse_status_zero_pivot(pivot_status_2, "expected rocsparse_status_zero_pivot");
        return rocsparse_status_success;
    }

    // Copy output from device to CPU
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);

    CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

    // Check z
    unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
    unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());

    return rocsparse_status_success;
}

#endif // TESTING_CSRILUSOLVE_HPP

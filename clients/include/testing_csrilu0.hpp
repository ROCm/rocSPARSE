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
#ifndef TESTING_CSRILU0_HPP
#define TESTING_CSRILU0_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <string>
#include <cmath>
#include <rocsparse.h>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_csrilu0_bad_arg(void)
{
    rocsparse_int m                    = 100;
    rocsparse_int nnz                  = 100;
    rocsparse_int safe_size            = 100;
    rocsparse_analysis_policy analysis = rocsparse_analysis_policy_reuse;
    rocsparse_solve_policy solve       = rocsparse_solve_policy_auto;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr descr = unique_ptr_descr->descr;

    std::unique_ptr<mat_info_struct> unique_ptr_mat_info(new mat_info_struct);
    rocsparse_mat_info info = unique_ptr_mat_info->info;

    auto dptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dbuffer_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
    T* dval             = (T*)dval_managed.get();
    void* dbuffer       = (void*)dbuffer_managed.get();

    if(!dval || !dptr || !dcol || !dbuffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing rocsparse_csrilu0_buffer_size
    size_t size;

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status = rocsparse_csrilu0_buffer_size(
            handle, m, nnz, descr, dval, dptr_null, dcol, info, &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        rocsparse_int* dcol_null = nullptr;

        status = rocsparse_csrilu0_buffer_size(
            handle, m, nnz, descr, dval, dptr, dcol_null, info, &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = rocsparse_csrilu0_buffer_size(
            handle, m, nnz, descr, dval_null, dptr, dcol, info, &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == buffer_size)
    {
        size_t* size_null = nullptr;

        status =
            rocsparse_csrilu0_buffer_size(handle, m, nnz, descr, dval, dptr, dcol, info, size_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: size is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrilu0_buffer_size(
            handle, m, nnz, descr_null, dval, dptr, dcol, info, &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrilu0_buffer_size(
            handle, m, nnz, descr, dval, dptr, dcol, info_null, &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrilu0_buffer_size(
            handle_null, m, nnz, descr, dval, dptr, dcol, info, &size);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrilu0_analysis

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status = rocsparse_csrilu0_analysis(
            handle, m, nnz, descr, dval, dptr_null, dcol, info, analysis, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        rocsparse_int* dcol_null = nullptr;

        status = rocsparse_csrilu0_analysis(
            handle, m, nnz, descr, dval, dptr, dcol_null, info, analysis, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = rocsparse_csrilu0_analysis(
            handle, m, nnz, descr, dval_null, dptr, dcol, info, analysis, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status = rocsparse_csrilu0_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, info, analysis, solve, dbuffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrilu0_analysis(
            handle, m, nnz, descr_null, dval, dptr, dcol, info, analysis, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrilu0_analysis(
            handle, m, nnz, descr, dval, dptr, dcol, info_null, analysis, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrilu0_analysis(
            handle_null, m, nnz, descr, dval, dptr, dcol, info, analysis, solve, dbuffer);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrilu0

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status =
            rocsparse_csrilu0(handle, m, nnz, descr, dval, dptr_null, dcol, info, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        rocsparse_int* dcol_null = nullptr;

        status =
            rocsparse_csrilu0(handle, m, nnz, descr, dval, dptr, dcol_null, info, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status =
            rocsparse_csrilu0(handle, m, nnz, descr, dval_null, dptr, dcol, info, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dbuffer)
    {
        void* dbuffer_null = nullptr;

        status =
            rocsparse_csrilu0(handle, m, nnz, descr, dval, dptr, dcol, info, solve, dbuffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: dbuffer is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status =
            rocsparse_csrilu0(handle, m, nnz, descr_null, dval, dptr, dcol, info, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status =
            rocsparse_csrilu0(handle, m, nnz, descr, dval, dptr, dcol, info_null, solve, dbuffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status =
            rocsparse_csrilu0(handle_null, m, nnz, descr, dval, dptr, dcol, info, solve, dbuffer);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrilu0_zero_pivot
    rocsparse_int position;

    // testing for(nullptr == position)
    {
        rocsparse_int* position_null = nullptr;

        status = rocsparse_csrilu0_zero_pivot(handle, info, position_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: position is nullptr");
    }
    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrilu0_zero_pivot(handle, info_null, &position);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrilu0_zero_pivot(handle_null, info, &position);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrilu0_clear

    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrilu0_clear(handle, info_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrilu0_clear(handle_null, info);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_csrilu0(Arguments argus)
{
    rocsparse_int safe_size       = 100;
    rocsparse_int m               = argus.M;
    rocsparse_index_base idx_base = argus.idx_base;
    std::string binfile           = "";
    std::string filename          = "";
    rocsparse_status status;
    size_t size;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(m == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m       = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    rocsparse_mat_descr descr = test_descr->descr;

    std::unique_ptr<mat_info_struct> unique_ptr_mat_info(new mat_info_struct);
    rocsparse_mat_info info = unique_ptr_mat_info->info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, idx_base));

    // Determine number of non-zero elements
    double scale = 0.02;
    if(m > 1000)
    {
        scale = 2.0 / m;
    }
    rocsparse_int nnz = m * scale * m;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || nnz <= 0)
    {
        auto dptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dcol_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto buffer_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
        rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
        T* dval             = (T*)dval_managed.get();
        void* buffer        = (void*)buffer_managed.get();

        if(!dval || !dptr || !dcol || !buffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dptr || !dcol || !dval || !buffer");
            return rocsparse_status_memory_error;
        }

        // Test rocsparse_csrilu0_buffer_size
        status =
            rocsparse_csrilu0_buffer_size(handle, m, nnz, descr, dval, dptr, dcol, info, &size);

        if(m < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        // Test rocsparse_csrilu0_analysis
        status = rocsparse_csrilu0_analysis(handle,
                                            m,
                                            nnz,
                                            descr,
                                            dval,
                                            dptr,
                                            dcol,
                                            info,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            buffer);

        if(m < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        // Test rocsparse_csrilu0
        status = rocsparse_csrilu0(
            handle, m, nnz, descr, dval, dptr, dcol, info, rocsparse_solve_policy_auto, buffer);

        if(m < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        // Test rocsparse_csrilu0_zero_pivot
        rocsparse_int zero_pivot;
        CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_zero_pivot(handle, info, &zero_pivot));

        // Zero pivot should be -1
        rocsparse_int res = -1;
        unit_check_general(1, 1, 1, &res, &zero_pivot);

        // Test rocsparse_csrilu0_clear
        CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_clear(handle, info));

        return rocsparse_status_success;
    }

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr;
    std::vector<rocsparse_int> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), m, m, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        m   = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
        nnz = hcsr_row_ptr[m];
    }
    else
    {
        std::vector<rocsparse_int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, m, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base) !=
               0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(m, m, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr.resize(m + 1, 0);
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr[0] = idx_base;
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
        }
    }

    // Allocate memory on device
    auto dptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto d_position_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int)), device_free};

    rocsparse_int* dptr       = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol       = (rocsparse_int*)dcol_managed.get();
    T* dval                   = (T*)dval_managed.get();
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
    CHECK_ROCSPARSE_ERROR(
        rocsparse_csrilu0_buffer_size(handle, m, nnz, descr, dval, dptr, dcol, info, &size));

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
                                                     descr,
                                                     dval,
                                                     dptr,
                                                     dcol,
                                                     info,
                                                     rocsparse_analysis_policy_reuse,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer));

    if(argus.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0(
            handle, m, nnz, descr, dval, dptr, dcol, info, rocsparse_solve_policy_auto, dbuffer));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hposition_1;
        rocsparse_status pivot_status_1;
        pivot_status_1 = rocsparse_csrilu0_zero_pivot(handle, info, &hposition_1);

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        rocsparse_status pivot_status_2;
        pivot_status_2 = rocsparse_csrilu0_zero_pivot(handle, info, d_position);

        // Copy output from device to CPU
        rocsparse_int hposition_2;
        std::vector<T> result(nnz);
        CHECK_HIP_ERROR(hipMemcpy(result.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(&hposition_2, d_position, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // Host csrilu0
        double cpu_time_used = get_time_us();

        rocsparse_int position_gold =
            csrilu0(m, hcsr_row_ptr.data(), hcsr_col_ind.data(), hcsr_val.data(), idx_base);

        cpu_time_used = get_time_us() - cpu_time_used;

        unit_check_general(1, 1, 1, &position_gold, &hposition_1);
        unit_check_general(1, 1, 1, &position_gold, &hposition_2);

        if(hposition_1 != -1)
        {
            verify_rocsparse_status_zero_pivot(pivot_status_1,
                                               "expected rocsparse_status_zero_pivot");
            return rocsparse_status_success;
        }

        if(hposition_2 != -1)
        {
            verify_rocsparse_status_zero_pivot(pivot_status_2,
                                               "expected rocsparse_status_zero_pivot");
            return rocsparse_status_success;
        }

        unit_check_general(1, nnz, 1, hcsr_val.data(), result.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_csrilu0(handle,
                              m,
                              nnz,
                              descr,
                              dval,
                              dptr,
                              dcol,
                              info,
                              rocsparse_solve_policy_auto,
                              dbuffer);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocsparse_csrilu0(handle,
                              m,
                              nnz,
                              descr,
                              dval,
                              dptr,
                              dcol,
                              info,
                              rocsparse_solve_policy_auto,
                              dbuffer);
        }

        // Convert to miliseconds per call
        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        // Bandwidth
        size_t int_data  = (m + 1 + nnz) * sizeof(rocsparse_int);
        size_t flt_data  = (nnz + nnz) * sizeof(T);
        double bandwidth = (int_data + flt_data) / gpu_time_used / 1e6;

        printf("m\t\tnnz\t\tGB/s\tmsec\n");
        printf("%8d\t%9d\t%0.2lf\t%0.2lf\n", m, nnz, bandwidth, gpu_time_used);
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_clear(handle, info));

    return rocsparse_status_success;
}

#endif // TESTING_CSRILU0_HPP

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
#ifndef TESTING_CSR2CSC_HPP
#define TESTING_CSR2CSC_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <rocsparse.h>
#include <algorithm>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_csr2csc_bad_arg(void)
{
    rocsparse_int m         = 100;
    rocsparse_int n         = 100;
    rocsparse_int nnz       = 100;
    rocsparse_int safe_size = 100;
    rocsparse_status status;

    size_t size = 0;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    auto csr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csr_col_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csr_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csc_row_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csc_col_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csc_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto buffer_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();
    rocsparse_int* csr_col_ind = (rocsparse_int*)csr_col_ind_managed.get();
    T* csr_val                 = (T*)csr_val_managed.get();
    rocsparse_int* csc_row_ind = (rocsparse_int*)csc_row_ind_managed.get();
    rocsparse_int* csc_col_ptr = (rocsparse_int*)csc_col_ptr_managed.get();
    T* csc_val                 = (T*)csc_val_managed.get();
    void* buffer               = (void*)buffer_managed.get();

    if(!csr_row_ptr || !csr_col_ind || !csr_val || !csc_row_ind || !csc_col_ptr || !csc_val ||
       !buffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing rocsparse_csr2csc_buffer_size()

    // Testing for (csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_csr2csc_buffer_size(
            handle, m, n, nnz, csr_row_ptr_null, csr_col_ind, rocsparse_action_numeric, &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        rocsparse_int* csr_col_ind_null = nullptr;

        status = rocsparse_csr2csc_buffer_size(
            handle, m, n, nnz, csr_row_ptr, csr_col_ind_null, rocsparse_action_numeric, &size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (buffer_size == nullptr)
    {
        size_t* buffer_size_null = nullptr;

        status = rocsparse_csr2csc_buffer_size(handle,
                                               m,
                                               n,
                                               nnz,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               rocsparse_action_numeric,
                                               buffer_size_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csr2csc_buffer_size(
            handle_null, m, n, nnz, csr_row_ptr, csr_col_ind, rocsparse_action_numeric, &size);
        verify_rocsparse_status_invalid_handle(status);
    }

    // Testing rocsparse_csr2csc()

    // Testing for (csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr_null,
                                   csr_col_ind,
                                   csc_val,
                                   csc_row_ind,
                                   csc_col_ptr,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        rocsparse_int* csr_col_ind_null = nullptr;

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind_null,
                                   csc_val,
                                   csc_row_ind,
                                   csc_col_ptr,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (csr_val == nullptr)
    {
        T* csr_val_null = nullptr;

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val_null,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csc_val,
                                   csc_row_ind,
                                   csc_col_ptr,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (csc_row_ind == nullptr)
    {
        rocsparse_int* csc_row_ind_null = nullptr;

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csc_val,
                                   csc_row_ind_null,
                                   csc_col_ptr,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csc_row_ind is nullptr");
    }

    // Testing for (csc_col_ptr == nullptr)
    {
        rocsparse_int* csc_col_ptr_null = nullptr;

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csc_val,
                                   csc_row_ind,
                                   csc_col_ptr_null,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csc_col_ptr is nullptr");
    }

    // Testing for (csc_val == nullptr)
    {
        T* csc_val_null = nullptr;

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csc_val_null,
                                   csc_row_ind,
                                   csc_col_ptr,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csc_val is nullptr");
    }

    // Testing for (buffer == nullptr)
    {
        void* buffer_null = nullptr;

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csc_val,
                                   csc_row_ind,
                                   csc_col_ptr,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csr2csc(handle_null,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csc_val,
                                   csc_row_ind,
                                   csc_col_ptr,
                                   rocsparse_action_numeric,
                                   rocsparse_index_base_zero,
                                   buffer);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_csr2csc(Arguments argus)
{
    rocsparse_int m               = argus.M;
    rocsparse_int n               = argus.N;
    rocsparse_int safe_size       = 100;
    rocsparse_index_base idx_base = argus.idx_base;
    rocsparse_action action       = argus.action;
    std::string binfile           = "";
    std::string filename          = "";
    rocsparse_status status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(m == -99 && n == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m = n = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    size_t size = 0;

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    rocsparse_int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto csr_row_ptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csr_col_ind_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csr_val_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto csc_row_ind_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csc_col_ptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csc_val_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto buffer_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();
        rocsparse_int* csr_col_ind = (rocsparse_int*)csr_col_ind_managed.get();
        T* csr_val                 = (T*)csr_val_managed.get();
        rocsparse_int* csc_row_ind = (rocsparse_int*)csc_row_ind_managed.get();
        rocsparse_int* csc_col_ptr = (rocsparse_int*)csc_col_ptr_managed.get();
        T* csc_val                 = (T*)csc_val_managed.get();
        void* buffer               = (void*)buffer_managed.get();

        if(!csr_row_ptr || !csr_col_ind || !csr_val || !csc_row_ind || !csc_col_ptr || !csc_val ||
           !buffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!csr_row_ptr || !csr_col_ind || !csr_val || "
                                            "!csc_row_ind || !csc_col_ptr || !csc_val || !buffer");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_csr2csc_buffer_size(
            handle, m, n, nnz, csr_row_ptr, csr_col_ind, action, &size);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");

            // Buffer size should be zero
            size_t four = 4;
            unit_check_general(1, 1, 1, &four, &size);
        }

        status = rocsparse_csr2csc(handle,
                                   m,
                                   n,
                                   nnz,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csc_val,
                                   csc_row_ind,
                                   csc_col_ptr,
                                   action,
                                   idx_base,
                                   buffer);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");
        }

        return rocsparse_status_success;
    }

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr;
    std::vector<rocsparse_int> hcsr_col_ind;
    std::vector<T> hcsr_val;

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        std::vector<rocsparse_int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base) !=
               0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base);
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

    // Allocate memory on the device
    auto dcsr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dcsr_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dcsc_row_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dcsc_col_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (n + 1)), device_free};
    auto dcsc_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    rocsparse_int* dcsr_row_ptr = (rocsparse_int*)dcsr_row_ptr_managed.get();
    rocsparse_int* dcsr_col_ind = (rocsparse_int*)dcsr_col_ind_managed.get();
    T* dcsr_val                 = (T*)dcsr_val_managed.get();
    rocsparse_int* dcsc_row_ind = (rocsparse_int*)dcsc_row_ind_managed.get();
    rocsparse_int* dcsc_col_ptr = (rocsparse_int*)dcsc_col_ptr_managed.get();
    T* dcsc_val                 = (T*)dcsc_val_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dcsc_row_ind || !dcsc_col_ptr || !dcsc_val)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || "
                                        "!dcsc_row_ind || !dcsc_col_ptr || !dcsc_val");
        return rocsparse_status_memory_error;
    }

    // Reset CSC arrays
    CHECK_HIP_ERROR(hipMemset(dcsc_row_ind, 0, sizeof(rocsparse_int) * nnz));
    CHECK_HIP_ERROR(hipMemset(dcsc_col_ptr, 0, sizeof(rocsparse_int) * (n + 1)));
    CHECK_HIP_ERROR(hipMemset(dcsc_val, 0, sizeof(T) * nnz));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain buffer size
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size(
        handle, m, n, nnz, dcsr_row_ptr, dcsr_col_ind, action, &size));

    // Allocate buffer on the device
    auto dbuffer_managed = rocsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dbuffer");
        return rocsparse_status_memory_error;
    }

    if(argus.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2csc(handle,
                                                m,
                                                n,
                                                nnz,
                                                dcsr_val,
                                                dcsr_row_ptr,
                                                dcsr_col_ind,
                                                dcsc_val,
                                                dcsc_row_ind,
                                                dcsc_col_ptr,
                                                action,
                                                idx_base,
                                                dbuffer));

        // Copy output from device to host
        std::vector<rocsparse_int> hcsc_row_ind(nnz);
        std::vector<rocsparse_int> hcsc_col_ptr(n + 1);
        std::vector<T> hcsc_val(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsc_row_ind.data(), dcsc_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsc_col_ptr.data(),
                                  dcsc_col_ptr,
                                  sizeof(rocsparse_int) * (n + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsc_val.data(), dcsc_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Host csr2csc conversion
        std::vector<rocsparse_int> hcsc_row_ind_gold(nnz);
        std::vector<rocsparse_int> hcsc_col_ptr_gold(n + 1, 0);
        std::vector<T> hcsc_val_gold(nnz);

        // Determine nnz per column
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++hcsc_col_ptr_gold[hcsr_col_ind[i] + 1 - idx_base];
        }

        // Scan
        for(rocsparse_int i = 0; i < n; ++i)
        {
            hcsc_col_ptr_gold[i + 1] += hcsc_col_ptr_gold[i];
        }

        // Fill row indices and values
        for(rocsparse_int i = 0; i < m; ++i)
        {
            for(rocsparse_int j = hcsr_row_ptr[i]; j < hcsr_row_ptr[i + 1]; ++j)
            {
                rocsparse_int col = hcsr_col_ind[j - idx_base] - idx_base;
                rocsparse_int idx = hcsc_col_ptr_gold[col];

                hcsc_row_ind_gold[idx] = i + idx_base;
                hcsc_val_gold[idx]     = hcsr_val[j - idx_base];

                ++hcsc_col_ptr_gold[col];
            }
        }

        // Shift column pointer array
        for(rocsparse_int i = n; i > 0; --i)
        {
            hcsc_col_ptr_gold[i] = hcsc_col_ptr_gold[i - 1] + idx_base;
        }

        hcsc_col_ptr_gold[0] = idx_base;

        // Unit check
        unit_check_general(1, nnz, 1, hcsc_row_ind_gold.data(), hcsc_row_ind.data());
        unit_check_general(1, n + 1, 1, hcsc_col_ptr_gold.data(), hcsc_col_ptr.data());

        // If action == rocsparse_action_numeric also check values
        if(action == rocsparse_action_numeric)
        {
            unit_check_general(1, nnz, 1, hcsc_val_gold.data(), hcsc_val.data());
        }
    }

    if(argus.timing)
    {
        rocsparse_int number_cold_calls = 2;
        rocsparse_int number_hot_calls  = argus.iters;

        for(rocsparse_int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_csr2csc(handle,
                              m,
                              n,
                              nnz,
                              dcsr_val,
                              dcsr_row_ptr,
                              dcsr_col_ind,
                              dcsc_val,
                              dcsc_row_ind,
                              dcsc_col_ptr,
                              rocsparse_action_numeric,
                              rocsparse_index_base_zero,
                              dbuffer);
        }

        double gpu_time_used = get_time_us();

        for(rocsparse_int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_csr2csc(handle,
                              m,
                              n,
                              nnz,
                              dcsr_val,
                              dcsr_row_ptr,
                              dcsr_col_ind,
                              dcsc_val,
                              dcsc_row_ind,
                              dcsc_col_ptr,
                              rocsparse_action_numeric,
                              rocsparse_index_base_zero,
                              dbuffer);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        printf("m\t\tn\t\tnnz\t\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\n", m, n, nnz, gpu_time_used);
    }

    return rocsparse_status_success;
}

#endif // TESTING_CSR2CSC_HPP

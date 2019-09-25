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
#ifndef TESTING_CSCSORT_HPP
#define TESTING_CSCSORT_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <rocsparse.h>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

void testing_cscsort_bad_arg(void)
{
    rocsparse_int    m         = 100;
    rocsparse_int    n         = 100;
    rocsparse_int    nnz       = 100;
    rocsparse_int    safe_size = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr           descr = unique_ptr_descr->descr;

    size_t buffer_size = 0;

    auto csc_col_ptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csc_row_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto perm_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto buffer_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    rocsparse_int* csc_col_ptr = (rocsparse_int*)csc_col_ptr_managed.get();
    rocsparse_int* csc_row_ind = (rocsparse_int*)csc_row_ind_managed.get();
    rocsparse_int* perm        = (rocsparse_int*)perm_managed.get();
    void*          buffer      = (void*)buffer_managed.get();

    if(!csc_col_ptr || !csc_row_ind || !perm || !buffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing cscsort_buffer_size for bad args

    // Testing for (csc_col_ptr == nullptr)
    {
        rocsparse_int* csc_col_ptr_null = nullptr;

        status = rocsparse_cscsort_buffer_size(
            handle, m, n, nnz, csc_col_ptr_null, csc_row_ind, &buffer_size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csc_col_ptr is nullptr");
    }

    // Testing for (csc_row_ind == nullptr)
    {
        rocsparse_int* csc_row_ind_null = nullptr;

        status = rocsparse_cscsort_buffer_size(
            handle, m, n, nnz, csc_col_ptr, csc_row_ind_null, &buffer_size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csc_row_ind is nullptr");
    }

    // Testing for (buffer_size == nullptr)
    {
        size_t* buffer_size_null = nullptr;

        status = rocsparse_cscsort_buffer_size(
            handle, m, n, nnz, csc_col_ptr, csc_row_ind, buffer_size_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_cscsort_buffer_size(
            handle_null, m, n, nnz, csc_col_ptr, csc_row_ind, &buffer_size);
        verify_rocsparse_status_invalid_handle(status);
    }

    // Testing cscsort for bad args

    // Testing for (csc_col_ptr == nullptr)
    {
        rocsparse_int* csc_col_ptr_null = nullptr;

        status = rocsparse_cscsort(
            handle, m, n, nnz, descr, csc_col_ptr_null, csc_row_ind, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csc_col_ptr is nullptr");
    }

    // Testing for (csc_row_ind == nullptr)
    {
        rocsparse_int* csc_row_ind_null = nullptr;

        status = rocsparse_cscsort(
            handle, m, n, nnz, descr, csc_col_ptr, csc_row_ind_null, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csc_row_ind is nullptr");
    }

    // Testing for (buffer == nullptr)
    {
        rocsparse_int* buffer_null = nullptr;

        status = rocsparse_cscsort(
            handle, m, n, nnz, descr, csc_col_ptr, csc_row_ind, perm, buffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
    }

    // Testing for (descr == nullptr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_cscsort(
            handle, m, n, nnz, descr_null, csc_col_ptr, csc_row_ind, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_cscsort(
            handle_null, m, n, nnz, descr, csc_col_ptr, csc_row_ind, perm, buffer);
        verify_rocsparse_status_invalid_handle(status);
    }
}

rocsparse_status testing_cscsort(Arguments argus)
{
    rocsparse_int        m         = argus.M;
    rocsparse_int        n         = argus.N;
    rocsparse_int        safe_size = 100;
    rocsparse_int        permute   = argus.temp;
    rocsparse_index_base idx_base  = argus.idx_base;
    std::string          binfile   = "";
    std::string          filename  = "";
    rocsparse_status     status;

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

    size_t buffer_size = 0;

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    rocsparse_int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr           descr = unique_ptr_descr->descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, idx_base));

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto csc_col_ptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csc_row_ind_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto perm_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto buffer_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        rocsparse_int* csc_col_ptr = (rocsparse_int*)csc_col_ptr_managed.get();
        rocsparse_int* csc_row_ind = (rocsparse_int*)csc_row_ind_managed.get();
        rocsparse_int* perm        = (rocsparse_int*)perm_managed.get();
        void*          buffer      = (void*)buffer_managed.get();

        if(!csc_col_ptr || !csc_row_ind || !perm || !buffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!csc_col_ptr || !csc_row_ind || !perm || !buffer");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_cscsort_buffer_size(
            handle, m, n, nnz, csc_col_ptr, csc_row_ind, &buffer_size);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");

            // Buffer size should be 4
            size_t four = 4;
            unit_check_general(1, 1, 1, &four, &buffer_size);
        }

        status
            = rocsparse_cscsort(handle, m, n, nnz, descr, csc_col_ptr, csc_row_ind, perm, buffer);

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

    // For testing, assemble a COO matrix and convert it to CSC first (on host)

    // Host structures
    std::vector<rocsparse_int> hcsc_col_ptr;
    std::vector<rocsparse_int> hcoo_col_ind;
    std::vector<rocsparse_int> hcsc_row_ind;
    std::vector<float>         hcsc_val;

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), n, m, nnz, hcsc_col_ptr, hcsc_row_ind, hcsc_val, idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        n = m = gen_2d_laplacian(argus.laplacian, hcsc_col_ptr, hcsc_row_ind, hcsc_val, idx_base);
        nnz   = hcsc_col_ptr[n];
    }
    else
    {
        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), n, m, nnz, hcoo_col_ind, hcsc_row_ind, hcsc_val, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(n, m, nnz, hcoo_col_ind, hcsc_row_ind, hcsc_val, idx_base);
        }

        // Convert COO to CSC
        hcsc_col_ptr.resize(n + 1, 0);
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++hcsc_col_ptr[hcoo_col_ind[i] + 1 - idx_base];
        }

        hcsc_col_ptr[0] = idx_base;
        for(rocsparse_int i = 0; i < n; ++i)
        {
            hcsc_col_ptr[i + 1] += hcsc_col_ptr[i];
        }
    }

    // Unsort CSC columns
    std::vector<rocsparse_int> hperm(nnz);
    std::vector<rocsparse_int> hcsc_row_ind_unsorted(nnz);
    std::vector<float>         hcsc_val_unsorted(nnz);

    hcsc_row_ind_unsorted = hcsc_row_ind;
    hcsc_val_unsorted     = hcsc_val;

    for(rocsparse_int i = 0; i < n; ++i)
    {
        rocsparse_int col_begin = hcsc_col_ptr[i] - idx_base;
        rocsparse_int col_end   = hcsc_col_ptr[i + 1] - idx_base;
        rocsparse_int col_nnz   = col_end - col_begin;

        for(rocsparse_int j = col_begin; j < col_end; ++j)
        {
            rocsparse_int rng = col_begin + rand() % col_nnz;

            rocsparse_int temp_row = hcsc_row_ind_unsorted[j];
            float         temp_val = hcsc_val_unsorted[j];

            hcsc_row_ind_unsorted[j] = hcsc_row_ind_unsorted[rng];
            hcsc_val_unsorted[j]     = hcsc_val_unsorted[rng];

            hcsc_row_ind_unsorted[rng] = temp_row;
            hcsc_val_unsorted[rng]     = temp_val;
        }
    }

    // Allocate memory on the device
    auto dcsc_col_ptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (n + 1)), device_free};
    auto dcsc_row_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dcsc_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dcsc_val_sorted_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dperm_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};

    rocsparse_int* dcsc_col_ptr    = (rocsparse_int*)dcsc_col_ptr_managed.get();
    rocsparse_int* dcsc_row_ind    = (rocsparse_int*)dcsc_row_ind_managed.get();
    float*         dcsc_val        = (float*)dcsc_val_managed.get();
    float*         dcsc_val_sorted = (float*)dcsc_val_sorted_managed.get();

    // Set permutation vector, if asked for
    rocsparse_int* dperm = permute ? (rocsparse_int*)dperm_managed.get() : nullptr;

    if(!dcsc_col_ptr || !dcsc_row_ind || !dcsc_val || !dcsc_val_sorted || (permute && !dperm))
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcsc_col_ptr || !dcsc_row_ind || !dcsc_val || "
                                        "!dcsc_val_sorted || (permute && !dperm)");
        return rocsparse_status_memory_error;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsc_col_ptr, hcsc_col_ptr.data(), sizeof(rocsparse_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsc_row_ind,
                              hcsc_row_ind_unsorted.data(),
                              sizeof(rocsparse_int) * nnz,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_val, hcsc_val_unsorted.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Obtain buffer size
        CHECK_ROCSPARSE_ERROR(rocsparse_cscsort_buffer_size(
            handle, m, n, nnz, dcsc_col_ptr, dcsc_row_ind, &buffer_size));

        // Allocate buffer on the device
        auto dbuffer_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};

        void* dbuffer = (void*)dbuffer_managed.get();

        if(!dbuffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error, "!dbuffer");
            return rocsparse_status_memory_error;
        }

        if(permute)
        {
            // Initialize perm with identity permutation
            CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnz, dperm));
        }

        // Sort CSC columns
        CHECK_ROCSPARSE_ERROR(rocsparse_cscsort(
            handle, m, n, nnz, descr, dcsc_col_ptr, dcsc_row_ind, dperm, dbuffer));

        if(permute)
        {
            // Sort CSC values
            CHECK_ROCSPARSE_ERROR(rocsparse_sgthr(
                handle, nnz, dcsc_val, dcsc_val_sorted, dperm, rocsparse_index_base_zero));
        }

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hcsc_row_ind_unsorted.data(),
                                  dcsc_row_ind,
                                  sizeof(rocsparse_int) * nnz,
                                  hipMemcpyDeviceToHost));

        if(permute)
        {
            CHECK_HIP_ERROR(hipMemcpy(hcsc_val_unsorted.data(),
                                      dcsc_val_sorted,
                                      sizeof(float) * nnz,
                                      hipMemcpyDeviceToHost));
        }

        // Unit check
        unit_check_general(1, nnz, 1, hcsc_row_ind.data(), hcsc_row_ind_unsorted.data());

        if(permute)
        {
            unit_check_general(1, nnz, 1, hcsc_val.data(), hcsc_val_unsorted.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Allocate buffer for cscsort
        rocsparse_cscsort_buffer_size(handle, m, n, nnz, dcsc_col_ptr, dcsc_row_ind, &buffer_size);

        auto dbuffer_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};
        void* dbuffer = (void*)dbuffer_managed.get();

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_cscsort(
                handle, m, n, nnz, descr, dcsc_col_ptr, dcsc_row_ind, nullptr, dbuffer);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_cscsort(
                handle, m, n, nnz, descr, dcsc_col_ptr, dcsc_row_ind, nullptr, dbuffer);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);
        std::cout << std::setw(12) << "m" << std::setw(12) << "n" << std::setw(12) << "nnz"
                  << std::setw(12) << "msec" << std::endl;
        std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz
                  << std::setw(12) << gpu_time_used << std::endl;
    }

    return rocsparse_status_success;
}

#endif // TESTING_CSCSORT_HPP

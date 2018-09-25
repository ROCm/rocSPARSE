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
#ifndef TESTING_CSRSORT_HPP
#define TESTING_CSRSORT_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <rocsparse.h>
#include <algorithm>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

void testing_csrsort_bad_arg(void)
{
    rocsparse_int m         = 100;
    rocsparse_int n         = 100;
    rocsparse_int nnz       = 100;
    rocsparse_int safe_size = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr descr = unique_ptr_descr->descr;

    size_t buffer_size = 0;

    auto csr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csr_col_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto perm_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto buffer_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();
    rocsparse_int* csr_col_ind = (rocsparse_int*)csr_col_ind_managed.get();
    rocsparse_int* perm        = (rocsparse_int*)perm_managed.get();
    void* buffer               = (void*)buffer_managed.get();

    if(!csr_row_ptr || !csr_col_ind || !perm || !buffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing csrsort_buffer_size for bad args

    // Testing for (csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_csrsort_buffer_size(
            handle, m, n, nnz, csr_row_ptr_null, csr_col_ind, &buffer_size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        rocsparse_int* csr_col_ind_null = nullptr;

        status = rocsparse_csrsort_buffer_size(
            handle, m, n, nnz, csr_row_ptr, csr_col_ind_null, &buffer_size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (buffer_size == nullptr)
    {
        size_t* buffer_size_null = nullptr;

        status = rocsparse_csrsort_buffer_size(
            handle, m, n, nnz, csr_row_ptr, csr_col_ind, buffer_size_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrsort_buffer_size(
            handle_null, m, n, nnz, csr_row_ptr, csr_col_ind, &buffer_size);
        verify_rocsparse_status_invalid_handle(status);
    }

    // Testing csrsort for bad args

    // Testing for (csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_csrsort(
            handle, m, n, nnz, descr, csr_row_ptr_null, csr_col_ind, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        rocsparse_int* csr_col_ind_null = nullptr;

        status = rocsparse_csrsort(
            handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind_null, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (buffer == nullptr)
    {
        rocsparse_int* buffer_null = nullptr;

        status = rocsparse_csrsort(
            handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, buffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
    }

    // Testing for (descr == nullptr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrsort(
            handle, m, n, nnz, descr_null, csr_row_ptr, csr_col_ind, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrsort(
            handle_null, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, buffer);
        verify_rocsparse_status_invalid_handle(status);
    }
}

rocsparse_status testing_csrsort(Arguments argus)
{
    rocsparse_int m               = argus.M;
    rocsparse_int n               = argus.N;
    rocsparse_int safe_size       = 100;
    rocsparse_int permute         = argus.temp;
    rocsparse_index_base idx_base = argus.idx_base;
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

    size_t buffer_size = 0;

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    rocsparse_int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr descr = unique_ptr_descr->descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, idx_base));

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto csr_row_ptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csr_col_ind_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto perm_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto buffer_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

        rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();
        rocsparse_int* csr_col_ind = (rocsparse_int*)csr_col_ind_managed.get();
        rocsparse_int* perm        = (rocsparse_int*)perm_managed.get();
        void* buffer               = (void*)buffer_managed.get();

        if(!csr_row_ptr || !csr_col_ind || !perm || !buffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!csr_row_ptr || !csr_col_ind || !perm || !buffer");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_csrsort_buffer_size(
            handle, m, n, nnz, csr_row_ptr, csr_col_ind, &buffer_size);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");

            // Buffer size should be zero
            size_t zero = 0;
            unit_check_general(1, 1, 1, &zero, &buffer_size);
        }

        status =
            rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, buffer);

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

    // For testing, assemble a COO matrix and convert it to CSR first (on host)

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr;
    std::vector<rocsparse_int> hcoo_row_ind;
    std::vector<rocsparse_int> hcsr_col_ind;
    std::vector<float> hcsr_val;

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

    // Unsort CSR columns
    std::vector<rocsparse_int> hperm(nnz);
    std::vector<rocsparse_int> hcsr_col_ind_unsorted(nnz);
    std::vector<float> hcsr_val_unsorted(nnz);

    hcsr_col_ind_unsorted = hcsr_col_ind;
    hcsr_val_unsorted     = hcsr_val;

    for(rocsparse_int i = 0; i < m; ++i)
    {
        rocsparse_int row_begin = hcsr_row_ptr[i] - idx_base;
        rocsparse_int row_end   = hcsr_row_ptr[i + 1] - idx_base;
        rocsparse_int row_nnz   = row_end - row_begin;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            rocsparse_int rng = row_begin + rand() % row_nnz;

            rocsparse_int temp_col = hcsr_col_ind_unsorted[j];
            float temp_val         = hcsr_val_unsorted[j];

            hcsr_col_ind_unsorted[j] = hcsr_col_ind_unsorted[rng];
            hcsr_val_unsorted[j]     = hcsr_val_unsorted[rng];

            hcsr_col_ind_unsorted[rng] = temp_col;
            hcsr_val_unsorted[rng]     = temp_val;
        }
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dcsr_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dcsr_val_sorted_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dperm_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};

    rocsparse_int* dcsr_row_ptr = (rocsparse_int*)dcsr_row_ptr_managed.get();
    rocsparse_int* dcsr_col_ind = (rocsparse_int*)dcsr_col_ind_managed.get();
    float* dcsr_val             = (float*)dcsr_val_managed.get();
    float* dcsr_val_sorted      = (float*)dcsr_val_sorted_managed.get();

    // Set permutation vector, if asked for
    rocsparse_int* dperm = permute ? (rocsparse_int*)dperm_managed.get() : nullptr;

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dcsr_val_sorted || (permute && !dperm))
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || "
                                        "!dcsr_val_sorted || (permute && !dperm)");
        return rocsparse_status_memory_error;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_col_ind,
                              hcsr_col_ind_unsorted.data(),
                              sizeof(rocsparse_int) * nnz,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val, hcsr_val_unsorted.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Obtain buffer size
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsort_buffer_size(
            handle, m, n, nnz, dcsr_row_ptr, dcsr_col_ind, &buffer_size));

        // Allocate buffer on the device
        auto dbuffer_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};

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

        // Sort CSR columns
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsort(
            handle, m, n, nnz, descr, dcsr_row_ptr, dcsr_col_ind, dperm, dbuffer));

        if(permute)
        {
            // Sort CSR values
            CHECK_ROCSPARSE_ERROR(rocsparse_sgthr(
                handle, nnz, dcsr_val, dcsr_val_sorted, dperm, rocsparse_index_base_zero));
        }

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_unsorted.data(),
                                  dcsr_col_ind,
                                  sizeof(rocsparse_int) * nnz,
                                  hipMemcpyDeviceToHost));

        if(permute)
        {
            CHECK_HIP_ERROR(hipMemcpy(hcsr_val_unsorted.data(),
                                      dcsr_val_sorted,
                                      sizeof(float) * nnz,
                                      hipMemcpyDeviceToHost));
        }

        // Unit check
        unit_check_general(1, nnz, 1, hcsr_col_ind.data(), hcsr_col_ind_unsorted.data());

        if(permute)
        {
            unit_check_general(1, nnz, 1, hcsr_val.data(), hcsr_val_unsorted.data());
        }
    }

    if(argus.timing)
    {
        rocsparse_int number_cold_calls = 2;
        rocsparse_int number_hot_calls  = argus.iters;

        // Allocate buffer for csrsort
        rocsparse_csrsort_buffer_size(handle, m, n, nnz, dcsr_row_ptr, dcsr_col_ind, &buffer_size);

        auto dbuffer_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};
        void* dbuffer = (void*)dbuffer_managed.get();

        for(rocsparse_int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_csrsort(
                handle, m, n, nnz, descr, dcsr_row_ptr, dcsr_col_ind, nullptr, dbuffer);
        }

        double gpu_time_used = get_time_us();

        for(rocsparse_int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_csrsort(
                handle, m, n, nnz, descr, dcsr_row_ptr, dcsr_col_ind, nullptr, dbuffer);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        printf("m\t\tn\t\tnnz\t\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\n", m, n, nnz, gpu_time_used);
    }
    return rocsparse_status_success;
}

#endif // TESTING_CSRSORT_HPP

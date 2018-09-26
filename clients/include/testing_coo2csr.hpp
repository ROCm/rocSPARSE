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
#ifndef TESTING_COO2CSR_HPP
#define TESTING_COO2CSR_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <rocsparse.h>
#include <algorithm>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

void testing_coo2csr_bad_arg(void)
{
    rocsparse_int m         = 100;
    rocsparse_int nnz       = 100;
    rocsparse_int safe_size = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    auto coo_row_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

    rocsparse_int* coo_row_ind = (rocsparse_int*)coo_row_ind_managed.get();
    rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();

    if(!coo_row_ind || !csr_row_ptr)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for(coo_row_ind == nullptr)
    {
        rocsparse_int* coo_row_ind_null = nullptr;

        status = rocsparse_coo2csr(
            handle, coo_row_ind_null, nnz, m, csr_row_ptr, rocsparse_index_base_zero);
        verify_rocsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }
    // Testing for(csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_coo2csr(
            handle, coo_row_ind, nnz, m, csr_row_ptr_null, rocsparse_index_base_zero);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }
    // Testing for(handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_coo2csr(
            handle_null, coo_row_ind, nnz, m, csr_row_ptr, rocsparse_index_base_zero);
        verify_rocsparse_status_invalid_handle(status);
    }
}

rocsparse_status testing_coo2csr(Arguments argus)
{
    rocsparse_int m               = argus.M;
    rocsparse_int n               = argus.N;
    rocsparse_int safe_size       = 100;
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
        auto coo_row_ind_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csr_row_ptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

        rocsparse_int* coo_row_ind = (rocsparse_int*)coo_row_ind_managed.get();
        rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();

        if(!coo_row_ind || !csr_row_ptr)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!coo_row_ind || !csr_row_ptr");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_coo2csr(handle, coo_row_ind, nnz, m, csr_row_ptr, idx_base);

        if(m < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");
        }

        return rocsparse_status_success;
    }

    // Host structures
    std::vector<rocsparse_int> hcoo_row_ind;
    std::vector<rocsparse_int> hcoo_col_ind;
    std::vector<float> hcoo_val;

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        std::vector<rocsparse_int> hptr(m + 1);
        if(read_bin_matrix(binfile.c_str(), m, n, nnz, hptr, hcoo_col_ind, hcoo_val, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }

        hcoo_row_ind.resize(nnz);

        // Convert to COO
        for(rocsparse_int i = 0; i < m; ++i)
        {
            for(rocsparse_int j = hptr[i]; j < hptr[i + 1]; ++j)
            {
                hcoo_row_ind[j - idx_base] = i + idx_base;
            }
        }
    }
    else if(argus.laplacian)
    {
        std::vector<rocsparse_int> hptr(m + 1);
        m = n = gen_2d_laplacian(argus.laplacian, hptr, hcoo_col_ind, hcoo_val, idx_base);
        nnz   = hptr[m];
        hcoo_row_ind.resize(nnz);

        // Convert to COO
        for(rocsparse_int i = 0; i < m; ++i)
        {
            for(rocsparse_int j = hptr[i]; j < hptr[i + 1]; ++j)
            {
                hcoo_row_ind[j - idx_base] = i + idx_base;
            }
        }
    }
    else
    {
        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, n, nnz, hcoo_row_ind, hcoo_col_ind, hcoo_val, idx_base) !=
               0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcoo_col_ind, hcoo_val, idx_base);
        }
    }

    std::vector<rocsparse_int> hcsr_row_ptr(m + 1);
    std::vector<rocsparse_int> hcsr_row_ptr_gold(m + 1, 0);

    // Allocate memory on the device
    auto dcoo_row_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dcsr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};

    rocsparse_int* dcoo_row_ind = (rocsparse_int*)dcoo_row_ind_managed.get();
    rocsparse_int* dcsr_row_ptr = (rocsparse_int*)dcsr_row_ptr_managed.get();

    if(!dcoo_row_ind || !dcsr_row_ptr)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcoo_row_ind || !dcsr_row_ptr");
        return rocsparse_status_memory_error;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcoo_row_ind, hcoo_row_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, dcoo_row_ind, nnz, m, dcsr_row_ptr, idx_base));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr.data(),
                                  dcsr_row_ptr,
                                  sizeof(rocsparse_int) * (m + 1),
                                  hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        // coo2csr on host
        for(int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr_gold[hcoo_row_ind[i] + 1 - idx_base];
        }

        hcsr_row_ptr_gold[0] = idx_base;
        for(int i = 0; i < m; ++i)
        {
            hcsr_row_ptr_gold[i + 1] += hcsr_row_ptr_gold[i];
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
    }

    if(argus.timing)
    {
        rocsparse_int number_cold_calls = 2;
        rocsparse_int number_hot_calls  = argus.iters;

        for(rocsparse_int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_coo2csr(handle, dcoo_row_ind, nnz, m, dcsr_row_ptr, idx_base);
        }

        double gpu_time_used = get_time_us();

        for(rocsparse_int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_coo2csr(handle, dcoo_row_ind, nnz, m, dcsr_row_ptr, idx_base);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        double bandwidth = sizeof(rocsparse_int) * (nnz + m + 1) / gpu_time_used / 1e6;

        printf("m\t\tn\t\tnnz\t\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\n", m, n, nnz, bandwidth, gpu_time_used);
    }
    return rocsparse_status_success;
}

#endif // TESTING_COO2CSR_HPP

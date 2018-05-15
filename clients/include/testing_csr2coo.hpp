/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_CSR2COO_HPP
#define TESTING_CSR2COO_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <rocsparse.h>
#include <algorithm>

using namespace rocsparse;
using namespace rocsparse_test;

void testing_csr2coo_bad_arg(void)
{
    rocsparse_int m         = 100;
    rocsparse_int nnz       = 100;
    rocsparse_int safe_size = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    auto csr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto coo_row_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

    rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();
    rocsparse_int* coo_row_ind = (rocsparse_int*)coo_row_ind_managed.get();

    if(!csr_row_ptr || !coo_row_ind)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing for(csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_csr2coo(
            handle, csr_row_ptr_null, nnz, m, coo_row_ind, rocsparse_index_base_zero);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }
    // Testing for(coo_row_ind == nullptr)
    {
        rocsparse_int* coo_row_ind_null = nullptr;

        status = rocsparse_csr2coo(
            handle, csr_row_ptr, nnz, m, coo_row_ind_null, rocsparse_index_base_zero);
        verify_rocsparse_status_invalid_pointer(status, "Error: coo_row_ind is nullptr");
    }
    // Testing for(handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csr2coo(
            handle_null, csr_row_ptr, nnz, m, coo_row_ind, rocsparse_index_base_zero);
        verify_rocsparse_status_invalid_handle(status);
    }
}

rocsparse_status testing_csr2coo(Arguments argus)
{
    rocsparse_int m               = argus.M;
    rocsparse_int n               = argus.N;
    rocsparse_int safe_size       = 100;
    rocsparse_index_base idx_base = argus.idx_base;
    rocsparse_status status;

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
        auto coo_row_ind_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

        rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();
        rocsparse_int* coo_row_ind = (rocsparse_int*)coo_row_ind_managed.get();

        if(!csr_row_ptr || !coo_row_ind)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!csr_row_ptr || !coo_row_ind");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_csr2coo(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base);

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

    // For testing, assemble a COO matrix and convert it to CSR first (on host)

    // Host structures
    std::vector<rocsparse_int> hcoo_row_ind(nnz);
    std::vector<rocsparse_int> hcoo_row_ind_gold(nnz);
    std::vector<rocsparse_int> hcoo_col_ind(nnz);
    std::vector<float> hcoo_val(nnz);

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    gen_matrix_coo(m, n, nnz, hcoo_row_ind_gold, hcoo_col_ind, hcoo_val, idx_base);

    // Convert COO to CSR
    std::vector<rocsparse_int> hcsr_row_ptr(m + 1);

    // csr2coo on host
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        ++hcsr_row_ptr[hcoo_row_ind_gold[i] + 1 - idx_base];
    }

    hcsr_row_ptr[0] = idx_base;
    for(rocsparse_int i = 0; i < m; ++i)
    {
        hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
    }

    // Allocate memory on the device
    auto dcsr_row_ptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcoo_row_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};

    rocsparse_int* dcsr_row_ptr = (rocsparse_int*)dcsr_row_ptr_managed.get();
    rocsparse_int* dcoo_row_ind = (rocsparse_int*)dcoo_row_ind_managed.get();

    if(!dcsr_row_ptr || !dcoo_row_ind)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcsr_row_ptr || !dcoo_row_ind");
        return rocsparse_status_memory_error;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_row_ind.data(), dcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, nnz, hcoo_row_ind_gold.data(), hcoo_row_ind.data());
    }

    if(argus.timing)
    {
        rocsparse_int number_cold_calls = 2;
        rocsparse_int number_hot_calls  = argus.iters;

        for(rocsparse_int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_csr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        double gpu_time_used = get_time_us();

        for(rocsparse_int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_csr2coo(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        double bandwidth = sizeof(rocsparse_int) * (nnz + m + 1) / gpu_time_used / 1e6;

        printf("m\t\tn\t\tnnz\t\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\n", m, n, nnz, bandwidth, gpu_time_used);
    }
    return rocsparse_status_success;
}

#endif // TESTING_CSR2COO_HPP

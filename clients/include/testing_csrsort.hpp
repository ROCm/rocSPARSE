/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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

        status = rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr_null, csr_col_ind, &buffer_size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        rocsparse_int* csr_col_ind_null = nullptr;

        status = rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, csr_col_ind_null, &buffer_size);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (buffer_size == nullptr)
    {
        size_t* buffer_size_null = nullptr;

        status = rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, csr_col_ind, buffer_size_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrsort_buffer_size(handle_null, m, n, nnz, csr_row_ptr, csr_col_ind, &buffer_size);
        verify_rocsparse_status_invalid_handle(status);
    }

    // Testing csrsort for bad args

    // Testing for (csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr_null, csr_col_ind, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        rocsparse_int* csr_col_ind_null = nullptr;

        status = rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind_null, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (perm == nullptr)
    {
        rocsparse_int* perm_null = nullptr;

        status = rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm_null, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: perm is nullptr");
    }

    // Testing for (buffer == nullptr)
    {
        rocsparse_int* buffer_null = nullptr;

        status = rocsparse_csrsort(handle, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, buffer_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
    }

    // Testing for (descr == nullptr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrsort(handle, m, n, nnz, descr_null, csr_row_ptr, csr_col_ind, perm, buffer);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrsort(handle_null, m, n, nnz, descr, csr_row_ptr, csr_col_ind, perm, buffer);
        verify_rocsparse_status_invalid_handle(status);
    }
}

rocsparse_status testing_csrsort(Arguments argus)
{
    rocsparse_int m               = argus.M;
    rocsparse_int n               = argus.N;
    rocsparse_int safe_size       = 100;
    rocsparse_status status;

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

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto csr_row_ptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csr_col_ind_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

        rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();
        rocsparse_int* csr_col_ind = (rocsparse_int*)csr_col_ind_managed.get();

        if(!csr_row_ptr || !csr_col_ind)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!csr_row_ptr || !csr_col_ind");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, csr_col_ind, &buffer_size);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");

            // Buffer size should be zero
            size_t zero = 0;
            unit_check_general(1, 1, &zero, &buffer_size);
        }

        return rocsparse_status_success;
    }

    // For testing, assemble a COO matrix and convert it to CSR first (on host)

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr(m + 1, 0);
    std::vector<rocsparse_int> hcoo_row_ind(nnz);
    std::vector<rocsparse_int> hcsr_col_ind(nnz);
    std::vector<float> hcsr_val(nnz);

    // Sample initial COO matrix on CPU
    srand(12345ULL);
    gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, rocsparse_index_base_zero);

    // Convert COO to CSR
    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        ++hcsr_row_ptr[hcoo_row_ind[i] + 1];
    }

    for(rocsparse_int i = 0; i < m; ++i)
    {
        hcsr_row_ptr[i + 1] += hcsr_row_ptr[i];
    }

    // Unsort CSR columns
    std::vector<rocsparse_int> hperm(nnz);
    std::vector<rocsparse_int> hcsr_col_ind_unsorted(nnz);
    std::vector<float> hcsr_val_unsorted(nnz);

    hcsr_col_ind_unsorted = hcsr_col_ind;
    hcsr_val_unsorted     = hcsr_val;

    for(rocsparse_int i = 0; i < m; ++i)
    {
        rocsparse_int row_begin = hcsr_row_ptr[i];
        rocsparse_int row_end   = hcsr_row_ptr[i + 1];
        rocsparse_int row_nnz   = row_end - row_begin;

        for(rocsparse_int j = row_begin; j < row_end; ++j)
        {
            rocsparse_int rng = row_begin + rand() % row_nnz;

            rocsparse_int temp_col  = hcsr_col_ind_unsorted[j];
            float temp_val          = hcsr_val_unsorted[j];

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
    auto dcsr_val_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dcsr_val_sorted_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(float) * nnz), device_free};
    auto dperm_managed = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};

    rocsparse_int* dcsr_row_ptr = (rocsparse_int*)dcsr_row_ptr_managed.get();
    rocsparse_int* dcsr_col_ind = (rocsparse_int*)dcsr_col_ind_managed.get();
    float* dcsr_val = (float*)dcsr_val_managed.get();
    float* dcsr_val_sorted = (float*)dcsr_val_sorted_managed.get();
    rocsparse_int* dperm = (rocsparse_int*)dperm_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dcsr_val_sorted || !dperm)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcsr_row_ptr || !dcsr_col_ind || "
                                        "!dcsr_val || !dcsr_val_sorted || !dperm");
        return rocsparse_status_memory_error;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_col_ind, hcsr_col_ind_unsorted.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val_unsorted.data(), sizeof(float) * nnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // Obtain buffer size
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsort_buffer_size(handle, m, n, nnz, dcsr_row_ptr, dcsr_col_ind, &buffer_size));

        // Allocate buffer on the device
        auto dbuffer_managed = rocsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};

        void* dbuffer = (void*)dbuffer_managed.get();

        if(!dbuffer)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error, "!dbuffer");
            return rocsparse_status_memory_error;
        }

        // Initialize perm with identity permutation
        CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnz, dperm));

        // Sort CSR columns
        CHECK_ROCSPARSE_ERROR(rocsparse_csrsort(handle, m, n, nnz, descr, dcsr_row_ptr, dcsr_col_ind, dperm, dbuffer));

        // Sort CSR values
        CHECK_ROCSPARSE_ERROR(rocsparse_sgthr(handle, nnz, dcsr_val, dcsr_val_sorted, dperm, rocsparse_index_base_zero));

        // Copy output from device to host
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind_unsorted.data(), dcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val_unsorted.data(), dcsr_val_sorted, sizeof(float) * nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, nnz, hcsr_col_ind.data(), hcsr_col_ind_unsorted.data());
        unit_check_general(1, nnz, hcsr_val.data(), hcsr_val_unsorted.data());
    }

    if(argus.timing)
    {
/* TODO
        rocsparse_int number_cold_calls = 2;
        rocsparse_int number_hot_calls  = argus.iters;

        for(rocsparse_int iter = 0; iter < number_cold_calls; ++iter)
        {
            rocsparse_csrsort(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        double gpu_time_used = get_time_us();

        for(rocsparse_int iter = 0; iter < number_hot_calls; ++iter)
        {
            rocsparse_csrsort(handle, dcsr_row_ptr, nnz, m, dcoo_row_ind, idx_base);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        double bandwidth = sizeof(rocsparse_int) * (nnz + m + 1) / gpu_time_used / 1e6;

        printf("m\t\tn\t\tnnz\t\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\n", m, n, nnz, bandwidth, gpu_time_used);
*/
    }
    return rocsparse_status_success;
}

#endif // TESTING_CSRSORT_HPP

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
#ifndef TESTING_ELL2CSR_HPP
#define TESTING_ELL2CSR_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <rocsparse.h>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

template <typename T>
void testing_ell2csr_bad_arg(void)
{
    rocsparse_int    m         = 100;
    rocsparse_int    n         = 100;
    rocsparse_int    ell_width = 100;
    rocsparse_int    safe_size = 100;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    rocsparse_mat_descr           csr_descr = unique_ptr_csr_descr->descr;

    std::unique_ptr<descr_struct> unique_ptr_ell_descr(new descr_struct);
    rocsparse_mat_descr           ell_descr = unique_ptr_ell_descr->descr;

    auto ell_col_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csr_row_ptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

    rocsparse_int* ell_col_ind = (rocsparse_int*)ell_col_ind_managed.get();
    rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();

    if(!ell_col_ind || !csr_row_ptr)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // ELL to CSR conversion is a two step process - test both functions for bad arguments

    // Step 1: Determine number of non-zero elements of CSR storage format
    rocsparse_int csr_nnz;

    // Testing for (ell_col_ind == nullptr)
    {
        rocsparse_int* ell_col_ind_null = nullptr;

        status = rocsparse_ell2csr_nnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind_null, csr_descr, csr_row_ptr, &csr_nnz);
        verify_rocsparse_status_invalid_pointer(status, "Error: ell_col_ind is nullptr");
    }

    // Testing for (csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_ell2csr_nnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr_null, &csr_nnz);
        verify_rocsparse_status_invalid_pointer(status, "Error: ell_width is nullptr");
    }

    // Testing for (csr_nnz == nullptr)
    {
        rocsparse_int* csr_nnz_null = nullptr;

        status = rocsparse_ell2csr_nnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, csr_nnz_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_nnz is nullptr");
    }

    // Testing for (ell_descr == nullptr)
    {
        rocsparse_mat_descr ell_descr_null = nullptr;

        status = rocsparse_ell2csr_nnz(
            handle, m, n, ell_descr_null, ell_width, ell_col_ind, csr_descr, csr_row_ptr, &csr_nnz);
        verify_rocsparse_status_invalid_pointer(status, "Error: ell_descr is nullptr");
    }

    // Testing for (csr_descr == nullptr)
    {
        rocsparse_mat_descr csr_descr_null = nullptr;

        status = rocsparse_ell2csr_nnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr_null, csr_row_ptr, &csr_nnz);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_ell2csr_nnz(
            handle_null, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, &csr_nnz);
        verify_rocsparse_status_invalid_handle(status);
    }

    // Allocate memory for ELL storage format
    auto ell_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto csr_col_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto csr_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T*             ell_val     = (T*)ell_val_managed.get();
    rocsparse_int* csr_col_ind = (rocsparse_int*)csr_col_ind_managed.get();
    T*             csr_val     = (T*)csr_val_managed.get();

    if(!ell_val || !csr_col_ind || !csr_val)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Step 2: Perform the actual conversion

    // Set ell_width to some valid value, to avoid invalid_size status
    ell_width = 10;

    // Testing for (ell_col_ind == nullptr)
    {
        rocsparse_int* ell_col_ind_null = nullptr;

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind_null,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_rocsparse_status_invalid_pointer(status, "Error: ell_col_ind is nullptr");
    }

    // Testing for (ell_val == nullptr)
    {
        T* ell_val_null = nullptr;

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val_null,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_rocsparse_status_invalid_pointer(status, "Error: ell_val is nullptr");
    }

    // Testing for (csr_row_ptr == nullptr)
    {
        rocsparse_int* csr_row_ptr_null = nullptr;

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr_null,
                                   csr_col_ind);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    // Testing for (csr_col_ind == nullptr)
    {
        rocsparse_int* csr_col_ind_null = nullptr;

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    // Testing for (csr_val == nullptr)
    {
        T* csr_val_null = nullptr;

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val_null,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    // Testing for (ell_descr == nullptr)
    {
        rocsparse_mat_descr ell_descr_null = nullptr;

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr_null,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_rocsparse_status_invalid_pointer(status, "Error: ell_descr is nullptr");
    }

    // Testing for (csr_descr == nullptr)
    {
        rocsparse_mat_descr csr_descr_null = nullptr;

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr_null,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_rocsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    // Testing for (handle == nullptr)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_ell2csr(handle_null,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_ell2csr(Arguments argus)
{
    rocsparse_int        m         = argus.M;
    rocsparse_int        n         = argus.N;
    rocsparse_int        safe_size = 100;
    rocsparse_index_base ell_base  = argus.idx_base;
    rocsparse_index_base csr_base  = argus.idx_base2;
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

    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    rocsparse_int nnz = m * scale * n;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_ell_descr(new descr_struct);
    rocsparse_mat_descr           ell_descr = unique_ptr_ell_descr->descr;

    // Set ELL matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(ell_descr, ell_base));

    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    rocsparse_mat_descr           csr_descr = unique_ptr_csr_descr->descr;

    // Set CSR matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto ell_col_ind_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto ell_val_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto csr_row_ptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};

        rocsparse_int* ell_col_ind = (rocsparse_int*)ell_col_ind_managed.get();
        T*             ell_val     = (T*)ell_val_managed.get();
        rocsparse_int* csr_row_ptr = (rocsparse_int*)csr_row_ptr_managed.get();

        if(!ell_col_ind || !ell_val || !csr_row_ptr)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!ell_col_ind || !ell_val || !csr_row_ptr");
            return rocsparse_status_memory_error;
        }

        rocsparse_int ell_width = safe_size;

        // Step 1 - obtain CSR nnz
        rocsparse_int csr_nnz;
        status = rocsparse_ell2csr_nnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, &csr_nnz);

        if(m < 0 || n < 0 || ell_width < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || ell_width < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && ell_width >= 0");
        }

        // Step 2 - perform actual conversion
        auto csr_col_ind_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto csr_val_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        rocsparse_int* csr_col_ind = (rocsparse_int*)csr_col_ind_managed.get();
        T*             csr_val     = (T*)csr_val_managed.get();

        if(!csr_col_ind || !csr_val)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!csr_col_ind || !csr_val");
            return rocsparse_status_memory_error;
        }

        status = rocsparse_ell2csr(handle,
                                   m,
                                   n,
                                   ell_descr,
                                   ell_width,
                                   ell_val,
                                   ell_col_ind,
                                   csr_descr,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind);

        if(m < 0 || n < 0 || ell_width < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || ell_width < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && ell_width >= 0");
        }

        return rocsparse_status_success;
    }

    // For testing, assemble a CSR matrix

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr_gold;
    std::vector<rocsparse_int> hcsr_col_ind_gold;
    std::vector<T>             hcsr_val_gold;

    // Sample initial CSR matrix on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(),
                           m,
                           n,
                           nnz,
                           hcsr_row_ptr_gold,
                           hcsr_col_ind_gold,
                           hcsr_val_gold,
                           csr_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(
            argus.laplacian, hcsr_row_ptr_gold, hcsr_col_ind_gold, hcsr_val_gold, csr_base);
        nnz = hcsr_row_ptr_gold[m];
    }
    else
    {
        std::vector<rocsparse_int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               m,
                               n,
                               nnz,
                               hcoo_row_ind,
                               hcsr_col_ind_gold,
                               hcsr_val_gold,
                               csr_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcsr_col_ind_gold, hcsr_val_gold, csr_base);
        }

        // Convert COO to CSR
        hcsr_row_ptr_gold.resize(m + 1, 0);
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptr_gold[hcoo_row_ind[i] + 1 - csr_base];
        }

        hcsr_row_ptr_gold[0] = csr_base;
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hcsr_row_ptr_gold[i + 1] += hcsr_row_ptr_gold[i];
        }
    }

    rocsparse_int csr_nnz_gold = nnz;

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dcsr_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    rocsparse_int* dcsr_row_ptr = (rocsparse_int*)dcsr_row_ptr_managed.get();
    rocsparse_int* dcsr_col_ind = (rocsparse_int*)dcsr_col_ind_managed.get();
    T*             dcsr_val     = (T*)dcsr_val_managed.get();

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val");
        return rocsparse_status_memory_error;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr,
                              hcsr_row_ptr_gold.data(),
                              sizeof(rocsparse_int) * (m + 1),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_col_ind,
                              hcsr_col_ind_gold.data(),
                              sizeof(rocsparse_int) * nnz,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val, hcsr_val_gold.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Convert CSR matrix to ELL format on GPU
    rocsparse_int ell_width;

    CHECK_ROCSPARSE_ERROR(
        rocsparse_csr2ell_width(handle, m, csr_descr, dcsr_row_ptr, ell_descr, &ell_width));

    auto dell_col_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (ell_width * m)), device_free};
    auto dell_val_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(T) * (ell_width * m)), device_free};

    rocsparse_int* dell_col_ind = (rocsparse_int*)dell_col_ind_managed.get();
    T*             dell_val     = (T*)dell_val_managed.get();

    if(!dell_col_ind || !dell_val)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dell_col_ind || !dell_val");
        return rocsparse_status_memory_error;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2ell(handle,
                                            m,
                                            csr_descr,
                                            dcsr_val,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            ell_descr,
                                            ell_width,
                                            dell_val,
                                            dell_col_ind));

    if(argus.unit_check)
    {
        // Determine csr non-zero entries
        rocsparse_int csr_nnz;

        auto dcsr_row_ptr_conv_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};

        rocsparse_int* dcsr_row_ptr_conv = (rocsparse_int*)dcsr_row_ptr_conv_managed.get();

        if(!dcsr_row_ptr_conv)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error, "!dcsr_row_ptr_conv");
            return rocsparse_status_memory_error;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz(handle,
                                                    m,
                                                    n,
                                                    ell_descr,
                                                    ell_width,
                                                    dell_col_ind,
                                                    csr_descr,
                                                    dcsr_row_ptr_conv,
                                                    &csr_nnz));

        // Check if CSR nnz does match
        unit_check_general(1, 1, 1, &csr_nnz_gold, &csr_nnz);

        // Allocate CSR column and values arrays
        auto dcsr_col_ind_conv_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * csr_nnz), device_free};
        auto dcsr_val_conv_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * csr_nnz), device_free};

        rocsparse_int* dcsr_col_ind_conv = (rocsparse_int*)dcsr_col_ind_conv_managed.get();
        T*             dcsr_val_conv     = (T*)dcsr_val_conv_managed.get();

        if(!dcsr_col_ind_conv || !dcsr_val_conv)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dcsr_col_ind_conv || !dcsr_val_conv");
            return rocsparse_status_memory_error;
        }

        // Perform actual CSR conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr(handle,
                                                m,
                                                n,
                                                ell_descr,
                                                ell_width,
                                                dell_val,
                                                dell_col_ind,
                                                csr_descr,
                                                dcsr_val_conv,
                                                dcsr_row_ptr_conv,
                                                dcsr_col_ind_conv));

        // Verification host structures
        std::vector<rocsparse_int> hcsr_row_ptr(m + 1);
        std::vector<rocsparse_int> hcsr_col_ind(csr_nnz);
        std::vector<T>             hcsr_val(csr_nnz);

        CHECK_HIP_ERROR(hipMemcpy(hcsr_row_ptr.data(),
                                  dcsr_row_ptr_conv,
                                  sizeof(rocsparse_int) * (m + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_col_ind.data(),
                                  dcsr_col_ind_conv,
                                  sizeof(rocsparse_int) * csr_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hcsr_val.data(), dcsr_val_conv, sizeof(T) * csr_nnz, hipMemcpyDeviceToHost));

        // Unit check
        unit_check_general(1, m + 1, 1, hcsr_row_ptr_gold.data(), hcsr_row_ptr.data());
        unit_check_general(1, csr_nnz, 1, hcsr_col_ind_gold.data(), hcsr_col_ind.data());
        unit_check_general(1, csr_nnz, 1, hcsr_val_gold.data(), hcsr_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            auto dcsr_row_ptr_conv_managed
                = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};

            rocsparse_int* dcsr_row_ptr_conv = (rocsparse_int*)dcsr_row_ptr_conv_managed.get();

            rocsparse_int csr_nnz;
            rocsparse_ell2csr_nnz(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  dell_col_ind,
                                  csr_descr,
                                  dcsr_row_ptr_conv,
                                  &csr_nnz);

            auto dcsr_col_ind_conv_managed
                = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * csr_nnz), device_free};
            auto dcsr_val_conv_managed
                = rocsparse_unique_ptr{device_malloc(sizeof(T) * csr_nnz), device_free};

            rocsparse_int* dcsr_col_ind_conv = (rocsparse_int*)dcsr_col_ind_conv_managed.get();
            T*             dcsr_val_conv     = (T*)dcsr_val_conv_managed.get();

            rocsparse_ell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              dell_val,
                              dell_col_ind,
                              csr_descr,
                              dcsr_val_conv,
                              dcsr_row_ptr_conv,
                              dcsr_col_ind_conv);
        }

        double gpu_time_used = get_time_us();

        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            auto dcsr_row_ptr_conv_managed
                = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};

            rocsparse_int* dcsr_row_ptr_conv = (rocsparse_int*)dcsr_row_ptr_conv_managed.get();

            rocsparse_int csr_nnz;
            rocsparse_ell2csr_nnz(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  dell_col_ind,
                                  csr_descr,
                                  dcsr_row_ptr_conv,
                                  &csr_nnz);

            auto dcsr_col_ind_conv_managed
                = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * csr_nnz), device_free};
            auto dcsr_val_conv_managed
                = rocsparse_unique_ptr{device_malloc(sizeof(T) * csr_nnz), device_free};

            rocsparse_int* dcsr_col_ind_conv = (rocsparse_int*)dcsr_col_ind_conv_managed.get();
            T*             dcsr_val_conv     = (T*)dcsr_val_conv_managed.get();

            rocsparse_ell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              dell_val,
                              dell_col_ind,
                              csr_descr,
                              dcsr_val_conv,
                              dcsr_row_ptr_conv,
                              dcsr_col_ind_conv);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);

        printf("m\t\tn\t\tnnz\t\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\n", m, n, nnz, gpu_time_used);
    }

    return rocsparse_status_success;
}

#endif // TESTING_ELL2CSR_HPP

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
#ifndef TESTING_CSRMV_HPP
#define TESTING_CSRMV_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <cmath>
#include <rocsparse.h>
#include <string>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_csrmv_bad_arg(void)
{
    rocsparse_int       n         = 100;
    rocsparse_int       m         = 100;
    rocsparse_int       nnz       = 100;
    rocsparse_int       safe_size = 100;
    T                   alpha     = 0.6;
    T                   beta      = 0.2;
    rocsparse_operation transA    = rocsparse_operation_none;
    rocsparse_status    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr           descr = unique_ptr_descr->descr;

    std::unique_ptr<mat_info_struct> unique_ptr_mat_info(new mat_info_struct);
    rocsparse_mat_info               info = unique_ptr_mat_info->info;

    auto dptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dy_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
    T*             dval = (T*)dval_managed.get();
    T*             dx   = (T*)dx_managed.get();
    T*             dy   = (T*)dy_managed.get();

    if(!dval || !dptr || !dcol || !dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing rocsparse_csrmv_analysis

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status = rocsparse_csrmv_analysis(
            handle, transA, m, n, nnz, descr, dval, dptr_null, dcol, info);
        verify_rocsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        rocsparse_int* dcol_null = nullptr;

        status = rocsparse_csrmv_analysis(
            handle, transA, m, n, nnz, descr, dval, dptr, dcol_null, info);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = rocsparse_csrmv_analysis(
            handle, transA, m, n, nnz, descr, dval_null, dptr, dcol, info);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrmv_analysis(
            handle, transA, m, n, nnz, descr_null, dval, dptr, dcol, info);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrmv_analysis(
            handle, transA, m, n, nnz, descr, dval, dptr, dcol, info_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrmv_analysis(
            handle_null, transA, m, n, nnz, descr, dval, dptr, dcol, info);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrmv

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr_null,
                                 dcol,
                                 nullptr,
                                 dx,
                                 &beta,
                                 dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        rocsparse_int* dcol_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol_null,
                                 nullptr,
                                 dx,
                                 &beta,
                                 dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval_null,
                                 dptr,
                                 dcol,
                                 nullptr,
                                 dx,
                                 &beta,
                                 dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dx)
    {
        T* dx_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 nullptr,
                                 dx_null,
                                 &beta,
                                 dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: dx is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 nullptr,
                                 dx,
                                 &beta,
                                 dy_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: dy is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 d_alpha_null,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 nullptr,
                                 dx,
                                 &beta,
                                 dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == d_beta)
    {
        T* d_beta_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 nullptr,
                                 dx,
                                 d_beta_null,
                                 dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrmv(handle,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr_null,
                                 dval,
                                 dptr,
                                 dcol,
                                 nullptr,
                                 dx,
                                 &beta,
                                 dy);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrmv(handle_null,
                                 transA,
                                 m,
                                 n,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 nullptr,
                                 dx,
                                 &beta,
                                 dy);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrmv_clear

    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrmv_clear(handle, info_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrmv_clear(handle_null, info);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
static T two_sum(T x, T y, T* sumk_err)
{
    T sumk_s = x + y;
    T bp     = sumk_s - x;
    (*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
    return sumk_s;
}

template <typename T>
rocsparse_status testing_csrmv(Arguments argus)
{
    rocsparse_int        safe_size  = 100;
    rocsparse_int        m          = argus.M;
    rocsparse_int        n          = argus.N;
    T                    h_alpha    = argus.alpha;
    T                    h_beta     = argus.beta;
    rocsparse_operation  transA     = argus.transA;
    rocsparse_index_base idx_base   = argus.idx_base;
    bool                 adaptive   = argus.bswitch;
    std::string          binfile    = "";
    std::string          filename   = "";
    std::string          rocalution = "";
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
        if(argus.rocalution != "")
        {
            rocalution = argus.rocalution;
        }
        else if(argus.filename != "")
        {
            filename = argus.filename;
        }
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle               handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    rocsparse_mat_descr           descr = test_descr->descr;

    std::unique_ptr<mat_info_struct> unique_ptr_mat_info(new mat_info_struct);
    rocsparse_mat_info               info = nullptr;

    if(adaptive)
    {
        info = unique_ptr_mat_info->info;
    }

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, idx_base));

    // Determine number of non-zero elements
    double scale = 0.02;
    if(m > 1000 || n > 1000)
    {
        scale = 2.0 / std::max(m, n);
    }
    rocsparse_int nnz = m * scale * n;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || nnz <= 0)
    {
        auto dptr_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dcol_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dx_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
        rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
        T*             dval = (T*)dval_managed.get();
        T*             dx   = (T*)dx_managed.get();
        T*             dy   = (T*)dy_managed.get();

        if(!dval || !dptr || !dcol || !dx || !dy)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dptr || !dcol || !dval || !dx || !dy");
            return rocsparse_status_memory_error;
        }

        if(adaptive)
        {
            // Test rocsparse_csrmv_analysis
            status = rocsparse_csrmv_analysis(
                handle, transA, m, n, nnz, descr, dval, dptr, dcol, info);

            if(m < 0 || n < 0 || nnz < 0)
            {
                verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
            }
            else
            {
                verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");
            }
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        status = rocsparse_csrmv(
            handle, transA, m, n, nnz, &h_alpha, descr, dval, dptr, dcol, info, dx, &h_beta, dy);

        if(m < 0 || n < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || n < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && n >= 0 && nnz >= 0");
        }

        if(adaptive)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
        }

        return rocsparse_status_success;
    }

    // Host structures
    std::vector<rocsparse_int> hcsr_row_ptr;
    std::vector<rocsparse_int> hcoo_row_ind;
    std::vector<rocsparse_int> hcol_ind;
    std::vector<T>             hval;

    // Initial Data on CPU
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(binfile.c_str(), m, n, nnz, hcsr_row_ptr, hcol_ind, hval, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(rocalution != "")
    {
        if(read_rocalution_matrix(
               rocalution.c_str(), m, n, nnz, hcsr_row_ptr, hcol_ind, hval, idx_base)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", rocalution.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        m = n = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcol_ind, hval, idx_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(), m, n, nnz, hcoo_row_ind, hcol_ind, hval, idx_base)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(m, n, nnz, hcoo_row_ind, hcol_ind, hval, idx_base);
        }

        // Convert COO to CSR
        if(!argus.laplacian)
        {
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
    }

    std::vector<T> hx(n);
    std::vector<T> hy_1(m);
    std::vector<T> hy_2(m);
    std::vector<T> hy_gold(m);

    rocsparse_init<T>(hx, 1, n);
    rocsparse_init<T>(hy_1, 1, m);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dptr_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (m + 1)), device_free};
    auto dcol_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dval_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_managed      = rocsparse_unique_ptr{device_malloc(sizeof(T) * n), device_free};
    auto dy_1_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto dy_2_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * m), device_free};
    auto d_alpha_managed = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed  = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dptr    = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol    = (rocsparse_int*)dcol_managed.get();
    T*             dval    = (T*)dval_managed.get();
    T*             dx      = (T*)dx_managed.get();
    T*             dy_1    = (T*)dy_1_managed.get();
    T*             dy_2    = (T*)dy_2_managed.get();
    T*             d_alpha = (T*)d_alpha_managed.get();
    T*             d_beta  = (T*)d_beta_managed.get();

    if(!dval || !dptr || !dcol || !dx || !dy_1 || !dy_2 || !d_alpha || !d_beta)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dval || !dptr || !dcol || !dx || "
                                        "!dy_1 || !dy_2 || !d_alpha || !d_beta");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcol, hcol_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * m, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(adaptive)
    {
        // csrmv analysis
        CHECK_ROCSPARSE_ERROR(
            rocsparse_csrmv_analysis(handle, transA, m, n, nnz, descr, dval, dptr, dcol, info));
    }

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * m, hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv(
            handle, transA, m, n, nnz, &h_alpha, descr, dval, dptr, dcol, info, dx, &h_beta, dy_1));

        // ROCSPARSE pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv(
            handle, transA, m, n, nnz, d_alpha, descr, dval, dptr, dcol, info, dx, d_beta, dy_2));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * m, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * m, hipMemcpyDeviceToHost));

        // CPU - do the csrmv row reduction in the same order as the GPU
        double cpu_time_used = get_time_us();

        // Different csrmv algorithms require different CPU summation
        if(adaptive)
        {
            for(rocsparse_int i = 0; i < m; ++i)
            {
                hy_gold[i] *= h_beta;
                T sum = hy_gold[i];
                T err = static_cast<T>(0);

                for(rocsparse_int j = hcsr_row_ptr[i] - idx_base;
                    j < hcsr_row_ptr[i + 1] - idx_base;
                    ++j)
                {
                    sum = two_sum(sum, h_alpha * hval[j] * hx[hcol_ind[j] - idx_base], &err);
                }

                hy_gold[i] = (T)(sum + err);
            }
        }
        else
        {
            // Query for wavefrontSize
            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, 0);

            rocsparse_int WF_SIZE;
            rocsparse_int nnz_per_row = nnz / m;

            if(prop.warpSize == 32)
            {
                if(nnz_per_row < 4)
                    WF_SIZE = 2;
                else if(nnz_per_row < 8)
                    WF_SIZE = 4;
                else if(nnz_per_row < 16)
                    WF_SIZE = 8;
                else if(nnz_per_row < 32)
                    WF_SIZE = 16;
                else
                    WF_SIZE = 32;
            }
            else if(prop.warpSize == 64)
            {
                if(nnz_per_row < 4)
                    WF_SIZE = 2;
                else if(nnz_per_row < 8)
                    WF_SIZE = 4;
                else if(nnz_per_row < 16)
                    WF_SIZE = 8;
                else if(nnz_per_row < 32)
                    WF_SIZE = 16;
                else if(nnz_per_row < 64)
                    WF_SIZE = 32;
                else
                    WF_SIZE = 64;
            }
            else
            {
                return rocsparse_status_internal_error;
            }

            for(rocsparse_int i = 0; i < m; ++i)
            {
                std::vector<T> sum(WF_SIZE, 0.0);

                for(rocsparse_int j = hcsr_row_ptr[i] - idx_base;
                    j < hcsr_row_ptr[i + 1] - idx_base;
                    j += WF_SIZE)
                {
                    for(rocsparse_int k = 0; k < WF_SIZE; ++k)
                    {
                        if(j + k < hcsr_row_ptr[i + 1] - idx_base)
                        {
                            sum[k] = fma(
                                h_alpha * hval[j + k], hx[hcol_ind[j + k] - idx_base], sum[k]);
                        }
                    }
                }

                for(rocsparse_int j = 1; j < WF_SIZE; j <<= 1)
                {
                    for(rocsparse_int k = 0; k < WF_SIZE - j; ++k)
                    {
                        sum[k] += sum[k + j];
                    }
                }

                if(h_beta == 0.0)
                {
                    hy_gold[i] = sum[0];
                }
                else
                {
                    hy_gold[i] = std::fma(h_beta, hy_gold[i], sum[0]);
                }
            }
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        if(adaptive)
        {
            unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
            unit_check_near(1, m, 1, hy_gold.data(), hy_2.data());
        }
        else
        {
            unit_check_general(1, m, 1, hy_gold.data(), hy_1.data());
            unit_check_general(1, m, 1, hy_gold.data(), hy_2.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_csrmv(handle,
                            transA,
                            m,
                            n,
                            nnz,
                            &h_alpha,
                            descr,
                            dval,
                            dptr,
                            dcol,
                            info,
                            dx,
                            &h_beta,
                            dy_1);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocsparse_csrmv(handle,
                            transA,
                            m,
                            n,
                            nnz,
                            &h_alpha,
                            descr,
                            dval,
                            dptr,
                            dcol,
                            info,
                            dx,
                            &h_beta,
                            dy_1);
        }

        // Convert to miliseconds per call
        gpu_time_used     = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);
        size_t flops      = (h_alpha != 1.0) ? 3.0 * nnz : 2.0 * nnz;
        flops             = (h_beta != 0.0) ? flops + m : flops;
        double gpu_gflops = flops / gpu_time_used / 1e6;
        size_t memtrans   = 2.0 * m + nnz;
        memtrans          = (h_beta != 0.0) ? memtrans + m : memtrans;
        double bandwidth
            = (memtrans * sizeof(T) + (m + 1 + nnz) * sizeof(rocsparse_int)) / gpu_time_used / 1e6;

        printf("m\t\tn\t\tnnz\t\talpha\tbeta\tGFlops\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
               m,
               n,
               nnz,
               h_alpha,
               h_beta,
               gpu_gflops,
               bandwidth,
               gpu_time_used);
    }

    if(adaptive)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    }

    return rocsparse_status_success;
}

#endif // TESTING_CSRMV_HPP

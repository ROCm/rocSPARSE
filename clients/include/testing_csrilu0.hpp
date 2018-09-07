/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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
/*
    rocsparse_int n            = 100;
    rocsparse_int m            = 100;
    rocsparse_int nnz          = 100;
    rocsparse_int safe_size    = 100;
    T alpha                    = 0.6;
    T beta                     = 0.2;
    rocsparse_operation transA = rocsparse_operation_none;
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
    auto dx_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dy_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
    T* dval             = (T*)dval_managed.get();
    T* dx               = (T*)dx_managed.get();
    T* dy               = (T*)dy_managed.get();

    if(!dval || !dptr || !dcol || !dx || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing rocsparse_csrilu0_analysis

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status = rocsparse_csrilu0_analysis(handle, transA, m, n, nnz, descr, dptr_null, dcol, info);
        verify_rocsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        rocsparse_int* dcol_null = nullptr;

        status = rocsparse_csrilu0_analysis(handle, transA, m, n, nnz, descr, dptr, dcol_null, info);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrilu0_analysis(handle, transA, m, n, nnz, descr_null, dptr, dcol, info);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrilu0_analysis(handle, transA, m, n, nnz, descr, dptr, dcol, info_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrilu0_analysis(handle_null, transA, m, n, nnz, descr, dptr, dcol, info);
        verify_rocsparse_status_invalid_handle(status);
    }

    // testing rocsparse_csrilu0

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle,
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

        status = rocsparse_csrilu0(handle_null,
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

    // testing rocsparse_csrilu0_analysis_clear

    // testing for(nullptr == info)
    {
        rocsparse_mat_info info_null = nullptr;

        status = rocsparse_csrilu0_analysis_clear(handle, info_null);
        verify_rocsparse_status_invalid_pointer(status, "Error: info is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrilu0_analysis_clear(handle_null, info);
        verify_rocsparse_status_invalid_handle(status);
    }
*/
}

template <typename T>
static int ILU0(int m, const int* ptr, const int* col, T* val, rocsparse_index_base idx_base)
{
    // pointer of upper part of each row
    std::vector<int> diag_offset(m);
    std::vector<int> nnz_entries(m, 0);

    // ai = 0 to N loop over all rows
    for(int ai = 0; ai < m; ++ai)
    {
        // ai-th row entries
        int row_start = ptr[ai] - idx_base;
        int row_end   = ptr[ai + 1] - idx_base;
        int j;

        // nnz position of ai-th row in val array
        for (j = row_start; j < row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = j;
        }

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for (j = row_start; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if (col[j] - idx_base < ai) {

                int col_j  = col[j] - idx_base;
                int diag_j = diag_offset[col_j];

                if (val[diag_j] != 0.0)
                {
                    // multiplication factor
                    val[j] = val[j] / val[diag_j];

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(int k = diag_j + 1; k < ptr[col_j + 1] - idx_base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if (nnz_entries[col[k] - idx_base] != 0)
                        {
                            val[nnz_entries[col[k] - idx_base]] -= val[j] * val[k];
                        }
                    }
                }
                else
                {
                    // Numerical zero diagonal
                    return col_j;
                }
            }
            else if(col[j] - idx_base == ai)
            {
                has_diag = true;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural zero digonal
            return ai;
        }

        // set diagonal pointer to diagonal element
        diag_offset[ai] = j;

        // clear nnz entries
        for (j=row_start; j<row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = 0;
        }
    }

    return -1;
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
        m = safe_size;
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
        auto buffer_managed = rocsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

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
        status = rocsparse_csrilu0_buffer_size(handle, m, nnz, descr, dptr, dcol, info, &size);

        if(m < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        // Test rocsparse_csrilu0_analysis
        status = rocsparse_csrilu0_analysis(handle, m, nnz, descr, dptr, dcol, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, buffer);

        if(m < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        status = rocsparse_csrilu0(handle, m, nnz, descr, dval, dptr, dcol, info, rocsparse_solve_policy_auto, buffer);

        if(m < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: m < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "m >= 0 && nnz >= 0");
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_clear(info));

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
        if(read_bin_matrix(binfile.c_str(), m, m, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        m = gen_2d_laplacian(argus.laplacian, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base);
        nnz   = hcsr_row_ptr[m];
    }
    else
    {
        std::vector<rocsparse_int> hcoo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), m, m, nnz, hcoo_row_ind, hcsr_col_ind, hcsr_val, idx_base) != 0)
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
    auto dval_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};

    rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
    T* dval             = (T*)dval_managed.get();

    if(!dval || !dptr || !dcol)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dval || !dptr || !dcol");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dptr, hcsr_row_ptr.data(), sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcol, hcsr_col_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain csrilu0 buffer size
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_buffer_size(handle, m, nnz, descr, dptr, dcol, info, &size));

    // Allocate buffer on the device
    auto dbuffer_managed = rocsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error, "!dbuffer");
        return rocsparse_status_memory_error;
    }

    // csrilu0 analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_analysis(handle, m, nnz, descr, dptr, dcol, info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, dbuffer));

    if(argus.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0(handle, m, nnz, descr, dval, dptr, dcol, info, rocsparse_solve_policy_auto, dbuffer));

        // Copy output from device to CPU
        std::vector<T> result(nnz);
        CHECK_HIP_ERROR(hipMemcpy(result.data(), dval, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // Host csrilu0
        double cpu_time_used = get_time_us();

        int zero_piv = ILU0(m, hcsr_row_ptr.data(), hcsr_col_ind.data(), hcsr_val.data(), idx_base);

        cpu_time_used = get_time_us() - cpu_time_used;

        // TODO compare zero pivot
        if(zero_piv == -1)
        {
//            unit_check_near(1, m, 1, hy_gold.data(), hy_1.data());
            unit_check_general(1, nnz, 1, hcsr_val.data(), result.data());
        }
    }
/*
    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_csrilu0(handle,
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
            rocsparse_csrilu0(handle,
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
        double bandwidth =
            (memtrans * sizeof(T) + (m + 1 + nnz) * sizeof(rocsparse_int)) / gpu_time_used / 1e6;

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
        CHECK_ROCSPARSE_ERROR(rocsparse_csrilu0_clear(info));
    }
*/
    return rocsparse_status_success;
}

#endif // TESTING_CSRILU0_HPP

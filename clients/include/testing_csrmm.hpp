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
#ifndef TESTING_CSRMM_HPP
#define TESTING_CSRMM_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <string>
#include <rocsparse.h>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_csrmm_bad_arg(void)
{

    rocsparse_int N            = 100;
    rocsparse_int M            = 100;
    rocsparse_int K            = 100;
    rocsparse_int ldb          = 100;
    rocsparse_int ldc          = 100;
    rocsparse_int nnz          = 100;
    rocsparse_int safe_size    = 100;
    T alpha                    = 0.6;
    T beta                     = 0.2;
    rocsparse_operation transA = rocsparse_operation_none;
    rocsparse_operation transB = rocsparse_operation_none;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    rocsparse_mat_descr descr = unique_ptr_descr->descr;

    auto dptr_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dcol_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dB_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dC_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
    rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
    T* dval             = (T*)dval_managed.get();
    T* dB               = (T*)dB_managed.get();
    T* dC               = (T*)dC_managed.get();

    if(!dval || !dptr || !dcol || !dB || !dC)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for(nullptr == dptr)
    {
        rocsparse_int* dptr_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr_null,
                                 dcol,
                                 dB,
                                 ldb,
                                 &beta,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: dptr is nullptr");
    }
    // testing for(nullptr == dcol)
    {
        rocsparse_int* dcol_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol_null,
                                 dB,
                                 ldb,
                                 &beta,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: dcol is nullptr");
    }
    // testing for(nullptr == dval)
    {
        T* dval_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval_null,
                                 dptr,
                                 dcol,
                                 dB,
                                 ldb,
                                 &beta,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: dval is nullptr");
    }
    // testing for(nullptr == dB)
    {
        T* dB_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 dB_null,
                                 ldb,
                                 &beta,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: dB is nullptr");
    }
    // testing for(nullptr == dC)
    {
        T* dC_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 dB,
                                 ldb,
                                 &beta,
                                 dC_null,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: dC is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 d_alpha_null,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 dB,
                                 ldb,
                                 &beta,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == d_beta)
    {
        T* d_beta_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 dB,
                                 ldb,
                                 d_beta_null,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: beta is nullptr");
    }
    // testing for(nullptr == descr)
    {
        rocsparse_mat_descr descr_null = nullptr;

        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr_null,
                                 dval,
                                 dptr,
                                 dcol,
                                 dB,
                                 ldb,
                                 &beta,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_pointer(status, "Error: descr is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_csrmm(handle_null,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 dB,
                                 ldb,
                                 &beta,
                                 dC,
                                 ldc);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_csrmm(Arguments argus)
{
    rocsparse_int safe_size       = 100;
    rocsparse_int M               = argus.M;
    rocsparse_int N               = argus.N;
    rocsparse_int K               = argus.K;
    rocsparse_int ldb             = argus.ldb;
    rocsparse_int ldc             = argus.ldc;
    T h_alpha                     = argus.alpha;
    T h_beta                      = argus.beta;
    rocsparse_operation transA    = argus.transA;
    rocsparse_operation transB    = argus.transB;
    rocsparse_index_base idx_base = argus.idx_base;
    std::string binfile           = "";
    std::string filename          = "";
    rocsparse_status status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    if(M == -99 && K == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        M = K = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    rocsparse_mat_descr descr = test_descr->descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, idx_base));

    // Determine number of non-zero elements
    double scale = 0.02;
    if(M > 1000 || K > 1000)
    {
        scale = 2.0 / std::max(M, K);
    }
    rocsparse_int nnz = M * scale * K;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || K <= 0 || nnz <= 0)
    {
        auto dptr_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dcol_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dval_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dB_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dC_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        rocsparse_int* dptr = (rocsparse_int*)dptr_managed.get();
        rocsparse_int* dcol = (rocsparse_int*)dcol_managed.get();
        T* dval             = (T*)dval_managed.get();
        T* dB               = (T*)dB_managed.get();
        T* dC               = (T*)dC_managed.get();

        if(!dval || !dptr || !dcol || !dB || !dC)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dptr || !dcol || !dval || !dB || !dC");
            return rocsparse_status_memory_error;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        status = rocsparse_csrmm(handle,
                                 transA,
                                 transB,
                                 M,
                                 N,
                                 K,
                                 nnz,
                                 &h_alpha,
                                 descr,
                                 dval,
                                 dptr,
                                 dcol,
                                 dB,
                                 ldb,
                                 &h_beta,
                                 dC,
                                 ldc);

        if(M < 0 || N < 0 || K < 0 || nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status,
                                                 "Error: M < 0 || N < 0 || K < 0 || nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "M >= 0 && N >= 0 && K >= 0 && nnz >= 0");
        }

        return rocsparse_status_success;
    }

    // Initialize random seed
    srand(12345ULL);

    // Host structures - CSR matrix A
    std::vector<rocsparse_int> hcsr_row_ptrA;
    std::vector<rocsparse_int> hcsr_col_indA;
    std::vector<T> hcsr_valA;

    // Initial Data on CPU
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), M, K, nnz, hcsr_row_ptrA, hcsr_col_indA, hcsr_valA, idx_base) != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return rocsparse_status_internal_error;
        }
    }
    else if(argus.laplacian)
    {
        M = K =
            gen_2d_laplacian(argus.laplacian, hcsr_row_ptrA, hcsr_col_indA, hcsr_valA, idx_base);
        nnz = hcsr_row_ptrA[M];
    }
    else
    {
        std::vector<rocsparse_int> hcoo_row_indA;

        if(filename != "")
        {
            if(read_mtx_matrix(filename.c_str(),
                               M,
                               K,
                               nnz,
                               hcoo_row_indA,
                               hcsr_col_indA,
                               hcsr_valA,
                               idx_base) != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return rocsparse_status_internal_error;
            }
        }
        else
        {
            gen_matrix_coo(M, K, nnz, hcoo_row_indA, hcsr_col_indA, hcsr_valA, idx_base);
        }

        // Convert COO to CSR
        hcsr_row_ptrA.resize(M + 1, 0);
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            ++hcsr_row_ptrA[hcoo_row_indA[i] + 1 - idx_base];
        }

        hcsr_row_ptrA[0] = idx_base;
        for(rocsparse_int i = 0; i < M; ++i)
        {
            hcsr_row_ptrA[i + 1] += hcsr_row_ptrA[i];
        }
    }

    if(transB == rocsparse_operation_none)
    {
        ldb = (transA == rocsparse_operation_none) ? K : M;
    }
    else
    {
        ldb = N;
    }

    ldc = (transA == rocsparse_operation_none) ? M : K;

    rocsparse_int Anrow = M;
    rocsparse_int Ancol = K;
    rocsparse_int Bnrow = ldb;
    rocsparse_int Bncol = (transB == rocsparse_operation_none) ? N : K;
    rocsparse_int Bnnz  = Bnrow * Bncol;
    rocsparse_int Cnrow = ldc;
    rocsparse_int Cncol = N;
    rocsparse_int Cnnz  = Cnrow * Cncol;

    // Host structures - Dense matrix B and C
    std::vector<T> hB(Bnnz);
    std::vector<T> hC_1(Cnnz);
    std::vector<T> hC_2(Cnnz);
    std::vector<T> hC_gold(Cnnz);

    rocsparse_init<T>(hB, Bnrow, Bncol);
    rocsparse_init<T>(hC_1, Cnrow, Cncol);

    // copy vector is easy in STL; hC_gold = hC_1: save a copy in hy_gold which will be output of
    // CPU
    hC_gold = hC_1;
    hC_2    = hC_1;

    // allocate memory on device
    auto dcsr_row_ptrA_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * (M + 1)), device_free};
    auto dcsr_col_indA_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dcsr_valA_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dB_managed        = rocsparse_unique_ptr{device_malloc(sizeof(T) * Bnnz), device_free};
    auto dC_1_managed      = rocsparse_unique_ptr{device_malloc(sizeof(T) * Cnnz), device_free};
    auto dC_2_managed      = rocsparse_unique_ptr{device_malloc(sizeof(T) * Cnnz), device_free};
    auto d_alpha_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto d_beta_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dcsr_row_ptrA = (rocsparse_int*)dcsr_row_ptrA_managed.get();
    rocsparse_int* dcsr_col_indA = (rocsparse_int*)dcsr_col_indA_managed.get();
    T* dcsr_valA                 = (T*)dcsr_valA_managed.get();
    T* dB                        = (T*)dB_managed.get();
    T* dC_1                      = (T*)dC_1_managed.get();
    T* dC_2                      = (T*)dC_2_managed.get();
    T* d_alpha                   = (T*)d_alpha_managed.get();
    T* d_beta                    = (T*)d_beta_managed.get();

    if(!dcsr_valA || !dcsr_row_ptrA || !dcsr_col_indA || !dB || !dC_1 || !d_alpha || !d_beta)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dcsr_valA || !dcsr_row_ptrA || !dcsr_col_indA || !dB || "
                                        "!dC_1 || !d_alpha || !d_beta");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptrA,
                              hcsr_row_ptrA.data(),
                              sizeof(rocsparse_int) * (M + 1),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_indA, hcsr_col_indA.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_valA, hcsr_valA.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * Bnnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC_1, hC_1.data(), sizeof(T) * Cnnz, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dC_2, hC_2.data(), sizeof(T) * Cnnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmm(handle,
                                              transA,
                                              transB,
                                              Anrow,
                                              Cncol,
                                              Ancol,
                                              nnz,
                                              &h_alpha,
                                              descr,
                                              dcsr_valA,
                                              dcsr_row_ptrA,
                                              dcsr_col_indA,
                                              dB,
                                              ldb,
                                              &h_beta,
                                              dC_1,
                                              ldc));

        // ROCSPARSE pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmm(handle,
                                              transA,
                                              transB,
                                              Anrow,
                                              Cncol,
                                              Ancol,
                                              nnz,
                                              d_alpha,
                                              descr,
                                              dcsr_valA,
                                              dcsr_row_ptrA,
                                              dcsr_col_indA,
                                              dB,
                                              ldb,
                                              d_beta,
                                              dC_2,
                                              ldc));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC_1, sizeof(T) * Cnnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hC_2.data(), dC_2, sizeof(T) * Cnnz, hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        for(rocsparse_int i = 0; i < Cnrow; ++i)
        {
            for(rocsparse_int j = 0; j < Cncol; ++j)
            {
                rocsparse_int Cidx = i + j * ldc;
                T sum              = hC_gold[Cidx] * h_beta;

                for(rocsparse_int k = hcsr_row_ptrA[i] - idx_base;
                    k < hcsr_row_ptrA[i + 1] - idx_base;
                    ++k)
                {
                    rocsparse_int Bidx = (transB == rocsparse_operation_none)
                                             ? (hcsr_col_indA[k] - idx_base + j * ldb)
                                             : (j + (hcsr_col_indA[k] - idx_base) * ldb);
                    sum += h_alpha * hcsr_valA[k] * hB[Bidx];
                }

                hC_gold[Cidx] = sum;
            }
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        unit_check_near(Cnrow, Cncol, ldc, hC_gold.data(), hC_1.data());
        unit_check_near(Cnrow, Cncol, ldc, hC_gold.data(), hC_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_csrmm(handle,
                            transA,
                            transB,
                            Anrow,
                            Cncol,
                            Ancol,
                            nnz,
                            &h_alpha,
                            descr,
                            dcsr_valA,
                            dcsr_row_ptrA,
                            dcsr_col_indA,
                            dB,
                            ldb,
                            &h_beta,
                            dC_1,
                            ldc);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocsparse_csrmm(handle,
                            transA,
                            transB,
                            Anrow,
                            Cncol,
                            Ancol,
                            nnz,
                            &h_alpha,
                            descr,
                            dcsr_valA,
                            dcsr_row_ptrA,
                            dcsr_col_indA,
                            dB,
                            ldb,
                            &h_beta,
                            dC_1,
                            ldc);
        }

        // Convert to miliseconds per call
        gpu_time_used     = (get_time_us() - gpu_time_used) / (number_hot_calls * 1e3);
        size_t flops      = 3.0 * nnz * Bncol;
        flops             = (h_beta != 0.0) ? flops + Cnnz : flops;
        double gpu_gflops = flops / gpu_time_used / 1e6;
        size_t memtrans   = nnz + Cnnz + Bnnz;
        memtrans          = (h_beta != 0.0) ? memtrans + Cnnz : memtrans;
        double bandwidth =
            (memtrans * sizeof(T) + (M + 1 + nnz) * sizeof(rocsparse_int)) / gpu_time_used / 1e6;

        printf("m\t\tn\t\tk\t\tnnz\t\talpha\tbeta\tGFlops\tGB/s\tmsec\n");
        printf("%8d\t%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
               M,
               N,
               K,
               nnz,
               h_alpha,
               h_beta,
               gpu_gflops,
               bandwidth,
               gpu_time_used);
    }

    return rocsparse_status_success;
}

#endif // TESTING_CSRMM_HPP

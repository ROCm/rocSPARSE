/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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

#include "testing.hpp"

template <typename T>
void testing_csrmv_managed_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Allocate managed memory
    rocsparse_int* csr_row_ptr;
    rocsparse_int* csr_col_ind;
    T*             csr_val;
    T*             x;
    T*             y;
    T*             alpha;
    T*             beta;

    CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_row_ptr, safe_size * sizeof(rocsparse_int)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_col_ind, safe_size * sizeof(rocsparse_int)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_val, safe_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&x, safe_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&y, safe_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&alpha, sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&beta, sizeof(T)));

    *alpha = 0.6;
    *beta  = 1.0;

    // Test rocsparse_csrmv_analysis()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(nullptr,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        csr_col_ind,
                                                        info),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        nullptr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        csr_col_ind,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        nullptr,
                                                        csr_row_ptr,
                                                        csr_col_ind,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        csr_val,
                                                        nullptr,
                                                        csr_col_ind,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        nullptr,
                                                        info),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(handle,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        csr_val,
                                                        csr_row_ptr,
                                                        csr_col_ind,
                                                        nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrmv()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(nullptr,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               beta,
                                               y),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               nullptr,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               beta,
                                               y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               nullptr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               beta,
                                               y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               descr,
                                               nullptr,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               beta,
                                               y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               descr,
                                               csr_val,
                                               nullptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               beta,
                                               y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               nullptr,
                                               info,
                                               x,
                                               beta,
                                               y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               nullptr,
                                               beta,
                                               y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               nullptr,
                                               y),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               alpha,
                                               descr,
                                               csr_val,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               info,
                                               x,
                                               beta,
                                               nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_csrmv_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(nullptr, info), rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);

    CHECK_HIP_ERROR(hipFree(csr_row_ptr));
    CHECK_HIP_ERROR(hipFree(csr_col_ind));
    CHECK_HIP_ERROR(hipFree(csr_val));
    CHECK_HIP_ERROR(hipFree(x));
    CHECK_HIP_ERROR(hipFree(y));
    CHECK_HIP_ERROR(hipFree(alpha));
    CHECK_HIP_ERROR(hipFree(beta));
}

template <typename T>
void testing_csrmv_managed(const Arguments& arg)
{
    rocsparse_int        M        = arg.M;
    rocsparse_int        N        = arg.N;
    rocsparse_operation  trans    = arg.transA;
    rocsparse_index_base base     = arg.baseA;
    uint32_t             adaptive = arg.algo;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info_ptr;

    // Differentiate between algorithm 0 (csrmv without analysis step) and
    //                       algorithm 1 (csrmv with analysis step)
    rocsparse_mat_info info = adaptive ? info_ptr : nullptr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate managed memory
        rocsparse_int* csr_row_ptr;
        rocsparse_int* csr_col_ind;
        T*             csr_val;
        T*             x;
        T*             y;
        T*             alpha;
        T*             beta;

        CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_row_ptr, safe_size * sizeof(rocsparse_int)));
        CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_col_ind, safe_size * sizeof(rocsparse_int)));
        CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_val, safe_size * sizeof(T)));
        CHECK_HIP_ERROR(hipMallocManaged((void**)&x, safe_size * sizeof(T)));
        CHECK_HIP_ERROR(hipMallocManaged((void**)&y, safe_size * sizeof(T)));
        CHECK_HIP_ERROR(hipMallocManaged((void**)&alpha, sizeof(T)));
        CHECK_HIP_ERROR(hipMallocManaged((void**)&beta, sizeof(T)));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // If adaptive, perform analysis step
        if(adaptive)
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_csrmv_analysis<T>(
                    handle, trans, M, N, safe_size, descr, csr_val, csr_row_ptr, csr_col_ind, info),
                (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(handle,
                                                   trans,
                                                   M,
                                                   N,
                                                   safe_size,
                                                   alpha,
                                                   descr,
                                                   csr_val,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   info,
                                                   x,
                                                   beta,
                                                   y),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        // If adaptive, clear data
        if(adaptive)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(handle, info), rocsparse_status_success);
        }

        CHECK_HIP_ERROR(hipFree(csr_row_ptr));
        CHECK_HIP_ERROR(hipFree(csr_col_ind));
        CHECK_HIP_ERROR(hipFree(csr_val));
        CHECK_HIP_ERROR(hipFree(x));
        CHECK_HIP_ERROR(hipFree(y));
        CHECK_HIP_ERROR(hipFree(alpha));
        CHECK_HIP_ERROR(hipFree(beta));

        return;
    }

    // Wavefront size
    int dev;
    hipGetDevice(&dev);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);

    bool type = (prop.warpSize == 32) ? true : adaptive;

    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, arg.timing ? false : type, full_rank);

    // Generate matrix
    std::vector<rocsparse_int> trow_ptr;
    std::vector<rocsparse_int> tcol_ind;
    std::vector<T>             tval;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(trow_ptr, tcol_ind, tval, M, N, nnz, base);

    // Allocate host memory for vectors
    std::vector<T> tx(N);
    std::vector<T> ty(M);

    // Initialize data on CPU
    rocsparse_init<T>(tx, 1, N, 1);
    rocsparse_init<T>(ty, 1, M, 1);

    // Allocate managed memory
    rocsparse_int* csr_row_ptr;
    rocsparse_int* csr_col_ind;
    T*             csr_val;
    T*             x;
    T*             y_1;
    T*             y_2;
    T*             alpha;
    T*             beta;

    CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_row_ptr, (M + 1) * sizeof(rocsparse_int)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_col_ind, nnz * sizeof(rocsparse_int)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&csr_val, nnz * sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&x, N * sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&y_1, M * sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&y_2, M * sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&alpha, sizeof(T)));
    CHECK_HIP_ERROR(hipMallocManaged((void**)&beta, sizeof(T)));

    // Copy data to managed arrays
    for(rocsparse_int i = 0; i < M + 1; i++)
    {
        csr_row_ptr[i] = trow_ptr[i];
    }

    for(rocsparse_int i = 0; i < nnz; i++)
    {
        csr_col_ind[i] = tcol_ind[i];
        csr_val[i]     = tval[i];
    }

    for(rocsparse_int i = 0; i < N; i++)
    {
        x[i] = tx[i];
    }

    for(rocsparse_int i = 0; i < M; i++)
    {
        y_1[i] = ty[i];
        y_2[i] = ty[i];
    }

    *alpha = arg.get_alpha<T>();
    *beta  = arg.get_beta<T>();

    // If adaptive, run analysis step
    if(adaptive)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(
            handle, trans, M, N, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info));
    }

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                 trans,
                                                 M,
                                                 N,
                                                 nnz,
                                                 alpha,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 info,
                                                 x,
                                                 beta,
                                                 y_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                 trans,
                                                 M,
                                                 N,
                                                 nnz,
                                                 alpha,
                                                 descr,
                                                 csr_val,
                                                 csr_row_ptr,
                                                 csr_col_ind,
                                                 info,
                                                 x,
                                                 beta,
                                                 y_2));

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // CPU y
        std::vector<T> y_gold(M);
        for(rocsparse_int i = 0; i < M; i++)
        {
            y_gold[i] = ty[i];
        }

        // CPU csrmv
        host_csrmv<rocsparse_int, rocsparse_int, T>(M,
                                                    nnz,
                                                    *alpha,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    csr_val,
                                                    x,
                                                    *beta,
                                                    &y_gold[0],
                                                    base,
                                                    adaptive);

        near_check_general<T>(1, M, 1, &y_gold[0], y_1);
        near_check_general<T>(1, M, 1, &y_gold[0], y_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                     trans,
                                                     M,
                                                     N,
                                                     nnz,
                                                     alpha,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     info,
                                                     x,
                                                     beta,
                                                     y_1));
        }

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmv<T>(handle,
                                                     trans,
                                                     M,
                                                     N,
                                                     nnz,
                                                     alpha,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     info,
                                                     x,
                                                     beta,
                                                     y_1));
        }

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops
            = spmv_gflop_count(M, nnz, *beta != static_cast<T>(0)) / gpu_time_used * 1e6;
        double gpu_gbyte
            = csrmv_gbyte_count<T>(M, N, nnz, *beta != static_cast<T>(0)) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "Algorithm" << std::setw(12) << "GFlop/s" << std::setw(12) << "GB/s"
                  << std::setw(12) << "msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz
                  << std::setw(12) << *alpha << std::setw(12) << *beta << std::setw(12)
                  << (adaptive ? "adaptive" : "stream") << std::setw(12) << gpu_gflops
                  << std::setw(12) << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // If adaptive, clear analysis data
    if(adaptive)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    }

    CHECK_HIP_ERROR(hipFree(csr_row_ptr));
    CHECK_HIP_ERROR(hipFree(csr_col_ind));
    CHECK_HIP_ERROR(hipFree(csr_val));
    CHECK_HIP_ERROR(hipFree(x));
    CHECK_HIP_ERROR(hipFree(y_1));
    CHECK_HIP_ERROR(hipFree(y_2));
    CHECK_HIP_ERROR(hipFree(alpha));
    CHECK_HIP_ERROR(hipFree(beta));
}

#define INSTANTIATE(TYPE)                                                    \
    template void testing_csrmv_managed_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrmv_managed<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

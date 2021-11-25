/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_gemvi_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    T h_alpha = static_cast<T>(1);
    T h_beta  = static_cast<T>(1);

    rocsparse_handle     handle = local_handle;
    rocsparse_operation  trans  = rocsparse_operation_none;
    rocsparse_index_base base   = rocsparse_index_base_zero;
    rocsparse_int        m      = safe_size;
    rocsparse_int        n      = safe_size;
    rocsparse_int        nnz    = safe_size;
    const T*             alpha  = &h_alpha;
    const T*             A      = (const T*)0x4;
    rocsparse_int        lda    = safe_size;
    const rocsparse_int* x_ind  = (const rocsparse_int*)0x4;
    const T*             x_val  = (const T*)0x4;
    const T*             beta   = &h_beta;
    T*                   y      = (T*)0x4;
    void*                buffer = (void*)0x4;

#define PARAMS handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, base, buffer

    auto_testing_bad_arg(rocsparse_gemvi<T>, PARAMS);

    {
        auto tmp = trans;
        trans    = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gemvi<T>(PARAMS), rocsparse_status_not_implemented);
        trans = tmp;
    }

    // nnz cannot be larger than n
    {
        auto tmp = nnz;
        nnz      = n + 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gemvi<T>(PARAMS), rocsparse_status_invalid_size);
        nnz = tmp;
    }

    // lda cannot be lesser than m in non transposed case
    {
        auto tmp = lda;
        lda      = m - 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gemvi<T>(PARAMS), rocsparse_status_invalid_size);
        lda = tmp;
    }

    // lda cannot be lesser than n in transposed case
    {
        auto tmp1 = lda;
        auto tmp2 = trans;
        lda       = n - 1;
        trans     = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_gemvi<T>(PARAMS), rocsparse_status_invalid_size);
        lda   = tmp1;
        trans = tmp2;
    }
}

template <typename T>
void testing_gemvi(const Arguments& arg)
{
    rocsparse_int        M     = arg.M;
    rocsparse_int        N     = arg.N;
    rocsparse_operation  trans = arg.transA;
    rocsparse_index_base base  = arg.baseA;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Vector sparsity of 33%
    rocsparse_int nnz = N * 0.33;
    rocsparse_int lda = (trans == rocsparse_operation_none) ? M : N;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N < 0)
    {
        device_vector<T> dy(std::max(100, M));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_gemvi_buffer_size<T>(handle, trans, M, N, nnz, &buffer_size));

        void* buffer;
        CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size));
        EXPECT_ROCSPARSE_STATUS(rocsparse_gemvi<T>(handle,
                                                   trans,
                                                   M,
                                                   N,
                                                   &h_alpha,
                                                   nullptr,
                                                   lda,
                                                   nnz,
                                                   nullptr,
                                                   nullptr,
                                                   &h_beta,
                                                   dy,
                                                   base,
                                                   buffer),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        CHECK_HIP_ERROR(hipFree(buffer));

        return;
    }

    // Allocate host memory
    host_vector<T>             hA(M * N);
    host_vector<T>             hx_val(nnz);
    host_vector<rocsparse_int> hx_ind(nnz);
    host_vector<T>             hy_1(M);
    host_vector<T>             hy_2(M);
    host_vector<T>             hy_gold(M);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, 1, std::max(N, ((rocsparse_int)1)));
    rocsparse_init<T>(hx_val, 1, nnz, 1);
    rocsparse_init<T>(hy_1, 1, M, 1);
    rocsparse_init<T>(hA, M, N, lda, 1);
    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<rocsparse_int> dx_ind(nnz);
    device_vector<T>             dx_val(nnz);
    device_vector<T>             dA(M * N);
    device_vector<T>             dy_1(M);
    device_vector<T>             dy_2(M);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * M * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_gemvi_buffer_size<T>(handle, trans, M, N, nnz, &buffer_size));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_gemvi<T>(handle,
                                                 trans,
                                                 M,
                                                 N,
                                                 &h_alpha,
                                                 dA,
                                                 lda,
                                                 nnz,
                                                 dx_val,
                                                 dx_ind,
                                                 &h_beta,
                                                 dy_1,
                                                 base,
                                                 buffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_gemvi<T>(handle,
                                                 trans,
                                                 M,
                                                 N,
                                                 d_alpha,
                                                 dA,
                                                 lda,
                                                 nnz,
                                                 dx_val,
                                                 dx_ind,
                                                 d_beta,
                                                 dy_2,
                                                 base,
                                                 buffer));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU gemvi
        host_gemvi<rocsparse_int, T>(
            M, N, h_alpha, hA, lda, nnz, hx_val, hx_ind, h_beta, hy_gold, base);

        hy_gold.near_check(hy_1);
        hy_gold.near_check(hy_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gemvi<T>(handle,
                                                     trans,
                                                     M,
                                                     N,
                                                     &h_alpha,
                                                     dA,
                                                     lda,
                                                     nnz,
                                                     dx_val,
                                                     dx_ind,
                                                     &h_beta,
                                                     dy_1,
                                                     base,
                                                     buffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gemvi<T>(handle,
                                                     trans,
                                                     M,
                                                     N,
                                                     &h_alpha,
                                                     dA,
                                                     lda,
                                                     nnz,
                                                     dx_val,
                                                     dx_ind,
                                                     &h_beta,
                                                     dy_1,
                                                     base,
                                                     buffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops = gemvi_gflop_count(M, nnz) / gpu_time_used * 1e6;
        double gpu_gbyte  = gemvi_gbyte_count<T>((trans == rocsparse_operation_none) ? M : N,
                                                nnz,
                                                h_beta != static_cast<T>(0))
                           / gpu_time_used * 1e6;

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "nnz",
                            nnz,
                            "trans",
                            rocsparse_operation2string(trans),
                            "alpha",
                            h_alpha,
                            "beta",
                            h_beta,
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            gpu_time_used / 1e3,
                            "iter",
                            number_hot_calls,
                            "verified",
                            (arg.unit_check ? "yes" : "no"));
    }

    CHECK_HIP_ERROR(hipFree(buffer));
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_gemvi_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gemvi<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

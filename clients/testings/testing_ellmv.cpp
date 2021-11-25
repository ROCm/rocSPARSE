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

#include "rocsparse_enum.hpp"
#include "testing.hpp"

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_ellmv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    const T h_alpha = static_cast<T>(1);
    const T h_beta  = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle            = local_handle;
    rocsparse_operation       trans             = rocsparse_operation_none;
    rocsparse_int             m                 = safe_size;
    rocsparse_int             n                 = safe_size;
    const T*                  alpha_device_host = &h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  ell_val           = (const T*)0x4;
    const rocsparse_int*      ell_col_ind       = (const rocsparse_int*)0x4;
    rocsparse_int             ell_width         = safe_size;
    const T*                  x                 = (const T*)0x4;
    const T*                  beta_device_host  = &h_beta;
    T*                        y                 = (T*)0x4;

#define PARAMS                                                                         \
    handle, trans, m, n, alpha_device_host, descr, ell_val, ell_col_ind, ell_width, x, \
        beta_device_host, y

    auto_testing_bad_arg(rocsparse_ellmv<T>, PARAMS);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_ellmv<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }

#undef PARAMS
}

template <typename T>
void testing_ellmv(const Arguments& arg)
{
    rocsparse_int        M     = arg.M;
    rocsparse_int        N     = arg.N;
    rocsparse_operation  trans = arg.transA;
    rocsparse_index_base base  = arg.baseA;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

#define PARAMS(alpha_, A_, x_, beta_, y_) \
    handle, trans, A_.m, A_.n, alpha_, descr, A_.val, A_.ind, A_.width, x_, beta_, y_

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {

        static const size_t safe_size = 100;

        device_ell_matrix<T> dA;
        device_vector<T>     dx, dy;

        dA.m     = M;
        dA.n     = N;
        dA.nnz   = safe_size;
        dA.width = safe_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_ellmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)),
            ((M < 0 || N < 0 || dA.width < 0) || ((M == 0 || N == 0) && dA.width != 0))
                ? rocsparse_status_invalid_size
                : rocsparse_status_success);

        dA.width = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_ellmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)),
            ((M < 0 || N < 0 || dA.width < 0) || ((M == 0 || N == 0) && dA.width != 0))
                ? rocsparse_status_invalid_size
                : rocsparse_status_success);

        return;
    }

    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_ell_matrix<T> hA;

    matrix_factory.init_ell(hA, M, N, base);

    host_dense_matrix<T> hx((trans == rocsparse_operation_none) ? N : M, 1);
    host_dense_matrix<T> hy((trans == rocsparse_operation_none) ? M : N, 1);

    rocsparse_matrix_utils::init(hx);
    rocsparse_matrix_utils::init(hy);

    device_ell_matrix<T>   dA(hA);
    device_dense_matrix<T> dx(hx), dy(hy);

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_ellmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));

        {
            host_dense_matrix<T> hy_copy(hy);
            // CPU ellmv
            host_ellmv<rocsparse_int, T>(
                trans, hA.m, hA.n, *h_alpha, hA.ind, hA.val, hA.width, hx, *h_beta, hy, hA.base);
            hy.near_check(dy);
            dy = hy_copy;
        }

        // Pointer mode device
        device_scalar<T> d_alpha(h_alpha), d_beta(h_beta);
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_ellmv<T>(PARAMS(d_alpha, dA, dx, d_beta, dy)));
        hy.near_check(dy);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_ellmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_ellmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(M, dA.nnz, *h_beta != static_cast<T>(0));
        double gbyte_count = ellmv_gbyte_count<T>(M, N, dA.nnz, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "ELL_nnz",
                            dA.nnz,
                            "ELL_width",
                            dA.width,
                            "alpha",
                            *h_alpha,
                            "beta",
                            *h_beta,
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            (arg.unit_check ? "yes" : "no"));
    }
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_ellmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_ellmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

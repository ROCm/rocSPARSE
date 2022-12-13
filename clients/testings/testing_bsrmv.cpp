/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_enum.hpp"

template <typename T>
void testing_bsrmv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 10;
    const T             h_alpha   = static_cast<T>(1);
    const T             h_beta    = static_cast<T>(1);
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    rocsparse_handle          handle            = local_handle;
    rocsparse_direction       dir               = rocsparse_direction_column;
    rocsparse_operation       trans             = rocsparse_operation_none;
    rocsparse_int             mb                = safe_size;
    rocsparse_int             nb                = safe_size;
    rocsparse_int             nnzb              = safe_size;
    const T*                  alpha_device_host = (const T*)&h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  bsr_val           = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind       = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim         = safe_size;
    rocsparse_mat_info        info              = local_info;
    const T*                  x                 = (const T*)0x4;
    const T*                  beta_device_host  = (const T*)&h_beta;
    T*                        y                 = (T*)0x4;

#define PARAMS_ANALYSIS \
    handle, dir, trans, mb, nb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, info

    auto_testing_bad_arg(rocsparse_bsrmv_ex_analysis<T>, PARAMS_ANALYSIS);

#define PARAMS                                                                                     \
    handle, dir, trans, mb, nb, nnzb, alpha_device_host, descr, bsr_val, bsr_row_ptr, bsr_col_ind, \
        block_dim, info, x, beta_device_host, y

    {
        static constexpr int num_exclusions  = 1;
        static constexpr int exclude_args[1] = {12};
        auto_testing_bad_arg(rocsparse_bsrmv_ex<T>, num_exclusions, exclude_args, PARAMS);
    }

    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex_clear(nullptr, info),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex<T>(PARAMS),
                                    rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    for(auto operation : rocsparse_operation_t::values)
    {
        if(operation != rocsparse_operation_none)
        {
            trans = operation;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex<T>(PARAMS),
                                    rocsparse_status_not_implemented);
        }
    }
    trans = rocsparse_operation_none;

#undef PARAMS_ANALYSIS
#undef PARAMS

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex_analysis<T>(handle,
                                                           dir,
                                                           trans,
                                                           mb,
                                                           nb,
                                                           nnzb,
                                                           descr,
                                                           nullptr,
                                                           bsr_row_ptr,
                                                           nullptr,
                                                           block_dim,
                                                           info),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex<T>(handle,
                                                  dir,
                                                  trans,
                                                  mb,
                                                  nb,
                                                  nnzb,
                                                  alpha_device_host,
                                                  descr,
                                                  nullptr,
                                                  bsr_row_ptr,
                                                  nullptr,
                                                  block_dim,
                                                  info,
                                                  x,
                                                  beta_device_host,
                                                  y),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_bsrmv(const Arguments& arg)
{
    rocsparse_int          M         = arg.M;
    rocsparse_int          N         = arg.N;
    rocsparse_direction    dir       = arg.direction;
    rocsparse_operation    trans     = arg.transA;
    rocsparse_index_base   base      = arg.baseA;
    rocsparse_int          block_dim = arg.block_dim;
    rocsparse_storage_mode storage   = arg.storage;

    host_scalar<T> h_alpha(arg.get_alpha<T>()), h_beta(arg.get_beta<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Set storage mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, storage));

    // Create matrix info
    rocsparse_local_mat_info info;

    // BSR dimensions
    rocsparse_int mb = (block_dim > 0) ? (M + block_dim - 1) / block_dim : 0;
    rocsparse_int nb = (block_dim > 0) ? (N + block_dim - 1) / block_dim : 0;
    // Argument sanity check before allocating invalid memory
#define PARAMS_ANALYSIS(A_)                                                                  \
    handle, A_.block_direction, trans, A_.mb, A_.nb, A_.nnzb, descr, A_.val, A_.ptr, A_.ind, \
        A_.row_block_dim, info
#define PARAMS(alpha_, A_, x_, beta_, y_)                                                    \
    handle, A_.block_direction, trans, A_.mb, A_.nb, A_.nnzb, alpha_, descr, A_.val, A_.ptr, \
        A_.ind, A_.row_block_dim, info, x_, beta_, y_

    if(mb == 0 || nb == 0 || M == 0 || N == 0 || block_dim == 0)
    {
        device_gebsr_matrix<T> dA;
        dA.block_direction = dir;
        dA.mb              = mb;
        dA.nb              = nb;
        dA.nnzb            = 10;
        dA.row_block_dim   = block_dim;
        dA.col_block_dim   = block_dim;

        device_dense_matrix<T> dx;
        device_dense_matrix<T> dy;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv_ex<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)),
                                rocsparse_status_success);
        return;
    }

    // Wavefront size
    int dev;
    CHECK_HIP_ERROR(hipGetDevice(&dev));

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

    bool                        type = (prop.warpSize == 32) ? (arg.timing ? false : true) : false;
    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, type, full_rank);

    //
    // Declare and initialize matrices.
    //
    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;

    matrix_factory.init_bsr(hA, dA, mb, nb, base);

    M = dA.mb * dA.row_block_dim;
    N = dA.nb * dA.col_block_dim;

    host_dense_matrix<T> hx(N, 1), hy(M, 1);

    rocsparse_matrix_utils::init(hx);
    rocsparse_matrix_utils::init(hy);

    device_dense_matrix<T> dx(hx), dy(hy);

    // bsrmv_analysis (Optional)
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrmv_ex_analysis<T>(PARAMS_ANALYSIS(dA)));

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrmv_ex<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));

        {
            host_dense_matrix<T> hy_copy(hy);
            // CPU bsrmv
            host_bsrmv<T, rocsparse_int, rocsparse_int, T, T, T>(dir,
                                                                 trans,
                                                                 hA.mb,
                                                                 hA.nb,
                                                                 hA.nnzb,
                                                                 *h_alpha,
                                                                 hA.ptr,
                                                                 hA.ind,
                                                                 hA.val,
                                                                 hA.row_block_dim,
                                                                 hx,
                                                                 *h_beta,
                                                                 hy,
                                                                 base);

            hy.near_check(dy);
            dy = hy_copy;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        device_scalar<T> d_alpha(h_alpha), d_beta(h_beta);
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrmv_ex<T>(PARAMS(d_alpha, dA, dx, d_beta, dy)));
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
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrmv_ex<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrmv_ex<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(
            M, dA.nnzb * dA.row_block_dim * dA.col_block_dim, *h_beta != static_cast<T>(0));
        double gbyte_count = bsrmv_gbyte_count<T>(
            dA.mb, dA.nb, dA.nnzb, dA.row_block_dim, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "BSR dim",
                            dA.row_block_dim,
                            "dir",
                            rocsparse_direction2string(dA.block_direction),
                            "alpha",
                            *h_alpha,
                            "beta",
                            *h_beta,
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }

#undef PARAMS_ANALYSIS
#undef PARAMS
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_bsrmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrmv_extra(const Arguments& arg) {}

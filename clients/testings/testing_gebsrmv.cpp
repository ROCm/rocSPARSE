/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_gebsrmv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 10;
    const T             h_alpha   = static_cast<T>(1);
    const T             h_beta    = static_cast<T>(1);
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

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
    rocsparse_int             row_block_dim     = safe_size;
    rocsparse_int             col_block_dim     = safe_size;
    const T*                  x                 = (const T*)0x4;
    const T*                  beta_device_host  = (const T*)&h_beta;
    T*                        y                 = (T*)0x4;

#define PARAMS                                                                                     \
    handle, dir, trans, mb, nb, nnzb, alpha_device_host, descr, bsr_val, bsr_row_ptr, bsr_col_ind, \
        row_block_dim, col_block_dim, x, beta_device_host, y

    auto_testing_bad_arg(rocsparse_gebsrmv<T>, PARAMS);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }

#undef PARAMS
}

template <typename T>
void testing_gebsrmv(const Arguments& arg)
{
    rocsparse_int        M     = arg.M;
    rocsparse_int        N     = arg.N;
    rocsparse_operation  trans = arg.transA;
    rocsparse_index_base base  = arg.baseA;

    host_scalar<T> h_alpha(arg.get_alpha<T>()), h_beta(arg.get_beta<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // BSR dimensions

    // Argument sanity check before allocating invalid memory
#define PARAMS(alpha_, A_, x_, beta_, y_)                                                    \
    handle, A_.block_direction, trans, A_.mb, A_.nb, A_.nnzb, alpha_, descr, A_.val, A_.ptr, \
        A_.ind, A_.row_block_dim, A_.col_block_dim, x_, beta_, y_

    {
        rocsparse_int row_block_dim = arg.row_block_dimA;
        rocsparse_int col_block_dim = arg.col_block_dimA;
        rocsparse_int mb = (row_block_dim > 0) ? (M + row_block_dim - 1) / row_block_dim : 0;
        rocsparse_int nb = (col_block_dim > 0) ? (N + col_block_dim - 1) / col_block_dim : 0;
        if(mb <= 0 || nb <= 0 || M <= 0 || N <= 0 || row_block_dim <= 0 || col_block_dim <= 0)
        {
            rocsparse_direction dir = arg.direction;

            device_gebsr_matrix<T> dA;
            dA.block_direction = dir;
            dA.mb              = mb;
            dA.nb              = nb;
            dA.nnzb            = 10;
            dA.row_block_dim   = row_block_dim;
            dA.col_block_dim   = col_block_dim;

            device_dense_matrix<T> dx;
            device_dense_matrix<T> dy;

            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)),
                                    (mb < 0 || nb < 0 || row_block_dim < 0 || col_block_dim < 0)
                                        ? rocsparse_status_invalid_size
                                        : rocsparse_status_success);
            return;
        }
    }

    // Wavefront size
    int dev;
    hipGetDevice(&dev);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);

    bool                  type      = (prop.warpSize == 32) ? (arg.timing ? false : true) : false;
    static constexpr bool full_rank = false;

    rocsparse_matrix_factory<T> matrix_factory(arg, type, full_rank);
    //
    // A
    //
    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;

    {
        rocsparse_int row_block_dim = arg.row_block_dimA;
        rocsparse_int col_block_dim = arg.col_block_dimA;
        rocsparse_int mb = (row_block_dim > 0) ? (M + row_block_dim - 1) / row_block_dim : 0;
        rocsparse_int nb = (col_block_dim > 0) ? (N + col_block_dim - 1) / col_block_dim : 0;
        matrix_factory.init_gebsr(hA, dA, mb, nb);
    }

    if(!arg.unit_check)
    {
        hA.~host_gebsr_matrix<T>();
    }

    M = dA.mb * dA.row_block_dim;
    N = dA.nb * dA.col_block_dim;

    //
    // X
    //
    host_dense_matrix<T> hx(N, 1);
    rocsparse_matrix_utils::init(hx);
    device_dense_matrix<T> dx(hx);
    if(!arg.unit_check)
    {
        hx.~host_dense_matrix<T>();
    }

    //
    // Y
    //
    host_dense_matrix<T> hy(M, 1);
    rocsparse_matrix_utils::init(hy);
    device_dense_matrix<T> dy(hy);
    if(!arg.unit_check)
    {
        hy.~host_dense_matrix<T>();
    }

#define PARAMS(alpha_, A_, x_, beta_, y_)                                                    \
    handle, A_.block_direction, trans, A_.mb, A_.nb, A_.nnzb, alpha_, descr, A_.val, A_.ptr, \
        A_.ind, A_.row_block_dim, A_.col_block_dim, x_, beta_, y_

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));

        {
            host_dense_matrix<T> hy_copy(hy);
            // CPU gebsrmv
            host_gebsrmv<T>(hA.block_direction,
                            trans,
                            hA.mb,
                            hA.nb,
                            hA.nnzb,
                            *h_alpha,
                            hA.ptr,
                            hA.ind,
                            hA.val,
                            hA.row_block_dim,
                            hA.col_block_dim,
                            hx,
                            *h_beta,
                            hy,
                            base);
            hy.near_check(dy);
            dy.transfer_from(hy_copy);
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        device_scalar<T> d_alpha(h_alpha), d_beta(h_beta);
        CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmv<T>(PARAMS(d_alpha, dA, dx, d_beta, dy)));
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
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsrmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(
            M, dA.nnzb * dA.row_block_dim * dA.col_block_dim, *h_beta != static_cast<T>(0));
        double gbyte_count = gebsrmv_gbyte_count<T>(
            dA.mb, dA.nb, dA.nnzb, dA.row_block_dim, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "BSR nnz",
                            dA.nnzb,
                            "rblockdim",
                            dA.row_block_dim,
                            "cblockdim",
                            dA.col_block_dim,
                            "dir",
                            rocsparse_direction2string(dA.block_direction),
                            "alpha",
                            *h_alpha,
                            "beta",
                            *h_beta,
                            "GFlop/s",
                            gpu_gflops,
                            "GB/s",
                            gpu_gbyte,
                            "msec",
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            (arg.unit_check ? "yes" : "no"));
    }

#undef PARAMS
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_gebsrmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsrmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

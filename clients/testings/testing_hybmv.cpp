/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_hybmv_bad_arg(const Arguments& arg)
{

    const T h_alpha = static_cast<T>(1);
    const T h_beta  = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;
    rocsparse_local_hyb_mat   local_hyb;

    rocsparse_handle          handle            = local_handle;
    rocsparse_operation       trans             = rocsparse_operation_none;
    const T*                  alpha_device_host = &h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const rocsparse_hyb_mat   hyb               = local_hyb;
    const T*                  x                 = (const T*)0x4;
    const T*                  beta_device_host  = &h_beta;
    T*                        y                 = (T*)0x4;

#define PARAMS handle, trans, alpha_device_host, descr, hyb, x, beta_device_host, y

    auto_testing_bad_arg(rocsparse_hybmv<T>, PARAMS);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_hybmv<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_hybmv<T>(PARAMS), rocsparse_status_not_implemented);

#undef PARAMS
}

template <typename T>
void testing_hybmv(const Arguments& arg)
{
    rocsparse_int           M              = arg.M;
    rocsparse_int           N              = arg.N;
    rocsparse_operation     trans          = arg.transA;
    rocsparse_index_base    base           = arg.baseA;
    rocsparse_hyb_partition part           = arg.part;
    rocsparse_int           user_ell_width = arg.algo;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create hyb matrix
    rocsparse_local_hyb_mat hyb;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

#define PARAMS(alpha_, x_, beta_, y_) handle, trans, alpha_, descr, hyb, x_, beta_, y_

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);

        if(!dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_hybmv<T>(handle, trans, h_alpha, descr, hyb, dx, h_beta, dy));
        return;
    }

    rocsparse_matrix_factory<T> matrix_factory(arg, arg.timing ? false : true, false);

    bool          conform;
    rocsparse_int nnz;
    matrix_factory.init_hyb(hyb, M, N, nnz, base, conform);
    if(!conform)
    {
        return;
    }

    host_dense_matrix<T> hx((trans == rocsparse_operation_none) ? N : M, 1);
    host_dense_matrix<T> hy((trans == rocsparse_operation_none) ? M : N, 1);

    rocsparse_matrix_utils::init_exact(hx);
    rocsparse_matrix_utils::init_exact(hy);

    device_dense_matrix<T> dx(hx), dy(hy);

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_hybmv<T>(PARAMS(h_alpha, dx, h_beta, dy)));

        {
            // CPU hybmv
            rocsparse_hyb_mat ptr  = hyb;
            test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

            rocsparse_int ell_width = dhyb->ell_width;
            rocsparse_int ell_nnz   = dhyb->ell_nnz;
            rocsparse_int coo_nnz   = dhyb->coo_nnz;

            host_ell_matrix<T> hA_ell;
            host_coo_matrix<T> hA_coo;

            if(ell_nnz > 0)
            {
                hA_ell.define(M, N, ell_width, base);
                hA_ell.ind.template transfer_from<memory_mode::device>(
                    (const rocsparse_int*)dhyb->ell_col_ind);
                hA_ell.val.template transfer_from<memory_mode::device>((const T*)dhyb->ell_val);
            }

            if(coo_nnz > 0)
            {
                hA_coo.define(M, N, coo_nnz, base);
                hA_coo.row_ind.template transfer_from<memory_mode::device>(
                    (const rocsparse_int*)dhyb->coo_row_ind);
                hA_coo.col_ind.template transfer_from<memory_mode::device>(
                    (const rocsparse_int*)dhyb->coo_col_ind);
                hA_coo.val.template transfer_from<memory_mode::device>((const T*)dhyb->coo_val);
            }

            host_dense_matrix<T> hy_copy(hy);
            // CPU hybmv
            host_hybmv<T>(trans,
                          M,
                          N,
                          *h_alpha,
                          hA_ell.nnz,
                          hA_ell.ind,
                          hA_ell.val,
                          hA_ell.width,
                          hA_coo.nnz,
                          hA_coo.row_ind,
                          hA_coo.col_ind,
                          hA_coo.val,
                          hx,
                          *h_beta,
                          hy,
                          base);

            hy.near_check(dy);
            dy.transfer_from(hy_copy);
        }

        // Pointer mode device
        device_scalar<T> d_alpha(h_alpha), d_beta(h_beta);
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_hybmv<T>(PARAMS(d_alpha, dx, d_beta, dy)));
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
            CHECK_ROCSPARSE_ERROR(rocsparse_hybmv<T>(PARAMS(h_alpha, dx, h_beta, dy)));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_hybmv<T>(PARAMS(h_alpha, dx, h_beta, dy)));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spmv_gflop_count(M, nnz, *h_beta != static_cast<T>(0));
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        if(part == rocsparse_hyb_partition_user)
        {
            {
                rocsparse_hyb_mat ptr  = hyb;
                test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);
                user_ell_width         = dhyb->ell_width;
            }

            display_timing_info("M",
                                M,
                                "N",
                                N,
                                "nnz",
                                nnz,
                                "alpha",
                                *h_alpha,
                                "beta",
                                *h_beta,
                                "partition",
                                rocsparse_partition2string(part),
                                "width",
                                user_ell_width,
                                s_timing_info_perf,
                                gpu_gflops,
                                s_timing_info_time,
                                get_gpu_time_msec(gpu_time_used));
        }
        else
        {

            display_timing_info("M",
                                M,
                                "N",
                                N,
                                "nnz",
                                nnz,
                                "alpha",
                                *h_alpha,
                                "beta",
                                *h_beta,
                                "partition",
                                rocsparse_partition2string(part),
                                s_timing_info_perf,
                                gpu_gflops,
                                s_timing_info_time,
                                get_gpu_time_msec(gpu_time_used));
        }
    }
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_hybmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_hybmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_hybmv_extra(const Arguments& arg) {}

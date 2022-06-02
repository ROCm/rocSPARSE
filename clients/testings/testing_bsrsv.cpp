/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
void testing_bsrsv_bad_arg(const Arguments& arg)
{

    static const size_t safe_size = 100;

    const T h_alpha = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_operation       trans       = rocsparse_operation_none;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nnzb        = safe_size;
    const T*                  alpha       = &h_alpha;
    const rocsparse_mat_descr descr       = local_descr;
    const T*                  bsr_val     = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_mat_info        info        = local_info;
    rocsparse_int             block_dim   = safe_size;
    const T*                  x           = (const T*)0x4;
    T*                        y           = (T*)0x4;
    rocsparse_analysis_policy analysis    = rocsparse_analysis_policy_reuse;
    rocsparse_solve_policy    solve       = rocsparse_solve_policy_auto;
    size_t                    buffer_size;
    size_t*                   p_buffer_size = &buffer_size;
    void*                     temp_buffer   = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                   \
    handle, dir, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, info, \
        p_buffer_size

#define PARAMS_ANALYSIS                                                                      \
    handle, dir, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, info, \
        analysis, solve, temp_buffer

#define PARAMS_SOLVE                                                                          \
    handle, dir, trans, mb, nnzb, alpha, descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, \
        info, x, y, solve, temp_buffer

    //
    // Call solve before analysis
    //
    auto_testing_bad_arg(rocsparse_bsrsv_buffer_size<T>, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_bsrsv_analysis<T>, PARAMS_ANALYSIS);
    auto_testing_bad_arg(rocsparse_bsrsv_solve<T>, PARAMS_SOLVE);

    //
    // Not implemented cases.
    //
    for(auto operation : rocsparse_operation_t::values)
    {
        if(operation != rocsparse_operation_none && operation != rocsparse_operation_transpose)
        {
            {
                auto tmp = trans;
                trans    = operation;
                EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS),
                                        rocsparse_status_not_implemented);
                EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE),
                                        rocsparse_status_not_implemented);
                trans = tmp;
            }
        }
    }

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE),
                                    rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_not_implemented);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE),
                            rocsparse_status_not_implemented);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE

    // Test rocsparse_bsrsv_zero_pivot()
    rocsparse_int position;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(nullptr, info, &position),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, nullptr, &position),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_bsrsv_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_clear(nullptr, info), rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     dir,
                                                     trans,
                                                     mb,
                                                     nnzb,
                                                     alpha,
                                                     descr,
                                                     nullptr,
                                                     bsr_row_ptr,
                                                     nullptr,
                                                     block_dim,
                                                     info,
                                                     x,
                                                     y,
                                                     solve,
                                                     temp_buffer),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_bsrsv(const Arguments& arg)
{
    auto                        tol       = get_near_check_tol<T>(arg);
    static constexpr bool       to_int    = false;
    static constexpr bool       full_rank = true;
    rocsparse_matrix_factory<T> matrix_factory(arg, to_int, full_rank);

    rocsparse_int M         = arg.M;
    rocsparse_int N         = arg.N;
    rocsparse_int block_dim = arg.block_dim;

    rocsparse_operation       trans = arg.transA;
    rocsparse_direction       dir   = arg.direction;
    rocsparse_diag_type       diag  = arg.diag;
    rocsparse_fill_mode       uplo  = arg.uplo;
    rocsparse_analysis_policy apol  = arg.apol;
    rocsparse_solve_policy    spol  = arg.spol;
    rocsparse_index_base      base  = arg.baseA;

    host_scalar<T> h_alpha(arg.get_alpha<T>());

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix diag type
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr, diag));

    // Set matrix fill mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr, uplo));

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

#define PARAMS_BUFFER_SIZE(A_)                                                                 \
    handle, dir, trans, A_.mb, A_.nnzb, descr, A_.val, A_.ptr, A_.ind, A_.row_block_dim, info, \
        &buffer_size
#define PARAMS_ANALYSIS(A_)                                                                    \
    handle, dir, trans, A_.mb, A_.nnzb, descr, A_.val, A_.ptr, A_.ind, A_.row_block_dim, info, \
        apol, spol, dbuffer
#define PARAMS_SOLVE(alpha_, A_, x_, y_)                                                         \
    handle, dir, trans, A_.mb, A_.nnzb, alpha_, descr, A_.val, A_.ptr, A_.ind, A_.row_block_dim, \
        info, x_, y_, spol, dbuffer

    // Set matrix diag type
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descr, diag));

    // Set matrix fill mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr, uplo));

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // BSR dimensions
    const size_t  safe_size = 16;
    rocsparse_int mb        = (block_dim > 0) ? ((M + block_dim - 1) / block_dim) : safe_size;
    rocsparse_int nb        = (block_dim > 0) ? ((N + block_dim - 1) / block_dim) : safe_size;

    // Argument sanity check before allocating invalid memory
    if(mb <= 0 || nb <= 0 || M <= 0 || N <= 0 || block_dim <= 0)
    {
        rocsparse_int          pivot;
        size_t                 buffer_size;
        device_vector<T>       dx, dy, dbuffer;
        device_gebsr_matrix<T> dA;
        dA.mb            = mb;
        dA.nb            = nb;
        dA.nnzb          = 10;
        dA.row_block_dim = block_dim;
        dA.col_block_dim = block_dim;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(PARAMS_BUFFER_SIZE(dA)),
                                (mb < 0 || nb < 0 || block_dim < 0) ? rocsparse_status_invalid_size
                                                                    : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)),
                                (mb < 0 || nb < 0 || block_dim < 0) ? rocsparse_status_invalid_size
                                                                    : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                                (mb < 0 || nb < 0 || block_dim < 0) ? rocsparse_status_invalid_size
                                                                    : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, &pivot),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_clear(handle, info), rocsparse_status_success);
        return;
    }
    // Allocate host memory for matrix
    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;

    // Sample matrix
    matrix_factory.init_bsr(hA, dA, mb, nb);

    M = dA.mb * dA.row_block_dim;
    N = dA.nb * dA.col_block_dim;

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    // Allocate host memory for vectors
    host_dense_matrix<T> hx(M, 1);
    rocsparse_matrix_utils::init_exact(hx);
    //    rocsparse_init<T>(hx.val,hx.m,1,1,0,1,static_cast<T>(-1),static_cast<T>(1));

    device_dense_matrix<T>     dx(hx), dy(M, 1);
    host_scalar<rocsparse_int> h_analysis_pivot, h_solve_pivot;

    // Obtain required buffer size
    void* dbuffer;
    {
        size_t buffer_size;
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_buffer_size<T>(PARAMS_BUFFER_SIZE(dA)));
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
    }

    if(arg.unit_check)
    {
        host_scalar<rocsparse_int> analysis_no_pivot(-1);
        host_dense_matrix<T>       hy(M, 1);
        // CPU csrsv
        host_bsrsv<T>(trans,
                      dir,
                      hA.mb,
                      hA.nnzb,
                      *h_alpha,
                      hA.ptr,
                      hA.ind,
                      hA.val,
                      hA.row_block_dim,
                      hx,
                      hy,
                      diag,
                      uplo,
                      base,
                      h_analysis_pivot,
                      h_solve_pivot);

        // Pointer mode host
        {
            host_scalar<rocsparse_int> analysis_pivot;
            host_scalar<rocsparse_int> solve_pivot;
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, analysis_pivot),
                                    rocsparse_status_success);
            analysis_no_pivot.unit_check(analysis_pivot);

            //
            // Call before analysis
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)),
                                    rocsparse_status_invalid_pointer);

            // Call it twice.
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            {
                auto st = rocsparse_bsrsv_zero_pivot(handle, info, analysis_pivot);
                EXPECT_ROCSPARSE_STATUS(st,
                                        (*analysis_pivot != -1) ? rocsparse_status_zero_pivot
                                                                : rocsparse_status_success);
            }
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)));
            {
                auto st = rocsparse_bsrsv_zero_pivot(handle, info, solve_pivot);
                EXPECT_ROCSPARSE_STATUS(st,
                                        (*solve_pivot != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);
            }
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            h_analysis_pivot.unit_check(analysis_pivot);
            h_solve_pivot.unit_check(solve_pivot);
        }

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            hy.near_check(dy, tol);
        }

        //
        // RESET MAT INFO.
        //
        info.reset();
        {
            device_scalar<rocsparse_int> d_analysis_pivot;
            device_scalar<rocsparse_int> d_solve_pivot;
            device_scalar<T>             d_alpha(h_alpha);

            // Pointer mode device
            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, d_analysis_pivot),
                                    rocsparse_status_success);
            analysis_no_pivot.unit_check(d_analysis_pivot);
            //
            // Call before analysis
            //
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE(d_alpha, dA, dx, dy)),
                                    rocsparse_status_invalid_pointer);

            // Call it twice.
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, d_analysis_pivot),
                                    (*h_analysis_pivot != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE(d_alpha, dA, dx, dy)));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, d_solve_pivot),
                                    (*h_solve_pivot != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            h_analysis_pivot.unit_check(d_analysis_pivot);
            h_solve_pivot.unit_check(d_solve_pivot);
        }

        if(*h_analysis_pivot == -1 && *h_solve_pivot == -1)
        {
            hy.near_check(dy, tol);
        }

        //
        // RESET MAT INFO
        //
        info.reset();

        //
        // A BIT MORE FOR CODE COVERAGE, WE ONLY DO ANALYSIS FOR INFO ASSIGNMENT.
        //
        {
            void*  buffer = nullptr;
            size_t buffer_size;
            int    boost       = arg.numericboost;
            T      h_boost_tol = static_cast<T>(arg.boosttol);
            T      h_boost_val = arg.get_boostval<T>();

            rocsparse_matrix_utils::bsrilu0<T>(descr,
                                               dA,
                                               info,
                                               apol,
                                               spol,
                                               boost,
                                               h_boost_val,
                                               h_boost_tol,
                                               &buffer_size,
                                               buffer,
                                               rocsparse_matrix_utils::bsrilu0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
            rocsparse_matrix_utils::bsrilu0<T>(descr,
                                               dA,
                                               info,
                                               apol,
                                               spol,
                                               boost,
                                               h_boost_val,
                                               h_boost_tol,
                                               &buffer_size,
                                               buffer,
                                               rocsparse_matrix_utils::bsrilu0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        }

        info.reset();

        //
        // A BIT MORE FOR CODE COVERAGE, WE ONLY DO ANALYSIS FOR INFO ASSIGNMENT.
        //
        {
            void*  buffer = nullptr;
            size_t buffer_size;
            rocsparse_matrix_utils::bsric0<T>(descr,
                                              dA,
                                              info,
                                              apol,
                                              spol,
                                              &buffer_size,
                                              buffer,
                                              rocsparse_matrix_utils::bsric0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));
            rocsparse_matrix_utils::bsric0<T>(descr,
                                              dA,
                                              info,
                                              apol,
                                              spol,
                                              &buffer_size,
                                              buffer,
                                              rocsparse_matrix_utils::bsric0_analysis);
            CHECK_HIP_ERROR(rocsparse_hipFree(buffer));

            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_zero_pivot(handle, info, h_analysis_pivot));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_zero_pivot(handle, info, h_solve_pivot));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_clear(handle, info));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(PARAMS_ANALYSIS(dA)));
        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(PARAMS_SOLVE(h_alpha, dA, dx, dy)));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gflop_count
            = csrsv_gflop_count(M, size_t(dA.nnzb) * dA.row_block_dim * dA.row_block_dim, diag);
        double gbyte_count = bsrsv_gbyte_count<T>(dA.mb, dA.nnzb, dA.row_block_dim);

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "nnz",
                            size_t(dA.nnzb) * dA.row_block_dim * dA.row_block_dim,
                            "alpha",
                            h_alpha,
                            "pivot",
                            std::min(*h_analysis_pivot, *h_solve_pivot),
                            "operation",
                            rocsparse_operation2string(trans),
                            "diag_type",
                            rocsparse_diagtype2string(diag),
                            "fill_mode",
                            rocsparse_fillmode2string(uplo),
                            "analysis_policy",
                            rocsparse_analysis2string(apol),
                            "solve_policy",
                            rocsparse_solve2string(spol),
                            "analysis time",
                            get_gpu_time_msec(gpu_analysis_time_used),
                            s_timing_info_perf,
                            gpu_gflops,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Clear bsrsv meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_clear(handle, info));

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_bsrsv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrsv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

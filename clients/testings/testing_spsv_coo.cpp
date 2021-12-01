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

template <typename I, typename T>
void testing_spsv_coo_bad_arg(const Arguments& arg)
{
    I m     = 100;
    I n     = 100;
    I nnz   = 100;
    T alpha = 0.6;

    rocsparse_operation  trans_A = rocsparse_operation_none;
    rocsparse_index_base base    = rocsparse_index_base_zero;
    rocsparse_spsv_alg   alg     = rocsparse_spsv_alg_default;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // SpSV structures
    rocsparse_local_spmat local_A(
        m, n, nnz, (void*)0x4, (void*)0x4, (void*)0x4, itype, base, ttype);
    rocsparse_local_dnvec local_x(m, (void*)0x4, ttype);
    rocsparse_local_dnvec local_y(m, (void*)0x4, ttype);

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {9, 10};

    rocsparse_handle      handle = local_handle;
    rocsparse_spmat_descr A      = local_A;
    rocsparse_dnvec_descr x      = local_x;
    rocsparse_dnvec_descr y      = local_y;

    size_t buffer_size;
    void*  temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                        \
    handle, trans_A, &alpha, A, x, y, ttype, alg, rocsparse_spsv_stage_buffer_size, &buffer_size, \
        temp_buffer

#define PARAMS_ANALYSIS                                                                          \
    handle, trans_A, &alpha, A, x, y, ttype, alg, rocsparse_spsv_stage_preprocess, &buffer_size, \
        temp_buffer

#define PARAMS_SOLVE                                                                          \
    handle, trans_A, &alpha, A, x, y, ttype, alg, rocsparse_spsv_stage_compute, &buffer_size, \
        temp_buffer

    auto_testing_bad_arg(rocsparse_spsv, nargs_to_exclude, args_to_exclude, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_spsv, nargs_to_exclude, args_to_exclude, PARAMS_ANALYSIS);
    auto_testing_bad_arg(rocsparse_spsv, nargs_to_exclude, args_to_exclude, PARAMS_SOLVE);

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS_SOLVE
}

template <typename I, typename T>
void testing_spsv_coo(const Arguments& arg)
{
    I                    M       = arg.M;
    I                    N       = arg.N;
    rocsparse_operation  trans_A = arg.transA;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_spsv_alg   alg     = arg.spsv_alg;
    rocsparse_diag_type  diag    = arg.diag;
    rocsparse_fill_mode  uplo    = arg.uplo;

    rocsparse_spsv_stage buffersize = rocsparse_spsv_stage_buffer_size;
    rocsparse_spsv_stage preprocess = rocsparse_spsv_stage_preprocess;
    rocsparse_spsv_stage compute    = rocsparse_spsv_stage_compute;

    T halpha = arg.get_alpha<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0)
    {
        // M == 0 means nnz can only be 0, too
        static const I safe_size = 100;

        // Allocate memory on device
        device_vector<I> dcoo_row_ind(safe_size);
        device_vector<I> dcoo_col_ind(safe_size);
        device_vector<T> dcoo_val(safe_size);
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);

        // Check SpSV when structures can be created
        if(M == 0 && M == N)
        {
            I nnz_A = 0;

            // Pointer mode
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            // Check structures
            rocsparse_local_spmat A(
                M, N, nnz_A, dcoo_row_ind, dcoo_col_ind, dcoo_val, itype, base, ttype);

            rocsparse_local_dnvec x(M, dx, ttype);
            rocsparse_local_dnvec y(M, dy, ttype);

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)),
                rocsparse_status_success);

            size_t buffer_size;
            EXPECT_ROCSPARSE_STATUS(rocsparse_spsv(handle,
                                                   trans_A,
                                                   &halpha,
                                                   A,
                                                   x,
                                                   y,
                                                   ttype,
                                                   alg,
                                                   buffersize,
                                                   &buffer_size,
                                                   nullptr),
                                    rocsparse_status_success);

            void* dbuffer;
            CHECK_HIP_ERROR(hipMalloc(&dbuffer, safe_size));

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spsv(
                    handle, trans_A, &halpha, A, x, y, ttype, alg, preprocess, nullptr, dbuffer),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spsv(
                    handle, trans_A, &halpha, A, x, y, ttype, alg, compute, &buffer_size, dbuffer),
                rocsparse_status_success);
            CHECK_HIP_ERROR(hipFree(dbuffer));
        }

        return;
    }

    rocsparse_matrix_factory<T, I> matrix_factory(arg);

    // Allocate host memory for matrix
    host_vector<I> hcoo_row_ind;
    host_vector<I> hcoo_col_ind;
    host_vector<T> hcoo_val;

    // Sample matrix
    I nnz_A;
    matrix_factory.init_coo(hcoo_row_ind, hcoo_col_ind, hcoo_val, M, N, nnz_A, base);

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    // Allocate host memory for vectors
    host_vector<T> hx(M);
    host_vector<T> hy_1(M);
    host_vector<T> hy_2(M);
    host_vector<T> hy_gold(M);

    // Initialize data on CPU
    rocsparse_init<T>(hx, M, 1, 1);
    rocsparse_init<T>(hy_1, M, 1, 1);

    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<I> dcoo_row_ind(nnz_A);
    device_vector<I> dcoo_col_ind(nnz_A);
    device_vector<T> dcoo_val(nnz_A);
    device_vector<T> dx(M);
    device_vector<T> dy_1(M);
    device_vector<T> dy_2(M);
    device_vector<T> dalpha(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind.data(), sizeof(I) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_col_ind, hcoo_col_ind.data(), sizeof(I) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcoo_val, hcoo_val.data(), sizeof(T) * nnz_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &halpha, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(M, N, nnz_A, dcoo_row_ind, dcoo_col_ind, dcoo_val, itype, base, ttype);
    rocsparse_local_dnvec x(M, dx, ttype);
    rocsparse_local_dnvec y1(M, dy_1, ttype);
    rocsparse_local_dnvec y2(M, dy_2, ttype);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_spmat_set_attribute(A, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    // Query SpSV buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsv(handle,
                                         trans_A,
                                         &halpha,
                                         A,
                                         x,
                                         y1,
                                         ttype,
                                         alg,
                                         rocsparse_spsv_stage_auto /*buffersize*/,
                                         &buffer_size,
                                         nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    // Perform analysis on host
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsv(handle,
                                         trans_A,
                                         &halpha,
                                         A,
                                         x,
                                         y1,
                                         ttype,
                                         alg,
                                         rocsparse_spsv_stage_auto /*preprocess*/,
                                         nullptr,
                                         dbuffer));

    // Perform analysis on device
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsv(handle,
                                         trans_A,
                                         dalpha,
                                         A,
                                         x,
                                         y2,
                                         ttype,
                                         alg,
                                         rocsparse_spsv_stage_auto /*preprocess*/,
                                         nullptr,
                                         dbuffer));

    if(arg.unit_check)
    {
        // Solve on host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsv(handle,
                                             trans_A,
                                             &halpha,
                                             A,
                                             x,
                                             y1,
                                             ttype,
                                             alg,
                                             rocsparse_spsv_stage_auto /*compute*/,
                                             &buffer_size,
                                             dbuffer));

        // Solve on device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsv(handle,
                                             trans_A,
                                             dalpha,
                                             A,
                                             x,
                                             y2,
                                             ttype,
                                             alg,
                                             rocsparse_spsv_stage_auto /*compute*/,
                                             &buffer_size,
                                             dbuffer));

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU coosv
        I analysis_pivot = -1;
        I solve_pivot    = -1;
        host_coosv(trans_A,
                   M,
                   nnz_A,
                   halpha,
                   hcoo_row_ind,
                   hcoo_col_ind,
                   hcoo_val,
                   hx,
                   hy_gold,
                   diag,
                   uplo,
                   base,
                   &analysis_pivot,
                   &solve_pivot);

        if(analysis_pivot == -1 && solve_pivot == -1)
        {
            hy_gold.near_check(hy_1);
            hy_gold.near_check(hy_2);
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
            CHECK_ROCSPARSE_ERROR(rocsparse_spsv(
                handle, trans_A, &halpha, A, x, y1, ttype, alg, compute, &buffer_size, dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spsv(
                handle, trans_A, &halpha, A, x, y1, ttype, alg, compute, &buffer_size, dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = spsv_gflop_count(M, nnz_A, diag);
        double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        double gbyte_count = coosv_gbyte_count<T>(M, nnz_A);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "nnz_A",
                            nnz_A,
                            "alpha",
                            halpha,
                            "Algorithm",
                            rocsparse_spsvalg2string(alg),
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

    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, TTYPE)                                               \
    template void testing_spsv_coo_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spsv_coo<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

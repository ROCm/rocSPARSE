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

template <typename T>
void testing_coo2dense_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create rocsparse descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    rocsparse_int             n           = safe_size;
    rocsparse_int             nnz         = safe_size;
    const rocsparse_mat_descr descr       = local_descr;
    const T*                  coo_val     = (const T*)0x4;
    const rocsparse_int*      coo_row_ind = (const rocsparse_int*)0x4;
    const rocsparse_int*      coo_col_ind = (const rocsparse_int*)0x4;
    T*                        A           = (T*)0x4;
    rocsparse_int             lda         = safe_size;

#define PARAMS handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, lda
    auto_testing_bad_arg(rocsparse_coo2dense<T>, PARAMS);
#undef PARAMS
}

template <typename T>
void testing_coo2dense(const Arguments& arg)
{
    rocsparse_int             M  = arg.M;
    rocsparse_int             N  = arg.N;
    rocsparse_int             LD = arg.denseld;
    rocsparse_local_handle    handle(arg);
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, arg.baseA));

    // Set matrix storage mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, arg.storage));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || LD < M)
    {
        rocsparse_status expected_status = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_size;

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_coo2dense<T>(
                handle, M, N, 100, descr, nullptr, nullptr, nullptr, (T*)nullptr, LD),
            expected_status);
        return;
    }

    rocsparse_matrix_factory<T> matrix_factory(arg);

    host_vector<rocsparse_int> h_coo_row_ind;
    host_vector<rocsparse_int> h_coo_col_ind;
    host_vector<T>             h_coo_val;

    int64_t coo_nnz = 0;
    matrix_factory.init_coo(h_coo_row_ind, h_coo_col_ind, h_coo_val, M, N, coo_nnz, arg.baseA);

    rocsparse_int nnz = rocsparse_convert_to_int(coo_nnz);

    device_vector<rocsparse_int> d_coo_row_ind(nnz);
    device_vector<rocsparse_int> d_coo_col_ind(nnz);
    device_vector<T>             d_coo_val(nnz);

    // Copy on host.
    CHECK_HIP_ERROR(hipMemcpy(
        d_coo_val, h_coo_val.data(), sizeof(T) * h_coo_val.size(), hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemcpy(d_coo_row_ind,
                              h_coo_row_ind.data(),
                              sizeof(rocsparse_int) * h_coo_row_ind.size(),
                              hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemcpy(d_coo_col_ind,
                              h_coo_col_ind.data(),
                              sizeof(rocsparse_int) * h_coo_col_ind.size(),
                              hipMemcpyHostToDevice));

    // // Random initialization of the matrix.
    host_vector<T> h_dense_val(LD * N);
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < LD; ++i)
        {
            h_dense_val[j * LD + i] = -2;
        }
    }

    device_vector<T> d_dense_val(LD * N);
    CHECK_HIP_ERROR(
        hipMemcpy(d_dense_val, h_dense_val.data(), sizeof(T) * LD * N, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_coo2dense<T>(handle,
                                                              M,
                                                              N,
                                                              nnz,
                                                              descr,
                                                              d_coo_val,
                                                              d_coo_row_ind,
                                                              d_coo_col_ind,
                                                              (T*)d_dense_val,
                                                              LD));

        host_vector<T> gpu_dense_val(LD * N);
        CHECK_HIP_ERROR(
            hipMemcpy(gpu_dense_val, d_dense_val, sizeof(T) * LD * N, hipMemcpyDeviceToHost));

        host_vector<T> cpu_dense_val(LD * N, -2);
        host_coo_to_dense(M,
                          N,
                          nnz,
                          rocsparse_get_mat_index_base(descr),
                          h_coo_val,
                          h_coo_row_ind,
                          h_coo_col_ind,
                          cpu_dense_val,
                          LD,
                          rocsparse_order_column);

        cpu_dense_val.unit_check(gpu_dense_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm-up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {

            CHECK_ROCSPARSE_ERROR(rocsparse_coo2dense<T>(handle,
                                                         M,
                                                         N,
                                                         nnz,
                                                         descr,
                                                         d_coo_val,
                                                         d_coo_row_ind,
                                                         d_coo_col_ind,
                                                         (T*)d_dense_val,
                                                         LD));
        }

        double gpu_time_used = get_time_us();
        {
            // Performance run
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_coo2dense<T>(handle,
                                                             M,
                                                             N,
                                                             nnz,
                                                             descr,
                                                             d_coo_val,
                                                             d_coo_row_ind,
                                                             d_coo_col_ind,
                                                             (T*)d_dense_val,
                                                             LD));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = coo2dense_gbyte_count<T>(M, N, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "LD",
                            LD,
                            "nnz",
                            nnz,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_coo2dense_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_coo2dense<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_coo2dense_extra(const Arguments& arg) {}

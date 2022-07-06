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
#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename I, typename J, typename T>
void testing_sparse_to_dense_csc_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle              handle      = local_handle;
    J                             m           = safe_size;
    J                             n           = safe_size;
    I                             nnz         = safe_size;
    I                             ld          = safe_size;
    void*                         csc_val     = (void*)0x4;
    void*                         csc_row_ind = (void*)0x4;
    void*                         csc_col_ptr = (void*)0x4;
    void*                         dense_val   = (void*)0x4;
    rocsparse_index_base          base        = rocsparse_index_base_zero;
    rocsparse_order               order       = rocsparse_order_column;
    rocsparse_sparse_to_dense_alg alg         = rocsparse_sparse_to_dense_alg_default;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Sparse and dense matrix structures
    rocsparse_local_spmat local_mat_A(m,
                                      n,
                                      nnz,
                                      csc_col_ptr,
                                      csc_row_ind,
                                      csc_val,
                                      itype,
                                      jtype,
                                      base,
                                      ttype,
                                      rocsparse_format_csc);
    rocsparse_local_dnmat local_mat_B(m, n, ld, dense_val, ttype, order);

    rocsparse_spmat_descr mat_A = local_mat_A;
    rocsparse_dnmat_descr mat_B = local_mat_B;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {4, 5};

#define PARAMS handle, mat_A, mat_B, alg, buffer_size, temp_buffer
    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_sparse_to_dense, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_sparse_to_dense, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_sparse_to_dense, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_sparse_to_dense, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS

    EXPECT_ROCSPARSE_STATUS(rocsparse_sparse_to_dense(handle, mat_A, mat_B, alg, nullptr, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename I, typename J, typename T>
void testing_sparse_to_dense_csc(const Arguments& arg)
{
    J                             m     = arg.M;
    J                             n     = arg.N;
    I                             ld    = arg.denseld;
    rocsparse_index_base          base  = arg.baseA;
    rocsparse_sparse_to_dense_alg alg   = arg.sparse_to_dense_alg;
    rocsparse_order               order = arg.order;

    I mn = (order == rocsparse_order_column) ? m : n;
    I nm = (order == rocsparse_order_column) ? n : m;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || ld < mn)
    {
        // M == N == 0 means nnz can only be 0, too

        static const I safe_size = 100;

        // Allocate memory on device
        device_vector<I> d_csc_col_ptr(safe_size);
        device_vector<J> d_csc_row_ind(safe_size);
        device_vector<T> d_csc_val(safe_size);
        device_vector<T> d_dense_val(safe_size);

        if(!d_csc_col_ptr || !d_csc_row_ind || !d_csc_val || !d_dense_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        if(m == 0 && n == 0 && ld >= mn)
        {
            I                     nnz = 0;
            rocsparse_local_spmat mat_A(m,
                                        n,
                                        nnz,
                                        d_csc_col_ptr,
                                        d_csc_row_ind,
                                        d_csc_val,
                                        itype,
                                        jtype,
                                        base,
                                        ttype,
                                        rocsparse_format_csc);
            rocsparse_local_dnmat mat_B(m, n, ld, d_dense_val, ttype, order);

            size_t buffer_size;
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_sparse_to_dense(handle, mat_A, mat_B, alg, &buffer_size, nullptr),
                rocsparse_status_success);

            void* dbuffer;
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, safe_size));
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_sparse_to_dense(handle, mat_A, mat_B, alg, &buffer_size, dbuffer),
                rocsparse_status_success);
            CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        }

        return;
    }

    // Allocate memory.
    host_vector<T>   h_dense_val(ld * nm);
    device_vector<T> d_dense_val(ld * nm);

    host_vector<I>   h_nnzPerRow(m);
    device_vector<I> d_nnzPerRow(m);
    if(!d_nnzPerRow || !d_dense_val || !h_nnzPerRow || !h_dense_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_seedrand();

    // Random initialization of the matrix.
    for(int i = 0; i < ld; ++i)
    {
        for(int j = 0; j < nm; ++j)
        {
            h_dense_val[j * (int)ld + i] = static_cast<T>(-2);
        }
    }

    for(int i = 0; i < mn; ++i)
    {
        for(int j = 0; j < nm; ++j)
        {
            h_dense_val[j * (int)ld + i] = random_generator<T>(0, 9);
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(
        hipMemcpy(d_dense_val, h_dense_val, sizeof(T) * ld * nm, hipMemcpyHostToDevice));

    // Allocate device memory for ro pointer array
    device_vector<I> d_csc_col_ptr(n + 1);

    rocsparse_local_dnmat mat_dense(m, n, ld, d_dense_val, ttype, order);
    rocsparse_local_spmat mat_sparse(
        m, n, 0, d_csc_col_ptr, nullptr, nullptr, itype, jtype, base, ttype, rocsparse_format_csc);

    // Find size of required temporary buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                    mat_dense,
                                                    mat_sparse,
                                                    rocsparse_dense_to_sparse_alg_default,
                                                    &buffer_size,
                                                    nullptr));

    // Allocate temporary buffer on device
    device_vector<J> d_temp_buffer(buffer_size);

    if(!d_temp_buffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Perform analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                    mat_dense,
                                                    mat_sparse,
                                                    rocsparse_dense_to_sparse_alg_default,
                                                    nullptr,
                                                    d_temp_buffer));

    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(mat_sparse, &num_rows_tmp, &num_cols_tmp, &nnz));

    // Allocate memory on device
    device_vector<J> d_csc_row_ind(nnz);
    device_vector<T> d_csc_val(nnz);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_csc_set_pointers(mat_sparse, d_csc_col_ptr, d_csc_row_ind, d_csc_val));

    // Complete conversion
    CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                    mat_dense,
                                                    mat_sparse,
                                                    rocsparse_dense_to_sparse_alg_default,
                                                    &buffer_size,
                                                    d_temp_buffer));

    host_vector<I> h_csc_col_ptr(n + 1);
    host_vector<J> h_csc_row_ind(nnz);
    host_vector<T> h_csc_val(nnz);

    CHECK_HIP_ERROR(
        hipMemcpy(h_csc_col_ptr.data(), d_csc_col_ptr, sizeof(I) * (n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(h_csc_row_ind.data(), d_csc_row_ind, sizeof(J) * nnz, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(h_csc_val.data(), d_csc_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

    if(arg.unit_check)
    {
        // Clear device dense matrix before re-computing it
        host_vector<T> temp(ld * nm, -2);
        CHECK_HIP_ERROR(hipMemcpy(d_dense_val, temp, sizeof(T) * ld * nm, hipMemcpyHostToDevice));

        // Find size of required temporary buffer
        size_t buffer_size2;
        CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_dense(handle,
                                                        mat_sparse,
                                                        mat_dense,
                                                        rocsparse_sparse_to_dense_alg_default,
                                                        &buffer_size2,
                                                        nullptr));

        // Allocate temporary buffer on device
        device_vector<J> d_temp_buffer2(buffer_size2);

        if(!d_temp_buffer2)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Complete conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_dense(handle,
                                                        mat_sparse,
                                                        mat_dense,
                                                        rocsparse_sparse_to_dense_alg_default,
                                                        &buffer_size2,
                                                        d_temp_buffer2));

        host_vector<T> gpu_dense_val(ld * nm);
        CHECK_HIP_ERROR(
            hipMemcpy(gpu_dense_val, d_dense_val, sizeof(T) * ld * nm, hipMemcpyDeviceToHost));

        host_vector<T> cpu_dense_val = h_dense_val;

        host_csx2dense<rocsparse_direction_column>(m,
                                                   n,
                                                   base,
                                                   order,
                                                   h_csc_val.data(),
                                                   h_csc_col_ptr.data(),
                                                   h_csc_row_ind.data(),
                                                   cpu_dense_val.data(),
                                                   ld);

        h_dense_val.unit_check(cpu_dense_val);
        h_dense_val.unit_check(gpu_dense_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Find size of required temporary buffer
        size_t buffer_size2;
        CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_dense(handle,
                                                        mat_sparse,
                                                        mat_dense,
                                                        rocsparse_sparse_to_dense_alg_default,
                                                        &buffer_size2,
                                                        nullptr));

        // Allocate temporary buffer on device
        device_vector<J> d_temp_buffer2(buffer_size2);

        if(!d_temp_buffer2)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Warm-up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_sparse_to_dense(handle,
                                                            mat_sparse,
                                                            mat_dense,
                                                            rocsparse_sparse_to_dense_alg_default,
                                                            &buffer_size2,
                                                            d_temp_buffer2));
        }

        double gpu_time_used = get_time_us();
        {
            // Performance run
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_sparse_to_dense(handle,
                                              mat_sparse,
                                              mat_dense,
                                              rocsparse_sparse_to_dense_alg_default,
                                              &buffer_size2,
                                              d_temp_buffer2));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = csx2dense_gbyte_count<rocsparse_direction_column, T>(m, n, nnz);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("order",
                            order,
                            "M",
                            m,
                            "N",
                            n,
                            "LD",
                            ld,
                            "nnz",
                            nnz,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TYPE)                                                          \
    template void testing_sparse_to_dense_csc_bad_arg<ITYPE, JTYPE, TYPE>(const Arguments& arg); \
    template void testing_sparse_to_dense_csc<ITYPE, JTYPE, TYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

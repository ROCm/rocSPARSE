/*! \file */
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
#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename I, typename T>
void testing_dense_to_sparse_coo_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle              handle      = local_handle;
    I                             m           = safe_size;
    I                             n           = safe_size;
    I                             nnz         = safe_size;
    I                             ld          = safe_size;
    void*                         dense_val   = (void*)0x4;
    void*                         coo_val     = (void*)0x4;
    void*                         coo_row_ind = (void*)0x4;
    void*                         coo_col_ind = (void*)0x4;
    rocsparse_index_base          base        = rocsparse_index_base_zero;
    rocsparse_order               order       = rocsparse_order_column;
    rocsparse_dense_to_sparse_alg alg         = rocsparse_dense_to_sparse_alg_default;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Dense and sparse matrix structures
    rocsparse_local_dnmat local_mat_A(m, n, ld, dense_val, ttype, order);
    rocsparse_local_spmat local_mat_B(
        m, n, nnz, coo_row_ind, coo_col_ind, coo_val, itype, base, ttype);

    rocsparse_dnmat_descr mat_A = local_mat_A;
    rocsparse_spmat_descr mat_B = local_mat_B;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {4, 5};

#define PARAMS handle, mat_A, mat_B, alg, buffer_size, temp_buffer
    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = (void*)0x4;
        auto_testing_bad_arg(rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = nullptr;
        auto_testing_bad_arg(rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS

    EXPECT_ROCSPARSE_STATUS(rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, nullptr, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename I, typename T>
void testing_dense_to_sparse_coo(const Arguments& arg)
{
    I                             m     = arg.M;
    I                             n     = arg.N;
    I                             ld    = arg.denseld;
    rocsparse_index_base          base  = arg.baseA;
    rocsparse_dense_to_sparse_alg alg   = arg.dense_to_sparse_alg;
    rocsparse_order               order = arg.order;

    I mn = (order == rocsparse_order_column) ? m : n;
    I nm = (order == rocsparse_order_column) ? n : m;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || ld < mn)
    {
        // M == N == 0 means nnz can only be 0, too
        I nnz = 0;

        static const I safe_size = 100;

        // Allocate memory on device
        device_vector<I> d_coo_row_ind(safe_size);
        device_vector<I> d_coo_col_ind(safe_size);
        device_vector<T> d_coo_val(safe_size);
        device_vector<T> d_dense_val(safe_size);

        if(!d_coo_row_ind || !d_coo_col_ind || !d_coo_val || !d_dense_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        if(m == 0 && n == 0 && ld >= mn)
        {
            rocsparse_local_dnmat mat_A(m, n, ld, d_dense_val, ttype, order);
            rocsparse_local_spmat mat_B(
                m, n, nnz, d_coo_row_ind, d_coo_col_ind, d_coo_val, itype, base, ttype);

            // Determine buffer size
            size_t buffer_size;
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, &buffer_size, nullptr),
                rocsparse_status_success);

            void* dbuffer;
            CHECK_HIP_ERROR(hipMalloc(&dbuffer, safe_size));

            // Perform analysis
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, nullptr, dbuffer),
                rocsparse_status_success);

            // Complete conversion
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, &buffer_size, dbuffer),
                rocsparse_status_success);
            CHECK_HIP_ERROR(hipFree(dbuffer));
        }

        return;
    }

    // Allocate memory.
    host_vector<T>   h_dense_val(ld * nm);
    device_vector<T> d_dense_val(ld * nm);

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

    rocsparse_local_dnmat mat_dense(m, n, ld, d_dense_val, ttype, order);
    rocsparse_local_spmat mat_sparse(m, n, 0, nullptr, nullptr, nullptr, itype, base, ttype);

    // Find size of required temporary buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                    mat_dense,
                                                    mat_sparse,
                                                    rocsparse_dense_to_sparse_alg_default,
                                                    &buffer_size,
                                                    nullptr));

    // Allocate temporary buffer on device
    device_vector<I> d_temp_buffer(buffer_size);

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
    device_vector<I> d_coo_row_ind(nnz);
    device_vector<I> d_coo_col_ind(nnz);
    device_vector<T> d_coo_val(nnz);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_coo_set_pointers(mat_sparse, d_coo_row_ind, d_coo_col_ind, d_coo_val));

    if(arg.unit_check)
    {
        // Complete conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                        mat_dense,
                                                        mat_sparse,
                                                        rocsparse_dense_to_sparse_alg_default,
                                                        &buffer_size,
                                                        d_temp_buffer));

        host_vector<I> h_coo_row_ind_gpu(nnz);
        host_vector<I> h_coo_col_ind_gpu(nnz);
        host_vector<T> h_coo_val_gpu(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            h_coo_row_ind_gpu.data(), d_coo_row_ind, nnz * sizeof(I), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_coo_col_ind_gpu.data(), d_coo_col_ind, nnz * sizeof(I), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(h_coo_val_gpu.data(), d_coo_val, nnz * sizeof(T), hipMemcpyDeviceToHost));

        host_vector<I> nnz_per_row(m);
        CHECK_HIP_ERROR(
            hipMemcpy(nnz_per_row.data(), d_temp_buffer, m * sizeof(I), hipMemcpyDeviceToHost));

        host_vector<I> h_coo_row_ind_cpu(nnz);
        host_vector<I> h_coo_col_ind_cpu(nnz);
        host_vector<T> h_coo_val_cpu(nnz);

        host_dense_to_coo(m,
                          n,
                          base,
                          h_dense_val,
                          ld,
                          order,
                          nnz_per_row,
                          h_coo_val_cpu,
                          h_coo_row_ind_cpu,
                          h_coo_col_ind_cpu);

        h_coo_row_ind_cpu.unit_check(h_coo_row_ind_gpu);
        h_coo_col_ind_cpu.unit_check(h_coo_col_ind_gpu);
        h_coo_val_cpu.unit_check(h_coo_val_gpu);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm-up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                            mat_dense,
                                                            mat_sparse,
                                                            rocsparse_dense_to_sparse_alg_default,
                                                            &buffer_size,
                                                            d_temp_buffer));
        }

        double gpu_time_used = get_time_us();
        {
            // Performance run
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_dense_to_sparse(handle,
                                              mat_dense,
                                              mat_sparse,
                                              rocsparse_dense_to_sparse_alg_default,
                                              &buffer_size,
                                              d_temp_buffer));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gbyte = dense2coo_gbyte_count<T>(m, n, (I)nnz) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);
        // clang-format off
        std::cout
      << std::setw(20) << "order"
	  << std::setw(20) << "M"
	  << std::setw(20) << "N"
	  << std::setw(20) << "LD"
	  << std::setw(20) << "nnz"
	  << std::setw(20) << "GB/s"
	  << std::setw(20) << "msec"
	  << std::setw(20) << "iter"
	  << std::setw(20) << "verified"
	  << std::endl;

        std::cout
      << std::setw(20) << order
	  << std::setw(20) << m
	  << std::setw(20) << n
	  << std::setw(20) << ld
	  << std::setw(20) << nnz
	  << std::setw(20) << gpu_gbyte
	  << std::setw(20) << gpu_time_used / 1e3
	  << std::setw(20) << number_hot_calls
	  << std::setw(20) << (arg.unit_check ? "yes" : "no")
	  << std::endl;
        // clang-format on
    }
}

#define INSTANTIATE(ITYPE, TYPE)                                                          \
    template void testing_dense_to_sparse_coo_bad_arg<ITYPE, TYPE>(const Arguments& arg); \
    template void testing_dense_to_sparse_coo<ITYPE, TYPE>(const Arguments& arg)
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

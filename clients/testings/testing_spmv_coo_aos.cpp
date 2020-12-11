/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

template <typename I, typename T>
void testing_spmv_coo_aos_bad_arg(const Arguments& arg)
{
    I m   = 100;
    I n   = 100;
    I nnz = 100;

    T alpha = 0.6;
    T beta  = 0.1;

    rocsparse_operation  trans = rocsparse_operation_none;
    rocsparse_index_base base  = rocsparse_index_base_zero;
    rocsparse_spmv_alg   alg   = rocsparse_spmv_default;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory on device
    device_vector<I> dcoo_ind(2 * nnz);
    device_vector<T> dcoo_val(nnz);
    device_vector<T> dx(n);
    device_vector<T> dy(m);

    if(!dcoo_ind || !dcoo_val || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // SpMV structures
    rocsparse_local_spmat A(m, n, nnz, dcoo_ind, dcoo_val, itype, base, ttype);
    rocsparse_local_dnvec x(n, dx, ttype);
    rocsparse_local_dnvec y(m, dy, ttype);

    // Test SpMV with invalid buffer
    size_t buffer_size;

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(nullptr, trans, &alpha, A, x, &beta, y, ttype, alg, &buffer_size, nullptr),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(handle, trans, nullptr, A, x, &beta, y, ttype, alg, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(
            handle, trans, &alpha, nullptr, x, &beta, y, ttype, alg, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(
            handle, trans, &alpha, A, nullptr, &beta, y, ttype, alg, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(handle, trans, &alpha, A, x, nullptr, y, ttype, alg, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(
            handle, trans, &alpha, A, x, &beta, nullptr, ttype, alg, &buffer_size, nullptr),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(handle, trans, &alpha, A, x, &beta, y, ttype, alg, nullptr, nullptr),
        rocsparse_status_invalid_pointer);

    // Test SpMV with valid buffer
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, 100));

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(nullptr, trans, &alpha, A, x, &beta, y, ttype, alg, &buffer_size, dbuffer),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(handle, trans, nullptr, A, x, &beta, y, ttype, alg, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(
            handle, trans, &alpha, nullptr, x, &beta, y, ttype, alg, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(
            handle, trans, &alpha, A, nullptr, &beta, y, ttype, alg, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(handle, trans, &alpha, A, x, nullptr, y, ttype, alg, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_spmv(
            handle, trans, &alpha, A, x, &beta, nullptr, ttype, alg, &buffer_size, dbuffer),
        rocsparse_status_invalid_pointer);

    CHECK_HIP_ERROR(hipFree(dbuffer));
}

template <typename I, typename T>
void testing_spmv_coo_aos(const Arguments& arg)
{
    I                     M         = arg.M;
    I                     N         = arg.N;
    I                     K         = arg.K;
    int32_t               dim_x     = arg.dimx;
    int32_t               dim_y     = arg.dimy;
    int32_t               dim_z     = arg.dimz;
    rocsparse_operation   trans     = arg.transA;
    rocsparse_index_base  base      = arg.baseA;
    rocsparse_spmv_alg    alg       = arg.spmv_alg;
    rocsparse_matrix_init mat       = arg.matrix;
    bool                  full_rank = false;
    std::string           filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        // M == N == 0 means nnz can only be 0, too
        I nnz = 0;

        static const I safe_size = 100;

        // Allocate memory on device
        device_vector<I> dcoo_ind(safe_size);
        device_vector<T> dcoo_val(safe_size);
        device_vector<T> dx(safe_size);
        device_vector<T> dy(safe_size);

        if(!dcoo_ind || !dcoo_val || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Check SpMV when structures can be created
        if(M == 0 && N == 0)
        {
            // Check structures
            rocsparse_local_spmat A(M, N, nnz, dcoo_ind, dcoo_val, itype, base, ttype);
            rocsparse_local_dnvec x(N, dx, ttype);
            rocsparse_local_dnvec y(M, dy, ttype);

            size_t buffer_size;
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spmv(
                    handle, trans, &h_alpha, A, x, &h_beta, y, ttype, alg, &buffer_size, nullptr),
                rocsparse_status_success);

            void* dbuffer;
            CHECK_HIP_ERROR(hipMalloc(&dbuffer, safe_size));
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_spmv(
                    handle, trans, &h_alpha, A, x, &h_beta, y, ttype, alg, &buffer_size, dbuffer),
                rocsparse_status_success);
            CHECK_HIP_ERROR(hipFree(dbuffer));
        }

        return;
    }

    // Allocate host memory for matrix
    host_vector<I> hcsr_row_ptr;
    host_vector<I> hcoo_col_ind;
    host_vector<T> hcoo_val;

    rocsparse_seedrand();

    // Sample matrix
    I nnz;
    rocsparse_init_csr_matrix(hcsr_row_ptr,
                              hcoo_col_ind,
                              hcoo_val,
                              M,
                              N,
                              K,
                              dim_x,
                              dim_y,
                              dim_z,
                              nnz,
                              base,
                              mat,
                              filename.c_str(),
                              arg.timing ? false : true,
                              full_rank);

    // Allocate host memory for vectors
    host_vector<T> hx(N);
    host_vector<T> hy_1(M);
    host_vector<T> hy_2(M);
    host_vector<T> hy_gold(M);

    // Initialize data on CPU
    rocsparse_init(hx, 1, N, 1);
    rocsparse_init(hy_1, 1, M, 1);
    hy_2    = hy_1;
    hy_gold = hy_1;

    // Convert CSR matrix to COO AoS
    host_vector<I> hcoo_ind(2 * nnz);
    host_csr_to_coo_aos(M, nnz, hcsr_row_ptr, hcoo_col_ind, hcoo_ind, base);

    // Allocate device memory
    device_vector<I> dcoo_ind(2 * nnz);
    device_vector<T> dcoo_val(nnz);
    device_vector<T> dx(N);
    device_vector<T> dy_1(M);
    device_vector<T> dy_2(M);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    if(!dcoo_ind || !dcoo_val || !dx || !dy_1 || !dy_2 || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dcoo_ind, hcoo_ind, sizeof(I) * 2 * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcoo_val, hcoo_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spmat A(M, N, nnz, dcoo_ind, dcoo_val, itype, base, ttype);
    rocsparse_local_dnvec x(N, dx, ttype);
    rocsparse_local_dnvec y1(M, dy_1, ttype);
    rocsparse_local_dnvec y2(M, dy_2, ttype);

    // Query SpMV buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmv(
        handle, trans, &h_alpha, A, x, &h_beta, y1, ttype, alg, &buffer_size, nullptr));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // SpMV

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(
            handle, trans, &h_alpha, A, x, &h_beta, y1, ttype, alg, &buffer_size, dbuffer));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_spmv(
            handle, trans, d_alpha, A, x, d_beta, y2, ttype, alg, &buffer_size, dbuffer));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU coomv aos
        host_coomv_aos<I, T>(M, nnz, h_alpha, hcoo_ind, hcoo_val, hx, h_beta, hy_gold, base);

        near_check_general<T>(1, M, 1, hy_gold, hy_1);
        near_check_general<T>(1, M, 1, hy_gold, hy_2);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv(
                handle, trans, &h_alpha, A, x, &h_beta, y1, ttype, alg, &buffer_size, dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spmv(
                handle, trans, &h_alpha, A, x, &h_beta, y1, ttype, alg, &buffer_size, dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops
            = spmv_gflop_count<I, T>(M, nnz, h_beta != static_cast<T>(0)) / gpu_time_used * 1e6;
        double gpu_gbyte
            = coomv_gbyte_count<I, T>(M, N, nnz, h_beta != static_cast<T>(0)) / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz
                  << std::setw(12) << h_alpha << std::setw(12) << h_beta << std::setw(12)
                  << gpu_gflops << std::setw(12) << gpu_gbyte << std::setw(12)
                  << gpu_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                   \
    template void testing_spmv_coo_aos_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmv_coo_aos<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

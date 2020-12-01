/*! \file */
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

template <typename T>
void testing_bsrmv_bad_arg(const Arguments& arg)
{
    static const size_t        safe_size = 100;
    static const rocsparse_int safe_dim  = 2;

    T h_alpha = 0.6;
    T h_beta  = 0.1;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Allocate memory on device
    device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dbsr_col_ind(safe_size);
    device_vector<T>             dbsr_val(safe_size);
    device_vector<T>             dx(safe_size);
    device_vector<T>             dy(safe_size);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dx || !dy)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_bsrmv()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(nullptr,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               nullptr,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               nullptr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               nullptr,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               nullptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               nullptr,
                                               safe_dim,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               nullptr,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               dx,
                                               nullptr,
                                               dy),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               safe_dim,
                                               dx,
                                               &h_beta,
                                               nullptr),
                            rocsparse_status_invalid_pointer);

    // Test bsr_dim -1 and 0
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               -1,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                               rocsparse_direction_column,
                                               rocsparse_operation_none,
                                               safe_size,
                                               safe_size,
                                               safe_size,
                                               &h_alpha,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind,
                                               0,
                                               dx,
                                               &h_beta,
                                               dy),
                            rocsparse_status_success);
}

template <typename T>
void testing_bsrmv(const Arguments& arg)
{

    rocsparse_int        M       = arg.M;
    rocsparse_int        N       = arg.N;
    rocsparse_direction  dir     = arg.direction;
    rocsparse_operation  trans   = arg.transA;
    rocsparse_index_base base    = arg.baseA;
    rocsparse_int        bsr_dim = arg.block_dim;
    T                    h_alpha = arg.get_alpha<T>();
    T                    h_beta  = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // BSR dimensions

    rocsparse_int mb = (bsr_dim > 0) ? (M + bsr_dim - 1) / bsr_dim : 0;
    rocsparse_int nb = (bsr_dim > 0) ? (N + bsr_dim - 1) / bsr_dim : 0;
    // Argument sanity check before allocating invalid memory
    if(mb <= 0 || nb <= 0 || M <= 0 || N <= 0 || bsr_dim <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);
        device_vector<T>             dx(safe_size);
        device_vector<T>             dy(safe_size);

        if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dx || !dy)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrmv<T>(handle,
                                                   dir,
                                                   trans,
                                                   mb,
                                                   nb,
                                                   safe_size,
                                                   &h_alpha,
                                                   descr,
                                                   dbsr_val,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   bsr_dim,
                                                   dx,
                                                   &h_beta,
                                                   dy),
                                (mb < 0 || nb < 0 || bsr_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    // Wavefront size
    int dev;
    hipGetDevice(&dev);

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev);

    bool                        type = (prop.warpSize == 32) ? (arg.timing ? false : true) : false;
    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, type, full_rank);

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz, base);

    // Update BSR block dimensions from generated matrix
    mb = (M + bsr_dim - 1) / bsr_dim;
    nb = (N + bsr_dim - 1) / bsr_dim;

    // Allocate host memory for vectors
    host_vector<T> hx(nb * bsr_dim);
    host_vector<T> hy_gold(mb * bsr_dim);

    // Initialize data on CPU
    // We need to initialize the padded entries (if any) with zero
    rocsparse_init<T>(hx, 1, nb * bsr_dim, 1);
    rocsparse_init<T>(hy_gold, 1, mb * bsr_dim, 1);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);
    device_vector<T>             dx(nb * bsr_dim);
    device_vector<T>             dy_1(mb * bsr_dim);
    device_vector<T>             dy_2(mb * bsr_dim);
    device_vector<T>             d_alpha(1);
    device_vector<T>             d_beta(1);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy_1 || !dy_2 || !d_alpha || !d_beta)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    // Padded x and y entries must be copied over too (as they are initialized with zero)
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * nb * bsr_dim, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_gold, sizeof(T) * mb * bsr_dim, hipMemcpyHostToDevice));

    // Convert CSR to BSR
    rocsparse_int                nnzb;
    device_vector<rocsparse_int> dbsr_row_ptr(mb + 1);

    if(!dbsr_row_ptr)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(
        handle, dir, M, N, descr, dcsr_row_ptr, dcsr_col_ind, bsr_dim, descr, dbsr_row_ptr, &nnzb));

    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val(nnzb * bsr_dim * bsr_dim);

    if(!dbsr_col_ind || !dbsr_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                               dir,
                                               M,
                                               N,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               bsr_dim,
                                               descr,
                                               dbsr_val,
                                               dbsr_row_ptr,
                                               dbsr_col_ind));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_gold, sizeof(T) * mb * bsr_dim, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrmv<T>(handle,
                                                 dir,
                                                 trans,
                                                 mb,
                                                 nb,
                                                 nnzb,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 bsr_dim,
                                                 dx,
                                                 &h_beta,
                                                 dy_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrmv<T>(handle,
                                                 dir,
                                                 trans,
                                                 mb,
                                                 nb,
                                                 nnzb,
                                                 d_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 bsr_dim,
                                                 dx,
                                                 d_beta,
                                                 dy_2));

        // Copy output to host
        host_vector<T> hy_1(M);
        host_vector<T> hy_2(M);

        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // Make BSR matrix available on host
        host_vector<rocsparse_int> hbsr_row_ptr(mb + 1);
        host_vector<rocsparse_int> hbsr_col_ind(nnzb);
        host_vector<T>             hbsr_val(nnzb * bsr_dim * bsr_dim);

        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr, dbsr_row_ptr, sizeof(rocsparse_int) * (mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind, dbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_val, dbsr_val, sizeof(T) * nnzb * bsr_dim * bsr_dim, hipMemcpyDeviceToHost));

        // CPU bsrmv
        host_bsrmv<T>(dir,
                      trans,
                      mb,
                      nb,
                      nnzb,
                      h_alpha,
                      hbsr_row_ptr,
                      hbsr_col_ind,
                      hbsr_val,
                      bsr_dim,
                      hx,
                      h_beta,
                      hy_gold,
                      base);

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
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrmv<T>(handle,
                                                     dir,
                                                     trans,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     bsr_dim,
                                                     dx,
                                                     &h_beta,
                                                     dy_1));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrmv<T>(handle,
                                                     dir,
                                                     trans,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     bsr_dim,
                                                     dx,
                                                     &h_beta,
                                                     dy_1));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gpu_gflops
            = spmv_gflop_count<T>(M, nnz, h_beta != static_cast<T>(0)) / gpu_time_used * 1e6;
        double gpu_gbyte = bsrmv_gbyte_count<T>(mb, nb, nnzb, bsr_dim, h_beta != static_cast<T>(0))
                           / gpu_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "BSR nnz"
                  << std::setw(12) << "BSR dim" << std::setw(12) << "dir" << std::setw(12)
                  << "alpha" << std::setw(12) << "beta" << std::setw(12) << "GFlop/s"
                  << std::setw(12) << "GB/s" << std::setw(12) << "msec" << std::setw(12) << "iter"
                  << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnzb
                  << std::setw(12) << bsr_dim << std::setw(12)
                  << (dir == rocsparse_direction_row ? "row" : "col") << std::setw(12) << h_alpha
                  << std::setw(12) << h_beta << std::setw(12) << gpu_gflops << std::setw(12)
                  << gpu_gbyte << std::setw(12) << gpu_time_used / 1e3 << std::setw(12)
                  << number_hot_calls << std::setw(12) << (arg.unit_check ? "yes" : "no")
                  << std::endl;
    }
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_bsrmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

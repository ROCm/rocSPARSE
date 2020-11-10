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

#include "utility.hpp"
#include <rocsparse.hpp>

#include "flops.hpp"
#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"
#include "utility.hpp"

template <typename T>
void testing_bsrsv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;
    static const size_t safe_dim  = 2;

    T h_alpha = 0.6;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Allocate memory on device
    device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
    device_vector<rocsparse_int> dbsr_col_ind(safe_size);
    device_vector<T>             dbsr_val(safe_size);
    device_vector<T>             dx(safe_size);
    device_vector<T>             dy(safe_size);
    device_vector<T>             dbuffer(safe_size);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dx || !dy || !dbuffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_bsrsv_buffer_size()
    size_t buffer_size;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(nullptr,
                                                           rocsparse_direction_column,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           safe_dim,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(handle,
                                                           rocsparse_direction_column,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           nullptr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           safe_dim,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(handle,
                                                           rocsparse_direction_column,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           nullptr,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           safe_dim,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(handle,
                                                           rocsparse_direction_column,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dbsr_val,
                                                           nullptr,
                                                           dbsr_col_ind,
                                                           safe_dim,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(handle,
                                                           rocsparse_direction_column,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           nullptr,
                                                           safe_dim,
                                                           info,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(handle,
                                                           rocsparse_direction_column,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           safe_dim,
                                                           nullptr,
                                                           &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(handle,
                                                           rocsparse_direction_column,
                                                           rocsparse_operation_none,
                                                           safe_size,
                                                           safe_size,
                                                           descr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           safe_dim,
                                                           info,
                                                           nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_bsrsv_analysis()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(nullptr,
                                                        rocsparse_direction_column,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dbsr_val,
                                                        dbsr_row_ptr,
                                                        dbsr_col_ind,
                                                        safe_dim,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(handle,
                                                        rocsparse_direction_column,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        nullptr,
                                                        dbsr_val,
                                                        dbsr_row_ptr,
                                                        dbsr_col_ind,
                                                        safe_dim,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(handle,
                                                        rocsparse_direction_column,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        nullptr,
                                                        dbsr_row_ptr,
                                                        dbsr_col_ind,
                                                        safe_dim,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(handle,
                                                        rocsparse_direction_column,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dbsr_val,
                                                        nullptr,
                                                        dbsr_col_ind,
                                                        safe_dim,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(handle,
                                                        rocsparse_direction_column,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dbsr_val,
                                                        dbsr_row_ptr,
                                                        nullptr,
                                                        safe_dim,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(handle,
                                                        rocsparse_direction_column,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dbsr_val,
                                                        dbsr_row_ptr,
                                                        dbsr_col_ind,
                                                        safe_dim,
                                                        nullptr,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(handle,
                                                        rocsparse_direction_column,
                                                        rocsparse_operation_none,
                                                        safe_size,
                                                        safe_size,
                                                        descr,
                                                        dbsr_val,
                                                        dbsr_row_ptr,
                                                        dbsr_col_ind,
                                                        safe_dim,
                                                        info,
                                                        rocsparse_analysis_policy_reuse,
                                                        rocsparse_solve_policy_auto,
                                                        nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_bsrsv_solve()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(nullptr,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     nullptr,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     nullptr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     nullptr,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     nullptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     nullptr,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     nullptr,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     nullptr,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     nullptr,
                                                     rocsparse_solve_policy_auto,
                                                     dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                     rocsparse_direction_column,
                                                     rocsparse_operation_none,
                                                     safe_size,
                                                     safe_size,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     safe_dim,
                                                     info,
                                                     dx,
                                                     dy,
                                                     rocsparse_solve_policy_auto,
                                                     nullptr),
                            rocsparse_status_invalid_pointer);

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
}

template <typename T>
void testing_bsrsv(const Arguments& arg)
{
    rocsparse_int             M         = arg.M;
    rocsparse_int             N         = arg.N;
    rocsparse_int             K         = arg.K;
    rocsparse_int             dim_x     = arg.dimx;
    rocsparse_int             dim_y     = arg.dimy;
    rocsparse_int             dim_z     = arg.dimz;
    rocsparse_direction       dir       = arg.direction;
    rocsparse_operation       trans     = arg.transA;
    rocsparse_diag_type       diag      = arg.diag;
    rocsparse_fill_mode       uplo      = arg.uplo;
    rocsparse_analysis_policy apol      = arg.apol;
    rocsparse_solve_policy    spol      = arg.spol;
    rocsparse_index_base      base      = arg.baseA;
    rocsparse_int             bsr_dim   = arg.block_dim;
    rocsparse_matrix_init     mat       = arg.matrix;
    bool                      full_rank = true;
    std::string               filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_alpha = arg.get_alpha<T>();

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

    // BSR dimensions
    rocsparse_int mb = (M + bsr_dim - 1) / bsr_dim;
    rocsparse_int nb = (N + bsr_dim - 1) / bsr_dim;

    // Argument sanity check before allocating invalid memory
    if(mb <= 0 || nb <= 0 || M <= 0 || N <= 0 || bsr_dim <= 0)
    {
        static const size_t safe_size = 100;
        size_t              buffer_size;
        rocsparse_int       pivot;

        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);
        device_vector<T>             dx(safe_size);
        device_vector<T>             dy(safe_size);
        device_vector<T>             dbuffer(safe_size);

        if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dx || !dy || !dbuffer)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_buffer_size<T>(handle,
                                                               dir,
                                                               trans,
                                                               mb,
                                                               safe_size,
                                                               descr,
                                                               dbsr_val,
                                                               dbsr_row_ptr,
                                                               dbsr_col_ind,
                                                               bsr_dim,
                                                               info,
                                                               &buffer_size),
                                (mb < 0 || nb < 0 || bsr_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_analysis<T>(handle,
                                                            dir,
                                                            trans,
                                                            mb,
                                                            safe_size,
                                                            descr,
                                                            dbsr_val,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            bsr_dim,
                                                            info,
                                                            apol,
                                                            spol,
                                                            dbuffer),
                                (mb < 0 || nb < 0 || bsr_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_solve<T>(handle,
                                                         dir,
                                                         trans,
                                                         mb,
                                                         safe_size,
                                                         &h_alpha,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         bsr_dim,
                                                         info,
                                                         dx,
                                                         dy,
                                                         spol,
                                                         dbuffer),
                                (mb < 0 || nb < 0 || bsr_dim < 0) ? rocsparse_status_invalid_size
                                                                  : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, &pivot),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_clear(handle, info), rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    rocsparse_seedrand();

    // Sample matrix
    rocsparse_int nnz;
    rocsparse_init_csr_matrix(hcsr_row_ptr,
                              hcsr_col_ind,
                              hcsr_val,
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
                              false,
                              full_rank);

    // Non-squared matrices are not supported
    if(M != N)
    {
        return;
    }

    // Update BSR block dimensions from generated matrix
    mb = (M + bsr_dim - 1) / bsr_dim;
    nb = (N + bsr_dim - 1) / bsr_dim;

    // Allocate host memory for vectors
    host_vector<T>             hx(nb * bsr_dim, 1.0);
    host_vector<rocsparse_int> h_analysis_pivot_1(1);
    host_vector<rocsparse_int> h_analysis_pivot_2(1);
    host_vector<rocsparse_int> h_analysis_pivot_gold(1);
    host_vector<rocsparse_int> h_solve_pivot_1(1);
    host_vector<rocsparse_int> h_solve_pivot_2(1);
    host_vector<rocsparse_int> h_solve_pivot_gold(1);

    // Initialize data on CPU
    //    rocsparse_init<T>(hx, 1, nb * bsr_dim, 1);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);
    device_vector<T>             dx(nb * bsr_dim);
    device_vector<T>             dy_1(mb * bsr_dim);
    device_vector<T>             dy_2(mb * bsr_dim);
    device_vector<T>             d_alpha(1);
    device_vector<rocsparse_int> d_analysis_pivot_2(1);
    device_vector<rocsparse_int> d_solve_pivot_2(1);

    if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dx || !dy_1 || !dy_2 || !d_alpha
       || !d_analysis_pivot_2 || !d_solve_pivot_2)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * nb * bsr_dim, hipMemcpyHostToDevice));

    // Convert CSR to BSR
    rocsparse_int                nnzb;
    device_vector<rocsparse_int> dbsr_row_ptr(mb + 1);

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(
        handle, dir, M, N, descr, dcsr_row_ptr, dcsr_col_ind, bsr_dim, descr, dbsr_row_ptr, &nnzb));

    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val(nnzb * bsr_dim * bsr_dim);

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

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_buffer_size<T>(handle,
                                                         dir,
                                                         trans,
                                                         mb,
                                                         nnzb,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         bsr_dim,
                                                         info,
                                                         &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // Perform analysis step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(handle,
                                                          dir,
                                                          trans,
                                                          mb,
                                                          nnzb,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          bsr_dim,
                                                          info,
                                                          apol,
                                                          spol,
                                                          dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, h_analysis_pivot_1),
                                (h_analysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(handle,
                                                          dir,
                                                          trans,
                                                          mb,
                                                          nnzb,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          bsr_dim,
                                                          info,
                                                          apol,
                                                          spol,
                                                          dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, d_analysis_pivot_2),
                                (h_analysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Perform solve step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(handle,
                                                       dir,
                                                       trans,
                                                       mb,
                                                       nnzb,
                                                       &h_alpha,
                                                       descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       bsr_dim,
                                                       info,
                                                       dx,
                                                       dy_1,
                                                       spol,
                                                       dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, h_solve_pivot_1),
                                (h_solve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(handle,
                                                       dir,
                                                       trans,
                                                       mb,
                                                       nnzb,
                                                       d_alpha,
                                                       descr,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       bsr_dim,
                                                       info,
                                                       dx,
                                                       dy_2,
                                                       spol,
                                                       dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrsv_zero_pivot(handle, info, d_solve_pivot_2),
                                (h_solve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Copy output to host
        host_vector<T> hy_1(M);
        host_vector<T> hy_2(M);

        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_analysis_pivot_2, d_analysis_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_solve_pivot_2, d_solve_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

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

        // CPU bsrsv
        host_vector<T> hy_gold(mb * bsr_dim);

        host_bsrsv<T>(trans,
                      dir,
                      mb,
                      nnzb,
                      h_alpha,
                      hbsr_row_ptr,
                      hbsr_col_ind,
                      hbsr_val,
                      bsr_dim,
                      hx,
                      hy_gold,
                      diag,
                      uplo,
                      base,
                      h_analysis_pivot_gold,
                      h_solve_pivot_gold);

        // Check pivots
        unit_check_general<rocsparse_int>(1, 1, 1, h_analysis_pivot_gold, h_analysis_pivot_1);
        unit_check_general<rocsparse_int>(1, 1, 1, h_analysis_pivot_gold, h_analysis_pivot_2);
        unit_check_general<rocsparse_int>(1, 1, 1, h_solve_pivot_gold, h_solve_pivot_1);
        unit_check_general<rocsparse_int>(1, 1, 1, h_solve_pivot_gold, h_solve_pivot_2);

        // Check solution vector if no pivot has been found
        if(h_analysis_pivot_gold[0] == -1 && h_solve_pivot_gold[0] == -1)
        {
            near_check_general<T>(1, M, 1, hy_gold, hy_1);
            near_check_general<T>(1, M, 1, hy_gold, hy_2);
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
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(handle,
                                                              dir,
                                                              trans,
                                                              mb,
                                                              nnzb,
                                                              descr,
                                                              dbsr_val,
                                                              dbsr_row_ptr,
                                                              dbsr_col_ind,
                                                              bsr_dim,
                                                              info,
                                                              apol,
                                                              spol,
                                                              dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(handle,
                                                           dir,
                                                           trans,
                                                           mb,
                                                           nnzb,
                                                           &h_alpha,
                                                           descr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           bsr_dim,
                                                           info,
                                                           dx,
                                                           dy_1,
                                                           spol,
                                                           dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_clear(handle, info));
        }

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_analysis<T>(handle,
                                                          dir,
                                                          trans,
                                                          mb,
                                                          nnzb,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          bsr_dim,
                                                          info,
                                                          apol,
                                                          spol,
                                                          dbuffer));

        gpu_analysis_time_used = get_time_us() - gpu_analysis_time_used;

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_solve<T>(handle,
                                                           dir,
                                                           trans,
                                                           mb,
                                                           nnzb,
                                                           &h_alpha,
                                                           descr,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           bsr_dim,
                                                           info,
                                                           dx,
                                                           dy_1,
                                                           spol,
                                                           dbuffer));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gpu_gflops = csrsv_gflop_count<T>(mb * bsr_dim, nnzb * bsr_dim * bsr_dim, diag)
                            / gpu_solve_time_used * 1e6;
        double gpu_gbyte = bsrsv_gbyte_count<T>(mb, nnzb, bsr_dim) / gpu_solve_time_used * 1e6;

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);
        // TODO
        std::cout << std::setw(12) << "M" << std::setw(12) << "nnz" << std::setw(12) << "alpha"
                  << std::setw(12) << "pivot" << std::setw(16) << "operation" << std::setw(12)
                  << "diag_type" << std::setw(12) << "fill_mode" << std::setw(16)
                  << "analysis_policy" << std::setw(16) << "solve_policy" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(16) << "analysis_msec"
                  << std::setw(16) << "solve_msec" << std::setw(12) << "iter" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << nnz << std::setw(12) << h_alpha
                  << std::setw(12) << std::min(h_analysis_pivot_gold[0], h_solve_pivot_gold[0])
                  << std::setw(16) << rocsparse_operation2string(trans) << std::setw(12)
                  << rocsparse_diagtype2string(diag) << std::setw(12)
                  << rocsparse_fillmode2string(uplo) << std::setw(16)
                  << rocsparse_analysis2string(apol) << std::setw(16)
                  << rocsparse_solve2string(spol) << std::setw(12) << gpu_gflops << std::setw(12)
                  << gpu_gbyte << std::setw(16) << gpu_analysis_time_used / 1e3 << std::setw(16)
                  << gpu_solve_time_used / 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Clear bsrsv meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrsv_clear(handle, info));

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_bsrsv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrsv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

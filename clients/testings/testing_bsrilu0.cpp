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

#include "rocsparse_data.hpp"
#include "rocsparse_datatype2string.hpp"
#include "rocsparse_test.hpp"
#include "testing_bsrilu0.hpp"
#include "type_dispatch.hpp"

#include <cctype>
#include <complex>
#include <cstring>
#include <type_traits>

#include <iomanip>

#include "flops.hpp"
#include "gbyte.hpp"
#include "rocsparse_check.hpp"
#include "rocsparse_host.hpp"
#include "rocsparse_init.hpp"
#include "rocsparse_math.hpp"
#include "rocsparse_random.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_vector.hpp"

#include "testing_csrilu0.hpp"

template <typename T>
void testing_bsrilu0_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

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
    device_vector<T>             dbuffer(safe_size);
    device_vector<T>             dboost_tol(1);
    device_vector<T>             dboost_val(1);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val || !dbuffer || !dboost_tol || !dboost_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test rocsparse_bsrilu0_buffer_size()
    size_t buffer_size;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(nullptr,
                                                             rocsparse_direction_row,
                                                             safe_size,
                                                             safe_size,
                                                             descr,
                                                             dbsr_val,
                                                             dbsr_row_ptr,
                                                             dbsr_col_ind,
                                                             safe_size,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                             rocsparse_direction_row,
                                                             safe_size,
                                                             safe_size,
                                                             nullptr,
                                                             dbsr_val,
                                                             dbsr_row_ptr,
                                                             dbsr_col_ind,
                                                             safe_size,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                             rocsparse_direction_row,
                                                             safe_size,
                                                             safe_size,
                                                             descr,
                                                             nullptr,
                                                             dbsr_row_ptr,
                                                             dbsr_col_ind,
                                                             safe_size,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                             rocsparse_direction_row,
                                                             safe_size,
                                                             safe_size,
                                                             descr,
                                                             dbsr_val,
                                                             nullptr,
                                                             dbsr_col_ind,
                                                             safe_size,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                             rocsparse_direction_row,
                                                             safe_size,
                                                             safe_size,
                                                             descr,
                                                             dbsr_val,
                                                             dbsr_row_ptr,
                                                             nullptr,
                                                             safe_size,
                                                             info,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                             rocsparse_direction_row,
                                                             safe_size,
                                                             safe_size,
                                                             descr,
                                                             dbsr_val,
                                                             dbsr_row_ptr,
                                                             dbsr_col_ind,
                                                             safe_size,
                                                             nullptr,
                                                             &buffer_size),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                             rocsparse_direction_row,
                                                             safe_size,
                                                             safe_size,
                                                             descr,
                                                             dbsr_val,
                                                             dbsr_row_ptr,
                                                             dbsr_col_ind,
                                                             safe_size,
                                                             info,
                                                             nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_bsrilu0_numeric_boost()
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_bsrilu0_numeric_boost<T>(nullptr, info, 1, get_boost_tol(dboost_tol), dboost_val),
        rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_numeric_boost<T>(
                                handle, nullptr, 1, get_boost_tol(dboost_tol), dboost_val),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_bsrilu0_numeric_boost<T>(handle, info, 1, get_boost_tol((T*)nullptr), dboost_val),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_bsrilu0_numeric_boost<T>(handle, info, 1, get_boost_tol(dboost_tol), nullptr),
        rocsparse_status_invalid_pointer);

    // Test rocsparse_bsrilu0_analysis()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(nullptr,
                                                          rocsparse_direction_row,
                                                          safe_size,
                                                          safe_size,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          safe_size,
                                                          info,
                                                          rocsparse_analysis_policy_reuse,
                                                          rocsparse_solve_policy_auto,
                                                          dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(handle,
                                                          rocsparse_direction_row,
                                                          safe_size,
                                                          safe_size,
                                                          nullptr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          safe_size,
                                                          info,
                                                          rocsparse_analysis_policy_reuse,
                                                          rocsparse_solve_policy_auto,
                                                          dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(handle,
                                                          rocsparse_direction_row,
                                                          safe_size,
                                                          safe_size,
                                                          descr,
                                                          nullptr,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          safe_size,
                                                          info,
                                                          rocsparse_analysis_policy_reuse,
                                                          rocsparse_solve_policy_auto,
                                                          dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(handle,
                                                          rocsparse_direction_row,
                                                          safe_size,
                                                          safe_size,
                                                          descr,
                                                          dbsr_val,
                                                          nullptr,
                                                          dbsr_col_ind,
                                                          safe_size,
                                                          info,
                                                          rocsparse_analysis_policy_reuse,
                                                          rocsparse_solve_policy_auto,
                                                          dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(handle,
                                                          rocsparse_direction_row,
                                                          safe_size,
                                                          safe_size,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          nullptr,
                                                          safe_size,
                                                          info,
                                                          rocsparse_analysis_policy_reuse,
                                                          rocsparse_solve_policy_auto,
                                                          dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(handle,
                                                          rocsparse_direction_row,
                                                          safe_size,
                                                          safe_size,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          safe_size,
                                                          nullptr,
                                                          rocsparse_analysis_policy_reuse,
                                                          rocsparse_solve_policy_auto,
                                                          dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(handle,
                                                          rocsparse_direction_row,
                                                          safe_size,
                                                          safe_size,
                                                          descr,
                                                          dbsr_val,
                                                          dbsr_row_ptr,
                                                          dbsr_col_ind,
                                                          safe_size,
                                                          info,
                                                          rocsparse_analysis_policy_reuse,
                                                          rocsparse_solve_policy_auto,
                                                          nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_bsrilu0()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(nullptr,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 info,
                                                 rocsparse_solve_policy_auto,
                                                 dbuffer),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 info,
                                                 rocsparse_solve_policy_auto,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 descr,
                                                 nullptr,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 info,
                                                 rocsparse_solve_policy_auto,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 descr,
                                                 dbsr_val,
                                                 nullptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 info,
                                                 rocsparse_solve_policy_auto,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 nullptr,
                                                 safe_size,
                                                 info,
                                                 rocsparse_solve_policy_auto,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 nullptr,
                                                 rocsparse_solve_policy_auto,
                                                 dbuffer),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(handle,
                                                 rocsparse_direction_row,
                                                 safe_size,
                                                 safe_size,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 safe_size,
                                                 info,
                                                 rocsparse_solve_policy_auto,
                                                 nullptr),
                            rocsparse_status_invalid_pointer);

    // Test rocsparse_bsrilu0_zero_pivot()
    rocsparse_int position;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(nullptr, info, &position),
                            rocsparse_status_invalid_handle);

    // Test rocsparse_bsrilu0_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_clear(nullptr, info),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_bsrilu0(const Arguments& arg)
{
    rocsparse_int             M           = arg.M;
    rocsparse_int             N           = arg.N;
    rocsparse_int             K           = arg.K;
    rocsparse_int             block_dim   = arg.block_dim;
    rocsparse_int             dim_x       = arg.dimx;
    rocsparse_int             dim_y       = arg.dimy;
    rocsparse_int             dim_z       = arg.dimz;
    rocsparse_analysis_policy apol        = arg.apol;
    rocsparse_solve_policy    spol        = arg.spol;
    int                       boost       = arg.numericboost;
    T                         h_boost_tol = static_cast<T>(arg.boosttol);
    rocsparse_index_base      base        = arg.baseA;
    rocsparse_direction       direction   = arg.direction;
    rocsparse_matrix_init     mat         = arg.matrix;
    bool                      full_rank   = true;
    std::string               filename
        = arg.timing ? arg.filename : rocsparse_exepath() + "../matrices/" + arg.filename + ".csr";

    T h_boost_val = arg.get_boostval<T>();

    rocsparse_int Mb = -1;
    rocsparse_int Nb = -1;
    if(block_dim > 0)
    {
        Mb = (M + block_dim - 1) / block_dim;
        Nb = (N + block_dim - 1) / block_dim;
    }

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || block_dim <= 0)
    {
        static const size_t safe_size = 100;
        size_t              buffer_size;
        rocsparse_int       pivot;

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                                 direction,
                                                                 Mb,
                                                                 safe_size,
                                                                 descr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 safe_size,
                                                                 info,
                                                                 &buffer_size),
                                (Mb < 0 || block_dim <= 0) ? rocsparse_status_invalid_size
                                                           : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_analysis<T>(handle,
                                                              direction,
                                                              Mb,
                                                              safe_size,
                                                              descr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              safe_size,
                                                              info,
                                                              apol,
                                                              spol,
                                                              nullptr),
                                (Mb < 0 || block_dim <= 0) ? rocsparse_status_invalid_size
                                                           : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0<T>(handle,
                                                     direction,
                                                     Mb,
                                                     safe_size,
                                                     descr,
                                                     nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     safe_size,
                                                     info,
                                                     spol,
                                                     nullptr),
                                (Mb < 0 || block_dim <= 0) ? rocsparse_status_invalid_size
                                                           : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, &pivot),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_clear(handle, info), rocsparse_status_success);

        return;
    }

    // Allocate host memory for original CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_orig;
    host_vector<rocsparse_int> hcsr_col_ind_orig;
    host_vector<T>             hcsr_val_orig;

    rocsparse_seedrand();

    // Generate CSR matrix on host (or read from file)
    rocsparse_int nnz;
    rocsparse_init_csr_matrix(hcsr_row_ptr_orig,
                              hcsr_col_ind_orig,
                              hcsr_val_orig,
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

    // M and N can be modified by rocsparse_init_csr_matrix if reading from a file
    Mb = (M + block_dim - 1) / block_dim;
    Nb = (N + block_dim - 1) / block_dim;

    // // Allocate device memory for original CSR matrix
    device_vector<rocsparse_int> dcsr_row_ptr_orig(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_orig(nnz);
    device_vector<T>             dcsr_val_orig(nnz);

    // Copy CSR matrix to device
    CHECK_HIP_ERROR(hipMemcpy(dcsr_row_ptr_orig,
                              hcsr_row_ptr_orig.data(),
                              sizeof(rocsparse_int) * (M + 1),
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_col_ind_orig,
                              hcsr_col_ind_orig.data(),
                              sizeof(rocsparse_int) * nnz,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_val_orig, hcsr_val_orig.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Allocate device memory for BSR row pointer array
    device_vector<rocsparse_int> dbsr_row_ptr(Mb + 1);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    // Convert sample CSR matrix to bsr
    rocsparse_int nnzb;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                direction,
                                                M,
                                                N,
                                                descr,
                                                dcsr_row_ptr_orig,
                                                dcsr_col_ind_orig,
                                                block_dim,
                                                descr,
                                                dbsr_row_ptr,
                                                &nnzb));

    // Allocate device memory for BSR col indices and values array
    device_vector<rocsparse_int> dbsr_col_ind(nnzb);
    device_vector<T>             dbsr_val_1(nnzb * block_dim * block_dim);
    device_vector<T>             dbsr_val_2(nnzb * block_dim * block_dim);

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                               direction,
                                               M,
                                               N,
                                               descr,
                                               dcsr_val_orig,
                                               dcsr_row_ptr_orig,
                                               dcsr_col_ind_orig,
                                               block_dim,
                                               descr,
                                               dbsr_val_1,
                                               dbsr_row_ptr,
                                               dbsr_col_ind));

    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_val_2, dbsr_val_1, sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyDeviceToDevice));

    // Allocate host memory for output BSR matrix
    host_vector<rocsparse_int> hbsr_row_ptr(Mb + 1);
    host_vector<rocsparse_int> hbsr_col_ind(nnzb);
    host_vector<T>             hbsr_val_orig(nnzb * block_dim * block_dim);
    host_vector<T>             hbsr_val_gold(nnzb * block_dim * block_dim);
    host_vector<T>             hbsr_val_1(nnzb * block_dim * block_dim);
    host_vector<T>             hbsr_val_2(nnzb * block_dim * block_dim);

    // Copy BSR matrix output to host
    CHECK_HIP_ERROR(hipMemcpy(hbsr_row_ptr.data(),
                              dbsr_row_ptr,
                              sizeof(rocsparse_int) * (Mb + 1),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hbsr_col_ind.data(), dbsr_col_ind, sizeof(rocsparse_int) * nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_val_orig.data(),
                              dbsr_val_1,
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_val_gold.data(),
                              dbsr_val_1,
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_val_1.data(),
                              dbsr_val_1,
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_val_2.data(),
                              dbsr_val_2,
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyDeviceToHost));

    // Allocate host memory for pivots
    host_vector<rocsparse_int> hanalysis_pivot_1(1);
    host_vector<rocsparse_int> hanalysis_pivot_2(1);
    host_vector<rocsparse_int> hanalysis_pivot_gold(1);
    host_vector<rocsparse_int> hsolve_pivot_1(1);
    host_vector<rocsparse_int> hsolve_pivot_2(1);
    host_vector<rocsparse_int> hsolve_pivot_gold(1);

    // Allocate device memory for pivots
    device_vector<rocsparse_int> danalysis_pivot_2(1);
    device_vector<rocsparse_int> dsolve_pivot_2(1);
    device_vector<T>             d_boost_tol(1);
    device_vector<T>             d_boost_val(1);

    if(!dbsr_row_ptr || !dbsr_col_ind || !dbsr_val_1 || !dbsr_val_2 || !danalysis_pivot_2
       || !dsolve_pivot_2 || !d_boost_tol || !d_boost_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_buffer_size<T>(handle,
                                                           direction,
                                                           Mb,
                                                           nnzb,
                                                           descr,
                                                           dbsr_val_1,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           block_dim,
                                                           info,
                                                           &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(d_boost_tol, &h_boost_tol, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_boost_val, &h_boost_val, sizeof(T), hipMemcpyHostToDevice));

        // Perform analysis step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_analysis<T>(handle,
                                                            direction,
                                                            Mb,
                                                            nnzb,
                                                            descr,
                                                            dbsr_val_1,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            info,
                                                            apol,
                                                            spol,
                                                            dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, hanalysis_pivot_1),
                                (hanalysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_analysis<T>(handle,
                                                            direction,
                                                            Mb,
                                                            nnzb,
                                                            descr,
                                                            dbsr_val_2,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            info,
                                                            apol,
                                                            spol,
                                                            dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, danalysis_pivot_2),
                                (hanalysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);

        // Perform solve step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_numeric_boost<T>(
            handle, info, boost, get_boost_tol(&h_boost_tol), &h_boost_val));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0<T>(handle,
                                                   direction,
                                                   Mb,
                                                   nnzb,
                                                   descr,
                                                   dbsr_val_1,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   block_dim,
                                                   info,
                                                   spol,
                                                   dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, hsolve_pivot_1),
                                (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_numeric_boost<T>(
            handle, info, boost, get_boost_tol(d_boost_tol), d_boost_val));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0<T>(handle,
                                                   direction,
                                                   Mb,
                                                   nnzb,
                                                   descr,
                                                   dbsr_val_2,
                                                   dbsr_row_ptr,
                                                   dbsr_col_ind,
                                                   block_dim,
                                                   info,
                                                   spol,
                                                   dbuffer));
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, dsolve_pivot_2),
                                (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val_1,
                                  dbsr_val_1,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val_2,
                                  dbsr_val_2,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hanalysis_pivot_2, danalysis_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hsolve_pivot_2, dsolve_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // CPU bsrilu0
        rocsparse_int numerical_pivot;
        rocsparse_int structural_pivot;
        host_bsrilu0<T>(direction,
                        Mb,
                        hbsr_row_ptr,
                        hbsr_col_ind,
                        hbsr_val_gold,
                        block_dim,
                        base,
                        &structural_pivot,
                        &numerical_pivot,
                        boost,
                        *get_boost_tol(&h_boost_tol),
                        h_boost_val);

        hanalysis_pivot_gold[0] = structural_pivot;

        // Solve pivot gives the first numerical or structural non-invertible block
        if(structural_pivot == -1)
        {
            hsolve_pivot_gold[0] = numerical_pivot;
        }
        else if(numerical_pivot == -1)
        {
            hsolve_pivot_gold[0] = structural_pivot;
        }
        else
        {
            hsolve_pivot_gold[0] = std::min(numerical_pivot, structural_pivot);
        }

        // Check pivots
        unit_check_general<rocsparse_int>(1, 1, 1, hanalysis_pivot_gold, hanalysis_pivot_1);
        unit_check_general<rocsparse_int>(1, 1, 1, hanalysis_pivot_gold, hanalysis_pivot_2);
        unit_check_general<rocsparse_int>(1, 1, 1, hsolve_pivot_gold, hsolve_pivot_1);
        unit_check_general<rocsparse_int>(1, 1, 1, hsolve_pivot_gold, hsolve_pivot_2);

        // Check solution vector if no pivot has been found
        if(hanalysis_pivot_gold[0] == -1 && hsolve_pivot_gold[0] == -1)
        {
            near_check_general<T>(1, nnzb * block_dim * block_dim, 1, hbsr_val_gold, hbsr_val_1);
            near_check_general<T>(1, nnzb * block_dim * block_dim, 1, hbsr_val_gold, hbsr_val_2);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_numeric_boost<T>(
            handle, info, 0, get_boost_tol((T*)nullptr), nullptr));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIP_ERROR(hipMemcpy(dbsr_val_1,
                                      hbsr_val_orig,
                                      sizeof(T) * nnzb * block_dim * block_dim,
                                      hipMemcpyHostToDevice));

            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_analysis<T>(handle,
                                                                direction,
                                                                Mb,
                                                                nnzb,
                                                                descr,
                                                                dbsr_val_1,
                                                                dbsr_row_ptr,
                                                                dbsr_col_ind,
                                                                block_dim,
                                                                info,
                                                                apol,
                                                                spol,
                                                                dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0<T>(handle,
                                                       direction,
                                                       Mb,
                                                       nnzb,
                                                       descr,
                                                       dbsr_val_1,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       block_dim,
                                                       info,
                                                       spol,
                                                       dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_clear(handle, info));
        }

        CHECK_HIP_ERROR(hipMemcpy(dbsr_val_1,
                                  hbsr_val_orig,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyHostToDevice));

        double gpu_analysis_time_used = get_time_us();

        // Analysis run
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_analysis<T>(handle,
                                                            direction,
                                                            Mb,
                                                            nnzb,
                                                            descr,
                                                            dbsr_val_1,
                                                            dbsr_row_ptr,
                                                            dbsr_col_ind,
                                                            block_dim,
                                                            info,
                                                            apol,
                                                            spol,
                                                            dbuffer));

        gpu_analysis_time_used = (get_time_us() - gpu_analysis_time_used);

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, hanalysis_pivot_1),
                                (hanalysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                             : rocsparse_status_success);

        double gpu_solve_time_used = 0;

        // Solve run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIP_ERROR(hipMemcpy(dbsr_val_1,
                                      hbsr_val_orig,
                                      sizeof(T) * nnzb * block_dim * block_dim,
                                      hipMemcpyHostToDevice));

            double temp = get_time_us();
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0<T>(handle,
                                                       direction,
                                                       Mb,
                                                       nnzb,
                                                       descr,
                                                       dbsr_val_1,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       block_dim,
                                                       info,
                                                       spol,
                                                       dbuffer));
            gpu_solve_time_used += (get_time_us() - temp);
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrilu0_zero_pivot(handle, info, hsolve_pivot_1),
                                (hsolve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                          : rocsparse_status_success);

        gpu_solve_time_used = gpu_solve_time_used / number_hot_calls;

        double gpu_gbyte = bsrilu0_gbyte_count<T>(Mb, block_dim, nnzb) / gpu_solve_time_used * 1e6;

        rocsparse_int pivot = -1;
        if(hanalysis_pivot_1[0] == -1)
        {
            pivot = hsolve_pivot_1[0];
        }
        else if(hsolve_pivot_1[0] == -1)
        {
            pivot = hanalysis_pivot_1[0];
        }
        else
        {
            pivot = std::min(hanalysis_pivot_1[0], hsolve_pivot_1[0]);
        }

        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::left);

        std::cout << std::setw(12) << "M" << std::setw(12) << "nnzb" << std::setw(12) << "block_dim"
                  << std::setw(12) << "pivot" << std::setw(16) << "direction" << std::setw(16)
                  << "analysis policy" << std::setw(16) << "solve policy" << std::setw(12) << "GB/s"
                  << std::setw(16) << "analysis msec" << std::setw(16) << "solve msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << M << std::setw(12) << nnzb << std::setw(12) << block_dim
                  << std::setw(12) << pivot << std::setw(16)
                  << rocsparse_direction2string(direction) << std::setw(16)
                  << rocsparse_analysis2string(apol) << std::setw(16)
                  << rocsparse_solve2string(spol) << std::setw(12) << gpu_gbyte << std::setw(16)
                  << gpu_analysis_time_used / 1e3 << std::setw(16) << gpu_solve_time_used / 1e3
                  << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }

    // Clear bsrilu0 meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrilu0_clear(handle, info));

    // Free buffer
    CHECK_HIP_ERROR(hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_bsrilu0_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrilu0<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

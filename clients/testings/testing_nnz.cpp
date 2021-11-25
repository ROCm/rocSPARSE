/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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

template <typename T>
void testing_nnz_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle                 = local_handle;
    rocsparse_direction       dir                    = rocsparse_direction_row;
    rocsparse_int             m                      = safe_size;
    rocsparse_int             n                      = safe_size;
    const rocsparse_mat_descr descr                  = local_descr;
    const T*                  A                      = (const T*)0x4;
    rocsparse_int             ld                     = safe_size;
    rocsparse_int*            nnz_per_row_columns    = (rocsparse_int*)0x4;
    rocsparse_int*            nnz_total_dev_host_ptr = (rocsparse_int*)0x4;

#define PARAMS handle, dir, m, n, descr, A, ld, nnz_per_row_columns, nnz_total_dev_host_ptr
    auto_testing_bad_arg(rocsparse_nnz<T>, PARAMS);
#undef PARAMS
}

template <typename T>
void testing_nnz(const Arguments& arg)
{
    rocsparse_int       M    = arg.M;
    rocsparse_int       N    = arg.N;
    rocsparse_direction dirA = arg.direction;
    rocsparse_int       LD   = arg.denseld;

    rocsparse_local_handle    handle;
    rocsparse_local_mat_descr descrA;

    //
    // Argument sanity check before allocating invalid memory
    //
    if(M <= 0 || N <= 0 || LD < M)
    {
        rocsparse_status expected_status = (((M == 0 && N >= 0) || (M >= 0 && N == 0)) && (LD >= M))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_size;

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_nnz<T>(handle, dirA, M, N, descrA, nullptr, LD, nullptr, nullptr),
            expected_status);

        if(rocsparse_status_success == expected_status)
        {
            rocsparse_int h_nnz = 77;
            CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_nnz<T>(handle, dirA, M, N, descrA, nullptr, LD, nullptr, nullptr),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_nnz<T>(handle, dirA, M, N, descrA, nullptr, LD, nullptr, &h_nnz),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(0 == h_nnz ? rocsparse_status_success
                                               : rocsparse_status_internal_error,
                                    rocsparse_status_success);

            h_nnz = 139;
            device_vector<rocsparse_int> d_nnz(1);
            CHECK_HIP_ERROR(hipMemcpy(
                (rocsparse_int*)d_nnz, &h_nnz, sizeof(rocsparse_int) * 1, hipMemcpyHostToDevice));

            CHECK_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_nnz<T>(handle, dirA, M, N, descrA, nullptr, LD, nullptr, nullptr),
                rocsparse_status_success);

            EXPECT_ROCSPARSE_STATUS(
                rocsparse_nnz<T>(
                    handle, dirA, M, N, descrA, nullptr, LD, nullptr, (rocsparse_int*)d_nnz),
                rocsparse_status_success);

            CHECK_HIP_ERROR(hipMemcpy(
                &h_nnz, (rocsparse_int*)d_nnz, sizeof(rocsparse_int) * 1, hipMemcpyDeviceToHost));

            EXPECT_ROCSPARSE_STATUS(0 == h_nnz ? rocsparse_status_success
                                               : rocsparse_status_internal_error,
                                    rocsparse_status_success);
        }

        return;
    }

    //
    // Create the dense matrix.
    //
    rocsparse_int MN = (dirA == rocsparse_direction_row) ? M : N;

    host_vector<T>             h_A(LD * N);
    host_vector<rocsparse_int> h_nnzPerRowColumn(MN);
    host_vector<rocsparse_int> hd_nnzPerRowColumn(MN);
    host_vector<rocsparse_int> h_nnzTotalDevHostPtr(1);
    host_vector<rocsparse_int> hd_nnzTotalDevHostPtr(1);

    // Allocate device memory
    device_vector<T>             d_A(LD * N);
    device_vector<rocsparse_int> d_nnzPerRowColumn(MN);
    device_vector<rocsparse_int> d_nnzTotalDevHostPtr(1);
    if(!h_nnzPerRowColumn || !d_nnzPerRowColumn || !d_A)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    //
    // Initialize a random matrix.
    //
    rocsparse_seedrand();

    //
    // Initialize the entire allocated memory.
    //
    for(rocsparse_int i = 0; i < LD; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            h_A[j * LD + i] = -1;
        }
    }

    //
    // Random initialization of the matrix.
    //
    for(rocsparse_int i = 0; i < M; ++i)
    {
        for(rocsparse_int j = 0; j < N; ++j)
        {
            h_A[j * LD + i] = random_generator<T>(0, 4);
        }
    }

    //
    // Transfer.
    //
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A, sizeof(T) * LD * N, hipMemcpyHostToDevice));

    //
    // Unit check.
    //
    if(arg.unit_check)
    {
        //
        // Compute the reference host first.
        //
        host_nnz<T>(dirA, M, N, h_A, LD, h_nnzPerRowColumn, h_nnzTotalDevHostPtr);

        //
        // Pointer mode device for nnz and call.
        //
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz<T>(
            handle, dirA, M, N, descrA, d_A, LD, d_nnzPerRowColumn, d_nnzTotalDevHostPtr));

        //
        // Transfer.
        //
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzPerRowColumn,
                                  d_nnzPerRowColumn,
                                  sizeof(rocsparse_int) * MN,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzTotalDevHostPtr,
                                  d_nnzTotalDevHostPtr,
                                  sizeof(rocsparse_int) * 1,
                                  hipMemcpyDeviceToHost));

        //
        // Check results.
        //
        hd_nnzPerRowColumn.unit_check(h_nnzPerRowColumn);
        hd_nnzTotalDevHostPtr.unit_check(h_nnzTotalDevHostPtr);

        //
        // Pointer mode host for nnz and call.
        //
        rocsparse_int dh_nnz;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_nnz<T>(handle, dirA, M, N, descrA, d_A, LD, d_nnzPerRowColumn, &dh_nnz));

        //
        // Transfer.
        //
        CHECK_HIP_ERROR(hipMemcpy(hd_nnzPerRowColumn,
                                  d_nnzPerRowColumn,
                                  sizeof(rocsparse_int) * MN,
                                  hipMemcpyDeviceToHost));

        //
        // Check results.
        //
        hd_nnzPerRowColumn.unit_check(h_nnzPerRowColumn);
        unit_check_scalar(dh_nnz, h_nnzTotalDevHostPtr[0]);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        //
        // Warm-up
        //
        rocsparse_int h_nnz;
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(
                rocsparse_nnz<T>(handle, dirA, M, N, descrA, d_A, LD, d_nnzPerRowColumn, &h_nnz));
        }

        double gpu_time_used = get_time_us();
        {
            //
            // Performance run
            //
            for(int iter = 0; iter < number_hot_calls; ++iter)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_nnz<T>(
                    handle, dirA, M, N, descrA, d_A, LD, d_nnzPerRowColumn, &h_nnz));
            }
        }
        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = nnz_gbyte_count<T>(M, N, dirA);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "LD",
                            LD,
                            "nnz",
                            h_nnz,
                            "dir",
                            rocsparse_direction2string(dirA),
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used),
                            "iter",
                            number_hot_calls,
                            "verified",
                            arg.unit_check ? "yes" : "no");
    }
}

#define INSTANTIATE(TYPE)                                          \
    template void testing_nnz_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_nnz<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

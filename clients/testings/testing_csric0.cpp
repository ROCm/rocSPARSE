/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_enum.hpp"
#include "testing.hpp"

static int const neg_pivot_level = 3;

template <typename T>
void testing_csric0_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    rocsparse_int             nnz         = safe_size;
    const rocsparse_mat_descr descr       = local_descr;
    T*                        csr_val     = (T*)0x4;
    const rocsparse_int*      csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_mat_info        info        = local_info;
    rocsparse_analysis_policy analysis    = rocsparse_analysis_policy_force;
    rocsparse_solve_policy    solve       = rocsparse_solve_policy_auto;
    size_t*                   buffer_size = (size_t*)0x4;
    void*                     temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE \
    handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size
#define PARAMS_ANALYSIS \
    handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer
#define PARAMS handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, solve, temp_buffer

    auto_testing_bad_arg(rocsparse_csric0_buffer_size<T>, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_csric0_analysis<T>, PARAMS_ANALYSIS);
    auto_testing_bad_arg(rocsparse_csric0<T>, PARAMS);

    for(auto val : rocsparse_matrix_type_t::values)
    {
        if(val != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, val));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csric0<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0<T>(PARAMS), rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_ANALYSIS
#undef PARAMS

    // Test rocsparse_csric0_zero_pivot()
    rocsparse_int position = -1;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(nullptr, info, &position),
                            rocsparse_status_invalid_handle);

    // Test rocsparse_csric0_negative_pivot()
    rocsparse_int negative_position = -1;
    if(neg_pivot_level >= 1)
    {
        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_negative_pivot(nullptr, info, &negative_position),
                                rocsparse_status_invalid_handle);
    };

    // Test rocsparse_csric0_clear()
    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_clear(nullptr, info), rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_csric0_buffer_size<T>(
            handle, safe_size, safe_size, descr, nullptr, csr_row_ptr, nullptr, info, buffer_size),
        rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_analysis<T>(handle,
                                                         safe_size,
                                                         safe_size,
                                                         descr,
                                                         nullptr,
                                                         csr_row_ptr,
                                                         nullptr,
                                                         info,
                                                         rocsparse_analysis_policy_reuse,
                                                         rocsparse_solve_policy_auto,
                                                         temp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_csric0<T>(handle,
                                                safe_size,
                                                safe_size,
                                                descr,
                                                nullptr,
                                                csr_row_ptr,
                                                nullptr,
                                                info,
                                                rocsparse_solve_policy_auto,
                                                temp_buffer),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_csric0(const Arguments& arg)
{
    rocsparse_int             M      = arg.M;
    rocsparse_int             N      = arg.N;
    rocsparse_analysis_policy apol   = arg.apol;
    rocsparse_solve_policy    spol   = arg.spol;
    rocsparse_index_base      base   = arg.baseA;
    bool                      to_int = arg.timing ? false : true;

    static constexpr bool       full_rank = true;
    rocsparse_matrix_factory<T> matrix_factory(arg, to_int, full_rank);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0)
    {
        static const size_t safe_size = 100;
        size_t              buffer_size;
        rocsparse_int       pivot;
        rocsparse_int       negative_pivot;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<T>             dbuffer(safe_size);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_buffer_size<T>(handle,
                                                                M,
                                                                safe_size,
                                                                descr,
                                                                dcsr_val,
                                                                dcsr_row_ptr,
                                                                dcsr_col_ind,
                                                                info,
                                                                &buffer_size),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_analysis<T>(handle,
                                                             M,
                                                             safe_size,
                                                             descr,
                                                             dcsr_val,
                                                             dcsr_row_ptr,
                                                             dcsr_col_ind,
                                                             info,
                                                             apol,
                                                             spol,
                                                             dbuffer),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0<T>(handle,
                                                    M,
                                                    safe_size,
                                                    descr,
                                                    dcsr_val,
                                                    dcsr_row_ptr,
                                                    dcsr_col_ind,
                                                    info,
                                                    spol,
                                                    dbuffer),
                                (M < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(handle, info, &pivot),
                                rocsparse_status_success);

        if(neg_pivot_level >= 2)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_negative_pivot(handle, info, &negative_pivot),
                                    rocsparse_status_success);
        };

        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_clear(handle, info), rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;
    host_vector<T>             hcsr_val_gold;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz, base);

    hcsr_val_gold = hcsr_val;

    // Allocate host memory for vectors
    host_vector<T> hcsr_val_1(nnz);
    host_vector<T> hcsr_val_2(nnz);

    host_vector<rocsparse_int> h_analysis_pivot_1(1);
    host_vector<rocsparse_int> h_analysis_pivot_2(1);

    host_vector<rocsparse_int> h_analysis_pivot_gold(1);
    host_vector<rocsparse_int> h_solve_pivot_1(1);
    host_vector<rocsparse_int> h_solve_pivot_2(1);
    host_vector<rocsparse_int> h_solve_pivot_gold(1);

    host_vector<rocsparse_int> h_solve_pivot_neg(1);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val_1(nnz);
    device_vector<T>             dcsr_val_2(nnz);
    device_vector<rocsparse_int> d_analysis_pivot_2(1);
    device_vector<rocsparse_int> d_solve_pivot_2(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val_1, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_csric0_buffer_size<T>(
        handle, M, nnz, descr, dcsr_val_1, dcsr_row_ptr, dcsr_col_ind, info, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dcsr_val_2, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

        // Perform analysis step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                           M,
                                                           nnz,
                                                           descr,
                                                           dcsr_val_1,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           apol,
                                                           spol,
                                                           dbuffer));
        {
            auto st = rocsparse_csric0_zero_pivot(handle, info, h_analysis_pivot_1);
            EXPECT_ROCSPARSE_STATUS(st,
                                    (h_analysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                                  : rocsparse_status_success);
        }

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                           M,
                                                           nnz,
                                                           descr,
                                                           dcsr_val_2,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           apol,
                                                           spol,
                                                           dbuffer));

        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(handle, info, d_analysis_pivot_2),
                                (h_analysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Perform solve step

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csric0<T>(
            handle, M, nnz, descr, dcsr_val_1, dcsr_row_ptr, dcsr_col_ind, info, spol, dbuffer));
        {
            auto st = rocsparse_csric0_zero_pivot(handle, info, h_solve_pivot_1);
            EXPECT_ROCSPARSE_STATUS(st,
                                    (h_solve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                               : rocsparse_status_success);
        }

        if(neg_pivot_level >= 3)
        {
            auto st = rocsparse_csric0_negative_pivot(handle, info, h_solve_pivot_neg);
            EXPECT_ROCSPARSE_STATUS(st,
                                    (h_solve_pivot_neg[0] != -1) ? rocsparse_status_negative_pivot
                                                                 : rocsparse_status_success);
        }

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csric0<T>(
            handle, M, nnz, descr, dcsr_val_2, dcsr_row_ptr, dcsr_col_ind, info, spol, dbuffer));

        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(handle, info, d_solve_pivot_2),
                                (h_solve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

        if(neg_pivot_level >= 4)
        {
           auto istat = rocsparse_csric0_negative_pivot( handle, info, h_solve_pivot_neg);
           std::cout << "testing_csric0:  istat = " << istat 
                     << " h_solve_pivot_neg[0] " << h_solve_pivot_neg[0]
                     << " line " << __LINE__
                     << "\n";
/*
            EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_negative_pivot(handle, info, d_solve_pivot_2),
                                    (h_solve_pivot_neg[0] != -1) ? rocsparse_status_negative_pivot
                                                                 : rocsparse_status_success);
*/
        };

        // Sync to force updated pivots
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val_1, dcsr_val_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val_2, dcsr_val_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_analysis_pivot_2, d_analysis_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_solve_pivot_2, d_solve_pivot_2, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // CPU csric0
        host_csric0<T>(M,
                       hcsr_row_ptr,
                       hcsr_col_ind,
                       hcsr_val_gold,
                       base,
                       h_analysis_pivot_gold,
                       h_solve_pivot_gold);

        // Check pivots
        h_analysis_pivot_gold.unit_check(h_analysis_pivot_1);
        h_analysis_pivot_gold.unit_check(h_analysis_pivot_2);
        h_solve_pivot_gold.unit_check(h_solve_pivot_1);
        h_solve_pivot_gold.unit_check(h_solve_pivot_2);

        // Check solution vector if no pivot has been found
        if(h_analysis_pivot_gold[0] == -1 && h_solve_pivot_gold[0] == -1)
        {
            hcsr_val_gold.near_check(hcsr_val_1);
            hcsr_val_gold.near_check(hcsr_val_2);
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
            CHECK_HIP_ERROR(
                hipMemcpy(dcsr_val_1, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

            CHECK_ROCSPARSE_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                               M,
                                                               nnz,
                                                               descr,
                                                               dcsr_val_1,
                                                               dcsr_row_ptr,
                                                               dcsr_col_ind,
                                                               info,
                                                               apol,
                                                               spol,
                                                               dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_csric0<T>(handle,
                                                      M,
                                                      nnz,
                                                      descr,
                                                      dcsr_val_1,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      info,
                                                      spol,
                                                      dbuffer));
            CHECK_ROCSPARSE_ERROR(rocsparse_csric0_clear(handle, info));
        }

        CHECK_HIP_ERROR(hipMemcpy(dcsr_val_1, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

        double gpu_analysis_time_used = get_time_us();

        CHECK_ROCSPARSE_ERROR(rocsparse_csric0_analysis<T>(handle,
                                                           M,
                                                           nnz,
                                                           descr,
                                                           dcsr_val_1,
                                                           dcsr_row_ptr,
                                                           dcsr_col_ind,
                                                           info,
                                                           apol,
                                                           spol,
                                                           dbuffer));
        gpu_analysis_time_used = (get_time_us() - gpu_analysis_time_used);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(handle, info, h_analysis_pivot_1),
                                (h_analysis_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                              : rocsparse_status_success);

        double gpu_solve_time_used = 0;

        // Solve run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(dcsr_val_1, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

            double temp = get_time_us();
            CHECK_ROCSPARSE_ERROR(rocsparse_csric0<T>(handle,
                                                      M,
                                                      nnz,
                                                      descr,
                                                      dcsr_val_1,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      info,
                                                      spol,
                                                      dbuffer));
            gpu_solve_time_used += (get_time_us() - temp);
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_csric0_zero_pivot(handle, info, h_solve_pivot_1),
                                (h_solve_pivot_1[0] != -1) ? rocsparse_status_zero_pivot
                                                           : rocsparse_status_success);

        if(neg_pivot_level >= 5)
        {
           auto istat = rocsparse_csric0_negative_pivot( handle, info, h_solve_pivot_neg);
           std::cout << "testing_csric0:  istat = " << istat 
                     << " h_solve_pivot_neg[0] " << h_solve_pivot_neg[0]
                     << " line " << __LINE__
                     << "\n";
     
                    
/*
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_csric0_negative_pivot(handle, info, h_solve_pivot_neg),
                (h_solve_pivot_neg[0] != -1) ? rocsparse_status_negative_pivot
                                             : rocsparse_status_success);
*/

        };

        gpu_solve_time_used = gpu_solve_time_used / number_hot_calls;

        double gbyte_count = csric0_gbyte_count<T>(M, nnz);

        rocsparse_int pivot = -1;
        if(h_analysis_pivot_1[0] == -1)
        {
            pivot = h_solve_pivot_1[0];
        }
        else if(h_solve_pivot_1[0] == -1)
        {
            pivot = h_analysis_pivot_1[0];
        }
        else
        {
            pivot = std::min(h_analysis_pivot_1[0], h_solve_pivot_1[0]);
        }

        double gpu_gbyte = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "nnz",
                            nnz,
                            "pivot",
                            pivot,
                            "analysis policy",
                            rocsparse_analysis2string(apol),
                            "solve policy",
                            rocsparse_solve2string(spol),
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            "analysis msec",
                            get_gpu_time_msec(gpu_analysis_time_used),
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Clear csric0 meta data
    CHECK_ROCSPARSE_ERROR(rocsparse_csric0_clear(handle, info));

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                             \
    template void testing_csric0_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csric0<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csric0_extra(const Arguments& arg) {}

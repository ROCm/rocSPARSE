/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename T>
void testing_coosort_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle      = local_handle;
    rocsparse_int    m           = safe_size;
    rocsparse_int    n           = safe_size;
    rocsparse_int    nnz         = safe_size;
    rocsparse_int*   coo_row_ind = (rocsparse_int*)0x4;
    rocsparse_int*   coo_col_ind = (rocsparse_int*)0x4;
    size_t*          buffer_size = (size_t*)0x4;
    void*            temp_buffer = (void*)0x4;

    int            nargs_to_exclude   = 1;
    const int      args_to_exclude[1] = {6};
    rocsparse_int* perm               = nullptr;

#define PARAMS_BUFFER_SIZE handle, m, n, nnz, coo_row_ind, coo_col_ind, buffer_size
#define PARAMS handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer
    auto_testing_bad_arg(rocsparse_coosort_buffer_size, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_coosort_by_row, nargs_to_exclude, args_to_exclude, PARAMS);
    auto_testing_bad_arg(rocsparse_coosort_by_column, nargs_to_exclude, args_to_exclude, PARAMS);
#undef PARAMS_BUFFER_SIZE
#undef PARAMS
}

template <typename T>
void testing_coosort(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M       = arg.M;
    rocsparse_int               N       = arg.N;
    bool                        permute = arg.algo;
    bool                        by_row  = arg.transA == rocsparse_operation_none;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcoo_row_ind(safe_size);
        device_vector<rocsparse_int> dcoo_col_ind(safe_size);
        device_vector<rocsparse_int> dbuffer(safe_size);

        size_t buffer_size;
        EXPECT_ROCSPARSE_STATUS(rocsparse_coosort_buffer_size(
                                    handle, M, N, 0, dcoo_row_ind, dcoo_col_ind, &buffer_size),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        if(by_row)
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_coosort_by_row(
                    handle, M, N, 0, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer),
                (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        }
        else
        {
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_coosort_by_column(
                    handle, M, N, 0, dcoo_row_ind, dcoo_col_ind, nullptr, dbuffer),
                (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        }

        return;
    }

    // Allocate host memory for COO matrix
    host_vector<rocsparse_int> hcoo_row_ind;
    host_vector<rocsparse_int> hcoo_col_ind;
    host_vector<T>             hcoo_val;
    host_vector<rocsparse_int> hcoo_row_ind_gold;
    host_vector<rocsparse_int> hcoo_col_ind_gold;
    host_vector<T>             hcoo_val_gold;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_coo(
        hcoo_row_ind, hcoo_col_ind, hcoo_val, M, N, nnz, rocsparse_index_base_zero);

    // Unsort COO matrix
    host_vector<rocsparse_int> hperm(nnz);
    hcoo_row_ind_gold = hcoo_row_ind;
    hcoo_col_ind_gold = hcoo_col_ind;
    hcoo_val_gold     = hcoo_val;

    for(rocsparse_int i = 0; i < nnz; ++i)
    {
        rocsparse_int rng = rand() % nnz;

        std::swap(hcoo_row_ind[i], hcoo_row_ind[rng]);
        std::swap(hcoo_col_ind[i], hcoo_col_ind[rng]);
        std::swap(hcoo_val[i], hcoo_val[rng]);
    }

    // Allocate device memory
    device_vector<rocsparse_int> dcoo_row_ind(nnz);
    device_vector<rocsparse_int> dcoo_col_ind(nnz);
    device_vector<T>             dcoo_val(nnz);
    device_vector<rocsparse_int> dperm(nnz);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_col_ind, hcoo_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcoo_val, hcoo_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_coosort_buffer_size(handle, M, N, nnz, dcoo_row_ind, dcoo_col_ind, &buffer_size));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Create permutation vector
        CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnz, dperm));

        // Sort COO matrix
        if(by_row)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_row(
                handle, M, N, nnz, dcoo_row_ind, dcoo_col_ind, permute ? dperm : nullptr, dbuffer));
        }
        else
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_column(
                handle, M, N, nnz, dcoo_row_ind, dcoo_col_ind, permute ? dperm : nullptr, dbuffer));

            // Sort host COO structure by column
            host_coosort_by_column<T>(M, nnz, hcoo_row_ind_gold, hcoo_col_ind_gold, hcoo_val_gold);
        }

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_row_ind, dcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcoo_col_ind, dcoo_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));

        hcoo_row_ind_gold.unit_check(hcoo_row_ind);
        hcoo_col_ind_gold.unit_check(hcoo_col_ind);

        // Permute, copy and check values, if requested
        if(permute)
        {
            device_vector<T> dcoo_val_sorted(nnz);

            CHECK_ROCSPARSE_ERROR(rocsparse_gthr<T>(
                handle, nnz, dcoo_val, dcoo_val_sorted, dperm, rocsparse_index_base_zero));
            CHECK_HIP_ERROR(
                hipMemcpy(hcoo_val, dcoo_val_sorted, sizeof(T) * nnz, hipMemcpyDeviceToHost));

            hcoo_val_gold.unit_check(hcoo_val);
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            if(by_row)
            {
                rocsparse_coosort_by_row(handle,
                                         M,
                                         N,
                                         nnz,
                                         dcoo_row_ind,
                                         dcoo_col_ind,
                                         permute ? dperm : nullptr,
                                         dbuffer);
            }
            else
            {
                rocsparse_coosort_by_column(handle,
                                            M,
                                            N,
                                            nnz,
                                            dcoo_row_ind,
                                            dcoo_col_ind,
                                            permute ? dperm : nullptr,
                                            dbuffer);
            }
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            if(by_row)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_row(handle,
                                                               M,
                                                               N,
                                                               nnz,
                                                               dcoo_row_ind,
                                                               dcoo_col_ind,
                                                               permute ? dperm : nullptr,
                                                               dbuffer));
            }
            else
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_coosort_by_column(handle,
                                                                  M,
                                                                  N,
                                                                  nnz,
                                                                  dcoo_row_ind,
                                                                  dcoo_col_ind,
                                                                  permute ? dperm : nullptr,
                                                                  dbuffer));
            }
        }

        double gbyte_count = coosort_gbyte_count<T>(nnz, permute);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            M,
                            "N",
                            N,
                            "nnz",
                            nnz,
                            "permute",
                            (permute ? "yes" : "no"),
                            "dir",
                            (by_row ? "row" : "column"),
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }

    // Clear buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_coosort_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_coosort<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

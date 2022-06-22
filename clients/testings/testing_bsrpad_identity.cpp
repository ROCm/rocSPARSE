/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

#include "auto_testing_bad_arg.hpp"

template <typename T>
void testing_bsrpad_identity_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_bsr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             block_dim   = safe_size;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    T*                        bsr_val     = (T*)0x4;
    rocsparse_int*            bsr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            bsr_col_ind = (rocsparse_int*)0x4;

#define PARAMS handle, m, mb, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind

    auto_testing_bad_arg(rocsparse_bsrpad_identity<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_identity<T>(PARAMS), rocsparse_status_not_implemented);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_sorted));

    mb = 3;
    m  = block_dim * mb + 1;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_identity<T>(PARAMS), rocsparse_status_invalid_size);
    mb = m = safe_size;

#undef PARAMS
}

template <typename T>
void testing_bsrpad_identity(const Arguments& arg)
{
    static constexpr bool       toint     = false;
    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, toint, full_rank);

    rocsparse_int        M         = arg.M;
    rocsparse_int        K         = arg.K; //size of m
    rocsparse_int        block_dim = arg.block_dim;
    rocsparse_index_base base      = arg.baseA;
    rocsparse_direction  direction = arg.direction;

    rocsparse_int Mb = -1;
    if(block_dim > 0)
    {
        Mb = (M + block_dim - 1) / block_dim;
    }

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr bsr_descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, base));

    // Argument sanity check before allocating invalid memory
    if(Mb <= 0 || block_dim <= 0 || K < 0 || Mb * block_dim < K)
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dbsr_row_ptr(safe_size);
        device_vector<rocsparse_int> dbsr_col_ind(safe_size);
        device_vector<T>             dbsr_val(safe_size);

        EXPECT_ROCSPARSE_STATUS(
            rocsparse_bsrpad_identity<T>(
                handle, K, Mb, block_dim, bsr_descr, dbsr_val, dbsr_row_ptr, dbsr_col_ind),
            (Mb < 0 || block_dim <= 0 || K < 0 || Mb * block_dim < K)
                ? rocsparse_status_invalid_size
                : rocsparse_status_success);

        return;
    }

    // Allocate host memory for BSR matrix
    host_gebsr_matrix<T> hbsrA(direction, Mb, Mb, 0, block_dim, block_dim, base);

    // Generate BSR matrix on host (or read from file)
    matrix_factory.init_bsr(hbsrA, Mb, Mb);

    // Convert to device memory
    device_gebsr_matrix<T> dbsr(hbsrA);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrpad_identity<T>(
            handle, K, Mb, block_dim, bsr_descr, dbsr.val, dbsr.ptr, dbsr.ind));

        // Copy output to host
        host_gebsr_matrix<T> hbsrC(dbsr);

        host_gebsr_matrix<T> hbsrC_gold(hbsrA);

        // CPU bsrpad_identity
        host_bsrpad_identity(K,
                             K,
                             Mb,
                             Mb,
                             block_dim,
                             hbsrC_gold.val,
                             hbsrC_gold.ptr,
                             hbsrC_gold.ind,
                             hbsrC_gold.base);
        hbsrC.unit_check(hbsrC_gold);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrpad_identity<T>(
                handle, K, Mb, block_dim, bsr_descr, dbsr.val, dbsr.ptr, dbsr.ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bsrpad_identity<T>(
                handle, K, Mb, block_dim, bsr_descr, dbsr.val, dbsr.ptr, dbsr.ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = 0;
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("K",
                            K,
                            "Mb",
                            Mb,
                            "blockdim",
                            block_dim,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                      \
    template void testing_bsrpad_identity_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrpad_identity<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

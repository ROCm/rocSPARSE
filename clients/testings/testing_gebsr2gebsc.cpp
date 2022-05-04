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
#include "auto_testing_bad_arg.hpp"
#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_gebsr2gebsc_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle        = local_handle;
    rocsparse_int        mb            = safe_size;
    rocsparse_int        nb            = safe_size;
    rocsparse_int        nnzb          = safe_size;
    const T*             bsr_val       = (const T*)0x4;
    const rocsparse_int* bsr_row_ptr   = (const rocsparse_int*)0x4;
    const rocsparse_int* bsr_col_ind   = (const rocsparse_int*)0x4;
    rocsparse_int        row_block_dim = safe_size;
    rocsparse_int        col_block_dim = safe_size;
    T*                   bsc_val       = (T*)0x4;
    rocsparse_int*       bsc_row_ind   = (rocsparse_int*)0x4;
    rocsparse_int*       bsc_col_ptr   = (rocsparse_int*)0x4;
    rocsparse_action     copy_values   = rocsparse_action_numeric;
    rocsparse_index_base idx_base      = rocsparse_index_base_zero;
    size_t*              buffer_size   = (size_t*)0x4;
    void*                temp_buffer   = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                 \
    handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, \
        buffer_size
#define PARAMS                                                                             \
    handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, \
        bsc_val, bsc_row_ind, bsc_col_ptr, copy_values, idx_base, temp_buffer
    auto_testing_bad_arg(rocsparse_gebsr2gebsc_buffer_size<T>, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_gebsr2gebsc<T>, PARAMS);
#undef PARAMS
#undef PARAMS_BUFFER_SIZE

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc_buffer_size<T>(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 safe_size,
                                                                 nullptr,
                                                                 bsr_row_ptr,
                                                                 nullptr,
                                                                 safe_size,
                                                                 safe_size,
                                                                 buffer_size),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc<T>(handle,
                                                     safe_size,
                                                     safe_size,
                                                     safe_size,
                                                     nullptr,
                                                     bsr_row_ptr,
                                                     nullptr,
                                                     safe_size,
                                                     safe_size,
                                                     bsc_val,
                                                     bsc_row_ind,
                                                     bsc_col_ptr,
                                                     rocsparse_action_numeric,
                                                     rocsparse_index_base_zero,
                                                     temp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc<T>(handle,
                                                     safe_size,
                                                     safe_size,
                                                     safe_size,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     safe_size,
                                                     safe_size,
                                                     nullptr,
                                                     nullptr,
                                                     bsc_col_ptr,
                                                     rocsparse_action_numeric,
                                                     rocsparse_index_base_zero,
                                                     temp_buffer),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_gebsr2gebsc(const Arguments& arg)
{
    rocsparse_action action = arg.action;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(arg.M <= 0 || arg.N <= 0 || arg.row_block_dimA <= 0 || arg.col_block_dimA <= 0)
    {
        rocsparse_int        M             = arg.M;
        rocsparse_int        N             = arg.N;
        rocsparse_int        row_block_dim = arg.row_block_dimA;
        rocsparse_int        col_block_dim = arg.col_block_dimA;
        rocsparse_index_base base          = arg.baseA;

        static const size_t safe_size = 100;

        // Allocate memory on device

        device_vector<T> dbuffer(safe_size);

        if(!dbuffer)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        size_t buffer_size;

        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc_buffer_size<T>(handle,
                                                                     M,
                                                                     N,
                                                                     safe_size,
                                                                     (const T*)nullptr,
                                                                     nullptr,
                                                                     nullptr,
                                                                     row_block_dim,
                                                                     col_block_dim,
                                                                     &buffer_size),
                                (M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0)
                                    ? rocsparse_status_invalid_size
                                    : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc<T>(handle,
                                                         M,
                                                         N,
                                                         safe_size,
                                                         (const T*)nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         row_block_dim,
                                                         col_block_dim,
                                                         (T*)nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         action,
                                                         base,
                                                         dbuffer),
                                (M < 0 || N < 0 || row_block_dim < 0 || col_block_dim < 0)
                                    ? rocsparse_status_invalid_size
                                    : rocsparse_status_success);

        return;
    }

    //
    // Declare the factory.
    //
    rocsparse_matrix_factory<T> factory(arg);

    //
    // Initialize the matrix.
    //
    host_gebsr_matrix<T> hbsr;
    factory.init_gebsr(hbsr);

    //
    // Allocate and transfer to device.
    //
    device_gebsr_matrix<T> dbsr(hbsr);

    //
    // Obtain required buffer size (from host)
    //
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsc_buffer_size<T>(handle,
                                                               dbsr.mb,
                                                               dbsr.nb,
                                                               dbsr.nnzb,
                                                               dbsr.val,
                                                               dbsr.ptr,
                                                               dbsr.ind,
                                                               dbsr.row_block_dim,
                                                               dbsr.col_block_dim,
                                                               &buffer_size));

    //
    // Allocate the buffer size.
    //
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    //
    // Allocate device bsc matrix.
    //
    device_gebsc_matrix<T> dbsc(dbsr.block_direction,
                                dbsr.mb,
                                dbsr.nb,
                                dbsr.nnzb,
                                dbsr.row_block_dim,
                                dbsr.col_block_dim,
                                dbsr.base);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsc<T>(handle,
                                                       dbsr.mb,
                                                       dbsr.nb,
                                                       dbsr.nnzb,
                                                       dbsr.val,
                                                       dbsr.ptr,
                                                       dbsr.ind,
                                                       dbsr.row_block_dim,
                                                       dbsr.col_block_dim,
                                                       dbsc.val,
                                                       dbsc.ind,
                                                       dbsc.ptr,
                                                       action,
                                                       dbsr.base,
                                                       dbuffer));

        //
        // Transfer to host.
        //
        host_gebsc_matrix<T> hbsc_from_device(dbsc);

        //
        // Allocate host bsc matrix.
        //
        host_gebsc_matrix<T> hbsc(hbsr.block_direction,
                                  hbsr.mb,
                                  hbsr.nb,
                                  hbsr.nnzb,
                                  hbsr.row_block_dim,
                                  hbsr.col_block_dim,
                                  hbsr.base);

        //
        // Now the results need to be validated with 2 steps:
        //
        host_gebsr_to_gebsc<T>(hbsr.mb,
                               hbsr.nb,
                               hbsr.nnzb,
                               hbsr.ptr,
                               hbsr.ind,
                               hbsr.val,
                               hbsr.row_block_dim,
                               hbsr.col_block_dim,
                               hbsc.ind,
                               hbsc.ptr,
                               hbsc.val,
                               action,
                               hbsr.base);

        hbsc.unit_check(hbsc_from_device, action == rocsparse_action_numeric);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsc<T>(handle,
                                                           dbsr.mb,
                                                           dbsr.nb,
                                                           dbsr.nnzb,
                                                           dbsr.val,
                                                           dbsr.ptr,
                                                           dbsr.ind,
                                                           dbsr.row_block_dim,
                                                           dbsr.col_block_dim,
                                                           dbsc.val,
                                                           dbsc.ind,
                                                           dbsc.ptr,
                                                           action,
                                                           dbsr.base,
                                                           dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gebsr2gebsc<T>(handle,
                                                           dbsr.mb,
                                                           dbsr.nb,
                                                           dbsr.nnzb,
                                                           dbsr.val,
                                                           dbsr.ptr,
                                                           dbsr.ind,
                                                           dbsr.row_block_dim,
                                                           dbsr.col_block_dim,
                                                           dbsc.val,
                                                           dbsc.ind,
                                                           dbsc.ptr,
                                                           action,
                                                           dbsr.base,
                                                           dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = gebsr2gebsc_gbyte_count<T>(
            dbsr.mb, dbsr.nb, dbsr.nnzb, dbsr.row_block_dim, dbsr.col_block_dim, action);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("Mb",
                            dbsr.mb,
                            "Nb",
                            dbsr.nb,
                            "nnzb",
                            dbsr.nnzb,
                            "rbdim",
                            dbsr.row_block_dim,
                            "cbdim",
                            dbsr.col_block_dim,
                            "action",
                            rocsparse_action2string(action),
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }
    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                                  \
    template void testing_gebsr2gebsc_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsr2gebsc<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

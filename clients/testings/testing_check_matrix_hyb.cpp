/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
void testing_check_matrix_hyb_bad_arg(const Arguments& arg)
{
    rocsparse_index_base   base        = rocsparse_index_base_zero;
    rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general;
    rocsparse_fill_mode    uplo        = rocsparse_fill_mode_lower;
    rocsparse_storage_mode storage     = rocsparse_storage_mode_sorted;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_hyb_mat local_hyb;

    rocsparse_handle        handle      = local_handle;
    const rocsparse_hyb_mat hyb         = local_hyb;
    void*                   temp_buffer = (void*)0x4;
    size_t                  buffer_size;
    rocsparse_data_status   data_status;

    auto_testing_bad_arg(rocsparse_check_matrix_hyb_buffer_size,
                         handle,
                         hyb,
                         base,
                         matrix_type,
                         uplo,
                         storage,
                         &buffer_size);

    int       nargs_to_exclude   = 1;
    const int args_to_exclude[1] = {7};
    auto_testing_bad_arg(rocsparse_check_matrix_hyb,
                         nargs_to_exclude,
                         args_to_exclude,
                         handle,
                         hyb,
                         base,
                         matrix_type,
                         uplo,
                         storage,
                         &data_status,
                         temp_buffer);
}

template <typename T>
void testing_check_matrix_hyb(const Arguments& arg)
{
    rocsparse_int          m           = arg.M;
    rocsparse_int          n           = arg.N;
    rocsparse_index_base   base        = arg.baseA;
    rocsparse_matrix_type  matrix_type = arg.matrix_type;
    rocsparse_fill_mode    uplo        = arg.uplo;
    rocsparse_storage_mode storage     = arg.storage;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create hyb matrix
    rocsparse_local_hyb_mat hyb;

    rocsparse_matrix_factory<T> matrix_factory(arg);

    // Generate (or load from file) HYB matrix
    bool          conform;
    rocsparse_int nnz;
    matrix_factory.init_hyb(hyb, m, n, nnz, base, conform);
    if(!conform)
    {
        return;
    }

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_hyb_buffer_size(
        handle, hyb, base, matrix_type, uplo, storage, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    rocsparse_data_status data_status;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_hyb(
        handle, hyb, base, matrix_type, uplo, storage, &data_status, dbuffer));

    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_hyb(
                handle, hyb, base, matrix_type, uplo, storage, &data_status, dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_hyb(
                handle, hyb, base, matrix_type, uplo, storage, &data_status, dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        double gbyte_count = check_matrix_hyb_gbyte_count<T>(dhyb->ell_nnz, dhyb->coo_nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info("M",
                            m,
                            "N",
                            n,
                            "ell nnz",
                            dhyb->ell_nnz,
                            "coo nnz",
                            dhyb->coo_nnz,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                                       \
    template void testing_check_matrix_hyb_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_check_matrix_hyb<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

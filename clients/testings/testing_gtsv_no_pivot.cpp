/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_gtsv_no_pivot_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle      = local_handle;
    rocsparse_int    m           = safe_size;
    rocsparse_int    n           = safe_size;
    rocsparse_int    ldb         = safe_size;
    const T*         dl          = (const T*)0x4;
    const T*         d           = (const T*)0x4;
    const T*         du          = (const T*)0x4;
    const T*         B1          = (const T*)0x4;
    T*               B2          = (T*)0x4;
    size_t*          buffer_size = (size_t*)0x4;
    void*            temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE handle, m, n, dl, d, du, B1, ldb, buffer_size
#define PARAMS_SOLVE handle, m, n, dl, d, du, B2, ldb, temp_buffer

    auto_testing_bad_arg(rocsparse_gtsv_no_pivot_buffer_size<T>, PARAMS_BUFFER_SIZE);
    auto_testing_bad_arg(rocsparse_gtsv_no_pivot<T>, PARAMS_SOLVE);

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_SOLVE
}

template <typename T>
void testing_gtsv_no_pivot(const Arguments& arg)
{
    rocsparse_int m   = arg.M;
    rocsparse_int n   = arg.N;
    rocsparse_int ldb = arg.denseld;

    // Create rocsparse handle
    rocsparse_local_handle handle;

#define PARAMS_BUFFER_SIZE handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size
#define PARAMS_SOLVE handle, m, n, ddl, dd, ddu, dB, ldb, dbuffer

    // Argument sanity check before allocating invalid memory
    if(m <= 1 || n <= 0 || ldb < m)
    {
        size_t buffer_size;
        T*     ddl     = nullptr;
        T*     dd      = nullptr;
        T*     ddu     = nullptr;
        T*     dB      = nullptr;
        void*  dbuffer = nullptr;

        EXPECT_ROCSPARSE_STATUS(rocsparse_gtsv_no_pivot_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                (m <= 1 || n < 0 || ldb < std::max(1, m))
                                    ? rocsparse_status_invalid_size
                                    : rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_gtsv_no_pivot<T>(PARAMS_SOLVE),
                                (m <= 1 || n < 0 || ldb < std::max(1, m))
                                    ? rocsparse_status_invalid_size
                                    : rocsparse_status_success);

        return;
    }

    rocsparse_seedrand();

    // Host tri-diagonal matrix
    host_vector<T> hdl(m);
    host_vector<T> hd(m);
    host_vector<T> hdu(m);

    // initialize tri-diagonal matrix
    for(rocsparse_int i = 0; i < m; ++i)
    {
        hdl[i] = random_generator<T>(1, 8);
        hd[i]  = random_generator<T>(17, 32);
        hdu[i] = random_generator<T>(1, 8);
    }

    hdl[0]     = 0.0f;
    hdu[m - 1] = 0.0f;

    // Host dense rhs
    host_vector<T> hB(ldb * n, static_cast<T>(7));

    for(rocsparse_int j = 0; j < n; ++j)
    {
        for(rocsparse_int i = 0; i < m; ++i)
        {
            hB[j * ldb + i] = random_generator<T>(-10, 10);
        }
    }

    host_vector<T> hB_original = hB;

    // Device tri-diagonal matrix
    device_vector<T> ddl(m);
    device_vector<T> dd(m);
    device_vector<T> ddu(m);

    // Device dense rhs
    device_vector<T> dB(ldb * n);

    // Copy to device
    ddl.transfer_from(hdl);
    dd.transfer_from(hd);
    ddu.transfer_from(hdu);
    dB.transfer_from(hB);

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_gtsv_no_pivot_buffer_size<T>(PARAMS_BUFFER_SIZE));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_gtsv_no_pivot<T>(PARAMS_SOLVE));

        hB.transfer_from(dB);

        // Check
        std::vector<T> hresult(ldb * n, static_cast<T>(7));
        for(rocsparse_int j = 0; j < n; j++)
        {
            hresult[ldb * j] = hd[0] * hB[ldb * j] + hdu[0] * hB[ldb * j + 1];
            hresult[ldb * j + m - 1]
                = hdl[m - 1] * hB[ldb * j + m - 2] + hd[m - 1] * hB[ldb * j + m - 1];
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(rocsparse_int i = 1; i < m - 1; i++)
            {
                hresult[ldb * j + i] = hdl[i] * hB[ldb * j + i - 1] + hd[i] * hB[ldb * j + i]
                                       + hdu[i] * hB[ldb * j + i + 1];
            }
        }

        near_check_segments<T>(ldb * n, hB_original.data(), hresult.data());
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gtsv_no_pivot<T>(PARAMS_SOLVE));
        }

        double gpu_solve_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_gtsv_no_pivot<T>(PARAMS_SOLVE));
        }

        gpu_solve_time_used = (get_time_us() - gpu_solve_time_used) / number_hot_calls;

        double gbyte_count = gtsv_gbyte_count<T>(m, n);

        double gpu_gbyte = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        display_timing_info("M",
                            m,
                            "N",
                            n,
                            s_timing_info_bandwidth,
                            gpu_gbyte,
                            s_timing_info_time,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));

#undef PARAMS_BUFFER_SIZE
#undef PARAMS_SOLVE
}

#define INSTANTIATE(TYPE)                                                    \
    template void testing_gtsv_no_pivot_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gtsv_no_pivot<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

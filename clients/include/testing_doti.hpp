/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef TESTING_DOTI_HPP
#define TESTING_DOTI_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <rocsparse.h>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_doti_bad_arg(void)
{
    rocsparse_int nnz       = 100;
    rocsparse_int safe_size = 100;

    rocsparse_index_base idx_base = rocsparse_index_base_zero;
    rocsparse_status     status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    auto dx_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dy_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T*             dx_val = (T*)dx_val_managed.get();
    rocsparse_int* dx_ind = (rocsparse_int*)dx_ind_managed.get();
    T*             dy     = (T*)dy_managed.get();

    if(!dx_ind || !dx_val || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    T result;

    // testing for (nullptr == dx_val)
    {
        T* dx_val_null = nullptr;

        status = rocsparse_doti(handle, nnz, dx_val_null, dx_ind, dy, &result, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: x_val is nullptr");
    }

    // testing for (nullptr == dx_ind)
    {
        rocsparse_int* dx_ind_null = nullptr;

        status = rocsparse_doti(handle, nnz, dx_val, dx_ind_null, dy, &result, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: x_ind is nullptr");
    }

    // testing for (nullptr == dy)
    {
        T* dy_null = nullptr;

        status = rocsparse_doti(handle, nnz, dx_val, dx_ind, dy_null, &result, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: y is nullptr");
    }

    // testing for (nullptr == result)
    {
        T* result_null = nullptr;

        status = rocsparse_doti(handle, nnz, dx_val, dx_ind, dy, result_null, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: result is nullptr");
    }

    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_doti(handle_null, nnz, dx_val, dx_ind, dy, &result, idx_base);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_doti(Arguments argus)
{
    rocsparse_int        N         = argus.N;
    rocsparse_int        nnz       = argus.nnz;
    rocsparse_int        safe_size = 100;
    rocsparse_index_base idx_base  = argus.idx_base;
    rocsparse_status     status;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle               handle = test_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
        auto dx_ind_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dx_val_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        rocsparse_int* dx_ind = (rocsparse_int*)dx_ind_managed.get();
        T*             dx_val = (T*)dx_val_managed.get();
        T*             dy     = (T*)dy_managed.get();

        if(!dx_ind || !dx_val || !dy)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dx_ind || !dx_val || !dy");
            return rocsparse_status_memory_error;
        }

        T result;

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        status = rocsparse_doti(handle, nnz, dx_val, dx_ind, dy, &result, idx_base);

        if(nnz < 0)
        {
            verify_rocsparse_status_invalid_size(status, "Error: nnz < 0");
        }
        else
        {
            verify_rocsparse_status_success(status, "nnz == 0");
        }

        return rocsparse_status_success;
    }

    // Host structures
    std::vector<rocsparse_int> hx_ind(nnz);
    std::vector<T>             hx_val(nnz);
    std::vector<T>             hy(N);

    T hresult_1;
    T hresult_2;
    T hresult_gold;

    // Initial Data on CPU
    srand(12345ULL);
    rocsparse_init_index(hx_ind.data(), nnz, 1, N);
    rocsparse_init<T>(hx_val, 1, nnz);
    rocsparse_init<T>(hy, 1, N);

    // allocate memory on device
    auto dx_ind_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dx_val_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_managed        = rocsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dresult_2_managed = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dx_ind    = (rocsparse_int*)dx_ind_managed.get();
    T*             dx_val    = (T*)dx_val_managed.get();
    T*             dy        = (T*)dy_managed.get();
    T*             dresult_2 = (T*)dresult_2_managed.get();

    if(!dx_ind || !dx_val || !dy || !dresult_2)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dx_ind || !dx_val || !dy || !dresult_2");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dx_ind, hx_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        // ROCSPARSE pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_doti(handle, nnz, dx_val, dx_ind, dy, &hresult_1, idx_base));

        // ROCSPARSE pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_doti(handle, nnz, dx_val, dx_ind, dy, dresult_2, idx_base));

        // copy output from device to CPU^
        CHECK_HIP_ERROR(hipMemcpy(&hresult_2, dresult_2, sizeof(T), hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        hresult_gold = static_cast<T>(0);
        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            hresult_gold += hy[hx_ind[i] - idx_base] * hx_val[i];
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        unit_check_general(1, 1, 1, &hresult_gold, &hresult_1);
        unit_check_general(1, 1, 1, &hresult_gold, &hresult_2);
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_doti(handle, nnz, dx_val, dx_ind, dy, &hresult_1, idx_base);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocsparse_doti(handle, nnz, dx_val, dx_ind, dy, &hresult_1, idx_base);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        double gpu_gflops = (2.0 * nnz) / 1e9 / gpu_time_used * 1e6 * 1;
        double bandwidth
            = (sizeof(rocsparse_int) * nnz + sizeof(T) * nnz * 2.0) / gpu_time_used / 1e3;

        printf("nnz\t\tGFlops\tGB/s\tusec\n");
        printf("%9d\t%0.2lf\t%0.2lf\t%0.2lf\n", nnz, gpu_gflops, bandwidth, gpu_time_used);
    }
    return rocsparse_status_success;
}

#endif // TESTING_DOTI_HPP

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
#ifndef TESTING_ROTI_HPP
#define TESTING_ROTI_HPP

#include "rocsparse_test_unique_ptr.hpp"
#include "rocsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <rocsparse.h>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_roti_bad_arg(void)
{
    rocsparse_int nnz       = 100;
    rocsparse_int safe_size = 100;
    T c                     = 3.7;
    T s                     = 1.2;

    rocsparse_index_base idx_base = rocsparse_index_base_zero;
    rocsparse_status status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle handle = unique_ptr_handle->handle;

    auto dx_val_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dy_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T* dx_val             = (T*)dx_val_managed.get();
    rocsparse_int* dx_ind = (rocsparse_int*)dx_ind_managed.get();
    T* dy                 = (T*)dy_managed.get();

    if(!dx_ind || !dx_val || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for(nullptr == dx_ind)
    {
        rocsparse_int* dx_ind_null = nullptr;

        status = rocsparse_roti(handle, nnz, dx_val, dx_ind_null, dy, &c, &s, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: x_ind is nullptr");
    }
    // testing for(nullptr == dx_val)
    {
        T* dx_val_null = nullptr;

        status = rocsparse_roti(handle, nnz, dx_val_null, dx_ind, dy, &c, &s, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: x_val is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = rocsparse_roti(handle, nnz, dx_val, dx_ind, dy_null, &c, &s, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: y is nullptr");
    }
    // testing for(nullptr == c)
    {
        T* dc_null = nullptr;

        status = rocsparse_roti(handle, nnz, dx_val, dx_ind, dy, dc_null, &s, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: c is nullptr");
    }
    // testing for(nullptr == s)
    {
        T* ds_null = nullptr;

        status = rocsparse_roti(handle, nnz, dx_val, dx_ind, dy, &c, ds_null, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: s is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_roti(handle_null, nnz, dx_val, dx_ind, dy, &c, &s, idx_base);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_roti(Arguments argus)
{
    rocsparse_int N               = argus.N;
    rocsparse_int nnz             = argus.nnz;
    T c                           = argus.alpha;
    T s                           = argus.beta;
    rocsparse_int safe_size       = 100;
    rocsparse_index_base idx_base = argus.idx_base;
    rocsparse_status status;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle handle = test_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
        auto dx_ind_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dx_val_managed =
            rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        rocsparse_int* dx_ind = (rocsparse_int*)dx_ind_managed.get();
        T* dx_val             = (T*)dx_val_managed.get();
        T* dy                 = (T*)dy_managed.get();

        if(!dx_ind || !dx_val || !dy)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dx_ind || !dx_val || !dy");
            return rocsparse_status_memory_error;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        status = rocsparse_roti(handle, nnz, dx_val, dx_ind, dy, &c, &s, idx_base);

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
    std::vector<T> hx_val_1(nnz);
    std::vector<T> hx_val_2(nnz);
    std::vector<T> hx_val_gold(nnz);
    std::vector<T> hy_1(N);
    std::vector<T> hy_2(N);
    std::vector<T> hy_gold(N);

    // Initial Data on CPU
    srand(12345ULL);
    rocsparse_init_index(hx_ind.data(), nnz, 1, N);
    rocsparse_init<T>(hx_val_1, 1, nnz);
    rocsparse_init<T>(hy_1, 1, N);

    hx_val_2    = hx_val_1;
    hx_val_gold = hx_val_1;
    hy_2        = hy_1;
    hy_gold     = hy_1;

    // allocate memory on device
    auto dx_ind_managed =
        rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dx_val_1_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_val_2_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_1_managed     = rocsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dy_2_managed     = rocsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dc_managed       = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto ds_managed       = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dx_ind = (rocsparse_int*)dx_ind_managed.get();
    T* dx_val_1           = (T*)dx_val_1_managed.get();
    T* dx_val_2           = (T*)dx_val_2_managed.get();
    T* dy_1               = (T*)dy_1_managed.get();
    T* dy_2               = (T*)dy_2_managed.get();
    T* dc                 = (T*)dc_managed.get();
    T* ds                 = (T*)ds_managed.get();

    if(!dx_ind || !dx_val_1 || !dx_val_2 || !dy_1 || !dy_2 || !dc || !ds)
    {
        verify_rocsparse_status_success(
            rocsparse_status_memory_error,
            "!dx_ind || !dx_val_1 || !dx_val_2 || !dy_1 || !dy_2 || !dc || !ds");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dx_ind, hx_ind.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val_1, hx_val_1.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(dx_val_2, hx_val_2.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, &c, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, &s, sizeof(T), hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_roti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base));

        // ROCSPARSE pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_roti(handle, nnz, dx_val_2, dx_ind, dy_2, dc, ds, idx_base));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_1.data(), dx_val_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_2.data(), dx_val_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * N, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * N, hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            rocsparse_int idx = hx_ind[i] - idx_base;

            T x = hx_val_gold[i];
            T y = hy_gold[idx];

            hx_val_gold[i] = c * x + s * y;
            hy_gold[idx]   = c * y - s * x;
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_1.data());
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_2.data());
        unit_check_general(1, N, 1, hy_gold.data(), hy_1.data());
        unit_check_general(1, N, 1, hy_gold.data(), hy_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(rocsparse_int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_roti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(rocsparse_int iter = 0; iter < number_hot_calls; iter++)
        {
            rocsparse_roti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base);
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;
        double gflops = nnz * 6.0 / gpu_time_used / 1e3;
        double bandwidth =
            (sizeof(rocsparse_int) * nnz + sizeof(T) * 2.0 * nnz) / gpu_time_used / 1e3;

        printf("nnz\t\tcosine\tsine\tGFlop/s\tGB/s\tusec\n");
        printf("%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
               nnz,
               c,
               s,
               gflops,
               bandwidth,
               gpu_time_used);
    }
    return rocsparse_status_success;
}

#endif // TESTING_ROTI_HPP

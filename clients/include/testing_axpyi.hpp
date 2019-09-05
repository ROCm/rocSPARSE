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
#ifndef TESTING_AXPYI_HPP
#define TESTING_AXPYI_HPP

#include "rocsparse.hpp"
#include "rocsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <rocsparse.h>

using namespace rocsparse;
using namespace rocsparse_test;

template <typename T>
void testing_axpyi_bad_arg(void)
{
    rocsparse_int nnz       = 100;
    rocsparse_int safe_size = 100;
    T             alpha     = 0.6;

    rocsparse_index_base idx_base = rocsparse_index_base_zero;
    rocsparse_status     status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    rocsparse_handle               handle = unique_ptr_handle->handle;

    auto dxVal_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dxInd_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
    auto dy_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T*             dxVal = (T*)dxVal_managed.get();
    rocsparse_int* dxInd = (rocsparse_int*)dxInd_managed.get();
    T*             dy    = (T*)dy_managed.get();

    if(!dxInd || !dxVal || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for(nullptr == dxInd)
    {
        rocsparse_int* dxInd_null = nullptr;

        status = rocsparse_axpyi(handle, nnz, &alpha, dxVal, dxInd_null, dy, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: xInd is nullptr");
    }
    // testing for(nullptr == dxVal)
    {
        T* dxVal_null = nullptr;

        status = rocsparse_axpyi(handle, nnz, &alpha, dxVal_null, dxInd, dy, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: xVal is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = rocsparse_axpyi(handle, nnz, &alpha, dxVal, dxInd, dy_null, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: y is nullptr");
    }
    // testing for(nullptr == d_alpha)
    {
        T* d_alpha_null = nullptr;

        status = rocsparse_axpyi(handle, nnz, d_alpha_null, dxVal, dxInd, dy, idx_base);
        verify_rocsparse_status_invalid_pointer(status, "Error: alpha is nullptr");
    }
    // testing for(nullptr == handle)
    {
        rocsparse_handle handle_null = nullptr;

        status = rocsparse_axpyi(handle_null, nnz, &alpha, dxVal, dxInd, dy, idx_base);
        verify_rocsparse_status_invalid_handle(status);
    }
}

template <typename T>
rocsparse_status testing_axpyi(Arguments argus)
{
    rocsparse_int        N         = argus.N;
    rocsparse_int        nnz       = argus.nnz;
    rocsparse_int        safe_size = 100;
    T                    h_alpha   = argus.alpha;
    rocsparse_index_base idx_base  = argus.idx_base;
    rocsparse_status     status;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    rocsparse_handle               handle = test_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
        auto dxInd_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * safe_size), device_free};
        auto dxVal_managed
            = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed = rocsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        rocsparse_int* dxInd = (rocsparse_int*)dxInd_managed.get();
        T*             dxVal = (T*)dxVal_managed.get();
        T*             dy    = (T*)dy_managed.get();

        if(!dxInd || !dxVal || !dy)
        {
            verify_rocsparse_status_success(rocsparse_status_memory_error,
                                            "!dxInd || !dxVal || !dy");
            return rocsparse_status_memory_error;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        status = rocsparse_axpyi(handle, nnz, &h_alpha, dxVal, dxInd, dy, idx_base);

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
    std::vector<rocsparse_int> hxInd(nnz);
    std::vector<T>             hxVal(nnz);
    std::vector<T>             hy_1(N);
    std::vector<T>             hy_2(N);
    std::vector<T>             hy_gold(N);

    // Initial Data on CPU
    srand(12345ULL);
    rocsparse_init_index(hxInd.data(), nnz, 1, N);
    rocsparse_init<T>(hxVal, 1, nnz);
    rocsparse_init<T>(hy_1, 1, N);

    // copy vector is easy in STL; hy_gold = hx: save a copy in hy_gold which will be output of CPU
    hy_2    = hy_1;
    hy_gold = hy_1;

    // allocate memory on device
    auto dxInd_managed
        = rocsparse_unique_ptr{device_malloc(sizeof(rocsparse_int) * nnz), device_free};
    auto dxVal_managed   = rocsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_1_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dy_2_managed    = rocsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto d_alpha_managed = rocsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    rocsparse_int* dxInd   = (rocsparse_int*)dxInd_managed.get();
    T*             dxVal   = (T*)dxVal_managed.get();
    T*             dy_1    = (T*)dy_1_managed.get();
    T*             dy_2    = (T*)dy_2_managed.get();
    T*             d_alpha = (T*)d_alpha_managed.get();

    if(!dxInd || !dxVal || !dy_1 || !dy_2 || !d_alpha)
    {
        verify_rocsparse_status_success(rocsparse_status_memory_error,
                                        "!dxInd || !dxVal || !dy_1 || !dy_2 || !d_alpha");
        return rocsparse_status_memory_error;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dxInd, hxInd.data(), sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dxVal, hxVal.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_axpyi(handle, nnz, &h_alpha, dxVal, dxInd, dy_1, idx_base));

        // ROCSPARSE pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_axpyi(handle, nnz, d_alpha, dxVal, dxInd, dy_2, idx_base));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * N, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * N, hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        for(rocsparse_int i = 0; i < nnz; ++i)
        {
            hy_gold[hxInd[i] - idx_base] += h_alpha * hxVal[i];
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general(1, N, 1, hy_gold.data(), hy_1.data());
            unit_check_general(1, N, 1, hy_gold.data(), hy_2.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            rocsparse_axpyi(handle, nnz, &h_alpha, dxVal, dxInd, dy_1, idx_base);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            rocsparse_axpyi(handle, nnz, &h_alpha, dxVal, dxInd, dy_1, idx_base);
        }

        gpu_time_used     = (get_time_us() - gpu_time_used) / number_hot_calls;
        double gpu_gflops = (2.0 * nnz) / 1e9 / gpu_time_used * 1e6 * 1;
        double bandwidth
            = (sizeof(rocsparse_int) * nnz + (sizeof(T) * (nnz + N))) / gpu_time_used / 1e3;

        printf("nnz\t\talpha\tGFlops\tGB/s\tusec\n");
        printf("%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
               nnz,
               h_alpha,
               gpu_gflops,
               bandwidth,
               gpu_time_used);
    }
    return rocsparse_status_success;
}

#endif // TESTING_AXPYI_HPP

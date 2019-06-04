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
#ifndef GUARD_ROCSPARSE_MANAGE_PTR
#define GUARD_ROCSPARSE_MANAGE_PTR

#include "arg_check.hpp"

#include <hip/hip_runtime_api.h>
#include <memory>
#include <rocsparse.h>

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

namespace rocsparse_test
{

    // device_malloc wraps hipMalloc and provides same API as malloc
    static void* device_malloc(size_t byte_size)
    {
        void* pointer;
        PRINT_IF_HIP_ERROR(hipMalloc(&pointer, byte_size));
        return pointer;
    }

    // device_free wraps hipFree and provides same API as free
    static void device_free(void* ptr)
    {
        PRINT_IF_HIP_ERROR(hipFree(ptr));
    }

    struct handle_struct
    {
        rocsparse_handle handle;
        handle_struct()
        {
            rocsparse_status status = rocsparse_create_handle(&handle);
            verify_rocsparse_status_success(status, "ERROR: handle_struct constructor");
        }

        ~handle_struct()
        {
            rocsparse_status status = rocsparse_destroy_handle(handle);
            verify_rocsparse_status_success(status, "ERROR: handle_struct destructor");
        }
    };

    struct descr_struct
    {
        rocsparse_mat_descr descr;
        descr_struct()
        {
            rocsparse_status status = rocsparse_create_mat_descr(&descr);
            verify_rocsparse_status_success(status, "ERROR: descr_struct constructor");
        }

        ~descr_struct()
        {
            rocsparse_status status = rocsparse_destroy_mat_descr(descr);
            verify_rocsparse_status_success(status, "ERROR: descr_struct destructor");
        }
    };

    struct hyb_struct
    {
        rocsparse_hyb_mat hyb;
        hyb_struct()
        {
            rocsparse_status status = rocsparse_create_hyb_mat(&hyb);
            verify_rocsparse_status_success(status, "ERROR: hyb_struct constructor");
        }

        ~hyb_struct()
        {
            rocsparse_status status = rocsparse_destroy_hyb_mat(hyb);
            verify_rocsparse_status_success(status, "ERROR: hyb_struct destructor");
        }
    };

    struct mat_info_struct
    {
        rocsparse_mat_info info;
        mat_info_struct()
        {
            rocsparse_status status = rocsparse_create_mat_info(&info);
            verify_rocsparse_status_success(status, "ERROR: mat_info_struct constructor");
        }

        ~mat_info_struct()
        {
            rocsparse_status status = rocsparse_destroy_mat_info(info);
            verify_rocsparse_status_success(status, "ERROR: mat_info_struct destructor");
        }
    };

} // namespace rocsparse_test

using rocsparse_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

#endif // GUARD_ROCSPARSE_MANAGE_PTR

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

#include "arg_check.hpp"

#include <iostream>
#include <hip/hip_runtime_api.h>
#include <rocsparse.h>

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif

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

void verify_rocsparse_status_invalid_pointer(rocsparse_status status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocsparse_status_invalid_pointer);
#else
    if(status != rocsparse_status_invalid_pointer)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != rocsparse_status_invalid_pointer, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_rocsparse_status_invalid_size(rocsparse_status status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocsparse_status_invalid_size);
#else
    if(status != rocsparse_status_invalid_size)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != rocsparse_status_invalid_size, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_rocsparse_status_invalid_value(rocsparse_status status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocsparse_status_invalid_value);
#else
    if(status != rocsparse_status_invalid_value)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != rocsparse_status_invalid_value, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_rocsparse_status_zero_pivot(rocsparse_status status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocsparse_status_zero_pivot);
#else
    if(status != rocsparse_status_zero_pivot)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != rocsparse_status_zero_pivot, ";
        std::cerr << message << std::endl;
    }
#endif
}

void verify_rocsparse_status_invalid_handle(rocsparse_status status)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocsparse_status_invalid_handle);
#else
    if(status != rocsparse_status_invalid_handle)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != rocsparse_status_invalid_handle" << std::endl;
    }
#endif
}

void verify_rocsparse_status_success(rocsparse_status status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, rocsparse_status_success);
#else
    if(status != rocsparse_status_success)
    {
        std::cerr << "rocSPARSE TEST ERROR: status != rocsparse_status_success, ";
        std::cerr << message << std::endl;
    }
#endif
}

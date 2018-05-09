/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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

void verify_rocsparse_status_invalid_pointer(rocsparse_status status,
                                             const char* message)
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

void verify_rocsparse_status_invalid_size(rocsparse_status status,
                                          const char* message)
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

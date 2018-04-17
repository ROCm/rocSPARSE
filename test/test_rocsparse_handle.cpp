/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <rocsparse.h>
#include <gtest/gtest.h>

#define ROCSPARSE_CHECK(x) ASSERT_EQ(x, ROCSPARSE_STATUS_SUCCESS)

TEST(Tests, handle)
{
    rocsparseHandle_t handle;
    ROCSPARSE_CHECK(rocsparseCreate(&handle));
    ROCSPARSE_CHECK(rocsparseDestroy(handle));
}

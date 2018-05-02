/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <rocsparse.h>
#include <gtest/gtest.h>

#define ROCSPARSE_CHECK(x) ASSERT_EQ(x, rocsparse_status_success)

TEST(Tests, handle)
{
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
}

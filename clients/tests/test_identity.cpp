/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_identity.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

int identity_N_range[] = {-3, 0, 33, 242, 623, 1000};

class parameterized_identity : public testing::TestWithParam<int>
{
    protected:
    parameterized_identity() {}
    virtual ~parameterized_identity() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_identity_arguments(int n)
{
    Arguments arg;
    arg.N      = n;
    arg.timing = 0;
    return arg;
}

TEST(identity_bad_arg, identity) { testing_identity_bad_arg(); }

TEST_P(parameterized_identity, identity)
{
    Arguments arg = setup_identity_arguments(GetParam());

    rocsparse_status status = testing_identity(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(identity, parameterized_identity, testing::ValuesIn(identity_N_range));

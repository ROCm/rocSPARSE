/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrsort.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int> csrsort_tuple;

int csrsort_M_range[] = {-1, 0, 10, 500, 872, 1000};
int csrsort_N_range[] = {-3, 0, 33, 242, 623, 1000};

class parameterized_csrsort : public testing::TestWithParam<csrsort_tuple>
{
    protected:
    parameterized_csrsort() {}
    virtual ~parameterized_csrsort() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrsort_arguments(csrsort_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(csrsort_bad_arg, csrsort) { testing_csrsort_bad_arg(); }

TEST_P(parameterized_csrsort, csrsort)
{
    Arguments arg = setup_csrsort_arguments(GetParam());

    rocsparse_status status = testing_csrsort(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csrsort,
                        parameterized_csrsort,
                        testing::Combine(testing::ValuesIn(csrsort_M_range),
                                         testing::ValuesIn(csrsort_N_range)));

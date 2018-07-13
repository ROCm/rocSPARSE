/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_coosort.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, rocsparse_operation, int, rocsparse_index_base> coosort_tuple;

int coosort_M_range[]               = {-1, 0, 10, 500, 3872, 10000};
int coosort_N_range[]               = {-3, 0, 33, 242, 1623, 10000};
rocsparse_operation coosort_trans[] = {rocsparse_operation_none, rocsparse_operation_transpose};
int coosort_perm[]                  = {0, 1};
rocsparse_index_base coosort_base[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_coosort : public testing::TestWithParam<coosort_tuple>
{
    protected:
    parameterized_coosort() {}
    virtual ~parameterized_coosort() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_coosort_arguments(coosort_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.trans    = std::get<2>(tup);
    arg.temp     = std::get<3>(tup);
    arg.idx_base = std::get<4>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(coosort_bad_arg, coosort) { testing_coosort_bad_arg(); }

TEST_P(parameterized_coosort, coosort)
{
    Arguments arg = setup_coosort_arguments(GetParam());

    rocsparse_status status = testing_coosort(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(coosort,
                        parameterized_coosort,
                        testing::Combine(testing::ValuesIn(coosort_M_range),
                                         testing::ValuesIn(coosort_N_range),
                                         testing::ValuesIn(coosort_trans),
                                         testing::ValuesIn(coosort_perm),
                                         testing::ValuesIn(coosort_base)));

/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_coo2csr.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, rocsparse_index_base> coo2csr_tuple;

int coo_M_range[] = {-1, 0, 10, 500, 872, 1000};
int coo_N_range[] = {-3, 0, 33, 242, 623, 1000};
rocsparse_index_base coo_idx_base_range[] = {rocsparse_index_base_zero,
                                             rocsparse_index_base_one};

class parameterized_coo2csr : public testing::TestWithParam<coo2csr_tuple>
{
    protected:
    parameterized_coo2csr() {}
    virtual ~parameterized_coo2csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_coo2csr_arguments(coo2csr_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(coo2csr_bad_arg, coo2csr)
{
    testing_coo2csr_bad_arg();
}

TEST_P(parameterized_coo2csr, coo2csr)
{
    Arguments arg = setup_coo2csr_arguments(GetParam());
    rocsparse_status status = testing_coo2csr(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(coo2csr, parameterized_coo2csr,
                        testing::Combine(testing::ValuesIn(coo_M_range),
                                         testing::ValuesIn(coo_N_range),
                                         testing::ValuesIn(coo_idx_base_range)));

/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csr2coo.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, rocsparse_index_base> csr2coo_tuple;

int coo_M_range[] = {-1, 0, 10, 500, 872, 1000};
int coo_N_range[] = {-3, 0, 33, 242, 623, 1000};

rocsparse_index_base coo_idx_base_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_csr2coo : public testing::TestWithParam<csr2coo_tuple>
{
    protected:
    parameterized_csr2coo() {}
    virtual ~parameterized_csr2coo() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2coo_arguments(csr2coo_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(csr2coo_bad_arg, csr2coo) { testing_csr2coo_bad_arg(); }

TEST_P(parameterized_csr2coo, csr2coo)
{
    Arguments arg = setup_csr2coo_arguments(GetParam());

    rocsparse_status status = testing_csr2coo(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csr2coo,
                        parameterized_csr2coo,
                        testing::Combine(testing::ValuesIn(coo_M_range),
                                         testing::ValuesIn(coo_N_range),
                                         testing::ValuesIn(coo_idx_base_range)));

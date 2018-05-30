/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csr2ell.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, rocsparse_index_base, rocsparse_index_base> csr2ell_tuple;

int csr2ell_M_range[] = {-1, 0, 10, 500, 872, 1000};
int csr2ell_N_range[] = {-3, 0, 33, 242, 623, 1000};

rocsparse_index_base csr2ell_csr_base_range[] = {rocsparse_index_base_zero,
                                                 rocsparse_index_base_one};
rocsparse_index_base csr2ell_ell_base_range[] = {rocsparse_index_base_zero,
                                                 rocsparse_index_base_one};

class parameterized_csr2ell : public testing::TestWithParam<csr2ell_tuple>
{
    protected:
    parameterized_csr2ell() {}
    virtual ~parameterized_csr2ell() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2ell_arguments(csr2ell_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.idx_base2 = std::get<3>(tup);
    arg.timing    = 0;
    return arg;
}

TEST(csr2ell_bad_arg, csr2ell) { testing_csr2ell_bad_arg<float>(); }

TEST_P(parameterized_csr2ell, csr2ell_float)
{
    Arguments arg = setup_csr2ell_arguments(GetParam());

    rocsparse_status status = testing_csr2ell<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csr2ell, csr2ell_double)
{
    Arguments arg = setup_csr2ell_arguments(GetParam());

    rocsparse_status status = testing_csr2ell<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csr2ell,
                        parameterized_csr2ell,
                        testing::Combine(testing::ValuesIn(csr2ell_M_range),
                                         testing::ValuesIn(csr2ell_N_range),
                                         testing::ValuesIn(csr2ell_csr_base_range),
                                         testing::ValuesIn(csr2ell_ell_base_range)));

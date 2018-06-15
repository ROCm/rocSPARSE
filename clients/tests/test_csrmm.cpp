/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrmm.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef rocsparse_index_base base;
typedef std::tuple<int, int, double, double, base> csrmm_tuple;

int csrmm_M_range[] = {-1, 0, 10, 500, 7111, 10000};
int csrmm_N_range[] = {-3, 0, 33, 842, 4441, 10000};

std::vector<double> csrmm_alpha_range = {2.0, 3.0};
std::vector<double> csrmm_beta_range  = {0.0, 1.0};

base csrmm_idxbase_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_csrmm : public testing::TestWithParam<csrmm_tuple>
{
    protected:
    parameterized_csrmm() {}
    virtual ~parameterized_csrmm() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrmm_arguments(csrmm_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.alpha    = std::get<2>(tup);
    arg.beta     = std::get<3>(tup);
    arg.idx_base = std::get<4>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(csrmm_bad_arg, csrmm_float) { testing_csrmm_bad_arg<float>(); }

TEST_P(parameterized_csrmm, csrmm_float)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    rocsparse_status status = testing_csrmm<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrmm, csrmm_double)
{
    Arguments arg = setup_csrmm_arguments(GetParam());

    rocsparse_status status = testing_csrmm<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csrmm,
                        parameterized_csrmm,
                        testing::Combine(testing::ValuesIn(csrmm_M_range),
                                         testing::ValuesIn(csrmm_N_range),
                                         testing::ValuesIn(csrmm_alpha_range),
                                         testing::ValuesIn(csrmm_beta_range),
                                         testing::ValuesIn(csrmm_idxbase_range)));

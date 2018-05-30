/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_ellmv.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef rocsparse_index_base base;
typedef std::tuple<int, int, double, double, base> ellmv_tuple;

int ell_M_range[] = {-1, 0, 10, 500, 7111, 10000};
int ell_N_range[] = {-3, 0, 33, 842, 4441, 10000};

std::vector<double> ell_alpha_range = {2.0, 3.0};
std::vector<double> ell_beta_range  = {0.0, 0.6};

base ell_idxbase_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_ellmv : public testing::TestWithParam<ellmv_tuple>
{
    protected:
    parameterized_ellmv() {}
    virtual ~parameterized_ellmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_ellmv_arguments(ellmv_tuple tup)
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

TEST(ellmv_bad_arg, ellmv_float) { testing_ellmv_bad_arg<float>(); }

TEST_P(parameterized_ellmv, ellmv_float)
{
    Arguments arg = setup_ellmv_arguments(GetParam());

    rocsparse_status status = testing_ellmv<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_ellmv, ellmv_double)
{
    Arguments arg = setup_ellmv_arguments(GetParam());

    rocsparse_status status = testing_ellmv<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(ellmv,
                        parameterized_ellmv,
                        testing::Combine(testing::ValuesIn(ell_M_range),
                                         testing::ValuesIn(ell_N_range),
                                         testing::ValuesIn(ell_alpha_range),
                                         testing::ValuesIn(ell_beta_range),
                                         testing::ValuesIn(ell_idxbase_range)));

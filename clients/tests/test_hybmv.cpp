/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_hybmv.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, double, double, rocsparse_index_base, rocsparse_hyb_partition, int> hybmv_tuple;

int hyb_M_range[] = {-1, 0, 10, 500, 7111, 10000};
int hyb_N_range[] = {-3, 0, 33, 842, 4441, 10000};

std::vector<double> hyb_alpha_range = {2.0, 3.0};
std::vector<double> hyb_beta_range  = {0.0, 1.0};

rocsparse_index_base hyb_idxbase_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

rocsparse_hyb_partition hyb_partition[] = {rocsparse_hyb_partition_auto,
                                           rocsparse_hyb_partition_max,
                                           rocsparse_hyb_partition_user};

int hyb_ELL_range[] = {-33, -1, 0, INT32_MAX};

class parameterized_hybmv : public testing::TestWithParam<hybmv_tuple>
{
    protected:
    parameterized_hybmv() {}
    virtual ~parameterized_hybmv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_hybmv_arguments(hybmv_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.alpha     = std::get<2>(tup);
    arg.beta      = std::get<3>(tup);
    arg.idx_base  = std::get<4>(tup);
    arg.part      = std::get<5>(tup);
    arg.ell_width = std::get<6>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(hybmv_bad_arg, hybmv_float) { testing_hybmv_bad_arg<float>(); }

TEST_P(parameterized_hybmv, hybmv_float)
{
    Arguments arg = setup_hybmv_arguments(GetParam());

    rocsparse_status status = testing_hybmv<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_hybmv, hybmv_double)
{
    Arguments arg = setup_hybmv_arguments(GetParam());

    rocsparse_status status = testing_hybmv<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(hybmv,
                        parameterized_hybmv,
                        testing::Combine(testing::ValuesIn(hyb_M_range),
                                         testing::ValuesIn(hyb_N_range),
                                         testing::ValuesIn(hyb_alpha_range),
                                         testing::ValuesIn(hyb_beta_range),
                                         testing::ValuesIn(hyb_idxbase_range),
                                         testing::ValuesIn(hyb_partition),
                                         testing::ValuesIn(hyb_ELL_range)));

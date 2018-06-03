/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_gthrz.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef rocsparse_index_base base;
typedef std::tuple<int, int, base> gthrz_tuple;

int gthrz_N_range[]   = {12000, 15332, 22031};
int gthrz_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};

base gthrz_idx_base_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_gthrz : public testing::TestWithParam<gthrz_tuple>
{
    protected:
    parameterized_gthrz() {}
    virtual ~parameterized_gthrz() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gthrz_arguments(gthrz_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(gthrz_bad_arg, gthrz_float) { testing_gthrz_bad_arg<float>(); }

TEST_P(parameterized_gthrz, gthrz_float)
{
    Arguments arg = setup_gthrz_arguments(GetParam());

    rocsparse_status status = testing_gthrz<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_gthrz, gthrz_double)
{
    Arguments arg = setup_gthrz_arguments(GetParam());

    rocsparse_status status = testing_gthrz<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(gthrz,
                        parameterized_gthrz,
                        testing::Combine(testing::ValuesIn(gthrz_N_range),
                                         testing::ValuesIn(gthrz_nnz_range),
                                         testing::ValuesIn(gthrz_idx_base_range)));

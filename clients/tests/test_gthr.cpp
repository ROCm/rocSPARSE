/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_gthr.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef rocsparse_index_base base;
typedef std::tuple<int, int, base> gthr_tuple;

int gthr_N_range[]   = {12000, 15332, 22031};
int gthr_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};

base gthr_idx_base_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_gthr : public testing::TestWithParam<gthr_tuple>
{
    protected:
    parameterized_gthr() {}
    virtual ~parameterized_gthr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_gthr_arguments(gthr_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(gthr_bad_arg, gthr_float) { testing_gthr_bad_arg<float>(); }

TEST_P(parameterized_gthr, gthr_float)
{
    Arguments arg = setup_gthr_arguments(GetParam());

    rocsparse_status status = testing_gthr<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_gthr, gthr_double)
{
    Arguments arg = setup_gthr_arguments(GetParam());

    rocsparse_status status = testing_gthr<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(gthr,
                        parameterized_gthr,
                        testing::Combine(testing::ValuesIn(gthr_N_range),
                                         testing::ValuesIn(gthr_nnz_range),
                                         testing::ValuesIn(gthr_idx_base_range)));

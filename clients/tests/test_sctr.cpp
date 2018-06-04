/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_sctr.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef rocsparse_index_base base;
typedef std::tuple<int, int, base> sctr_tuple;

int sctr_N_range[]   = {12000, 15332, 22031};
int sctr_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};

base sctr_idx_base_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_sctr : public testing::TestWithParam<sctr_tuple>
{
    protected:
    parameterized_sctr() {}
    virtual ~parameterized_sctr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_sctr_arguments(sctr_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(sctr_bad_arg, sctr_float) { testing_sctr_bad_arg<float>(); }

TEST_P(parameterized_sctr, sctr_float)
{
    Arguments arg = setup_sctr_arguments(GetParam());

    rocsparse_status status = testing_sctr<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_sctr, sctr_double)
{
    Arguments arg = setup_sctr_arguments(GetParam());

    rocsparse_status status = testing_sctr<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(sctr,
                        parameterized_sctr,
                        testing::Combine(testing::ValuesIn(sctr_N_range),
                                         testing::ValuesIn(sctr_nnz_range),
                                         testing::ValuesIn(sctr_idx_base_range)));

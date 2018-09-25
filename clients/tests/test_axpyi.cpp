/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "testing_axpyi.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef rocsparse_index_base base;
typedef std::tuple<int, int, double, base> axpyi_tuple;

int axpyi_N_range[]   = {12000, 15332, 22031};
int axpyi_nnz_range[] = {-1, 0, 5, 10, 500, 1000, 7111, 10000};

std::vector<double> axpyi_alpha_range = {1.0, 0.0};

base axpyi_idx_base_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

class parameterized_axpyi : public testing::TestWithParam<axpyi_tuple>
{
    protected:
    parameterized_axpyi() {}
    virtual ~parameterized_axpyi() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_axpyi_arguments(axpyi_tuple tup)
{
    Arguments arg;
    arg.N        = std::get<0>(tup);
    arg.nnz      = std::get<1>(tup);
    arg.alpha    = std::get<2>(tup);
    arg.idx_base = std::get<3>(tup);
    arg.timing   = 0;
    return arg;
}

TEST(axpyi_bad_arg, axpyi_float) { testing_axpyi_bad_arg<float>(); }

TEST_P(parameterized_axpyi, axpyi_float)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    rocsparse_status status = testing_axpyi<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_axpyi, axpyi_double)
{
    Arguments arg = setup_axpyi_arguments(GetParam());

    rocsparse_status status = testing_axpyi<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(axpyi,
                        parameterized_axpyi,
                        testing::Combine(testing::ValuesIn(axpyi_N_range),
                                         testing::ValuesIn(axpyi_nnz_range),
                                         testing::ValuesIn(axpyi_alpha_range),
                                         testing::ValuesIn(axpyi_idx_base_range)));

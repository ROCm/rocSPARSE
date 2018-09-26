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

#include "testing_csrsv.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>
#include <string>

typedef rocsparse_index_base base;
typedef rocsparse_operation op;
typedef rocsparse_diag_type diag;
typedef rocsparse_fill_mode fill;

typedef std::tuple<int, double, base, op, diag, fill> csrsv_tuple;
typedef std::tuple<double, base, op, diag, fill, std::string> csrsv_bin_tuple;

int csrsv_M_range[] = {-1, 0, 50, 647};

double csrsv_alpha_range[] = {1.0, 2.3, -3.7};

base csrsv_idxbase_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
op csrsv_op_range[]        = {rocsparse_operation_none};
diag csrsv_diag_range[]    = {rocsparse_diag_type_non_unit};
fill csrsv_fill_range[]    = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};

std::string csrsv_bin[] = {"rma10.bin",
                           "mac_econ_fwd500.bin",
                           "mc2depi.bin",
                           "scircuit.bin",
                           "ASIC_320k.bin",
                           "bmwcra_1.bin",
                           "nos1.bin",
                           "nos2.bin",
                           "nos3.bin",
                           "nos4.bin",
                           "nos5.bin",
                           "nos6.bin"};

class parameterized_csrsv : public testing::TestWithParam<csrsv_tuple>
{
    protected:
    parameterized_csrsv() {}
    virtual ~parameterized_csrsv() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrsv_bin : public testing::TestWithParam<csrsv_bin_tuple>
{
    protected:
    parameterized_csrsv_bin() {}
    virtual ~parameterized_csrsv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrsv_arguments(csrsv_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.alpha     = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.transA    = std::get<3>(tup);
    arg.diag_type = std::get<4>(tup);
    arg.fill_mode = std::get<5>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_csrsv_arguments(csrsv_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.alpha     = std::get<0>(tup);
    arg.idx_base  = std::get<1>(tup);
    arg.transA    = std::get<2>(tup);
    arg.diag_type = std::get<3>(tup);
    arg.fill_mode = std::get<4>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Get current executables absolute path
    char path_exe[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path_exe, sizeof(path_exe) - 1);
    if(len < 14)
    {
        path_exe[0] = '\0';
    }
    else
    {
        path_exe[len - 14] = '\0';
    }

    // Matrices are stored at the same path in matrices directory
    arg.filename = std::string(path_exe) + "matrices/" + bin_file;

    return arg;
}

TEST(csrsv_bad_arg, csrsv_float) { testing_csrsv_bad_arg<float>(); }

TEST_P(parameterized_csrsv, csrsv_float)
{
    Arguments arg = setup_csrsv_arguments(GetParam());

    rocsparse_status status = testing_csrsv<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrsv, csrsv_double)
{
    Arguments arg = setup_csrsv_arguments(GetParam());

    rocsparse_status status = testing_csrsv<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrsv_bin, csrsv_bin_float)
{
    Arguments arg = setup_csrsv_arguments(GetParam());

    rocsparse_status status = testing_csrsv<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrsv_bin, csrsv_bin_double)
{
    Arguments arg = setup_csrsv_arguments(GetParam());

    rocsparse_status status = testing_csrsv<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csrsv,
                        parameterized_csrsv,
                        testing::Combine(testing::ValuesIn(csrsv_M_range),
                                         testing::ValuesIn(csrsv_alpha_range),
                                         testing::ValuesIn(csrsv_idxbase_range),
                                         testing::ValuesIn(csrsv_op_range),
                                         testing::ValuesIn(csrsv_diag_range),
                                         testing::ValuesIn(csrsv_fill_range)));

INSTANTIATE_TEST_CASE_P(csrsv_bin,
                        parameterized_csrsv_bin,
                        testing::Combine(testing::ValuesIn(csrsv_alpha_range),
                                         testing::ValuesIn(csrsv_idxbase_range),
                                         testing::ValuesIn(csrsv_op_range),
                                         testing::ValuesIn(csrsv_diag_range),
                                         testing::ValuesIn(csrsv_fill_range),
                                         testing::ValuesIn(csrsv_bin)));

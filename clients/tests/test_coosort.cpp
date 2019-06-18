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

#include "testing_coosort.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocsparse.h>
#include <string>
#include <vector>

typedef std::tuple<int, int, rocsparse_operation, int, rocsparse_index_base>    coosort_tuple;
typedef std::tuple<rocsparse_operation, int, rocsparse_index_base, std::string> coosort_bin_tuple;

int                  coosort_M_range[] = {-1, 0, 10, 500, 3872, 10000};
int                  coosort_N_range[] = {-3, 0, 33, 242, 1623, 10000};
rocsparse_operation  coosort_trans[]   = {rocsparse_operation_none, rocsparse_operation_transpose};
int                  coosort_perm[]    = {0, 1};
rocsparse_index_base coosort_base[]    = {rocsparse_index_base_zero, rocsparse_index_base_one};

std::string coosort_bin[] = {"rma10.bin",
                             "mac_econ_fwd500.bin",
                             "bibd_22_8.bin",
                             "mc2depi.bin",
                             "scircuit.bin",
                             "ASIC_320k.bin",
                             "bmwcra_1.bin",
                             "nos1.bin",
                             "nos2.bin",
                             "nos3.bin",
                             "nos4.bin",
                             "nos5.bin",
                             "nos6.bin",
                             "nos7.bin"};

class parameterized_coosort : public testing::TestWithParam<coosort_tuple>
{
protected:
    parameterized_coosort() {}
    virtual ~parameterized_coosort() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_coosort_bin : public testing::TestWithParam<coosort_bin_tuple>
{
protected:
    parameterized_coosort_bin() {}
    virtual ~parameterized_coosort_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_coosort_arguments(coosort_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.transA   = std::get<2>(tup);
    arg.temp     = std::get<3>(tup);
    arg.idx_base = std::get<4>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_coosort_arguments(coosort_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.N        = -99;
    arg.transA   = std::get<0>(tup);
    arg.temp     = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

    // Get current executables absolute path
    char    path_exe[PATH_MAX];
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

TEST(coosort_bad_arg, coosort)
{
    testing_coosort_bad_arg();
}

TEST_P(parameterized_coosort, coosort)
{
    Arguments arg = setup_coosort_arguments(GetParam());

    rocsparse_status status = testing_coosort(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_coosort_bin, coosort_bin)
{
    Arguments arg = setup_coosort_arguments(GetParam());

    rocsparse_status status = testing_coosort(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(coosort,
                        parameterized_coosort,
                        testing::Combine(testing::ValuesIn(coosort_M_range),
                                         testing::ValuesIn(coosort_N_range),
                                         testing::ValuesIn(coosort_trans),
                                         testing::ValuesIn(coosort_perm),
                                         testing::ValuesIn(coosort_base)));

INSTANTIATE_TEST_CASE_P(coosort_bin,
                        parameterized_coosort_bin,
                        testing::Combine(testing::ValuesIn(coosort_trans),
                                         testing::ValuesIn(coosort_perm),
                                         testing::ValuesIn(coosort_base),
                                         testing::ValuesIn(coosort_bin)));

/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include "testing_csrgemm_b.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocsparse.h>
#include <string>
#include <unistd.h>
#include <vector>

typedef rocsparse_index_base base;

typedef std::tuple<rocsparse_int, rocsparse_int, double, base, base> csrgemm_b_tuple;
typedef std::tuple<double, base, base, std::string>                  csrgemm_b_bin_tuple;

double csrgemm_b_beta_range[] = {0.0, 1.3};

rocsparse_int csrgemm_b_M_range[] = {-1, 0, 50, 647, 1799};
rocsparse_int csrgemm_b_N_range[] = {-1, 0, 13, 523, 3712};

base csrgemm_b_idxbaseC_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
base csrgemm_b_idxbaseD_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

std::string csrgemm_b_bin[] = {"rma10.bin",
                               "mac_econ_fwd500.bin",
                               "mc2depi.bin",
                               "scircuit.bin",
                               "bmwcra_1.bin",
                               "nos1.bin",
                               "nos2.bin",
                               "nos3.bin",
                               "nos4.bin",
                               "nos5.bin",
                               "nos6.bin",
                               "nos7.bin",
                               "amazon0312.bin",
                               "Chebyshev4.bin",
                               "sme3Dc.bin",
                               //                             "webbase-1M.bin",
                               "shipsec1.bin"};

class parameterized_csrgemm_b : public testing::TestWithParam<csrgemm_b_tuple>
{
protected:
    parameterized_csrgemm_b() { }
    virtual ~parameterized_csrgemm_b() { }
    virtual void SetUp() { }
    virtual void TearDown() { }
};

class parameterized_csrgemm_b_bin : public testing::TestWithParam<csrgemm_b_bin_tuple>
{
protected:
    parameterized_csrgemm_b_bin() { }
    virtual ~parameterized_csrgemm_b_bin() { }
    virtual void SetUp() { }
    virtual void TearDown() { }
};

Arguments setup_csrgemm_b_arguments(csrgemm_b_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.beta      = std::get<2>(tup);
    arg.idx_base3 = std::get<3>(tup);
    arg.idx_base4 = std::get<4>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_csrgemm_b_arguments(csrgemm_b_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.beta      = std::get<0>(tup);
    arg.idx_base3 = std::get<1>(tup);
    arg.idx_base4 = std::get<2>(tup);
    arg.timing    = 0;

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
    arg.filename = std::string(path_exe) + "../matrices/" + bin_file;

    return arg;
}

TEST(csrgemm_b_bad_arg, csrgemm_b_float)
{
    testing_csrgemm_b_bad_arg<float>();
}

TEST_P(parameterized_csrgemm_b, csrgemm_b_float)
{
    Arguments arg = setup_csrgemm_b_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_b<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrgemm_b, csrgemm_b_double)
{
    Arguments arg = setup_csrgemm_b_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_b<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrgemm_b_bin, csrgemm_b_bin_float)
{
    Arguments arg = setup_csrgemm_b_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_b<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrgemm_b_bin, csrgemm_b_bin_double)
{
    Arguments arg = setup_csrgemm_b_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_b<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csrgemm_b,
                        parameterized_csrgemm_b,
                        testing::Combine(testing::ValuesIn(csrgemm_b_M_range),
                                         testing::ValuesIn(csrgemm_b_N_range),
                                         testing::ValuesIn(csrgemm_b_beta_range),
                                         testing::ValuesIn(csrgemm_b_idxbaseC_range),
                                         testing::ValuesIn(csrgemm_b_idxbaseD_range)));

INSTANTIATE_TEST_CASE_P(csrgemm_b_bin,
                        parameterized_csrgemm_b_bin,
                        testing::Combine(testing::ValuesIn(csrgemm_b_beta_range),
                                         testing::ValuesIn(csrgemm_b_idxbaseC_range),
                                         testing::ValuesIn(csrgemm_b_idxbaseD_range),
                                         testing::ValuesIn(csrgemm_b_bin)));

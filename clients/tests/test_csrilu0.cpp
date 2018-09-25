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

#include "testing_csrilu0.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>
#include <string>

typedef rocsparse_index_base base;
typedef std::tuple<int, base> csrilu0_tuple;
typedef std::tuple<base, std::string> csrilu0_bin_tuple;

int csrilu0_M_range[] = {-1, 0, 50, 647};

base csrilu0_idxbase_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

std::string csrilu0_bin[] = {"rma10.bin",
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
                             "nos6.bin",
                             "nos7.bin"};

class parameterized_csrilu0 : public testing::TestWithParam<csrilu0_tuple>
{
    protected:
    parameterized_csrilu0() {}
    virtual ~parameterized_csrilu0() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrilu0_bin : public testing::TestWithParam<csrilu0_bin_tuple>
{
    protected:
    parameterized_csrilu0_bin() {}
    virtual ~parameterized_csrilu0_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrilu0_arguments(csrilu0_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.idx_base = std::get<1>(tup);
    arg.timing   = 0;
    return arg;
}

Arguments setup_csrilu0_arguments(csrilu0_bin_tuple tup)
{
    Arguments arg;
    arg.M        = -99;
    arg.idx_base = std::get<0>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<1>(tup);

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

TEST(csrilu0_bad_arg, csrilu0_float) { testing_csrilu0_bad_arg<float>(); }

TEST_P(parameterized_csrilu0, csrilu0_float)
{
    Arguments arg = setup_csrilu0_arguments(GetParam());

    rocsparse_status status = testing_csrilu0<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrilu0, csrilu0_double)
{
    Arguments arg = setup_csrilu0_arguments(GetParam());

    rocsparse_status status = testing_csrilu0<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrilu0_bin, csrilu0_bin_float)
{
    Arguments arg = setup_csrilu0_arguments(GetParam());

    rocsparse_status status = testing_csrilu0<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrilu0_bin, csrilu0_bin_double)
{
    Arguments arg = setup_csrilu0_arguments(GetParam());

    rocsparse_status status = testing_csrilu0<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csrilu0,
                        parameterized_csrilu0,
                        testing::Combine(testing::ValuesIn(csrilu0_M_range),
                                         testing::ValuesIn(csrilu0_idxbase_range)));

INSTANTIATE_TEST_CASE_P(csrilu0_bin,
                        parameterized_csrilu0_bin,
                        testing::Combine(testing::ValuesIn(csrilu0_idxbase_range),
                                         testing::ValuesIn(csrilu0_bin)));

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

#include "testing_csrilusv.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocsparse.h>
#include <string>
#include <unistd.h>
#include <vector>

typedef rocsparse_index_base      base;
typedef rocsparse_analysis_policy analysis;

typedef std::tuple<base, analysis, std::string> csrilusv_bin_tuple;

base     csrilusv_idxbase_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
analysis csrilusv_analysis_range[]
    = {rocsparse_analysis_policy_reuse, rocsparse_analysis_policy_force};

std::string csrilusv_bin[] = {"mac_econ_fwd500.bin",
                              "mc2depi.bin",
                              "scircuit.bin",
                              //                              "bmwcra_1.bin",
                              "nos1.bin",
                              "nos3.bin",
                              "nos4.bin",
                              "nos5.bin",
                              "nos6.bin",
                              "nos7.bin",
                              "amazon0312.bin",
                              "sme3Dc.bin"};

class parameterized_csrilusv_bin : public testing::TestWithParam<csrilusv_bin_tuple>
{
protected:
    parameterized_csrilusv_bin() {}
    virtual ~parameterized_csrilusv_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrilusv_arguments(csrilusv_bin_tuple tup)
{
    Arguments arg;
    arg.idx_base = std::get<0>(tup);
    arg.analysis = std::get<1>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<2>(tup);

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

TEST_P(parameterized_csrilusv_bin, csrilusv_bin_float)
{
    Arguments arg = setup_csrilusv_arguments(GetParam());

    rocsparse_status status = testing_csrilusv<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrilusv_bin, csrilusv_bin_double)
{
    Arguments arg = setup_csrilusv_arguments(GetParam());

    rocsparse_status status = testing_csrilusv<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(DISABLED_csrilusv_bin,
                        parameterized_csrilusv_bin,
                        testing::Combine(testing::ValuesIn(csrilusv_idxbase_range),
                                         testing::ValuesIn(csrilusv_analysis_range),
                                         testing::ValuesIn(csrilusv_bin)));

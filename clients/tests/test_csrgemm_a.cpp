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

#include "testing_csrgemm_a.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocsparse.h>
#include <string>
#include <unistd.h>
#include <vector>

typedef rocsparse_index_base base;
typedef rocsparse_operation  trans;

typedef std::
    tuple<rocsparse_int, rocsparse_int, rocsparse_int, double, base, base, base, trans, trans>
                                                                        csrgemm_a_tuple;
typedef std::tuple<double, base, base, base, trans, trans, std::string> csrgemm_a_bin_tuple;

double csrgemm_a_alpha_range[] = {0.0, 2.7};

rocsparse_int csrgemm_a_M_range[] = {-1, 0, 50, 647, 1799};
rocsparse_int csrgemm_a_N_range[] = {-1, 0, 13, 523, 3712};
rocsparse_int csrgemm_a_K_range[] = {-1, 0, 50, 254, 1942};

base csrgemm_a_idxbaseA_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
base csrgemm_a_idxbaseB_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
base csrgemm_a_idxbaseC_range[] = {rocsparse_index_base_zero, rocsparse_index_base_one};

trans csrgemm_a_transA_range[] = {rocsparse_operation_none};
trans csrgemm_a_transB_range[] = {rocsparse_operation_none};

std::string csrgemm_a_bin[] = {"rma10.bin",
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

class parameterized_csrgemm_a : public testing::TestWithParam<csrgemm_a_tuple>
{
protected:
    parameterized_csrgemm_a() {}
    virtual ~parameterized_csrgemm_a() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csrgemm_a_bin : public testing::TestWithParam<csrgemm_a_bin_tuple>
{
protected:
    parameterized_csrgemm_a_bin() {}
    virtual ~parameterized_csrgemm_a_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csrgemm_a_arguments(csrgemm_a_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.K         = std::get<2>(tup);
    arg.alpha     = std::get<3>(tup);
    arg.idx_base  = std::get<4>(tup);
    arg.idx_base2 = std::get<5>(tup);
    arg.idx_base3 = std::get<6>(tup);
    arg.transA    = std::get<7>(tup);
    arg.transB    = std::get<8>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_csrgemm_a_arguments(csrgemm_a_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.K         = -99;
    arg.alpha     = std::get<0>(tup);
    arg.idx_base  = std::get<1>(tup);
    arg.idx_base2 = std::get<2>(tup);
    arg.idx_base3 = std::get<3>(tup);
    arg.transA    = std::get<4>(tup);
    arg.transB    = std::get<5>(tup);
    arg.timing    = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<6>(tup);

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

TEST(csrgemm_a_bad_arg, csrgemm_a_float)
{
    testing_csrgemm_a_bad_arg<float>();
}

TEST_P(parameterized_csrgemm_a, csrgemm_a_float)
{
    Arguments arg = setup_csrgemm_a_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_a<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrgemm_a, csrgemm_a_double)
{
    Arguments arg = setup_csrgemm_a_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_a<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrgemm_a_bin, csrgemm_a_bin_float)
{
    Arguments arg = setup_csrgemm_a_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_a<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csrgemm_a_bin, csrgemm_a_bin_double)
{
    Arguments arg = setup_csrgemm_a_arguments(GetParam());

    rocsparse_status status = testing_csrgemm_a<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csrgemm_a,
                        parameterized_csrgemm_a,
                        testing::Combine(testing::ValuesIn(csrgemm_a_M_range),
                                         testing::ValuesIn(csrgemm_a_N_range),
                                         testing::ValuesIn(csrgemm_a_K_range),
                                         testing::ValuesIn(csrgemm_a_alpha_range),
                                         testing::ValuesIn(csrgemm_a_idxbaseA_range),
                                         testing::ValuesIn(csrgemm_a_idxbaseB_range),
                                         testing::ValuesIn(csrgemm_a_idxbaseC_range),
                                         testing::ValuesIn(csrgemm_a_transA_range),
                                         testing::ValuesIn(csrgemm_a_transB_range)));

INSTANTIATE_TEST_CASE_P(csrgemm_a_bin,
                        parameterized_csrgemm_a_bin,
                        testing::Combine(testing::ValuesIn(csrgemm_a_alpha_range),
                                         testing::ValuesIn(csrgemm_a_idxbaseA_range),
                                         testing::ValuesIn(csrgemm_a_idxbaseB_range),
                                         testing::ValuesIn(csrgemm_a_idxbaseC_range),
                                         testing::ValuesIn(csrgemm_a_transA_range),
                                         testing::ValuesIn(csrgemm_a_transB_range),
                                         testing::ValuesIn(csrgemm_a_bin)));

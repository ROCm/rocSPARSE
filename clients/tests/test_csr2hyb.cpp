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

#include "testing_csr2hyb.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocsparse.h>
#include <string>
#include <vector>

typedef std::tuple<rocsparse_int, rocsparse_int, rocsparse_index_base, rocsparse_hyb_partition, int>
    csr2hyb_tuple;
typedef std::tuple<rocsparse_index_base, rocsparse_hyb_partition, int, std::string>
    csr2hyb_bin_tuple;

rocsparse_int csr2hyb_M_range[] = {-1, 0, 10, 500, 872, 1000};
rocsparse_int csr2hyb_N_range[] = {-3, 0, 33, 242, 623, 1000};

rocsparse_index_base csr2hyb_idx_base_range[]
    = {rocsparse_index_base_zero, rocsparse_index_base_one};

rocsparse_hyb_partition csr2hyb_partition[]
    = {rocsparse_hyb_partition_auto, rocsparse_hyb_partition_max, rocsparse_hyb_partition_user};

int csr2hyb_ELL_range[] = {-33, -1, 0, INT32_MAX};

std::string csr2hyb_bin[] = {"rma10.bin",
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
                             "nos7.bin",
                             "amazon0312.bin",
                             "Chebyshev4.bin",
                             "sme3Dc.bin",
                             "webbase-1M.bin",
                             "shipsec1.bin"};

class parameterized_csr2hyb : public testing::TestWithParam<csr2hyb_tuple>
{
protected:
    parameterized_csr2hyb() {}
    virtual ~parameterized_csr2hyb() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_csr2hyb_bin : public testing::TestWithParam<csr2hyb_bin_tuple>
{
protected:
    parameterized_csr2hyb_bin() {}
    virtual ~parameterized_csr2hyb_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2hyb_arguments(csr2hyb_tuple tup)
{
    Arguments arg;
    arg.M         = std::get<0>(tup);
    arg.N         = std::get<1>(tup);
    arg.idx_base  = std::get<2>(tup);
    arg.part      = std::get<3>(tup);
    arg.ell_width = std::get<4>(tup);
    arg.timing    = 0;
    return arg;
}

Arguments setup_csr2hyb_arguments(csr2hyb_bin_tuple tup)
{
    Arguments arg;
    arg.M         = -99;
    arg.N         = -99;
    arg.idx_base  = std::get<0>(tup);
    arg.part      = std::get<1>(tup);
    arg.ell_width = std::get<2>(tup);
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

TEST(csr2hyb_bad_arg, csr2hyb)
{
    testing_csr2hyb_bad_arg<float>();
}

TEST_P(parameterized_csr2hyb, csr2hyb_float)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    rocsparse_status status = testing_csr2hyb<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csr2hyb, csr2hyb_double)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    rocsparse_status status = testing_csr2hyb<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csr2hyb_bin, csr2hyb_bin_float)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    rocsparse_status status = testing_csr2hyb<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csr2hyb_bin, csr2hyb_bin_double)
{
    Arguments arg = setup_csr2hyb_arguments(GetParam());

    rocsparse_status status = testing_csr2hyb<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csr2hyb,
                        parameterized_csr2hyb,
                        testing::Combine(testing::ValuesIn(csr2hyb_M_range),
                                         testing::ValuesIn(csr2hyb_N_range),
                                         testing::ValuesIn(csr2hyb_idx_base_range),
                                         testing::ValuesIn(csr2hyb_partition),
                                         testing::ValuesIn(csr2hyb_ELL_range)));

INSTANTIATE_TEST_CASE_P(csr2hyb_bin,
                        parameterized_csr2hyb_bin,
                        testing::Combine(testing::ValuesIn(csr2hyb_idx_base_range),
                                         testing::ValuesIn(csr2hyb_partition),
                                         testing::ValuesIn(csr2hyb_ELL_range),
                                         testing::ValuesIn(csr2hyb_bin)));

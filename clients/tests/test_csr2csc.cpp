/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csr2csc.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, rocsparse_action, rocsparse_index_base, std::string> csr2csc_tuple;

int csr2csc_M_range[] = {-99, -1, 0, 10, 500, 872, 1000};
int csr2csc_N_range[] = {-99, 0, 33, 242, 623, 1000};

rocsparse_action csr2csc_action_range[] = {rocsparse_action_numeric, rocsparse_action_symbolic};

rocsparse_index_base csr2csc_csr_base_range[] = {rocsparse_index_base_zero,
                                                 rocsparse_index_base_one};

std::string csr2csc_bin[] = {"rma10.bin",
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

class parameterized_csr2csc : public testing::TestWithParam<csr2csc_tuple>
{
    protected:
    parameterized_csr2csc() {}
    virtual ~parameterized_csr2csc() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2csc_arguments(csr2csc_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.action   = std::get<2>(tup);
    arg.idx_base = std::get<3>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<4>(tup);

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

TEST(csr2csc_bad_arg, csr2csc) { testing_csr2csc_bad_arg<float>(); }

TEST_P(parameterized_csr2csc, csr2csc_float)
{
    Arguments arg = setup_csr2csc_arguments(GetParam());

    rocsparse_status status = testing_csr2csc<float>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

TEST_P(parameterized_csr2csc, csr2csc_double)
{
    Arguments arg = setup_csr2csc_arguments(GetParam());

    rocsparse_status status = testing_csr2csc<double>(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csr2csc,
                        parameterized_csr2csc,
                        testing::Combine(testing::ValuesIn(csr2csc_M_range),
                                         testing::ValuesIn(csr2csc_N_range),
                                         testing::ValuesIn(csr2csc_action_range),
                                         testing::ValuesIn(csr2csc_csr_base_range),
                                         testing::ValuesIn(csr2csc_bin)));

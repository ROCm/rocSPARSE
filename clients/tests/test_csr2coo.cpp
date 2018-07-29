/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csr2coo.hpp"
#include "utility.hpp"

#include <rocsparse.h>
#include <gtest/gtest.h>
#include <vector>

typedef std::tuple<int, int, rocsparse_index_base, std::string> csr2coo_tuple;

int csr2coo_M_range[] = {-99, -1, 0, 10, 500, 872, 1000};
int csr2coo_N_range[] = {-99, 0, 33, 242, 623, 1000};

rocsparse_index_base csr2coo_idx_base_range[] = {rocsparse_index_base_zero,
                                                 rocsparse_index_base_one};

std::string csr2coo_bin[] = {"rma10.bin",
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

class parameterized_csr2coo : public testing::TestWithParam<csr2coo_tuple>
{
    protected:
    parameterized_csr2coo() {}
    virtual ~parameterized_csr2coo() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_csr2coo_arguments(csr2coo_tuple tup)
{
    Arguments arg;
    arg.M        = std::get<0>(tup);
    arg.N        = std::get<1>(tup);
    arg.idx_base = std::get<2>(tup);
    arg.timing   = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

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

TEST(csr2coo_bad_arg, csr2coo) { testing_csr2coo_bad_arg(); }

TEST_P(parameterized_csr2coo, csr2coo)
{
    Arguments arg = setup_csr2coo_arguments(GetParam());

    rocsparse_status status = testing_csr2coo(arg);
    EXPECT_EQ(status, rocsparse_status_success);
}

INSTANTIATE_TEST_CASE_P(csr2coo,
                        parameterized_csr2coo,
                        testing::Combine(testing::ValuesIn(csr2coo_M_range),
                                         testing::ValuesIn(csr2coo_N_range),
                                         testing::ValuesIn(csr2coo_idx_base_range),
                                         testing::ValuesIn(csr2coo_bin)));

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

#include "rocsparse.hpp"
#include "utility.hpp"

// Level1
#include "testing_axpyi.hpp"
#include "testing_dotci.hpp"
#include "testing_doti.hpp"
#include "testing_gthr.hpp"
#include "testing_gthrz.hpp"
#include "testing_roti.hpp"
#include "testing_sctr.hpp"

// Level2
#include "testing_coomv.hpp"
#include "testing_csrmv.hpp"
#include "testing_csrsv.hpp"
#include "testing_ellmv.hpp"
#include "testing_hybmv.hpp"

// Level3
#include "testing_csrmm.hpp"

// Extra
#include "testing_csrgemm.hpp"

// Preconditioner
#include "testing_csrilu0.hpp"

// Conversion
#include "testing_coo2csr.hpp"
#include "testing_coosort.hpp"
#include "testing_cscsort.hpp"
#include "testing_csr2coo.hpp"
#include "testing_csr2csc.hpp"
#include "testing_csr2ell.hpp"
#include "testing_csr2hyb.hpp"
#include "testing_csrsort.hpp"
#include "testing_ell2csr.hpp"
#include "testing_identity.hpp"

#include <boost/program_options.hpp>
#include <iostream>
#include <rocsparse.h>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    Arguments arg;
    arg.unit_check = 0;
    arg.timing     = 1;
    arg.alphai     = 0.0;
    arg.betai      = 0.0;

    std::string function;
    std::string filename;
    std::string rocalution;
    char        precision = 's';
    char        transA;
    char        transB;
    int         baseA;
    int         baseB;
    int         baseC;
    int         baseD;
    int         action;
    int         part;
    char        diag;
    char        uplo;
    char        apol;

    std::vector<rocsparse_int> laplace(3, 0);

    rocsparse_int device_id;

    po::options_description desc("rocsparse client command line options");
    desc.add_options()("help,h", "produces this help message")
        // clang-format off
        ("sizem,m",
         po::value<rocsparse_int>(&arg.M)->default_value(128),
         "Specific matrix size testing: sizem is only applicable to SPARSE-2 "
         "& SPARSE-3: the number of rows.")

        ("sizen,n",
         po::value<rocsparse_int>(&arg.N)->default_value(128),
         "Specific matrix/vector size testing: SPARSE-1: the length of the "
         "dense vector. SPARSE-2 & SPARSE-3: the number of columns")

        ("sizek,k",
         po::value<rocsparse_int>(&arg.K)->default_value(128),
         "Specific matrix/vector size testing: SPARSE-3: the number of columns")

        ("sizennz,z",
         po::value<rocsparse_int>(&arg.nnz)->default_value(32),
         "Specific vector size testing, LEVEL-1: the number of non-zero elements "
         "of the sparse vector.")

        ("mtx",
         po::value<std::string>(&filename)->default_value(""), "read from matrix "
         "market (.mtx) format. This will override parameters -m, -n, and -z.")

        ("rocalution",
         po::value<std::string>(&rocalution)->default_value(""),
         "read from rocalution matrix binary file. This will override parameter --mtx")

        ("laplacian-dim",
         po::value<std::vector<rocsparse_int> >(&laplace)->multitoken(), "assemble "
         "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
         "will override parameters -m, -n, -z and --mtx.")

        ("alpha", 
          po::value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta", 
          po::value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("transposeA",
          po::value<char>(&transA)->default_value('N'),
          "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
          po::value<char>(&transB)->default_value('N'),
          "N = no transpose, T = transpose, C = conjugate transpose, (default = N)")

        ("indexbaseA",
          po::value<int>(&baseA)->default_value(0),
          "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("indexbaseB",
          po::value<int>(&baseB)->default_value(0),
          "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("indexbaseC",
          po::value<int>(&baseC)->default_value(0),
          "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("indexbaseD",
          po::value<int>(&baseD)->default_value(0),
          "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("action",
          po::value<int>(&action)->default_value(0),
          "0 = rocsparse_action_numeric, 1 = rocsparse_action_symbolic, (default: 0)")

        ("hybpart",
          po::value<int>(&part)->default_value(0),
          "0 = rocsparse_hyb_partition_auto, 1 = rocsparse_hyb_partition_user,\n"
          "2 = rocsparse_hyb_partition_max, (default: 0)")

        ("diag",
          po::value<char>(&diag)->default_value('N'),
          "N = non-unit diagonal, U = unit diagonal, (default = N)")

        ("uplo",
          po::value<char>(&uplo)->default_value('L'),
          "L = lower fill, U = upper fill, (default = L)")

        ("apolicy",
          po::value<char>(&apol)->default_value('R'),
          "R = reuse meta data, F = force re-build, (default = R)")

//        ("spolicy",
//          po::value<char>(&spol)->default_value('A'),
//          "A = auto, (default = A)")

        ("function,f",
         po::value<std::string>(&function)->default_value("axpyi"),
         "SPARSE function to test. Options:\n"
         "  Level1: axpyi, doti, dotci, gthr, gthrz, roti, sctr\n"
         "  Level2: coomv, csrmv, csrsv, ellmv, hybmv\n"
         "  Level3: csrmm\n"
         "  Extra: csrgemm\n"
         "  Preconditioner: csrilu0\n"
         "  Conversion: csr2coo, csr2csc, csr2ell,\n"
         "              csr2hyb, coo2csr, ell2csr\n"
         "  Sorting: cscsort, csrsort, coosort\n"
         "  Misc: identity")

        ("precision,r",
         po::value<char>(&precision)->default_value('s'), "Options: s,d,c,z")

        ("verify,v",
         po::value<rocsparse_int>(&arg.unit_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         po::value<int>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("device,d",
         po::value<rocsparse_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if(precision != 's' && precision != 'd' && precision != 'c' && precision != 'z')
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }

    if(transA == 'N')
    {
        arg.transA = rocsparse_operation_none;
    }
    else if(transA == 'T')
    {
        arg.transA = rocsparse_operation_transpose;
        ;
    }
    else if(transA == 'C')
    {
        arg.transA = rocsparse_operation_conjugate_transpose;
    }

    if(transB == 'N')
    {
        arg.transB = rocsparse_operation_none;
    }
    else if(transB == 'T')
    {
        arg.transB = rocsparse_operation_transpose;
        ;
    }
    else if(transB == 'C')
    {
        arg.transB = rocsparse_operation_conjugate_transpose;
    }

    arg.baseA = (baseA == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
    arg.baseB = (baseB == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
    arg.baseC = (baseC == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
    arg.baseD = (baseD == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;

    arg.action = (action == 0) ? rocsparse_action_numeric : rocsparse_action_symbolic;
    arg.part   = (part == 0)
                   ? rocsparse_hyb_partition_auto
                   : (part == 1) ? rocsparse_hyb_partition_user : rocsparse_hyb_partition_max;
    arg.diag = (diag == 'N') ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit;
    arg.uplo = (uplo == 'L') ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
    arg.apol = (apol == 'R') ? rocsparse_analysis_policy_reuse : rocsparse_analysis_policy_force;
    arg.spol = rocsparse_solve_policy_auto;

    // Set laplace dimensions
    arg.dimx = laplace[0];
    arg.dimy = laplace[1];
    arg.dimz = laplace[2];

    // rocALUTION parameter overrides filename parameter
    if(rocalution != "")
    {
        strcpy(arg.filename, rocalution.c_str());
        arg.matrix = rocsparse_matrix_file_rocalution;
    }
    else if(arg.dimx != 0 && arg.dimy != 0 && arg.dimz != 0)
    {
        arg.matrix = rocsparse_matrix_laplace_3d;
    }
    else if(arg.dimx != 0 && arg.dimy != 0)
    {
        arg.matrix = rocsparse_matrix_laplace_2d;
    }
    else if(filename != "")
    {
        strcpy(arg.filename, filename.c_str());
        arg.matrix = rocsparse_matrix_file_mtx;
    }
    else
    {
        arg.matrix = rocsparse_matrix_random;
    }

    // Device query
    int devs;
    if(hipGetDeviceCount(&devs) != hipSuccess)
    {
        std::cerr << "Error: cannot get device count" << std::endl;
        return -1;
    }

    std::cout << "Query device success: there are " << devs << " devices" << std::endl;

    for(int i = 0; i < devs; ++i)
    {
        hipDeviceProp_t prop;

        if(hipGetDeviceProperties(&prop, i) != hipSuccess)
        {
            std::cerr << "Error: cannot get device properties" << std::endl;
            return -1;
        }

        std::cout << "Device ID " << i << ": " << prop.name << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
        std::cout << "with " << (prop.totalGlobalMem >> 20) << "MB memory, clock rate "
                  << prop.clockRate / 1000 << "MHz @ computing capability " << prop.major << "."
                  << prop.minor << std::endl;
        std::cout << "maxGridDimX " << prop.maxGridSize[0] << ", sharedMemPerBlock "
                  << (prop.sharedMemPerBlock >> 10) << "KB, maxThreadsPerBlock "
                  << prop.maxThreadsPerBlock << std::endl;
        std::cout << "wavefrontSize " << prop.warpSize << std::endl;
        std::cout << "-------------------------------------------------------------------------"
                  << std::endl;
    }

    // Set device
    if(hipSetDevice(device_id) != hipSuccess || device_id >= devs)
    {
        std::cerr << "Error: cannot set device ID " << device_id << std::endl;
        return -1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);

    std::cout << "Using device ID " << device_id << " (" << prop.name << ") for rocSPARSE"
              << std::endl;
    std::cout << "-------------------------------------------------------------------------"
              << std::endl;

    // Print version
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    int  ver;
    char rev[64];

    rocsparse_get_version(handle, &ver);
    rocsparse_get_git_rev(handle, rev);

    std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << "-" << rev << std::endl;

    rocsparse_destroy_handle(handle);

    /* ============================================================================================
     */
    if(arg.M < 0 || arg.N < 0)
    {
        std::cerr << "Invalid dimension" << std::endl;
        return -1;
    }

    // Level1
    if(function == "axpyi")
    {
        if(precision == 's')
            testing_axpyi<float>(arg);
        else if(precision == 'd')
            testing_axpyi<double>(arg);
        else if(precision == 'c')
            testing_axpyi<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_axpyi<rocsparse_double_complex>(arg);
    }
    else if(function == "doti")
    {
        if(precision == 's')
            testing_doti<float>(arg);
        else if(precision == 'd')
            testing_doti<double>(arg);
        else if(precision == 'c')
            testing_doti<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_doti<rocsparse_double_complex>(arg);
    }
    else if(function == "dotci")
    {
        if(precision == 's')
            testing_doti<float>(arg);
        else if(precision == 'd')
            testing_doti<double>(arg);
        else if(precision == 'c')
            testing_dotci<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_dotci<rocsparse_double_complex>(arg);
    }
    else if(function == "gthr")
    {
        if(precision == 's')
            testing_gthr<float>(arg);
        else if(precision == 'd')
            testing_gthr<double>(arg);
        else if(precision == 'c')
            testing_gthr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gthr<rocsparse_double_complex>(arg);
    }
    else if(function == "gthrz")
    {
        if(precision == 's')
            testing_gthrz<float>(arg);
        else if(precision == 'd')
            testing_gthrz<double>(arg);
        else if(precision == 'c')
            testing_gthrz<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gthrz<rocsparse_double_complex>(arg);
    }
    else if(function == "roti")
    {
        if(precision == 's')
            testing_roti<float>(arg);
        else if(precision == 'd')
            testing_roti<double>(arg);
    }
    else if(function == "sctr")
    {
        if(precision == 's')
            testing_sctr<float>(arg);
        else if(precision == 'd')
            testing_sctr<double>(arg);
        else if(precision == 'c')
            testing_sctr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_sctr<rocsparse_double_complex>(arg);
    }
    else if(function == "coomv")
    {
        if(precision == 's')
            testing_coomv<float>(arg);
        else if(precision == 'd')
            testing_coomv<double>(arg);
        else if(precision == 'c')
            testing_coomv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_coomv<rocsparse_double_complex>(arg);
    }
    else if(function == "csrmv")
    {
        arg.algo = 1;
        if(precision == 's')
            testing_csrmv<float>(arg);
        else if(precision == 'd')
            testing_csrmv<double>(arg);
        else if(precision == 'c')
            testing_csrmv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrmv<rocsparse_double_complex>(arg);
    }
    else if(function == "csrsv")
    {
        if(precision == 's')
            testing_csrsv<float>(arg);
        else if(precision == 'd')
            testing_csrsv<double>(arg);
        else if(precision == 'c')
            testing_csrsv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrsv<rocsparse_double_complex>(arg);
    }
    else if(function == "ellmv")
    {
        if(precision == 's')
            testing_ellmv<float>(arg);
        else if(precision == 'd')
            testing_ellmv<double>(arg);
        else if(precision == 'c')
            testing_ellmv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_ellmv<rocsparse_double_complex>(arg);
    }
    else if(function == "hybmv")
    {
        if(precision == 's')
            testing_hybmv<float>(arg);
        else if(precision == 'd')
            testing_hybmv<double>(arg);
        else if(precision == 'c')
            testing_hybmv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_hybmv<rocsparse_double_complex>(arg);
    }
    else if(function == "csrmm")
    {
        if(precision == 's')
            testing_csrmm<float>(arg);
        else if(precision == 'd')
            testing_csrmm<double>(arg);
        else if(precision == 'c')
            testing_csrmm<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrmm<rocsparse_double_complex>(arg);
    }
    else if(function == "csrgemm")
    {
        // TODO workaround until fully implemented
        if(arg.beta == 0.0)
        {
            arg.beta = -99;
        }
        if(precision == 's')
            testing_csrgemm<float>(arg);
        else if(precision == 'd')
            testing_csrgemm<double>(arg);
        else if(precision == 'c')
            testing_csrgemm<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrgemm<rocsparse_double_complex>(arg);
    }
    else if(function == "csrilu0")
    {
        if(precision == 's')
            testing_csrilu0<float>(arg);
        else if(precision == 'd')
            testing_csrilu0<double>(arg);
        else if(precision == 'c')
            testing_csrilu0<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrilu0<rocsparse_double_complex>(arg);
    }
    else if(function == "csr2coo")
    {
        testing_csr2coo<float>(arg);
    }
    else if(function == "csr2csc")
    {
        if(precision == 's')
            testing_csr2csc<float>(arg);
        else if(precision == 'd')
            testing_csr2csc<double>(arg);
        else if(precision == 'c')
            testing_csr2csc<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csr2csc<rocsparse_double_complex>(arg);
    }
    else if(function == "csr2ell")
    {
        if(precision == 's')
            testing_csr2ell<float>(arg);
        else if(precision == 'd')
            testing_csr2ell<double>(arg);
        else if(precision == 'c')
            testing_csr2ell<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csr2ell<rocsparse_double_complex>(arg);
    }
    else if(function == "csr2hyb")
    {
        if(precision == 's')
            testing_csr2hyb<float>(arg);
        else if(precision == 'd')
            testing_csr2hyb<double>(arg);
        else if(precision == 'c')
            testing_csr2hyb<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csr2hyb<rocsparse_double_complex>(arg);
    }
    else if(function == "coo2csr")
    {
        testing_coo2csr<float>(arg);
    }
    else if(function == "ell2csr")
    {
        if(precision == 's')
            testing_ell2csr<float>(arg);
        else if(precision == 'd')
            testing_ell2csr<double>(arg);
        else if(precision == 'c')
            testing_ell2csr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_ell2csr<rocsparse_double_complex>(arg);
    }
    else if(function == "csrsort")
    {
        testing_csrsort<float>(arg);
    }
    else if(function == "cscsort")
    {
        testing_cscsort<float>(arg);
    }
    else if(function == "coosort")
    {
        testing_coosort<float>(arg);
    }
    else if(function == "identity")
    {
        testing_identity<float>(arg);
    }
    else
    {
        std::cerr << "Invalid value for --function" << std::endl;
        return -1;
    }
    return 0;
}

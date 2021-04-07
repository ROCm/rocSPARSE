/*! \file */
/* ************************************************************************
* Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
#include "testing_bsrmv.hpp"
#include "testing_bsrsv.hpp"
#include "testing_csrmv_managed.hpp"
#include "testing_csrsv.hpp"
#include "testing_gebsrmv.hpp"
#include "testing_gemvi.hpp"
#include "testing_hybmv.hpp"
#include "testing_spmv_coo.hpp"
#include "testing_spmv_coo_aos.hpp"
#include "testing_spmv_csr.hpp"
#include "testing_spmv_ell.hpp"

// Level3
#include "testing_bsrmm.hpp"
#include "testing_csrmm.hpp"
#include "testing_csrsm.hpp"
#include "testing_gebsrmm.hpp"
#include "testing_gemmi.hpp"
#include "testing_sddmm.hpp"
#include "testing_spmm_coo.hpp"
#include "testing_spmm_csr.hpp"

// Extra
#include "testing_csrgeam.hpp"
#include "testing_spgemm_csr.hpp"

// Preconditioner
#include "testing_bsric0.hpp"
#include "testing_bsrilu0.hpp"
#include "testing_csric0.hpp"
#include "testing_csrilu0.hpp"

// Conversion
#include "testing_bsr2csr.hpp"
#include "testing_coo2csr.hpp"
#include "testing_coo2dense.hpp"
#include "testing_coosort.hpp"
#include "testing_csc2dense.hpp"
#include "testing_cscsort.hpp"
#include "testing_csr2bsr.hpp"
#include "testing_csr2coo.hpp"
#include "testing_csr2csc.hpp"
#include "testing_csr2csr_compress.hpp"
#include "testing_csr2dense.hpp"
#include "testing_csr2ell.hpp"
#include "testing_csr2gebsr.hpp"
#include "testing_csr2hyb.hpp"
#include "testing_csrsort.hpp"
#include "testing_dense2coo.hpp"
#include "testing_dense2csc.hpp"
#include "testing_dense2csr.hpp"
#include "testing_dense_to_sparse_coo.hpp"
#include "testing_dense_to_sparse_csc.hpp"
#include "testing_dense_to_sparse_csr.hpp"
#include "testing_ell2csr.hpp"
#include "testing_gebsr2csr.hpp"
#include "testing_gebsr2gebsc.hpp"
#include "testing_gebsr2gebsr.hpp"
#include "testing_hyb2csr.hpp"
#include "testing_identity.hpp"
#include "testing_nnz.hpp"
#include "testing_prune_csr2csr.hpp"
#include "testing_prune_csr2csr_by_percentage.hpp"
#include "testing_prune_dense2csr.hpp"
#include "testing_prune_dense2csr_by_percentage.hpp"
#include "testing_sparse_to_dense_coo.hpp"
#include "testing_sparse_to_dense_csc.hpp"
#include "testing_sparse_to_dense_csr.hpp"

#include <iostream>
#include <rocsparse.h>
#include <unordered_set>

#include "program_options.hpp"

int main(int argc, char* argv[])
{
    Arguments arg;
    arg.unit_check          = 0;
    arg.timing              = 1;
    arg.alphai              = 0.0;
    arg.betai               = 0.0;
    arg.threshold           = 0.0;
    arg.percentage          = 0.0;
    arg.sddmm_alg           = rocsparse_sddmm_alg_default;
    arg.spmv_alg            = rocsparse_spmv_alg_default;
    arg.spmm_alg            = rocsparse_spmm_alg_default;
    arg.spgemm_alg          = rocsparse_spgemm_alg_default;
    arg.sparse_to_dense_alg = rocsparse_sparse_to_dense_alg_default;
    arg.dense_to_sparse_alg = rocsparse_dense_to_sparse_alg_default;

    std::string   function;
    std::string   filename;
    std::string   rocalution;
    char          indextype = 's';
    char          precision = 's';
    char          transA;
    char          transB;
    int           baseA;
    int           baseB;
    int           baseC;
    int           baseD;
    int           action;
    int           part;
    char          diag;
    char          uplo;
    char          apol;
    rocsparse_int dir;
    rocsparse_int order;
    rocsparse_int format;

    rocsparse_int device_id;

    // clang-format off

    options_description desc("rocsparse client command line options");
    desc.add_options() ("help,h", "produces this help message")
        // clang-format off
        ("sizem,m",
        value<rocsparse_int>(&arg.M)->default_value(128),
        "Specific matrix size testing: sizem is only applicable to SPARSE-2 "
        "& SPARSE-3: the number of rows.")

        ("sizen,n",
        value<rocsparse_int>(&arg.N)->default_value(128),
        "Specific matrix/vector size testing: SPARSE-1: the length of the "
        "dense vector. SPARSE-2 & SPARSE-3: the number of columns")

        ("sizek,k",
        value<rocsparse_int>(&arg.K)->default_value(128),
        "Specific matrix/vector size testing: SPARSE-3: the number of columns")

        ("sizennz,z",
        value<rocsparse_int>(&arg.nnz)->default_value(32),
        "Specific vector size testing, LEVEL-1: the number of non-zero elements "
        "of the sparse vector.")

        ("blockdim",
        value<rocsparse_int>(&arg.block_dim)->default_value(2),
        "BSR block dimension (default: 2)")

        ("row-blockdimA",
        value<rocsparse_int>(&arg.row_block_dimA)->default_value(2),
        "General BSR row block dimension (default: 2)")

        ("col-blockdimA",
        value<rocsparse_int>(&arg.col_block_dimA)->default_value(2),
        "General BSR col block dimension (default: 2)")

        ("row-blockdimB",
        value<rocsparse_int>(&arg.row_block_dimB)->default_value(2),
        "General BSR row block dimension (default: 2)")

        ("col-blockdimB",
        value<rocsparse_int>(&arg.col_block_dimB)->default_value(2),
        "General BSR col block dimension (default: 2)")

        ("mtx",
        value<std::string>(&filename)->default_value(""), "read from matrix "
        "market (.mtx) format. This will override parameters -m, -n, and -z.")

        ("rocalution",
        value<std::string>(&rocalution)->default_value(""),
        "read from rocalution matrix binary file. This will override parameter --mtx")

        ("dimx",
        value<rocsparse_int>(&arg.dimx)->default_value(0.0), "assemble "
        "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
        "will override parameters -m, -n, -z and --mtx.")

        ("dimy",
        value<rocsparse_int>(&arg.dimy)->default_value(0.0), "assemble "
        "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
        "will override parameters -m, -n, -z and --mtx.")

        ("dimz",
        value<rocsparse_int>(&arg.dimz)->default_value(0.0), "assemble "
        "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
        "will override parameters -m, -n, -z and --mtx.")

        ("alpha",
        value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
        value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("threshold",
        value<double>(&arg.threshold)->default_value(1.0), "specifies the scalar threshold")

        ("percentage",
        value<double>(&arg.percentage)->default_value(0.0), "specifies the scalar percentage")

        ("transposeA",
        value<char>(&transA)->default_value('N'),
        "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
        value<char>(&transB)->default_value('N'),
        "N = no transpose, T = transpose, C = conjugate transpose, (default = N)")

        ("indexbaseA",
        value<int>(&baseA)->default_value(0),
        "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("indexbaseB",
        value<int>(&baseB)->default_value(0),
        "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("indexbaseC",
        value<int>(&baseC)->default_value(0),
        "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("indexbaseD",
        value<int>(&baseD)->default_value(0),
        "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

        ("action",
        value<int>(&action)->default_value(0),
        "0 = rocsparse_action_numeric, 1 = rocsparse_action_symbolic, (default: 0)")

        ("hybpart",
        value<int>(&part)->default_value(0),
        "0 = rocsparse_hyb_partition_auto, 1 = rocsparse_hyb_partition_user,\n"
        "2 = rocsparse_hyb_partition_max, (default: 0)")

        ("diag",
        value<char>(&diag)->default_value('N'),
        "N = non-unit diagonal, U = unit diagonal, (default = N)")

        ("uplo",
        value<char>(&uplo)->default_value('L'),
        "L = lower fill, U = upper fill, (default = L)")

        ("apolicy",
        value<char>(&apol)->default_value('R'),
        "R = reuse meta data, F = force re-build, (default = R)")

        ("function,f",
        value<std::string>(&function)->default_value("axpyi"),
        "SPARSE function to test. Options:\n"
        "  Level1: axpyi, doti, dotci, gthr, gthrz, roti, sctr\n"
        "  Level2: bsrmv, bsrsv, coomv, coomv_aos, csrmv, csrmv_managed, csrsv, ellmv, hybmv, gebsrmv, gemvi\n"
        "  Level3: bsrmm, gebsrmm, csrmm, coomm, csrsm, gemmi, sddmm\n"
        "  Extra: csrgeam, csrgemm\n"
        "  Preconditioner: bsric0, bsrilu0, csric0, csrilu0\n"
        "  Conversion: csr2coo, csr2csc, gebsr2gebsc, csr2ell, csr2hyb, csr2bsr, csr2gebsr\n"
        "              coo2csr, ell2csr, hyb2csr, dense2csr, dense2coo, prune_dense2csr, prune_dense2csr_by_percentage, dense2csc\n"
        "              csr2dense, csc2dense, coo2dense, bsr2csr, gebsr2csr, gebsr2gebsr, csr2csr_compress, prune_csr2csr, prune_csr2csr_by_percentage\n"
        "              sparse_to_dense_coo, sparse_to_dense_csr, sparse_to_dense_csc, dense_to_sparse_coo, dense_to_sparse_csr, dense_to_sparse_csc\n"
        "  Sorting: cscsort, csrsort, coosort\n"
        "  Misc: identity, nnz")

        ("indextype",
        value<char>(&indextype)->default_value('s'),
        "Specify index types to be int32_t (s), int64_t (d) or mixed (m). Options: s,d,m")

        ("precision,r",
        value<char>(&precision)->default_value('s'), "Options: s,d,c,z")

        ("verify,v",
        value<rocsparse_int>(&arg.unit_check)->default_value(0),
        "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
        value<int>(&arg.iters)->default_value(10),
        "Iterations to run inside timing loop")

        ("device,d",
        value<rocsparse_int>(&device_id)->default_value(0),
        "Set default device to be used for subsequent program runs")

        ("direction",
        value<rocsparse_int>(&dir)->default_value(rocsparse_direction_row),
        "Indicates whether a dense matrix should be parsed by rows or by columns, assuming column-major storage: row = 0, column = 1 (default: 0)")

        ("order",
        value<rocsparse_int>(&order)->default_value(rocsparse_order_column),
        "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

        ("format",
        value<rocsparse_int>(&format)->default_value(rocsparse_format_coo),
        "Indicates wther a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3 or ell format: 4 (default:0)")

        ("denseld",
        value<rocsparse_int>(&arg.denseld)->default_value(128),
        "Indicates the leading dimension of a dense matrix >= M, assuming a column-oriented storage.");

    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if(dir != rocsparse_direction_row && dir != rocsparse_direction_column)
    {
        std::cerr << "Invalid value for --direction" << std::endl;
        return -1;
    }

    if(order != rocsparse_order_row && order != rocsparse_order_column)
    {
        std::cerr << "Invalid value for --order" << std::endl;
        return -1;
    }

    if(format != rocsparse_format_csr && format != rocsparse_format_coo
       && format != rocsparse_format_coo_aos && format != rocsparse_format_ell
       && format != rocsparse_format_csc)
    {
        std::cerr << "Invalid value for --format" << std::endl;
        return -1;
    }

    if(indextype != 's' && indextype != 'd' && indextype != 'm')
    {
        std::cerr << "Invalid value for --indextype" << std::endl;
        return -1;
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
    arg.part   = (part == 0)   ? rocsparse_hyb_partition_auto
                 : (part == 1) ? rocsparse_hyb_partition_user
                               : rocsparse_hyb_partition_max;
    arg.diag   = (diag == 'N') ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit;
    arg.uplo   = (uplo == 'L') ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
    arg.apol   = (apol == 'R') ? rocsparse_analysis_policy_reuse : rocsparse_analysis_policy_force;
    arg.spol   = rocsparse_solve_policy_auto;
    arg.direction
        = (dir == rocsparse_direction_row) ? rocsparse_direction_row : rocsparse_direction_column;
    arg.order  = (order == rocsparse_order_row) ? rocsparse_order_row : rocsparse_order_column;
    arg.format = (rocsparse_format)format;

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

    arg.matrix_init_kind = rocsparse_matrix_init_kind_default;

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

    if(arg.block_dim < 1)
    {
        std::cerr << "Invalid value for --blockdim" << std::endl;
        return -1;
    }

    if(arg.row_block_dimA < 1)
    {
        std::cerr << "Invalid value for --row-blockdimA" << std::endl;
        return -1;
    }

    if(arg.col_block_dimA < 1)
    {
        std::cerr << "Invalid value for --col-blockdimA" << std::endl;
        return -1;
    }

    if(arg.row_block_dimB < 1)
    {
        std::cerr << "Invalid value for --row-blockdimB" << std::endl;
        return -1;
    }

    if(arg.col_block_dimB < 1)
    {
        std::cerr << "Invalid value for --col-blockdimB" << std::endl;
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
    else if(function == "bsrmv")
    {
        if(precision == 's')
            testing_bsrmv<float>(arg);
        else if(precision == 'd')
            testing_bsrmv<double>(arg);
        else if(precision == 'c')
            testing_bsrmv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_bsrmv<rocsparse_double_complex>(arg);
    }
    else if(function == "bsrsv")
    {
        if(precision == 's')
            testing_bsrsv<float>(arg);
        else if(precision == 'd')
            testing_bsrsv<double>(arg);
        else if(precision == 'c')
            testing_bsrsv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_bsrsv<rocsparse_double_complex>(arg);
    }
    else if(function == "coomv")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_spmv_coo<int32_t, float>(arg);
            else if(indextype == 'd')
                testing_spmv_coo<int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_spmv_coo<int32_t, double>(arg);
            else if(indextype == 'd')
                testing_spmv_coo<int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_spmv_coo<int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_coo<int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_spmv_coo<int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_coo<int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "coomv_aos")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_spmv_coo_aos<int32_t, float>(arg);
            else if(indextype == 'd')
                testing_spmv_coo_aos<int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_spmv_coo_aos<int32_t, double>(arg);
            else if(indextype == 'd')
                testing_spmv_coo_aos<int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_spmv_coo_aos<int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_coo_aos<int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_spmv_coo_aos<int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_coo_aos<int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "csrmv")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_spmv_csr<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_spmv_csr<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_spmv_csr<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_spmv_csr<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_spmv_csr<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_spmv_csr<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_spmv_csr<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_spmv_csr<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_csr<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_spmv_csr<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_spmv_csr<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_csr<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "csrmv_managed")
    {
        if(precision == 's')
            testing_csrmv_managed<float>(arg);
        else if(precision == 'd')
            testing_csrmv_managed<double>(arg);
        else if(precision == 'c')
            testing_csrmv_managed<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrmv_managed<rocsparse_double_complex>(arg);
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
        {
            if(indextype == 's')
                testing_spmv_ell<int32_t, float>(arg);
            else if(indextype == 'd')
                testing_spmv_ell<int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_spmv_ell<int32_t, double>(arg);
            else if(indextype == 'd')
                testing_spmv_ell<int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_spmv_ell<int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_ell<int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_spmv_ell<int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_spmv_ell<int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "gemvi")
    {
        if(precision == 's')
            testing_gemvi<float>(arg);
        else if(precision == 'd')
            testing_gemvi<double>(arg);
        else if(precision == 'c')
            testing_gemvi<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gemvi<rocsparse_double_complex>(arg);
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
    else if(function == "gebsrmv")
    {
        if(precision == 's')
            testing_gebsrmv<float>(arg);
        else if(precision == 'd')
            testing_gebsrmv<double>(arg);
        else if(precision == 'c')
            testing_gebsrmv<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gebsrmv<rocsparse_double_complex>(arg);
    }
    else if(function == "bsrmm")
    {
        if(precision == 's')
            testing_bsrmm<float>(arg);
        else if(precision == 'd')
            testing_bsrmm<double>(arg);
        else if(precision == 'c')
            testing_bsrmm<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_bsrmm<rocsparse_double_complex>(arg);
    }
    else if(function == "gebsrmm")
    {
        if(precision == 's')
            testing_gebsrmm<float>(arg);
        else if(precision == 'd')
            testing_gebsrmm<double>(arg);
        else if(precision == 'c')
            testing_gebsrmm<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gebsrmm<rocsparse_double_complex>(arg);
    }
    else if(function == "csrmm")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_spmm_csr<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_spmm_csr<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_spmm_csr<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_spmm_csr<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_spmm_csr<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_spmm_csr<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_spmm_csr<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_spmm_csr<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_spmm_csr<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_spmm_csr<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_spmm_csr<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_spmm_csr<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "coomm")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_spmm_coo<int32_t, float>(arg);
            else if(indextype == 'd')
                testing_spmm_coo<int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_spmm_coo<int32_t, double>(arg);
            else if(indextype == 'd')
                testing_spmm_coo<int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_spmm_coo<int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_spmm_coo<int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_spmm_coo<int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_spmm_coo<int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "csrsm")
    {
        if(precision == 's')
            testing_csrsm<float>(arg);
        else if(precision == 'd')
            testing_csrsm<double>(arg);
        else if(precision == 'c')
            testing_csrsm<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrsm<rocsparse_double_complex>(arg);
    }
    else if(function == "gemmi")
    {
        if(precision == 's')
            testing_gemmi<float>(arg);
        else if(precision == 'd')
            testing_gemmi<double>(arg);
        else if(precision == 'c')
            testing_gemmi<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gemmi<rocsparse_double_complex>(arg);
    }
    else if(function == "csrgeam")
    {
        if(precision == 's')
            testing_csrgeam<float>(arg);
        else if(precision == 'd')
            testing_csrgeam<double>(arg);
        else if(precision == 'c')
            testing_csrgeam<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrgeam<rocsparse_double_complex>(arg);
    }
    else if(function == "csrgemm")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_spgemm_csr<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_spgemm_csr<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_spgemm_csr<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_spgemm_csr<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_spgemm_csr<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_spgemm_csr<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_spgemm_csr<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_spgemm_csr<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_spgemm_csr<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_spgemm_csr<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_spgemm_csr<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_spgemm_csr<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "sddmm")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_sddmm<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_sddmm<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_sddmm<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_sddmm<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_sddmm<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_sddmm<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_sddmm<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_sddmm<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_sddmm<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_sddmm<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_sddmm<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_sddmm<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "bsric0")
    {
        if(precision == 's')
            testing_bsric0<float>(arg);
        else if(precision == 'd')
            testing_bsric0<double>(arg);
        else if(precision == 'c')
            testing_bsric0<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_bsric0<rocsparse_double_complex>(arg);
    }
    else if(function == "bsrilu0")
    {
        if(precision == 's')
            testing_bsrilu0<float>(arg);
        else if(precision == 'd')
            testing_bsrilu0<double>(arg);
        else if(precision == 'c')
            testing_bsrilu0<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_bsrilu0<rocsparse_double_complex>(arg);
    }
    else if(function == "csric0")
    {
        if(precision == 's')
            testing_csric0<float>(arg);
        else if(precision == 'd')
            testing_csric0<double>(arg);
        else if(precision == 'c')
            testing_csric0<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csric0<rocsparse_double_complex>(arg);
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
    else if(function == "nnz")
    {
        if(precision == 's')
            testing_nnz<float>(arg);
        else if(precision == 'd')
            testing_nnz<double>(arg);
        else if(precision == 'c')
            testing_nnz<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_nnz<rocsparse_double_complex>(arg);
    }
    else if(function == "dense2csr")
    {
        if(precision == 's')
            testing_dense2csr<float>(arg);
        else if(precision == 'd')
            testing_dense2csr<double>(arg);
        else if(precision == 'c')
            testing_dense2csr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_dense2csr<rocsparse_double_complex>(arg);
    }
    else if(function == "dense2coo")
    {
        if(precision == 's')
            testing_dense2coo<float>(arg);
        else if(precision == 'd')
            testing_dense2coo<double>(arg);
        else if(precision == 'c')
            testing_dense2coo<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_dense2coo<rocsparse_double_complex>(arg);
    }
    else if(function == "prune_dense2csr")
    {
        if(precision == 's')
            testing_prune_dense2csr<float>(arg);
        else if(precision == 'd')
            testing_prune_dense2csr<double>(arg);
    }
    else if(function == "prune_dense2csr_by_percentage")
    {
        if(precision == 's')
            testing_prune_dense2csr_by_percentage<float>(arg);
        else if(precision == 'd')
            testing_prune_dense2csr_by_percentage<double>(arg);
    }
    else if(function == "dense2csc")
    {
        if(precision == 's')
            testing_dense2csc<float>(arg);
        else if(precision == 'd')
            testing_dense2csc<double>(arg);
        else if(precision == 'c')
            testing_dense2csc<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_dense2csc<rocsparse_double_complex>(arg);
    }
    else if(function == "csr2dense")
    {
        if(precision == 's')
            testing_csr2dense<float>(arg);
        else if(precision == 'd')
            testing_csr2dense<double>(arg);
        else if(precision == 'c')
            testing_csr2dense<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csr2dense<rocsparse_double_complex>(arg);
    }
    else if(function == "csc2dense")
    {
        if(precision == 's')
            testing_csc2dense<float>(arg);
        else if(precision == 'd')
            testing_csc2dense<double>(arg);
        else if(precision == 'c')
            testing_csc2dense<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csc2dense<rocsparse_double_complex>(arg);
    }
    else if(function == "coo2dense")
    {
        if(precision == 's')
            testing_coo2dense<float>(arg);
        else if(precision == 'd')
            testing_coo2dense<double>(arg);
        else if(precision == 'c')
            testing_coo2dense<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_coo2dense<rocsparse_double_complex>(arg);
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
    else if(function == "gebsr2gebsc")
    {
        if(precision == 's')
            testing_gebsr2gebsc<float>(arg);
        else if(precision == 'd')
            testing_gebsr2gebsc<double>(arg);
        else if(precision == 'c')
            testing_gebsr2gebsc<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gebsr2gebsc<rocsparse_double_complex>(arg);
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
    else if(function == "csr2bsr")
    {
        if(precision == 's')
            testing_csr2bsr<float>(arg);
        else if(precision == 'd')
            testing_csr2bsr<double>(arg);
        else if(precision == 'c')
            testing_csr2bsr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csr2bsr<rocsparse_double_complex>(arg);
    }
    else if(function == "csr2gebsr")
    {
        if(precision == 's')
            testing_csr2gebsr<float>(arg);
        else if(precision == 'd')
            testing_csr2gebsr<double>(arg);
        else if(precision == 'c')
            testing_csr2gebsr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csr2gebsr<rocsparse_double_complex>(arg);
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
    else if(function == "hyb2csr")
    {
        if(precision == 's')
            testing_hyb2csr<float>(arg);
        else if(precision == 'd')
            testing_hyb2csr<double>(arg);
        else if(precision == 'c')
            testing_hyb2csr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_hyb2csr<rocsparse_double_complex>(arg);
    }
    else if(function == "bsr2csr")
    {
        if(precision == 's')
            testing_bsr2csr<float>(arg);
        else if(precision == 'd')
            testing_bsr2csr<double>(arg);
        else if(precision == 'c')
            testing_bsr2csr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_bsr2csr<rocsparse_double_complex>(arg);
    }
    else if(function == "gebsr2csr")
    {
        if(precision == 's')
            testing_gebsr2csr<float>(arg);
        else if(precision == 'd')
            testing_gebsr2csr<double>(arg);
        else if(precision == 'c')
            testing_gebsr2csr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gebsr2csr<rocsparse_double_complex>(arg);
    }
    else if(function == "gebsr2gebsr")
    {
        if(precision == 's')
            testing_gebsr2gebsr<float>(arg);
        else if(precision == 'd')
            testing_gebsr2gebsr<double>(arg);
        else if(precision == 'c')
            testing_gebsr2gebsr<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_gebsr2gebsr<rocsparse_double_complex>(arg);
    }
    else if(function == "csr2csr_compress")
    {
        if(precision == 's')
            testing_csr2csr_compress<float>(arg);
        else if(precision == 'd')
            testing_csr2csr_compress<double>(arg);
        else if(precision == 'c')
            testing_csr2csr_compress<rocsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csr2csr_compress<rocsparse_double_complex>(arg);
    }
    else if(function == "prune_csr2csr")
    {
        if(precision == 's')
            testing_prune_csr2csr<float>(arg);
        else if(precision == 'd')
            testing_prune_csr2csr<double>(arg);
    }
    else if(function == "prune_csr2csr_by_percentage")
    {
        if(precision == 's')
            testing_prune_csr2csr_by_percentage<float>(arg);
        else if(precision == 'd')
            testing_prune_csr2csr_by_percentage<double>(arg);
    }
    else if(function == "dense_to_sparse_coo")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_dense_to_sparse_coo<int32_t, float>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_coo<int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_dense_to_sparse_coo<int32_t, double>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_coo<int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_dense_to_sparse_coo<int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_coo<int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_dense_to_sparse_coo<int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_coo<int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "dense_to_sparse_csr")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csr<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csr<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csr<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csr<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csr<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csr<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csr<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csr<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csr<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csr<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csr<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csr<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "dense_to_sparse_csc")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csc<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csc<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csc<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csc<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csc<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csc<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csc<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csc<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csc<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_dense_to_sparse_csc<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_dense_to_sparse_csc<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_dense_to_sparse_csc<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "sparse_to_dense_coo")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_sparse_to_dense_coo<int32_t, float>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_coo<int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_sparse_to_dense_coo<int32_t, double>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_coo<int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_sparse_to_dense_coo<int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_coo<int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_sparse_to_dense_coo<int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_coo<int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "sparse_to_dense_csr")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csr<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csr<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csr<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csr<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csr<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csr<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csr<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csr<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csr<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csr<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csr<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csr<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
    }
    else if(function == "sparse_to_dense_csc")
    {
        if(precision == 's')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csc<int32_t, int32_t, float>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csc<int64_t, int32_t, float>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csc<int64_t, int64_t, float>(arg);
        }
        else if(precision == 'd')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csc<int32_t, int32_t, double>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csc<int64_t, int32_t, double>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csc<int64_t, int64_t, double>(arg);
        }
        else if(precision == 'c')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csc<int32_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csc<int64_t, int32_t, rocsparse_float_complex>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csc<int64_t, int64_t, rocsparse_float_complex>(arg);
        }
        else if(precision == 'z')
        {
            if(indextype == 's')
                testing_sparse_to_dense_csc<int32_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'm')
                testing_sparse_to_dense_csc<int64_t, int32_t, rocsparse_double_complex>(arg);
            else if(indextype == 'd')
                testing_sparse_to_dense_csc<int64_t, int64_t, rocsparse_double_complex>(arg);
        }
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

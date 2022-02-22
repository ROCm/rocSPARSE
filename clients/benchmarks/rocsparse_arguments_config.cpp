/*! \file */
/* ************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

#include "rocsparse_arguments_config.hpp"

rocsparse_arguments_config::rocsparse_arguments_config()
{
    //
    // Arguments must be a C-compatible struct so cppcheck complains about non-initialized member variables.
    // Then we need to initialize.
    {
        this->M              = static_cast<rocsparse_int>(0);
        this->N              = static_cast<rocsparse_int>(0);
        this->K              = static_cast<rocsparse_int>(0);
        this->nnz            = static_cast<rocsparse_int>(0);
        this->block_dim      = static_cast<rocsparse_int>(0);
        this->row_block_dimA = static_cast<rocsparse_int>(0);
        this->col_block_dimA = static_cast<rocsparse_int>(0);
        this->row_block_dimB = static_cast<rocsparse_int>(0);
        this->col_block_dimB = static_cast<rocsparse_int>(0);
        this->dimx           = static_cast<rocsparse_int>(0);
        this->dimy           = static_cast<rocsparse_int>(0);
        this->dimz           = static_cast<rocsparse_int>(0);
        this->index_type_I   = static_cast<rocsparse_indextype>(0);
        this->index_type_J   = static_cast<rocsparse_indextype>(0);
        this->compute_type   = static_cast<rocsparse_datatype>(0);
        this->alpha          = static_cast<double>(0);
        this->alphai         = static_cast<double>(0);
        this->beta           = static_cast<double>(0);
        this->betai          = static_cast<double>(0);
        this->threshold      = static_cast<double>(0);
        this->percentage     = static_cast<double>(0);
        this->transA         = static_cast<rocsparse_operation>(0);
        this->transB         = static_cast<rocsparse_operation>(0);
        this->baseA          = static_cast<rocsparse_index_base>(0);
        this->baseB          = static_cast<rocsparse_index_base>(0);
        this->baseC          = static_cast<rocsparse_index_base>(0);
        this->baseD          = static_cast<rocsparse_index_base>(0);
        this->action         = static_cast<rocsparse_action>(0);
        this->part           = static_cast<rocsparse_hyb_partition>(0);
        this->matrix_type    = static_cast<rocsparse_matrix_type>(0);
        this->diag           = static_cast<rocsparse_diag_type>(0);
        this->uplo           = static_cast<rocsparse_fill_mode>(0);
        this->storage        = static_cast<rocsparse_storage_mode>(0);
        this->apol           = static_cast<rocsparse_analysis_policy>(0);
        this->spol           = static_cast<rocsparse_solve_policy>(0);
        this->direction      = static_cast<rocsparse_direction>(0);
        this->order          = static_cast<rocsparse_order>(0);
        this->format         = static_cast<rocsparse_format>(0);

        this->sddmm_alg           = rocsparse_sddmm_alg_default;
        this->spmv_alg            = rocsparse_spmv_alg_default;
        this->spsv_alg            = rocsparse_spsv_alg_default;
        this->spsm_alg            = rocsparse_spsm_alg_default;
        this->spmm_alg            = rocsparse_spmm_alg_default;
        this->spgemm_alg          = rocsparse_spgemm_alg_default;
        this->sparse_to_dense_alg = rocsparse_sparse_to_dense_alg_default;
        this->dense_to_sparse_alg = rocsparse_dense_to_sparse_alg_default;

        this->gtsv_interleaved_alg = static_cast<rocsparse_gtsv_interleaved_alg>(0);
        this->gpsv_interleaved_alg = static_cast<rocsparse_gpsv_interleaved_alg>(0);
        this->matrix               = static_cast<rocsparse_matrix_init>(0);
        this->matrix_init_kind     = static_cast<rocsparse_matrix_init_kind>(0);
        this->unit_check           = static_cast<rocsparse_int>(0);
        this->timing               = static_cast<rocsparse_int>(1);
        this->iters                = static_cast<rocsparse_int>(0);
        this->denseld              = static_cast<rocsparse_int>(0);
        this->batch_count          = static_cast<rocsparse_int>(0);
        this->batch_stride         = static_cast<rocsparse_int>(0);
        this->algo                 = static_cast<uint32_t>(0);
        this->numericboost         = static_cast<int>(0);
        this->boosttol             = static_cast<double>(0);
        this->boostval             = static_cast<double>(0);
        this->boostvali            = static_cast<double>(0);
        this->tolm                 = static_cast<double>(0);
        this->filename[0]          = '\0';
        this->function[0]          = '\0';
        this->name[0]              = '\0';
        this->category[0]          = '\0';
    }

    this->precision = 's';
    this->indextype = 's';
}

void rocsparse_arguments_config::set_description(options_description& desc)
{
    desc.add_options()("help,h", "produces this help message")
        // clang-format off
    ("sizem,m",
     value<rocsparse_int>(&this->M)->default_value(128),
     "Specific matrix size testing: sizem is only applicable to SPARSE-2 "
     "& SPARSE-3: the number of rows.")

    ("sizen,n",
     value<rocsparse_int>(&this->N)->default_value(128),
     "Specific matrix/vector size testing: SPARSE-1: the length of the "
     "dense vector. SPARSE-2 & SPARSE-3: the number of columns")

    ("sizek,k",
     value<rocsparse_int>(&this->K)->default_value(128),
     "Specific matrix/vector size testing: SPARSE-3: the number of columns")

    ("sizennz,z",
     value<rocsparse_int>(&this->nnz)->default_value(32),
     "Specific vector size testing, LEVEL-1: the number of non-zero elements "
     "of the sparse vector.")

    ("blockdim",
     value<rocsparse_int>(&this->block_dim)->default_value(2),
     "BSR block dimension (default: 2)")

    ("row-blockdimA",
     value<rocsparse_int>(&this->row_block_dimA)->default_value(2),
     "General BSR row block dimension (default: 2)")

    ("col-blockdimA",
     value<rocsparse_int>(&this->col_block_dimA)->default_value(2),
     "General BSR col block dimension (default: 2)")

    ("row-blockdimB",
     value<rocsparse_int>(&this->row_block_dimB)->default_value(2),
     "General BSR row block dimension (default: 2)")

    ("col-blockdimB",
     value<rocsparse_int>(&this->col_block_dimB)->default_value(2),
     "General BSR col block dimension (default: 2)")

    ("mtx",
     value<std::string>(&this->b_matrixmarket)->default_value(""), "read from matrix "
     "market (.mtx) format. This will override parameters -m, -n, and -z.")

    ("rocalution",
     value<std::string>(&this->b_rocalution)->default_value(""),
     "read from rocalution matrix binary file. This will override parameter --mtx")

    ("rocsparseio",
     value<std::string>(&this->b_rocsparseio)->default_value(""),
     "read from rocsparseio matrix binary file. This will override parameter --rocalution")

    ("file",
     value<std::string>(&this->b_file)->default_value(""),
     "read from file with file extension detection. This will override parameters --rocsparseio,rocalution,mtx")

    ("dimx",
     value<rocsparse_int>(&this->dimx)->default_value(0.0), "assemble "
     "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
     "will override parameters -m, -n, -z and --mtx.")

    ("dimy",
     value<rocsparse_int>(&this->dimy)->default_value(0.0), "assemble "
     "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
     "will override parameters -m, -n, -z and --mtx.")

    ("dimz",
     value<rocsparse_int>(&this->dimz)->default_value(0.0), "assemble "
     "laplacian matrix with dimensions <dimx dimy dimz>. dimz is optional. This "
     "will override parameters -m, -n, -z and --mtx.")

    ("alpha",
     value<double>(&this->alpha)->default_value(1.0), "specifies the scalar alpha")

    ("beta",
     value<double>(&this->beta)->default_value(0.0), "specifies the scalar beta")

    ("threshold",
     value<double>(&this->threshold)->default_value(1.0), "specifies the scalar threshold")

    ("percentage",
     value<double>(&this->percentage)->default_value(0.0), "specifies the scalar percentage")

    ("transposeA",
     value<char>(&this->b_transA)->default_value('N'),
     "N = no transpose, T = transpose, C = conjugate transpose")

    ("transposeB",
     value<char>(&this->b_transB)->default_value('N'),
     "N = no transpose, T = transpose, C = conjugate transpose, (default = N)")

    ("indexbaseA",
     value<int>(&this->b_baseA)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseB",
     value<int>(&this->b_baseB)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseC",
     value<int>(&this->b_baseC)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseD",
     value<int>(&this->b_baseD)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("action",
     value<int>(&this->b_action)->default_value(0),
     "0 = rocsparse_action_numeric, 1 = rocsparse_action_symbolic, (default: 0)")

    ("hybpart",
     value<int>(&this->b_part)->default_value(0),
     "0 = rocsparse_hyb_partition_auto, 1 = rocsparse_hyb_partition_user,\n"
     "2 = rocsparse_hyb_partition_max, (default: 0)")

    ("matrix_type",
     value<int>(&this->b_matrix_type)->default_value(0),
     "0 = rocsparse_matrix_type_general, 1 = rocsparse_matrix_type_symmetric,\n"
     "2 = rocsparse_matrix_type_hermitian, 3 = rocsparse_matrix_type_triangular, (default: 0)")

    ("diag",
     value<char>(&this->b_diag)->default_value('N'),
     "N = non-unit diagonal, U = unit diagonal, (default = N)")

    ("uplo",
     value<char>(&this->b_uplo)->default_value('L'),
     "L = lower fill, U = upper fill, (default = L)")

    ("storage",
     value<int>(&this->b_storage)->default_value(0),
     "0 = rocsparse_storage_mode_sorted, 1 = rocsparse_storage_mode_unsorted, (default = 0)")

    ("apolicy",
     value<char>(&this->b_apol)->default_value('R'),
     "R = reuse meta data, F = force re-build, (default = R)")

    ("function,f",
     value<std::string>(&this->function_name)->default_value("axpyi"),
     "SPARSE function to test. Options:\n"
     "  Level1: axpyi, doti, dotci, gthr, gthrz, roti, sctr\n"
     "  Level2: bsrmv, bsrxmv, bsrsv, coomv, coomv_aos, csrmv, csrmv_managed, csrsv, coosv, ellmv, hybmv, gebsrmv, gemvi\n"
     "  Level3: bsrmm, bsrsm, gebsrmm, csrmm, coomm, csrsm, coosm, gemmi, sddmm\n"
     "  Extra: csrgeam, csrgemm, csrgemm_reuse\n"
     "  Preconditioner: bsric0, bsrilu0, csric0, csrilu0, gtsv, gtsv_no_pivot, gtsv_no_pivot_strided_batch, gtsv_interleaved_batch, gpsv_interleaved_batch\n"
     "  Conversion: csr2coo, csr2csc, gebsr2gebsc, csr2ell, csr2hyb, csr2bsr, csr2gebsr\n"
     "              coo2csr, ell2csr, hyb2csr, dense2csr, dense2coo, prune_dense2csr, prune_dense2csr_by_percentage, dense2csc\n"
     "              csr2dense, csc2dense, coo2dense, bsr2csr, gebsr2csr, gebsr2gebsr, csr2csr_compress, prune_csr2csr, prune_csr2csr_by_percentage\n"
     "              sparse_to_dense_coo, sparse_to_dense_csr, sparse_to_dense_csc, dense_to_sparse_coo, dense_to_sparse_csr, dense_to_sparse_csc\n"
     "  Sorting: cscsort, csrsort, coosort\n"
     "  Misc: identity, nnz")

    ("indextype",
     value<char>(&this->indextype)->default_value('s'),
     "Specify index types to be int32_t (s), int64_t (d) or mixed (m). Options: s,d,m")

    ("precision,r",
     value<char>(&this->precision)->default_value('s'), "Options: s,d,c,z")

    ("verify,v",
     value<rocsparse_int>(&this->unit_check)->default_value(0),
     "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

    ("iters,i",
     value<int>(&this->iters)->default_value(10),
     "Iterations to run inside timing loop")

    ("device,d",
     value<rocsparse_int>(&this->device_id)->default_value(0),
     "Set default device to be used for subsequent program runs")

    ("direction",
     value<rocsparse_int>(&this->b_dir)->default_value(rocsparse_direction_row),
     "Indicates whether a dense matrix should be parsed by rows or by columns, assuming column-major storage: row = 0, column = 1 (default: 0)")

    ("order",
     value<rocsparse_int>(&this->b_order)->default_value(rocsparse_order_column),
     "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

    ("format",
     value<rocsparse_int>(&this->b_format)->default_value(rocsparse_format_coo),
     "Indicates wther a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3 or ell format: 4 (default:0)")

    ("denseld",
     value<rocsparse_int>(&this->denseld)->default_value(128),
     "Indicates the leading dimension of a dense matrix >= M, assuming a column-oriented storage.")

    ("batch_count",
     value<rocsparse_int>(&this->batch_count)->default_value(128),
     "Indicates the batch count for batched routines.")

    ("batch_stride",
     value<rocsparse_int>(&this->batch_stride)->default_value(128),
     "Indicates the batch stride for batched routines.")

    ("gtsv_interleaved_alg",
      value<rocsparse_int>(&this->b_gtsv_interleaved_alg)->default_value(rocsparse_gtsv_interleaved_alg_default),
      "Indicates what algorithm to use when running rocsparse_gtsv_interleaved_batch. Possibly choices are thomas: 1, lu: 2, qr: 3 (default:3)");
}

int rocsparse_arguments_config::parse(int&argc,char**&argv, options_description&desc)
{
  variables_map vm;
  store(parse_command_line(argc, argv, desc,  sizeof(rocsparse_arguments_config)), vm);
  notify(vm);

  if(vm.count("help"))
  {
    std::cout << desc << std::endl;
    return -2;
  }

  if(this->b_dir != rocsparse_direction_row && this->b_dir != rocsparse_direction_column)
  {
    std::cerr << "Invalid value for --direction" << std::endl;
    return -1;
  }

  if(this->b_order != rocsparse_order_row && this->b_order != rocsparse_order_column)
  {
    std::cerr << "Invalid value for --order" << std::endl;
    return -1;
  }

  if(this->b_format != rocsparse_format_csr && this->b_format != rocsparse_format_coo
     && this->b_format != rocsparse_format_coo_aos && this->b_format != rocsparse_format_ell
     && this->b_format != rocsparse_format_csc)
  {
    std::cerr << "Invalid value for --format" << std::endl;
    return -1;
  }

  if(this->indextype != 's' && this->indextype != 'd' && this->indextype != 'm')
  {
    std::cerr << "Invalid value for --indextype" << std::endl;
    return -1;
  }

  if(this->precision != 's' && this->precision != 'd' && this->precision != 'c' && this->precision != 'z')
  {
    std::cerr << "Invalid value for --precision" << std::endl;
    return -1;
  }

  if(this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_default
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_thomas
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_lu
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_qr)
  {
      std::cerr << "Invalid value for --gtsv_interleaved_alg" << std::endl;
      return -1;
  }

  if(this->b_transA == 'N')
  {
    this->transA = rocsparse_operation_none;
  }
  else if(this->b_transA == 'T')
  {
    this->transA = rocsparse_operation_transpose;
  }
  else if(this->b_transA == 'C')
  {
    this->transA = rocsparse_operation_conjugate_transpose;
  }

  if(this->b_transB == 'N')
  {
    this->transB = rocsparse_operation_none;
  }
  else if(this->b_transB == 'T')
  {
    this->transB = rocsparse_operation_transpose;
  }
  else if(this->b_transB == 'C')
  {
    this->transB = rocsparse_operation_conjugate_transpose;
  }
  sprintf(this->function,"%s",this->function_name.c_str());
  this->baseA = (this->b_baseA == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
  this->baseB = (this->b_baseB == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
  this->baseC = (this->b_baseC == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
  this->baseD = (this->b_baseD == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;

  this->action      = (this->b_action == 0) ? rocsparse_action_numeric : rocsparse_action_symbolic;
  this->part        = (this->b_part == 0)   ? rocsparse_hyb_partition_auto
    : (this->b_part == 1) ? rocsparse_hyb_partition_user
    : rocsparse_hyb_partition_max;
  this->matrix_type = (this->b_matrix_type == 0)   ? rocsparse_matrix_type_general
    : (this->b_matrix_type == 1) ? rocsparse_matrix_type_symmetric
    : (this->b_matrix_type == 2) ? rocsparse_matrix_type_hermitian
    : rocsparse_matrix_type_triangular;
  this->diag        = (this->b_diag == 'N') ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit;
  this->uplo        = (this->b_uplo == 'L') ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
  this->storage     = (this->b_storage == 0) ? rocsparse_storage_mode_sorted : rocsparse_storage_mode_unsorted;
  this->apol = (this->b_apol == 'R') ? rocsparse_analysis_policy_reuse : rocsparse_analysis_policy_force;
  this->spol = rocsparse_solve_policy_auto;
  this->direction
    = (this->b_dir == rocsparse_direction_row) ? rocsparse_direction_row : rocsparse_direction_column;
  this->order  = (this->b_order == rocsparse_order_row) ? rocsparse_order_row : rocsparse_order_column;
  this->format = (rocsparse_format)this->b_format;
  this->gtsv_interleaved_alg = (rocsparse_gtsv_interleaved_alg)this->b_gtsv_interleaved_alg;


  // rocALUTION parameter overrides filename parameter
  if(this->b_file != "")
  {
    strcpy(this->filename, this->b_file.c_str());
  }
  else if(this->b_rocsparseio != "")
    {
      strcpy(this->filename, this->b_rocsparseio.c_str());
      this->matrix = rocsparse_matrix_file_rocsparseio;
    }
  else if(this->b_rocalution != "")
  {
    strcpy(this->filename, this->b_rocalution.c_str());
    this->matrix = rocsparse_matrix_file_rocalution;
  }
  else if(this->dimx != 0 && this->dimy != 0 && this->dimz != 0)
  {
    this->matrix = rocsparse_matrix_laplace_3d;
  }
  else if(this->dimx != 0 && this->dimy != 0)
  {
    this->matrix = rocsparse_matrix_laplace_2d;
  }
  else if(this->b_matrixmarket != "")
  {
    strcpy(this->filename, this->b_matrixmarket.c_str());
    this->matrix = rocsparse_matrix_file_mtx;
  }
  else
  {
    this->matrix = rocsparse_matrix_random;
  }

  this->matrix_init_kind = rocsparse_matrix_init_kind_default;
  /* ============================================================================================
   */
  if(this->M < 0 || this->N < 0)
  {
    std::cerr << "Invalid dimension" << std::endl;
    return -1;
  }

  if(this->block_dim < 1)
  {
    std::cerr << "Invalid value for --blockdim" << std::endl;
    return -1;
  }

  if(this->row_block_dimA < 1)
  {
    std::cerr << "Invalid value for --row-blockdimA" << std::endl;
    return -1;
  }

  if(this->col_block_dimA < 1)
  {
    std::cerr << "Invalid value for --col-blockdimA" << std::endl;
    return -1;
  }

  if(this->row_block_dimB < 1)
  {
    std::cerr << "Invalid value for --row-blockdimB" << std::endl;
    return -1;
  }

  if(this->col_block_dimB < 1)
  {
    std::cerr << "Invalid value for --col-blockdimB" << std::endl;
    return -1;
  }
  return 0;
}

int rocsparse_arguments_config::parse_no_default(int&argc,char**&argv, options_description&desc)
{
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if(vm.count("help"))
  {
    std::cout << desc << std::endl;
    return -2;
  }

  if(this->b_dir != rocsparse_direction_row && this->b_dir != rocsparse_direction_column)
  {
    std::cerr << "Invalid value for --direction" << std::endl;
    return -1;
  }

  if(this->b_order != rocsparse_order_row && this->b_order != rocsparse_order_column)
  {
    std::cerr << "Invalid value for --order" << std::endl;
    return -1;
  }

  if(this->b_format != rocsparse_format_csr && this->b_format != rocsparse_format_coo
     && this->b_format != rocsparse_format_coo_aos && this->b_format != rocsparse_format_ell
     && this->b_format != rocsparse_format_csc)
  {
    std::cerr << "Invalid value for --format" << std::endl;
    return -1;
  }

  if(this->indextype != 's' && this->indextype != 'd' && this->indextype != 'm')
  {
    std::cerr << "Invalid value for --indextype" << std::endl;
    return -1;
  }

  if(this->precision != 's' && this->precision != 'd' && this->precision != 'c' && this->precision != 'z')
  {
    std::cerr << "Invalid value for --precision" << std::endl;
    return -1;
  }

  if(this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_default
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_thomas
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_lu
       && this->b_gtsv_interleaved_alg != rocsparse_gtsv_interleaved_alg_qr)
  {
      std::cerr << "Invalid value for --gtsv_interleaved_alg" << std::endl;
      return -1;
  }

  if(b_transA == 'N')
  {
    this->transA = rocsparse_operation_none;
  }
  else if(b_transA == 'T')
  {
    this->transA = rocsparse_operation_transpose;
  }
  else if(b_transA == 'C')
  {
    this->transA = rocsparse_operation_conjugate_transpose;
  }

  if(b_transB == 'N')
  {
    this->transB = rocsparse_operation_none;
  }
  else if(b_transB == 'T')
  {
    this->transB = rocsparse_operation_transpose;
  }
  else if(b_transB == 'C')
  {
    this->transB = rocsparse_operation_conjugate_transpose;
  }

  sprintf(this->function,"%s",this->function_name.c_str());
  this->baseA = (b_baseA == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
  this->baseB = (b_baseB == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
  this->baseC = (b_baseC == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;
  this->baseD = (b_baseD == 0) ? rocsparse_index_base_zero : rocsparse_index_base_one;

  this->action = (b_action == 0) ? rocsparse_action_numeric : rocsparse_action_symbolic;
  this->part   = (b_part == 0)   ? rocsparse_hyb_partition_auto
    : (b_part == 1) ? rocsparse_hyb_partition_user
    : rocsparse_hyb_partition_max;
  this->diag   = (b_diag == 'N') ? rocsparse_diag_type_non_unit : rocsparse_diag_type_unit;
  this->uplo   = (b_uplo == 'L') ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper;
  this->storage= (b_storage == 0) ? rocsparse_storage_mode_sorted : rocsparse_storage_mode_unsorted;
  this->apol   = (b_apol == 'R') ? rocsparse_analysis_policy_reuse : rocsparse_analysis_policy_force;
  this->spol   = rocsparse_solve_policy_auto;
  this->direction
    = (b_dir == rocsparse_direction_row) ? rocsparse_direction_row : rocsparse_direction_column;
  this->order  = (b_order == rocsparse_order_row) ? rocsparse_order_row : rocsparse_order_column;
  this->format = (rocsparse_format)b_format;
  this->gtsv_interleaved_alg = (rocsparse_gtsv_interleaved_alg)this->b_gtsv_interleaved_alg;

  // rocALUTION parameter overrides filename parameter
  if(b_file != "")
  {
    strcpy(this->filename, b_file.c_str());
    const char * p = b_file.c_str();
    const char * q = nullptr;
    while(*p!='\0') { if (*p=='.') q = p; ++p;}
    if (q!=nullptr)
      {
	std::cerr << "extension is not detected in filename '"<< b_file <<"' " << std::endl;
	return -1;
      }
    if (!strcmp(q,".mtx"))
      {
	this->matrix = rocsparse_matrix_file_mtx;
      }
    else if (!strcmp(q,".bin"))
      {
	this->matrix = rocsparse_matrix_file_rocsparseio;
      }
    else if (!strcmp(q,".csr"))
      {
	this->matrix = rocsparse_matrix_file_rocalution;
      }
  }
  else if(b_rocsparseio != "")
  {
    strcpy(this->filename, b_rocsparseio.c_str());
    this->matrix = rocsparse_matrix_file_rocsparseio;
  }
  else if(b_rocalution != "")
  {
    strcpy(this->filename, b_rocalution.c_str());
    this->matrix = rocsparse_matrix_file_rocalution;
  }
  else if(this->dimx != 0 && this->dimy != 0 && this->dimz != 0)
  {
    this->matrix = rocsparse_matrix_laplace_3d;
  }
  else if(this->dimx != 0 && this->dimy != 0)
  {
    this->matrix = rocsparse_matrix_laplace_2d;
  }
  else if(b_matrixmarket != "")
  {
    strcpy(this->filename, b_matrixmarket.c_str());
    this->matrix = rocsparse_matrix_file_mtx;
  }
  else
  {
    this->matrix = rocsparse_matrix_random;
  }

  this->matrix_init_kind = rocsparse_matrix_init_kind_default;
  /* ============================================================================================
   */
  if(this->M < 0 || this->N < 0)
  {
    std::cerr << "Invalid dimension" << std::endl;
    return -1;
  }

  if(this->block_dim < 1)
  {
    std::cerr << "Invalid value for --blockdim" << std::endl;
    return -1;
  }

  if(this->row_block_dimA < 1)
  {
    std::cerr << "Invalid value for --row-blockdimA" << std::endl;
    return -1;
  }

  if(this->col_block_dimA < 1)
  {
    std::cerr << "Invalid value for --col-blockdimA" << std::endl;
    return -1;
  }

  if(this->row_block_dimB < 1)
  {
    std::cerr << "Invalid value for --row-blockdimB" << std::endl;
    return -1;
  }

  if(this->col_block_dimB < 1)
  {
    std::cerr << "Invalid value for --col-blockdimB" << std::endl;
    return -1;
  }
  return 0;
}




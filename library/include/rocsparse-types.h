/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

/*! \file
 * \brief rocsparse-types.h defines data types used by rocsparse
 */

#pragma once
#ifndef _ROCSPARSE_TYPES_H_
#define _ROCSPARSE_TYPES_H_

#include "rocsparse-complex-types.h"

#include <stddef.h>
#include <stdint.h>

/*! \ingroup types_module
 *  \brief Specifies whether int32 or int64 is used.
 */
#if defined(rocsparse_ILP64)
typedef int64_t rocsparse_int;
#else
typedef int32_t rocsparse_int;
#endif

/* Forward declaration of hipStream_t */
typedef struct ihipStream_t* hipStream_t;

/*! \ingroup types_module
 *  \brief Handle to the rocSPARSE library context queue.
 *
 *  \details
 *  The rocSPARSE handle is a structure holding the rocSPARSE library context. It must
 *  be initialized using rocsparse_create_handle() and the returned handle must be
 *  passed to all subsequent library function calls. It should be destroyed at the end
 *  using rocsparse_destroy_handle().
 */
typedef struct _rocsparse_handle* rocsparse_handle;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The rocSPARSE matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using rocsparse_create_mat_descr() and the returned
 *  descriptor must be passed to all subsequent library calls that involve the matrix.
 *  It should be destroyed at the end using rocsparse_destroy_mat_descr().
 */
typedef struct _rocsparse_mat_descr* rocsparse_mat_descr;

/*! \ingroup types_module
 *  \brief HYB matrix storage format.
 *
 *  \details
 *  The rocSPARSE HYB matrix structure holds the HYB matrix. It must be initialized using
 *  rocsparse_create_hyb_mat() and the returned HYB matrix must be passed to all
 *  subsequent library calls that involve the matrix. It should be destroyed at the end
 *  using rocsparse_destroy_hyb_mat().
 */
typedef struct _rocsparse_hyb_mat* rocsparse_hyb_mat;

/*! \ingroup types_module
 *  \brief Info structure to hold all matrix meta data.
 *
 *  \details
 *  The rocSPARSE matrix info is a structure holding all matrix information that is
 *  gathered during analysis routines. It must be initialized using
 *  rocsparse_create_mat_info() and the returned info structure must be passed to all
 *  subsequent library calls that require additional matrix information. It should be
 *  destroyed at the end using rocsparse_destroy_mat_info().
 */
typedef struct _rocsparse_mat_info* rocsparse_mat_info;

// Generic API
typedef struct _rocsparse_spvec_descr* rocsparse_spvec_descr;
typedef struct _rocsparse_spmat_descr* rocsparse_spmat_descr;
typedef struct _rocsparse_dnvec_descr* rocsparse_dnvec_descr;
typedef struct _rocsparse_dnmat_descr* rocsparse_dnmat_descr;

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup types_module
 *  \brief Specify whether the matrix is to be transposed or not.
 *
 *  \details
 *  The \ref rocsparse_operation indicates the operation performed with the given matrix.
 */
typedef enum rocsparse_operation_
{
    rocsparse_operation_none                = 111, /**< Operate with matrix. */
    rocsparse_operation_transpose           = 112, /**< Operate with transpose. */
    rocsparse_operation_conjugate_transpose = 113 /**< Operate with conj. transpose. */
} rocsparse_operation;

/*! \ingroup types_module
 *  \brief Specify the matrix index base.
 *
 *  \details
 *  The \ref rocsparse_index_base indicates the index base of the indices. For a
 *  given \ref rocsparse_mat_descr, the \ref rocsparse_index_base can be set using
 *  rocsparse_set_mat_index_base(). The current \ref rocsparse_index_base of a matrix
 *  can be obtained by rocsparse_get_mat_index_base().
 */
typedef enum rocsparse_index_base_
{
    rocsparse_index_base_zero = 0, /**< zero based indexing. */
    rocsparse_index_base_one  = 1 /**< one based indexing. */
} rocsparse_index_base;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 *  The \ref rocsparse_matrix_type indices the type of a matrix. For a given
 *  \ref rocsparse_mat_descr, the \ref rocsparse_matrix_type can be set using
 *  rocsparse_set_mat_type(). The current \ref rocsparse_matrix_type of a matrix can be
 *  obtained by rocsparse_get_mat_type().
 */
typedef enum rocsparse_matrix_type_
{
    rocsparse_matrix_type_general    = 0, /**< general matrix type. */
    rocsparse_matrix_type_symmetric  = 1, /**< symmetric matrix type. */
    rocsparse_matrix_type_hermitian  = 2, /**< hermitian matrix type. */
    rocsparse_matrix_type_triangular = 3 /**< triangular matrix type. */
} rocsparse_matrix_type;

/*! \ingroup types_module
 *  \brief Indicates if the diagonal entries are unity.
 *
 *  \details
 *  The \ref rocsparse_diag_type indicates whether the diagonal entries of a matrix are
 *  unity or not. If \ref rocsparse_diag_type_unit is specified, all present diagonal
 *  values will be ignored. For a given \ref rocsparse_mat_descr, the
 *  \ref rocsparse_diag_type can be set using rocsparse_set_mat_diag_type(). The current
 *  \ref rocsparse_diag_type of a matrix can be obtained by
 *  rocsparse_get_mat_diag_type().
 */
typedef enum rocsparse_diag_type_
{
    rocsparse_diag_type_non_unit = 0, /**< diagonal entries are non-unity. */
    rocsparse_diag_type_unit     = 1 /**< diagonal entries are unity */
} rocsparse_diag_type;

/*! \ingroup types_module
 *  \brief Specify the matrix fill mode.
 *
 *  \details
 *  The \ref rocsparse_fill_mode indicates whether the lower or the upper part is stored
 *  in a sparse triangular matrix. For a given \ref rocsparse_mat_descr, the
 *  \ref rocsparse_fill_mode can be set using rocsparse_set_mat_fill_mode(). The current
 *  \ref rocsparse_fill_mode of a matrix can be obtained by
 *  rocsparse_get_mat_fill_mode().
 */
typedef enum rocsparse_fill_mode_
{
    rocsparse_fill_mode_lower = 0, /**< lower triangular part is stored. */
    rocsparse_fill_mode_upper = 1 /**< upper triangular part is stored. */
} rocsparse_fill_mode;

/*! \ingroup types_module
 *  \brief Specify where the operation is performed on.
 *
 *  \details
 *  The \ref rocsparse_action indicates whether the operation is performed on the full
 *  matrix, or only on the sparsity pattern of the matrix.
 */
typedef enum rocsparse_action_
{
    rocsparse_action_symbolic = 0, /**< Operate only on indices. */
    rocsparse_action_numeric  = 1 /**< Operate on data and indices. */
} rocsparse_action;

/*! \ingroup types_module
 *  \brief Specify the matrix direction.
 *
 *  \details
 *  The \ref rocsparse_direction indicates whether a dense matrix should be parsed by
 *  rows or by columns, assuming column-major storage.
 */
typedef enum rocsparse_direction_
{
    rocsparse_direction_row    = 0, /**< Parse the matrix by rows. */
    rocsparse_direction_column = 1 /**< Parse the matrix by columns. */
} rocsparse_direction;

/*! \ingroup types_module
 *  \brief HYB matrix partitioning type.
 *
 *  \details
 *  The \ref rocsparse_hyb_partition type indicates how the hybrid format partitioning
 *  between COO and ELL storage formats is performed.
 */
typedef enum rocsparse_hyb_partition_
{
    rocsparse_hyb_partition_auto = 0, /**< automatically decide on ELL nnz per row. */
    rocsparse_hyb_partition_user = 1, /**< user given ELL nnz per row. */
    rocsparse_hyb_partition_max  = 2 /**< max ELL nnz per row, no COO part. */
} rocsparse_hyb_partition;

/*! \ingroup types_module
 *  \brief Specify policy in analysis functions.
 *
 *  \details
 *  The \ref rocsparse_analysis_policy specifies whether gathered analysis data should be
 *  re-used or not. If meta data from a previous e.g. rocsparse_csrilu0_analysis() call
 *  is available, it can be re-used for subsequent calls to e.g.
 *  rocsparse_csrsv_analysis() and greatly improve performance of the analysis function.
 */
typedef enum rocsparse_analysis_policy_
{
    rocsparse_analysis_policy_reuse = 0, /**< try to re-use meta data. */
    rocsparse_analysis_policy_force = 1 /**< force to re-build meta data. */
} rocsparse_analysis_policy;

/*! \ingroup types_module
 *  \brief Specify policy in triangular solvers and factorizations.
 *
 *  \details
 *  This is a placeholder.
 */
typedef enum rocsparse_solve_policy_
{
    rocsparse_solve_policy_auto = 0 /**< automatically decide on level information. */
} rocsparse_solve_policy;

/*! \ingroup types_module
 *  \brief Indicates if the pointer is device pointer or host pointer.
 *
 *  \details
 *  The \ref rocsparse_pointer_mode indicates whether scalar values are passed by
 *  reference on the host or device. The \ref rocsparse_pointer_mode can be changed by
 *  rocsparse_set_pointer_mode(). The currently used pointer mode can be obtained by
 *  rocsparse_get_pointer_mode().
 */
typedef enum rocsparse_pointer_mode_
{
    rocsparse_pointer_mode_host   = 0, /**< scalar pointers are in host memory. */
    rocsparse_pointer_mode_device = 1 /**< scalar pointers are in device memory. */
} rocsparse_pointer_mode;

/*! \ingroup types_module
 *  \brief Indicates if layer is active with bitmask.
 *
 *  \details
 *  The \ref rocsparse_layer_mode bit mask indicates the logging characteristics.
 */
typedef enum rocsparse_layer_mode
{
    rocsparse_layer_mode_none      = 0x0, /**< layer is not active. */
    rocsparse_layer_mode_log_trace = 0x1, /**< layer is in logging mode. */
    rocsparse_layer_mode_log_bench = 0x2 /**< layer is in benchmarking mode. */
} rocsparse_layer_mode;

/*! \ingroup types_module
 *  \brief List of rocsparse status codes definition.
 *
 *  \details
 *  This is a list of the \ref rocsparse_status types that are used by the rocSPARSE
 *  library.
 */
typedef enum rocsparse_status_
{
    rocsparse_status_success         = 0, /**< success. */
    rocsparse_status_invalid_handle  = 1, /**< handle not initialized, invalid or null. */
    rocsparse_status_not_implemented = 2, /**< function is not implemented. */
    rocsparse_status_invalid_pointer = 3, /**< invalid pointer parameter. */
    rocsparse_status_invalid_size    = 4, /**< invalid size parameter. */
    rocsparse_status_memory_error    = 5, /**< failed memory allocation, copy, dealloc. */
    rocsparse_status_internal_error  = 6, /**< other internal library failure. */
    rocsparse_status_invalid_value   = 7, /**< invalid value parameter. */
    rocsparse_status_arch_mismatch   = 8, /**< device arch is not supported. */
    rocsparse_status_zero_pivot      = 9, /**< encountered zero pivot. */
    rocsparse_status_not_initialized = 10, /**< descriptor has not been initialized. */
    rocsparse_status_type_mismatch   = 11 /**< index types do not match. */
} rocsparse_status;

/*! \ingroup types_module
 *  \brief List of rocsparse index types.
 *
 *  \details
 *  Indicates the index width of a rocsparse index type.
 */
typedef enum rocsparse_indextype_
{
    rocsparse_indextype_u16 = 1, /**< 16 bit unsigned integer. */
    rocsparse_indextype_i32 = 2, /**< 32 bit signed integer. */
    rocsparse_indextype_i64 = 3 /**< 64 bit signed integer. */
} rocsparse_indextype;

/*! \ingroup types_module
 *  \brief List of rocsparse data types.
 *
 *  \details
 *  Indicates the precision width of data stored in a rocsparse type.
 */
typedef enum rocsparse_datatype_
{
    rocsparse_datatype_f32_r = 151, /**< 32 bit floating point, real. */
    rocsparse_datatype_f64_r = 152, /**< 64 bit floating point, real. */
    rocsparse_datatype_f32_c = 154, /**< 32 bit floating point, complex. */
    rocsparse_datatype_f64_c = 155 /**< 64 bit floating point, complex. */
} rocsparse_datatype;

/*! \ingroup types_module
 *  \brief List of sparse matrix formats.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_format types that are used to describe a
 *  sparse matrix.
 */
typedef enum rocsparse_format_
{
    rocsparse_format_coo     = 0, /**< COO sparse matrix format. */
    rocsparse_format_coo_aos = 1, /**< COO AoS sparse matrix format. */
    rocsparse_format_csr     = 2, /**< CSR sparse matrix format. */
    rocsparse_format_csc     = 3, /**< CSC sparse matrix format. */
    rocsparse_format_ell     = 4 /**< ELL sparse matrix format. */
} rocsparse_format;

/*! \ingroup types_module
 *  \brief List of dense matrix ordering.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_order types that are used to describe the
 *  memory layout of a dense matrix
 */
typedef enum rocsparse_order_
{
    rocsparse_order_row    = 0, /**< Row major. */
    rocsparse_order_column = 1 /**< Column major. */
} rocsparse_order;

/*! \ingroup types_module
 *  \brief List of SpMV algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spmv_alg types that are used to perform
 *  matrix vector product.
 */
typedef enum rocsparse_spmv_alg_
{
    rocsparse_spmv_alg_default      = 0, /**< Default SpMV algorithm for the given format. */
    rocsparse_spmv_alg_coo          = 1, /**< COO SpMV algorithm for COO matrices. */
    rocsparse_spmv_alg_csr_adaptive = 2, /**< CSR SpMV algorithm 1 (adaptive) for CSR matrices. */
    rocsparse_spmv_alg_csr_stream   = 3, /**< CSR SpMV algorithm 2 (stream) for CSR matrices. */
    rocsparse_spmv_alg_ell          = 4 /**< ELL SpMV algorithm for ELL matrices. */
} rocsparse_spmv_alg;

/*! \ingroup types_module
*  \brief List of SpMM algorithms.
*
*  \details
*  This is a list of supported \ref rocsparse_spmm_alg types that are used to perform
*  matrix vector product.
*/
typedef enum rocsparse_spmm_alg_
{
    rocsparse_spmm_alg_default = 0, /**< Default SpMM algorithm for the given format. */
    rocsparse_spmm_alg_csr     = 1, /**< SpMM algorithm for CSR format. */
    rocsparse_spmm_alg_coo_segmented
    = 2, /**< SpMM algorithm for COO format using segmented scan. */
    rocsparse_spmm_alg_coo_atomic = 3 /**< SpMM algorithm for COO format using atomics. */
} rocsparse_spmm_alg;

/*! \ingroup types_module
 *  \brief List of sparse to dense algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_sparse_to_dense_alg types that are used to perform
 *  sparse to dense conversion.
 */
typedef enum rocsparse_sparse_to_dense_alg_
{
    rocsparse_sparse_to_dense_alg_default
    = 0, /**< Default sparse to dense algorithm for the given format. */
} rocsparse_sparse_to_dense_alg;

/*! \ingroup types_module
 *  \brief List of dense to sparse algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_dense_tosparse_alg types that are used to perform
 *  dense to sparse conversion.
 */
typedef enum rocsparse_dense_to_sparse_alg_
{
    rocsparse_dense_to_sparse_alg_default
    = 0, /**< Default dense to sparse algorithm for the given format. */
} rocsparse_dense_to_sparse_alg;

/*! \ingroup types_module
 *  \brief List of SpGEMM stages.
 *
 *  \details
 *  This is a list of possible stages during SpGEMM computation. Typical order is
 *  rocsparse_spgemm_buffer_size, rocsparse_spgemm_nnz, rocsparse_spgemm_compute.
 */
typedef enum rocsparse_spgemm_stage_
{
    rocsparse_spgemm_stage_auto        = 0, /**< Automatic stage detection. */
    rocsparse_spgemm_stage_buffer_size = 1, /**< Returns the required buffer size. */
    rocsparse_spgemm_stage_nnz         = 2, /**< Computes number of non-zero entries. */
    rocsparse_spgemm_stage_compute     = 3 /**< Performs the actual SpGEMM computation. */
} rocsparse_spgemm_stage;

/*! \ingroup types_module
 *  \brief List of SpGEMM algorithms.
 *
 *  \details
 *  This is a list of supported \ref rocsparse_spgemm_alg types that are used to perform
 *  sparse matrix sparse matrix product.
 */
typedef enum rocsparse_spgemm_alg_
{
    rocsparse_spgemm_alg_default = 0 /**< Default SpGEMM algorithm for the given format. */
} rocsparse_spgemm_alg;

#ifdef __cplusplus
}
#endif

#endif /* _ROCSPARSE_TYPES_H_ */

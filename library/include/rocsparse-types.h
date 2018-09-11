/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

/*! \file
 * \brief rocsparse-types.h defines data types used by rocsparse
 */

#pragma once
#ifndef _ROCSPARSE_TYPES_H_
#define _ROCSPARSE_TYPES_H_

#include <stdint.h>

/*! \brief Specifies whether int32 or int64 is used. */
#if defined(rocsparse_ILP64)
typedef int64_t rocsparse_int;
#else
typedef int32_t rocsparse_int;
#endif



/*! \brief Handle to the rocSPARSE library context queue. */
typedef struct _rocsparse_handle* rocsparse_handle;
/*! \brief Descriptor of the matrix. */
typedef struct _rocsparse_mat_descr* rocsparse_mat_descr;
/*! \brief HYB matrix storage format. */
typedef struct _rocsparse_hyb_mat* rocsparse_hyb_mat;
/*! \brief Info structure to hold all matrix meta data. */
typedef struct _rocsparse_mat_info* rocsparse_mat_info;

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Specify whether the matrix is to be transposed or not. */
typedef enum rocsparse_operation_ {
    rocsparse_operation_none                = 111, /**< Operate with matrix. */
    rocsparse_operation_transpose           = 112, /**< Operate with transpose. */
    rocsparse_operation_conjugate_transpose = 113  /**< Operate with conj. transpose. */
} rocsparse_operation;

/*! \brief Specify the matrix index base. */
typedef enum rocsparse_index_base_ {
    rocsparse_index_base_zero = 0, /**< zero based indexing. */
    rocsparse_index_base_one  = 1  /**< one based indexing. */
} rocsparse_index_base;

/*! \brief Specify the matrix type. */
typedef enum rocsparse_matrix_type_ {
    rocsparse_matrix_type_general    = 0, /**< general matrix type. */
    rocsparse_matrix_type_symmetric  = 1, /**< symmetric matrix type. */
    rocsparse_matrix_type_hermitian  = 2, /**< hermitian matrix type. */
    rocsparse_matrix_type_triangular = 3  /**< triangular matrix type. */
} rocsparse_matrix_type;

/*! \brief Indicates if the diagonal entries are unity. */
typedef enum rocsparse_diag_type_ {
    rocsparse_diag_type_non_unit = 0, /**< diagonal entries are non-unity. */
    rocsparse_diag_type_unit     = 1  /**< diagonal entries are unity */
} rocsparse_diag_type;

/*! \brief Specify the matrix fill mode. */
typedef enum rocsparse_fill_mode_ {
    rocsparse_fill_mode_lower = 0, /**< lower triangular part is stored. */
    rocsparse_fill_mode_upper = 1  /**< upper triangular part is stored. */
} rocsparse_fill_mode;

/*! \brief Specify where the operation is performed on. */
typedef enum rocsparse_action_ {
    rocsparse_action_symbolic = 0, /**< Operate only on indices. */
    rocsparse_action_numeric  = 1  /**< Operate on data and indices. */
} rocsparse_action;

/*! \brief HYB matrix partitioning type. */
typedef enum rocsparse_hyb_partition_ {
    rocsparse_hyb_partition_auto = 0, /**< automatically decide on ELL nnz per row. */
    rocsparse_hyb_partition_user = 1, /**< user given ELL nnz per row. */
    rocsparse_hyb_partition_max  = 2  /**< max ELL nnz per row, no COO part. */
} rocsparse_hyb_partition;

/*! \brief Specify policy in triangular solvers and factorizations. */
typedef enum rocsparse_solve_policy_ {
    rocsparse_solve_policy_auto = 0 /**< automatically decide on level information. */
} rocsparse_solve_policy;

/*! \brief Specify policy in analysis functions. */
typedef enum rocsparse_analysis_policy_ {
    rocsparse_analysis_policy_reuse = 0, /**< try to re-use meta data. */
    rocsparse_analysis_policy_force = 1  /**< force to re-build meta data. */
} rocsparse_analysis_policy;

/*! \brief Indicates if the pointer is device pointer or host pointer. */
typedef enum rocsparse_pointer_mode_ {
    rocsparse_pointer_mode_host   = 0, /**< scalar pointers are in host memory. */
    rocsparse_pointer_mode_device = 1  /**< scalar pointers are in device memory. */
} rocsparse_pointer_mode;

/*! \brief Indicates if layer is active with bitmask. */
typedef enum rocsparse_layer_mode {
    rocsparse_layer_mode_none      = 0b0000000000, /**< layer is not active. */
    rocsparse_layer_mode_log_trace = 0b0000000001, /**< layer is in logging mode. */
    rocsparse_layer_mode_log_bench = 0b0000000010, /**< layer is in benchmarking mode. */
} rocsparse_layer_mode;

/*! \brief rocsparse status codes definition. */
typedef enum rocsparse_status_ {
    rocsparse_status_success         = 0, /**< success. */
    rocsparse_status_invalid_handle  = 1, /**< handle not initialized, invalid or null. */
    rocsparse_status_not_implemented = 2, /**< function is not implemented. */
    rocsparse_status_invalid_pointer = 3, /**< invalid pointer parameter. */
    rocsparse_status_invalid_size    = 4, /**< invalid size parameter. */
    rocsparse_status_memory_error    = 5, /**< failed memory allocation, copy, dealloc. */
    rocsparse_status_internal_error  = 6, /**< other internal library failure. */
    rocsparse_status_invalid_value   = 7, /**< invalid value parameter. */
    rocsparse_status_arch_mismatch   = 8, /**< device arch is not supported. */
    rocsparse_status_zero_pivot      = 9  /**< encountered zero pivot. */
} rocsparse_status;

#ifdef __cplusplus
}
#endif

#endif // _ROCSPARSE_TYPES_H_

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

/*! \brief To specify whether int32 or int64 is used
 */
#if defined(rocsparse_ILP64)
typedef int64_t rocsparse_int;
#else
typedef int32_t rocsparse_int;
#endif

typedef struct _rocsparse_handle* rocsparse_handle;
typedef struct _rocsparse_mat_descr* rocsparse_mat_descr;
typedef struct _rocsparse_hyb_mat* rocsparse_hyb_mat;
typedef struct _rocsparse_mat_info* rocsparse_mat_info;

#ifdef __cplusplus
extern "C" {
#endif

/* ==================================================================================== */

/*! parameter constants. */

/*! \brief Used to specify whether the matrix is to be transposed or not. */
typedef enum rocsparse_operation_ {
    rocsparse_operation_none                = 111, /**< Operate with matrix. */
    rocsparse_operation_transpose           = 112, /**< Operate with transpose. */
    rocsparse_operation_conjugate_transpose = 113  /**< Operate with conj. transpose. */
} rocsparse_operation;

/*! \brief Used to specify the matrix index base. */
typedef enum rocsparse_index_base_ {
    rocsparse_index_base_zero = 0, /**< zero based indexing. */
    rocsparse_index_base_one  = 1  /**< one based indexing. */
} rocsparse_index_base;

/*! \brief Used to specify the matrix type. */
typedef enum rocsparse_matrix_type_ {
    rocsparse_matrix_type_general    = 0, /**< general matrix type. */
    rocsparse_matrix_type_symmetric  = 1, /**< symmetric matrix type. */
    rocsparse_matrix_type_hermitian  = 2, /**< hermitian matrix type. */
    rocsparse_matrix_type_triangular = 3  /**< triangular matrix type. */
} rocsparse_matrix_type;

/*! \brief Used to specify where the operation is performed on. */
typedef enum rocsparse_action_ {
    rocsparse_action_symbolic = 0, /**< Operate only on indices. */
    rocsparse_action_numeric  = 1  /**< Operate on data and indices. */
} rocsparse_action;

/*! \brief HYB matrix partition type. */
typedef enum rocsparse_hyb_partition_ {
    rocsparse_hyb_partition_auto = 0, /**< automatically decide on ELL nnz per row. */
    rocsparse_hyb_partition_user = 1, /**< user given ELL nnz per row. */
    rocsparse_hyb_partition_max  = 2  /**< max ELL nnz per row, no COO part. */
} rocsparse_hyb_partition;

/* ==================================================================================== */
/**
 *   @brief rocsparse status codes definition.
 */
typedef enum rocsparse_status_ {
    rocsparse_status_success         = 0, /**< success. */
    rocsparse_status_invalid_handle  = 1, /**< handle not initialized, invalid or null. */
    rocsparse_status_not_implemented = 2, /**< function is not implemented. */
    rocsparse_status_invalid_pointer = 3, /**< invalid pointer parameter. */
    rocsparse_status_invalid_size    = 4, /**< invalid size parameter. */
    rocsparse_status_memory_error    = 5, /**< failed memory allocation, copy, dealloc. */
    rocsparse_status_internal_error  = 6, /**< other internal library failure. */
    rocsparse_status_invalid_value   = 7, /**< invalid value parameter. */
    rocsparse_status_arch_mismatch   = 8  /**< device arch is not supported. */
} rocsparse_status;

/*! \brief Indicates the pointer is device pointer or host pointer. */
typedef enum rocsparse_pointer_mode_ {
    rocsparse_pointer_mode_host   = 0, /**< scalar pointers are in host memory. */
    rocsparse_pointer_mode_device = 1  /**< scalar pointers are in device memory. */
} rocsparse_pointer_mode;

/*! \brief Indicates if layer is active with bitmask.*/
typedef enum rocsparse_layer_mode {
    rocsparse_layer_mode_none      = 0b0000000000, /**< layer is not active. */
    rocsparse_layer_mode_log_trace = 0b0000000001, /**< layer is in logging mode. */
    rocsparse_layer_mode_log_bench = 0b0000000010, /**< layer is in benchmarking mode. */
} rocsparse_layer_mode;

#ifdef __cplusplus
}
#endif

#endif // _ROCSPARSE_TYPES_H_

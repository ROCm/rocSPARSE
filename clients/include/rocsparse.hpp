/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _ROCSPARSE_HPP_
#define _ROCSPARSE_HPP_

#include <rocsparse.h>

namespace rocsparse {

template <typename T>
rocsparse_status rocsparse_axpyi(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const T* alpha,
                                 const T* x_val,
                                 const rocsparse_int* x_ind,
                                 T* y,
                                 rocsparse_index_base idx_base);

template <typename T>
rocsparse_status rocsparse_coomv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const T* coo_val,
                                 const rocsparse_int* coo_row_ind,
                                 const rocsparse_int* coo_col_ind,
                                 const T* x,
                                 const T* beta,
                                 T* y);
template <typename T>
rocsparse_status rocsparse_csrmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const T* csr_val,
                                 const rocsparse_int* csr_row_ptr,
                                 const rocsparse_int* csr_col_ind,
                                 const T* x,
                                 const T* beta,
                                 T* y);

template <typename T>
rocsparse_status rocsparse_hybmv(rocsparse_handle handle,
                                 rocsparse_operation trans,
                                 const T* alpha,
                                 const rocsparse_mat_descr descr,
                                 const rocsparse_hyb_mat hyb,
                                 const T* x,
                                 const T* beta,
                                 T* y);

template <typename T>
rocsparse_status rocsparse_csr2ell(rocsparse_handle handle,
                                   rocsparse_int m,
                                   const rocsparse_mat_descr csr_descr,
                                   const T* csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int ell_width,
                                   T* ell_val,
                                   rocsparse_int* ell_col_ind);

template <typename T>
rocsparse_status rocsparse_csr2hyb(rocsparse_handle handle,
                                   rocsparse_int m,
                                   rocsparse_int n,
                                   const rocsparse_mat_descr descr,
                                   const T* csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   rocsparse_hyb_mat hyb,
                                   rocsparse_int user_ell_width,
                                   rocsparse_hyb_partition partition_type);
}

#endif // _ROCSPARSE_HPP_

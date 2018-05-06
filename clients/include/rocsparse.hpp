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
                                 const T *alpha,
                                 const T *xVal,
                                 const rocsparse_int *xInd,
                                 T *y,
                                 rocsparse_index_base idxBase);

template <typename T>
rocsparse_status rocsparse_csrmv(rocsparse_handle handle,
                                 rocsparse_operation transA, 
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const T *alpha,
                                 const rocsparse_mat_descr descrA,
                                 const T *csrValA,
                                 const rocsparse_int *csrRowPtrA,
                                 const rocsparse_int *csrColIndA,
                                 const T *x,
                                 const T *beta,
                                 T *y);

}

#endif // _ROCSPARSE_HPP_

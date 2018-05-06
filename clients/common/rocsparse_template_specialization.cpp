/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.hpp"

#include <rocsparse.h>

namespace rocsparse {

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const float *alpha,
                                 const float *xVal,
                                 const rocsparse_int *xInd,
                                 float *y,
                                 rocsparse_index_base idxBase)
{
    return rocsparse_saxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle handle,
                                 rocsparse_int nnz,
                                 const double *alpha,
                                 const double *xVal,
                                 const rocsparse_int *xInd,
                                 double *y,
                                 rocsparse_index_base idxBase)
{
    return rocsparse_daxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle handle,
                                 rocsparse_operation transA, 
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const float *alpha,
                                 const rocsparse_mat_descr descrA,
                                 const float *csrValA,
                                 const rocsparse_int *csrRowPtrA,
                                 const rocsparse_int *csrColIndA,
                                 const float *x,
                                 const float *beta,
                                 float *y)
{
    return rocsparse_scsrmv(handle, transA, m, n, nnz, alpha,
                            descrA, csrValA, csrRowPtrA, csrColIndA,
                            x, beta, y);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle handle,
                                 rocsparse_operation transA, 
                                 rocsparse_int m,
                                 rocsparse_int n,
                                 rocsparse_int nnz,
                                 const double *alpha,
                                 const rocsparse_mat_descr descrA,
                                 const double *csrValA,
                                 const rocsparse_int *csrRowPtrA,
                                 const rocsparse_int *csrColIndA,
                                 const double *x,
                                 const double *beta,
                                 double *y)
{
    return rocsparse_dcsrmv(handle, transA, m, n, nnz, alpha,
                            descrA, csrValA, csrRowPtrA, csrColIndA,
                            x, beta, y);
}

} // namespace rocsparse

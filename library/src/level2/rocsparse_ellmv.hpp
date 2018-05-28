/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSPARSE_ELLMV_HPP
#define ROCSPARSE_ELLMV_HPP

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "ellmv_device.h"

#include <hip/hip_runtime.h>

/*! \brief SPARSE Level 2 API

    \details
    ellmv  multiplies the dense vector x[i] with scalar alpha and sparse m x n
    matrix A that is defined in ELL storage format and add the result to y[i]
    that is multiplied by beta, for  i = 1 , … , n

        y := alpha * op(A) * x + beta * y,

    @param[in]
    handle      rocsparse_handle.
                handle to the rocsparse library context queue.
    @param[in]
    trans       operation type of A.
    @param[in]
    m           number of rows of A.
    @param[in]
    n           number of columns of A.
    @param[in]
    nnz         number of non-zero entries of A.
    @param[in]
    alpha       scalar alpha.
    @param[in]
    descr       descriptor of A.
    @param[in]
    ell_val     array of nnz elements of A.
    @param[in]
    ell_col_ind array of nnz elements containing the column indices of A.
    @param[in]
    x           array of n elements (op(A) = A) or m elements (op(A) = A^T or
                op(A) = A^H).
    @param[in]
    beta        scalar beta.
    @param[inout]
    y           array of m elements (op(A) = A) or n elements (op(A) = A^T or
                op(A) = A^H).

    ********************************************************************/
template <typename T>
rocsparse_status rocsparse_ellmv_template(rocsparse_handle handle,
                                          rocsparse_operation trans,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int nnz,
                                          const T* alpha,
                                          const rocsparse_mat_descr descr,
                                          const T* ell_val,
                                          const rocsparse_int* ell_col_ind,
                                          const T* x,
                                          const T* beta,
                                          T* y)
{
    return rocsparse_status_not_implemented;
}

#endif // ROCSPARSE_ELLMV_HPP

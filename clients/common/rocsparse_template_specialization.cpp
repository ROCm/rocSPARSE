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

} // namespace rocsparse

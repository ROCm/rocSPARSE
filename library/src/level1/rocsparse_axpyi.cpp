/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_axpyi.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_saxpyi(rocsparse_handle handle,
                                             rocsparse_int nnz,
                                             const float* alpha,
                                             const float* x_val,
                                             const rocsparse_int* x_ind,
                                             float* y,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_axpyi_template<float>(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

extern "C" rocsparse_status rocsparse_daxpyi(rocsparse_handle handle,
                                             rocsparse_int nnz,
                                             const double* alpha,
                                             const double* x_val,
                                             const rocsparse_int* x_ind,
                                             double* y,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_axpyi_template<double>(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

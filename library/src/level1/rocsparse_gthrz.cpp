/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_gthrz.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sgthrz(rocsparse_handle handle,
                                             rocsparse_int nnz,
                                             float* y,
                                             float* x_val,
                                             const rocsparse_int* x_ind,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_gthrz_template<float>(handle, nnz, y, x_val, x_ind, idx_base);
}

extern "C" rocsparse_status rocsparse_dgthrz(rocsparse_handle handle,
                                             rocsparse_int nnz,
                                             double* y,
                                             double* x_val,
                                             const rocsparse_int* x_ind,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_gthrz_template<double>(handle, nnz, y, x_val, x_ind, idx_base);
}

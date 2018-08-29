/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_gthr.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sgthr(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const float* y,
                                            float* x_val,
                                            const rocsparse_int* x_ind,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_gthr_template<float>(handle, nnz, y, x_val, x_ind, idx_base);
}

extern "C" rocsparse_status rocsparse_dgthr(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const double* y,
                                            double* x_val,
                                            const rocsparse_int* x_ind,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_gthr_template<double>(handle, nnz, y, x_val, x_ind, idx_base);
}

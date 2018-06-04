/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_sctr.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_ssctr(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const float* x_val,
                                            const rocsparse_int* x_ind,
                                            float* y,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_sctr_template<float>(handle, nnz, x_val, x_ind, y, idx_base);
}

extern "C" rocsparse_status rocsparse_dsctr(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const double* x_val,
                                            const rocsparse_int* x_ind,
                                            double* y,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_sctr_template<double>(handle, nnz, x_val, x_ind, y, idx_base);
}

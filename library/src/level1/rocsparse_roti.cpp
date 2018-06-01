/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_roti.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sroti(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            float* x_val,
                                            const rocsparse_int* x_ind,
                                            float* y,
                                            const float* c,
                                            const float* s,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_roti_template<float>(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

extern "C" rocsparse_status rocsparse_droti(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            double* x_val,
                                            const rocsparse_int* x_ind,
                                            double* y,
                                            const double* c,
                                            const double* s,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_roti_template<double>(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

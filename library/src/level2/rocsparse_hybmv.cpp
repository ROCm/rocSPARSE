/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_hybmv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_shybmv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             const float* alpha,
                                             const rocsparse_mat_descr descr,
                                             const rocsparse_hyb_mat hyb,
                                             const float* x,
                                             const float* beta,
                                             float* y)
{
    return rocsparse_hybmv_template(handle, trans, alpha, descr, hyb, x, beta, y);
}

extern "C" rocsparse_status rocsparse_dhybmv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             const double* alpha,
                                             const rocsparse_mat_descr descr,
                                             const rocsparse_hyb_mat hyb,
                                             const double* x,
                                             const double* beta,
                                             double* y)
{
    return rocsparse_hybmv_template(handle, trans, alpha, descr, hyb, x, beta, y);
}

/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "rocsparse_doti.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sdoti(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const float* x_val,
                                            const rocsparse_int* x_ind,
                                            const float* y,
                                            float* result,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_doti_template<float>(handle, nnz, x_val, x_ind, y, result, idx_base);
}

extern "C" rocsparse_status rocsparse_ddoti(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const double* x_val,
                                            const rocsparse_int* x_ind,
                                            const double* y,
                                            double* result,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_doti_template<double>(handle, nnz, x_val, x_ind, y, result, idx_base);
}
/*
extern "C" rocsparse_status rocsparse_sdoti(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const rocsparse_float_complex* x_val,
                                            const rocsparse_int* x_ind,
                                            const rocsparse_float_complex* y,
                                            rocsparse_float_complex* result,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_doti_template<rocsparse_float_complex>(handle, nnz, x_val, x_ind, y, result, idx_base);
}

extern "C" rocsparse_status rocsparse_ddoti(rocsparse_handle handle,
                                            rocsparse_int nnz,
                                            const rocsparse_double_complex* x_val,
                                            const rocsparse_int* x_ind,
                                            const rocsparse_double_complex* y,
                                            rocsparse_double_complex* result,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_doti_template<rocsparse_double_complex>(handle, nnz, x_val, x_ind, y, result, idx_base);
}
*/

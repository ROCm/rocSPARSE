/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "definitions.h"
#include "rocsparse.h"
#include "rocsparse_csrmv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrmv_analysis(rocsparse_handle handle,
                                                      rocsparse_operation trans,
                                                      rocsparse_int m,
                                                      rocsparse_int n,
                                                      rocsparse_int nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const float* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      rocsparse_mat_info info)
{
    return rocsparse_csrmv_analysis_template<float>(handle,
                                                    trans,
                                                    m,
                                                    n,
                                                    nnz,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    info);
}

extern "C" rocsparse_status rocsparse_dcsrmv_analysis(rocsparse_handle handle,
                                                      rocsparse_operation trans,
                                                      rocsparse_int m,
                                                      rocsparse_int n,
                                                      rocsparse_int nnz,
                                                      const rocsparse_mat_descr descr,
                                                      const double* csr_val,
                                                      const rocsparse_int* csr_row_ptr,
                                                      const rocsparse_int* csr_col_ind,
                                                      rocsparse_mat_info info)
{
    return rocsparse_csrmv_analysis_template<double>(handle,
                                                     trans,
                                                     m,
                                                     n,
                                                     nnz,
                                                     descr,
                                                     csr_val,
                                                     csr_row_ptr,
                                                     csr_col_ind,
                                                     info);
}

extern "C" rocsparse_status rocsparse_csrmv_clear(rocsparse_handle handle, rocsparse_mat_info info)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle, "rocsparse_csrmv_clear", (const void*&)info);

    // Destroy csrmv info struct
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_csrmv_info(info->csrmv_info));
    info->csrmv_info = nullptr;

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_scsrmv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const float* alpha,
                                             const rocsparse_mat_descr descr,
                                             const float* csr_val,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             rocsparse_mat_info info,
                                             const float* x,
                                             const float* beta,
                                             float* y)
{
    return rocsparse_csrmv_template<float>(handle,
                                           trans,
                                           m,
                                           n,
                                           nnz,
                                           alpha,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           x,
                                           beta,
                                           y);
}

extern "C" rocsparse_status rocsparse_dcsrmv(rocsparse_handle handle,
                                             rocsparse_operation trans,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int nnz,
                                             const double* alpha,
                                             const rocsparse_mat_descr descr,
                                             const double* csr_val,
                                             const rocsparse_int* csr_row_ptr,
                                             const rocsparse_int* csr_col_ind,
                                             rocsparse_mat_info info,
                                             const double* x,
                                             const double* beta,
                                             double* y)
{
    return rocsparse_csrmv_template<double>(handle,
                                            trans,
                                            m,
                                            n,
                                            nnz,
                                            alpha,
                                            descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            x,
                                            beta,
                                            y);
}

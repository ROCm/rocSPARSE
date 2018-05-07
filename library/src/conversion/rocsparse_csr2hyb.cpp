/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "csr2hyb_device.h"

#include <hip/hip_runtime.h>

template <typename T>
__global__
void csr2ell_kernel(rocsparse_int m,
                    const T *csr_val,
                    const rocsparse_int *csr_row_ptr,
                    const rocsparse_int *csr_col_ind,
                    rocsparse_int ell_width,
                    rocsparse_int *ell_col_ind,
                    T *ell_val)
{
    csr2ell_device(m, csr_val, csr_row_ptr, csr_col_ind,
                   ell_width, ell_col_ind, ell_val);
}

template <typename T>
rocsparse_status rocsparse_csr2hyb_template(rocsparse_handle handle,
                                            rocsparse_int m,
                                            rocsparse_int n,
                                            const rocsparse_mat_descr descr,
                                            const T *csr_val,
                                            const rocsparse_int *csr_row_ptr,
                                            const rocsparse_int *csr_col_ind,
                                            rocsparse_hyb_mat hyb,
                                            rocsparse_int user_ell_width,
                                            rocsparse_hyb_partition partition_type)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if (descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsr2hyb"),
              m,
              n,
              (const void*&) descr,
              (const void*&) csr_val,
              (const void*&) csr_row_ptr,
              (const void*&) csr_col_ind,
              (const void*&) hyb,
              user_ell_width,
              partition_type);

    // Check matrix type
    if (descr->base != rocsparse_index_base_zero)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    if (descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }
    if (partition_type != rocsparse_hyb_partition_max)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if (m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if (n < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if (csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if (m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Clear HYB structure if already allocated
    hyb->m = m;
    hyb->n = n;
    hyb->partition = partition_type;
    hyb->ell_nnz   = 0;
    hyb->ell_width = 0;
    hyb->coo_nnz   = 0;

    if (hyb->ell_col_ind)
    {
        hipFree(hyb->ell_col_ind);
    }
    if (hyb->ell_val)
    {
        hipFree(hyb->ell_val);
    }
    if (hyb->coo_row_ind)
    {
        hipFree(hyb->coo_row_ind);
    }
    if (hyb->coo_col_ind)
    {
        hipFree(hyb->coo_col_ind);
    }
    if (hyb->coo_val)
    {
        hipFree(hyb->coo_val);
    }

#define CSR2ELL_DIM 256
    dim3 csr2ell_blocks((m-1)/CSR2ELL_DIM+1);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    //TODO we take max partition
    if (partition_type == rocsparse_hyb_partition_max)
    {
        // ELL part only, compute maximum non-zeros per row

        //TODO reduction with rocPRIM to compute maxrow
        rocsparse_int *hbuf = (rocsparse_int*) malloc(sizeof(rocsparse_int)*(m+1));
        hipMemcpy(hbuf, csr_row_ptr, sizeof(rocsparse_int)*(m+1), hipMemcpyDeviceToHost);

        for (rocsparse_int i=0; i<m; ++i)
        {
            rocsparse_int rownnz = hbuf[i+1] - hbuf[i];
            hyb->ell_width = rownnz > hyb->ell_width ? rownnz : hyb->ell_width;
        }
        free(hbuf);
        // END TODO
    }
    else
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Compute ELL non-zeros
    hyb->ell_nnz = hyb->ell_width * m;

    // Allocate ELL part
    hipMalloc((void**) &hyb->ell_col_ind, sizeof(rocsparse_int)*hyb->ell_nnz);
    hipMalloc(&hyb->ell_val, sizeof(T)*hyb->ell_nnz);




    hipLaunchKernelGGL((csr2ell_kernel<T>),
                       csr2ell_blocks, csr2ell_threads, 0, stream,
                       m, csr_val, csr_row_ptr, csr_col_ind,
                       hyb->ell_width, hyb->ell_col_ind, (T*) hyb->ell_val);

#undef CSR2ELL_DIM






    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C"
rocsparse_status rocsparse_scsr2hyb(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr descr,
                                    const float *csr_val,
                                    const rocsparse_int *csr_row_ptr,
                                    const rocsparse_int *csr_col_ind,
                                    rocsparse_hyb_mat hyb,
                                    rocsparse_int user_ell_width,
                                    rocsparse_hyb_partition partition_type)
{
    return rocsparse_csr2hyb_template(handle, m, n,
                                      descr, csr_val, csr_row_ptr, csr_col_ind,
                                      hyb, user_ell_width, partition_type);
}

extern "C"
rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle handle,
                                    rocsparse_int m,
                                    rocsparse_int n,
                                    const rocsparse_mat_descr descr,
                                    const double *csr_val,
                                    const rocsparse_int *csr_row_ptr,
                                    const rocsparse_int *csr_col_ind,
                                    rocsparse_hyb_mat hyb,
                                    rocsparse_int user_ell_width,
                                    rocsparse_hyb_partition partition_type)
{
    return rocsparse_csr2hyb_template(handle, m, n,
                                      descr, csr_val, csr_row_ptr, csr_col_ind,
                                      hyb, user_ell_width, partition_type);
}

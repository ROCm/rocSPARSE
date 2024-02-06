/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "../level2/rocsparse_csrsv.hpp"
#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "common.h"
#include "control.h"
#include "utility.h"
#include <rocprim/rocprim.hpp>

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsm_buffer_size_core(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   J                         m,
                                                   J                         nrhs,
                                                   I                         nnz,
                                                   const T*                  alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  B,
                                                   int64_t                   ldb,
                                                   rocsparse_order           order_B,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   size_t*                   buffer_size)
{

    if(nrhs == 1)
    {
        //
        // Call csrsv.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_buffer_size_template(
            handle, trans_A, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));
        *buffer_size += ((sizeof(T) * m - 1) / 256 + 1) * 256;
        return rocsparse_status_success;
    }

    hipStream_t stream = handle->stream;

    // max_nnz
    *buffer_size = 256;

    // Each thread block performs at most blockdim columns of the
    // rhs matrix. Therefore, the number of blocks depend on nrhs
    // and the blocksize.
    // Because of this, we might need a larger done_array compared
    // to csrsv.
    int blockdim = 512;
    while(nrhs <= blockdim && blockdim > 32)
    {
        blockdim >>= 1;
    }

    blockdim <<= 1;
    int narrays = (nrhs - 1) / blockdim + 1;

    // int done_array
    *buffer_size += ((sizeof(int) * m * narrays - 1) / 256 + 1) * 256;

    // workspace
    *buffer_size += ((sizeof(J) * m - 1) / 256 + 1) * 256;

    // int workspace2
    *buffer_size += ((sizeof(int) * m - 1) / 256 + 1) * 256;

    size_t rocprim_size;
    int*   ptr1 = reinterpret_cast<int*>(buffer_size);
    I*     ptr2 = reinterpret_cast<I*>(buffer_size);
    J*     ptr3 = reinterpret_cast<J*>(buffer_size);

    rocprim::double_buffer<int> dummy1(ptr1, ptr1);
    rocprim::double_buffer<I>   dummy2(ptr2, ptr2);
    rocprim::double_buffer<J>   dummy3(ptr3, ptr3);

    RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
        nullptr, rocprim_size, dummy1, dummy3, m, 0, rocsparse::clz(m), stream));

    // rocprim buffer
    *buffer_size += rocprim_size;

    // Additional buffer to store transpose of B, if trans_B == rocsparse_operation_none
    if(trans_B == rocsparse_operation_none && order_B == rocsparse_order_column)
    {
        *buffer_size += ((sizeof(T) * m * nrhs - 1) / 256 + 1) * 256;
    }

    // Additional buffer to store transpose A, if transA != rocsparse_operation_none
    if(trans_A == rocsparse_operation_transpose
       || trans_A == rocsparse_operation_conjugate_transpose)
    {
        size_t transpose_size;

        // Determine rocprim buffer size
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            nullptr, transpose_size, dummy3, dummy2, nnz, 0, rocsparse::clz(m), stream));

        // rocPRIM does not support in-place sorting, so we need an additional buffer
        transpose_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
        transpose_size += ((std::max(sizeof(I), sizeof(T)) * nnz - 1) / 256 + 1) * 256;

        *buffer_size += transpose_size;
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_operation       trans_B,
                                                          int64_t                   m,
                                                          int64_t                   nrhs,
                                                          int64_t                   nnz,
                                                          const void*               alpha,
                                                          const rocsparse_mat_descr descr,
                                                          const void*               csr_val,
                                                          const void*               csr_row_ptr,
                                                          const void*               csr_col_ind,
                                                          const void*               B,
                                                          int64_t                   ldb,
                                                          rocsparse_order           order_B,
                                                          rocsparse_mat_info        info,
                                                          rocsparse_solve_policy    policy,
                                                          size_t*                   buffer_size)
{

    if(m == 0 || nrhs == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status csrsm_buffer_size_checkarg(rocsparse_handle          handle, //0
                                                       rocsparse_operation       trans_A, //1
                                                       rocsparse_operation       trans_B, //2
                                                       int64_t                   m, //3
                                                       int64_t                   nrhs, //4
                                                       int64_t                   nnz, //5
                                                       const void*               alpha, //6
                                                       const rocsparse_mat_descr descr, //7
                                                       const void*               csr_val, //8
                                                       const void*               csr_row_ptr, //9
                                                       const void*               csr_col_ind, //10
                                                       const void*               B, //11
                                                       int64_t                   ldb, //12
                                                       rocsparse_order    order_B, // non-classified
                                                       rocsparse_mat_info info, //13
                                                       rocsparse_solve_policy policy, //14
                                                       size_t*                buffer_size) //15
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, nrhs);
        ROCSPARSE_CHECKARG_SIZE(5, nnz);
        ROCSPARSE_CHECKARG(12,
                           ldb,
                           (trans_B == rocsparse_operation_none && ldb < m),
                           rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG(12,
                           ldb,
                           ((trans_B == rocsparse_operation_transpose
                             || trans_B == rocsparse_operation_conjugate_transpose)
                            && ldb < nrhs),
                           rocsparse_status_invalid_size);

        const rocsparse_status status = rocsparse::csrsm_buffer_size_quickreturn(handle,
                                                                                 trans_A,
                                                                                 trans_B,
                                                                                 m,
                                                                                 nrhs,
                                                                                 nnz,
                                                                                 alpha,
                                                                                 descr,
                                                                                 csr_val,
                                                                                 csr_row_ptr,
                                                                                 csr_col_ind,
                                                                                 B,
                                                                                 ldb,
                                                                                 order_B,
                                                                                 info,
                                                                                 policy,
                                                                                 buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(7, descr);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_val);

        ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(13, info);
        ROCSPARSE_CHECKARG_ENUM(14, policy);
        ROCSPARSE_CHECKARG_POINTER(15, buffer_size);

        ROCSPARSE_CHECKARG_POINTER(6, alpha);
        ROCSPARSE_CHECKARG_POINTER(11, B);
        return rocsparse_status_continue;
    }

    template <typename I, typename J, typename T>
    static rocsparse_status csrsm_buffer_size_impl(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   J                         m,
                                                   J                         nrhs,
                                                   I                         nnz,
                                                   const T*                  alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  csr_val,
                                                   const I*                  csr_row_ptr,
                                                   const J*                  csr_col_ind,
                                                   const T*                  B,
                                                   int64_t                   ldb,
                                                   rocsparse_order           order_B,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   size_t*                   buffer_size)
    {

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrsm_buffer_size"),
                             trans_A,
                             trans_B,
                             m,
                             nrhs,
                             nnz,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)B,
                             ldb,
                             order_B,
                             (const void*&)info,
                             policy,
                             (const void*&)buffer_size);

        const rocsparse_status status = rocsparse::csrsm_buffer_size_checkarg(handle,
                                                                              trans_A,
                                                                              trans_B,
                                                                              m,
                                                                              nrhs,
                                                                              nnz,
                                                                              alpha,
                                                                              descr,
                                                                              csr_val,
                                                                              csr_row_ptr,
                                                                              csr_col_ind,
                                                                              B,
                                                                              ldb,
                                                                              order_B,
                                                                              info,
                                                                              policy,
                                                                              buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_buffer_size_core(handle,
                                                                    trans_A,
                                                                    trans_B,
                                                                    m,
                                                                    nrhs,
                                                                    nnz,
                                                                    alpha,
                                                                    descr,
                                                                    csr_val,
                                                                    csr_row_ptr,
                                                                    csr_col_ind,
                                                                    B,
                                                                    ldb,
                                                                    order_B,
                                                                    info,
                                                                    policy,
                                                                    buffer_size));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                            \
    template rocsparse_status rocsparse::csrsm_buffer_size_core(rocsparse_handle          handle,   \
                                                                rocsparse_operation       trans_A,  \
                                                                rocsparse_operation       trans_B,  \
                                                                JTYPE                     m,        \
                                                                JTYPE                     nrhs,     \
                                                                ITYPE                     nnz,      \
                                                                const TTYPE*              alpha,    \
                                                                const rocsparse_mat_descr descr,    \
                                                                const TTYPE*              csr_val,  \
                                                                const ITYPE*           csr_row_ptr, \
                                                                const JTYPE*           csr_col_ind, \
                                                                const TTYPE*           B,           \
                                                                int64_t                ldb,         \
                                                                rocsparse_order        order_B,     \
                                                                rocsparse_mat_info     info,        \
                                                                rocsparse_solve_policy policy,      \
                                                                size_t*                buffer_size);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, ITYPE, JTYPE, TTYPE)                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                      \
                                     rocsparse_operation       trans_A,                     \
                                     rocsparse_operation       trans_B,                     \
                                     JTYPE                     m,                           \
                                     JTYPE                     nrhs,                        \
                                     ITYPE                     nnz,                         \
                                     const TTYPE*              alpha,                       \
                                     const rocsparse_mat_descr descr,                       \
                                     const TTYPE*              csr_val,                     \
                                     const ITYPE*              csr_row_ptr,                 \
                                     const JTYPE*              csr_col_ind,                 \
                                     const TTYPE*              B,                           \
                                     JTYPE                     ldb,                         \
                                     rocsparse_mat_info        info,                        \
                                     rocsparse_solve_policy    policy,                      \
                                     size_t*                   buffer_size)                 \
    try                                                                                     \
    {                                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_buffer_size_impl(handle,                 \
                                                                    trans_A,                \
                                                                    trans_B,                \
                                                                    m,                      \
                                                                    nrhs,                   \
                                                                    nnz,                    \
                                                                    alpha,                  \
                                                                    descr,                  \
                                                                    csr_val,                \
                                                                    csr_row_ptr,            \
                                                                    csr_col_ind,            \
                                                                    B,                      \
                                                                    ldb,                    \
                                                                    rocsparse_order_column, \
                                                                    info,                   \
                                                                    policy,                 \
                                                                    buffer_size));          \
        return rocsparse_status_success;                                                    \
    }                                                                                       \
    catch(...)                                                                              \
    {                                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                                       \
    }

C_IMPL(rocsparse_scsrsm_buffer_size, int32_t, int32_t, float);
C_IMPL(rocsparse_dcsrsm_buffer_size, int32_t, int32_t, double);
C_IMPL(rocsparse_ccsrsm_buffer_size, int32_t, int32_t, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsm_buffer_size, int32_t, int32_t, rocsparse_double_complex);

#undef C_IMPL

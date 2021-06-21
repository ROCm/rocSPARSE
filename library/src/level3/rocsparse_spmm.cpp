/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include "rocsparse_coomm.hpp"
#include "rocsparse_csrmm.hpp"

#define RETURN_SPMM(itype, jtype, ctype, ...)                                           \
    {                                                                                   \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_r)                                        \
            return rocsparse_spmm_template<int32_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_r)                                        \
            return rocsparse_spmm_template<int32_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_c)                                        \
            return rocsparse_spmm_template<int32_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_c)                                        \
            return rocsparse_spmm_template<int32_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_r)                                        \
            return rocsparse_spmm_template<int64_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_r)                                        \
            return rocsparse_spmm_template<int64_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_c)                                        \
            return rocsparse_spmm_template<int64_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_c)                                        \
            return rocsparse_spmm_template<int64_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f32_r)                                        \
            return rocsparse_spmm_template<int64_t, int64_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f64_r)                                        \
            return rocsparse_spmm_template<int64_t, int64_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f32_c)                                        \
            return rocsparse_spmm_template<int64_t, int64_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f64_c)                                        \
            return rocsparse_spmm_template<int64_t, int64_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                           \
    }

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spmm_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans_A,
                                         rocsparse_operation         trans_B,
                                         const void*                 alpha,
                                         const rocsparse_spmat_descr mat_A,
                                         const rocsparse_dnmat_descr mat_B,
                                         const void*                 beta,
                                         const rocsparse_dnmat_descr mat_C,
                                         rocsparse_spmm_alg          alg,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    rocsparse_spmm_alg algorithm = alg;
    if(algorithm == rocsparse_spmm_alg_default)
    {
        if(mat_A->format == rocsparse_format_coo)
        {
            algorithm = rocsparse_spmm_alg_coo_atomic;
        }
        else if(mat_A->format == rocsparse_format_csr)
        {
            algorithm = rocsparse_spmm_alg_csr;
        }
    }

    // If temp_buffer is nullptr, return buffer_size
    if(temp_buffer == nullptr)
    {
        // We do not need a buffer
        *buffer_size = 4;

        // Run CSR analysis step when format is CSR
        if(mat_A->format == rocsparse_format_csr)
        {
            // If merge algorithm is selected a buffer is required
            if(alg == rocsparse_spmm_alg_csr_merge)
            {
                J m = (J)mat_C->rows;
                J n = (J)mat_C->cols;
                J k = trans_A == rocsparse_operation_none ? (J)mat_A->cols : (J)mat_A->rows;

                return rocsparse_csrmm_buffer_size_template(handle,
                                                            trans_A,
                                                            alg,
                                                            m,
                                                            n,
                                                            k,
                                                            (I)mat_A->nnz,
                                                            mat_A->descr,
                                                            (const T*)mat_A->val_data,
                                                            (const I*)mat_A->row_data,
                                                            (const J*)mat_A->col_data,
                                                            buffer_size);
            }
        }

        return rocsparse_status_success;
    }

    // If buffer_size is nullptr, return temp_buffer
    if(buffer_size == nullptr)
    {
        // Run CSR analysis step when format is CSR
        if(mat_A->format == rocsparse_format_csr)
        {
            // If merge algorithm is selected and analysis step is required
            if(alg == rocsparse_spmm_alg_csr_merge)
            {
                J m = (J)mat_C->rows;
                J n = (J)mat_C->cols;
                J k = trans_A == rocsparse_operation_none ? (J)mat_A->cols : (J)mat_A->rows;

                return rocsparse_csrmm_analysis_template(handle,
                                                         trans_A,
                                                         alg,
                                                         m,
                                                         n,
                                                         k,
                                                         (I)mat_A->nnz,
                                                         mat_A->descr,
                                                         (const T*)mat_A->val_data,
                                                         (const I*)mat_A->row_data,
                                                         (const J*)mat_A->col_data,
                                                         temp_buffer);
            }
        }

        return rocsparse_status_success;
    }

    // COO
    if(mat_A->format == rocsparse_format_coo)
    {
        return rocsparse_coomm_template(handle,
                                        trans_A,
                                        trans_B,
                                        mat_B->order,
                                        mat_C->order,
                                        algorithm,
                                        (I)mat_A->rows,
                                        (I)mat_C->cols,
                                        (I)mat_A->cols,
                                        (I)mat_A->nnz,
                                        (const T*)alpha,
                                        mat_A->descr,
                                        (const T*)mat_A->val_data,
                                        (const I*)mat_A->row_data,
                                        (const I*)mat_A->col_data,
                                        (const T*)mat_B->values,
                                        (I)mat_B->ld,
                                        (const T*)beta,
                                        (T*)mat_C->values,
                                        (I)mat_C->ld);
    }

    // CSR
    if(mat_A->format == rocsparse_format_csr)
    {
        J m = (J)mat_C->rows;
        J n = (J)mat_C->cols;
        J k = trans_A == rocsparse_operation_none ? (J)mat_A->cols : (J)mat_A->rows;

        return rocsparse_csrmm_template(handle,
                                        trans_A,
                                        trans_B,
                                        mat_B->order,
                                        mat_C->order,
                                        algorithm,
                                        m,
                                        n,
                                        k,
                                        (I)mat_A->nnz,
                                        (const T*)alpha,
                                        mat_A->descr,
                                        (const T*)mat_A->val_data,
                                        (const I*)mat_A->row_data,
                                        (const J*)mat_A->col_data,
                                        (const T*)mat_B->values,
                                        (J)mat_B->ld,
                                        (const T*)beta,
                                        (T*)mat_C->values,
                                        (J)mat_C->ld,
                                        temp_buffer);
    }

    return rocsparse_status_not_implemented;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr mat_A,
                                           const rocsparse_dnmat_descr mat_B,
                                           const void*                 beta,
                                           const rocsparse_dnmat_descr mat_C,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_spmm_alg          alg,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spmm",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)mat_A,
              (const void*&)mat_B,
              (const void*&)beta,
              (const void*&)mat_C,
              compute_type,
              alg,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat_A);
    RETURN_IF_NULLPTR(mat_B);
    RETURN_IF_NULLPTR(mat_C);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        RETURN_IF_NULLPTR(buffer_size);
    }

    // Check for valid temp_buffer pointer only if buffer_size is nullptr
    if(buffer_size == nullptr)
    {
        RETURN_IF_NULLPTR(temp_buffer);
    }

    // Check if descriptors are initialized
    if(mat_A->init == false || mat_B->init == false || mat_C->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Check for matching types while we do not support mixed precision computation
    if(compute_type != mat_A->data_type || compute_type != mat_B->data_type
       || compute_type != mat_C->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    RETURN_SPMM(mat_A->row_type,
                mat_A->col_type,
                compute_type,
                handle,
                trans_A,
                trans_B,
                alpha,
                mat_A,
                mat_B,
                beta,
                mat_C,
                alg,
                buffer_size,
                temp_buffer);

    return rocsparse_status_not_implemented;
}

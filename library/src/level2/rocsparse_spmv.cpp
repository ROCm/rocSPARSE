/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_coomv.hpp"
#include "rocsparse_coomv_aos.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_ellmv.hpp"

#define RETURN_SPMV(itype, jtype, ctype, ...)                                           \
    {                                                                                   \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_r)                                        \
            return rocsparse_spmv_template<int32_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_r)                                        \
            return rocsparse_spmv_template<int32_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_c)                                        \
            return rocsparse_spmv_template<int32_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_c)                                        \
            return rocsparse_spmv_template<int32_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_r)                                        \
            return rocsparse_spmv_template<int64_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_r)                                        \
            return rocsparse_spmv_template<int64_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f32_c)                                        \
            return rocsparse_spmv_template<int64_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32         \
           && ctype == rocsparse_datatype_f64_c)                                        \
            return rocsparse_spmv_template<int64_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f32_r)                                        \
            return rocsparse_spmv_template<int64_t, int64_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f64_r)                                        \
            return rocsparse_spmv_template<int64_t, int64_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f32_c)                                        \
            return rocsparse_spmv_template<int64_t, int64_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                           \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64         \
           && ctype == rocsparse_datatype_f64_c)                                        \
            return rocsparse_spmv_template<int64_t, int64_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                           \
    }

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spmv_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans,
                                         const void*                 alpha,
                                         const rocsparse_spmat_descr mat,
                                         const rocsparse_dnvec_descr x,
                                         const void*                 beta,
                                         const rocsparse_dnvec_descr y,
                                         rocsparse_spmv_alg          alg,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    // If temp_buffer is nullptr, return buffer_size
    if(temp_buffer == nullptr)
    {
        // We do not need a buffer
        *buffer_size = 4;

        // Run CSR analysis step when format is CSR
        if(mat->format == rocsparse_format_csr)
        {
            // If algorithm 1 or default is selected and analysis step is required
            if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive)
               && mat->analysed == false)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_csrmv_analysis_template<I, J, T>(handle,
                                                                trans,
                                                                (J)mat->rows,
                                                                (J)mat->cols,
                                                                (I)mat->nnz,
                                                                mat->descr,
                                                                (const T*)mat->val_data,
                                                                (const I*)mat->row_data,
                                                                (const J*)mat->col_data,
                                                                mat->info)));

                mat->analysed = true;
            }
        }

        return rocsparse_status_success;
    }

    // COO
    if(mat->format == rocsparse_format_coo)
    {
        return rocsparse_coomv_template<I, T>(handle,
                                              trans,
                                              (I)mat->rows,
                                              (I)mat->cols,
                                              (I)mat->nnz,
                                              (const T*)alpha,
                                              mat->descr,
                                              (const T*)mat->val_data,
                                              (const I*)mat->row_data,
                                              (const I*)mat->col_data,
                                              (const T*)x->values,
                                              (const T*)beta,
                                              (T*)y->values);
    }

    // COO (AoS)
    if(mat->format == rocsparse_format_coo_aos)
    {
        return rocsparse_coomv_aos_template<I, T>(handle,
                                                  trans,
                                                  (I)mat->rows,
                                                  (I)mat->cols,
                                                  (I)mat->nnz,
                                                  (const T*)alpha,
                                                  mat->descr,
                                                  (const T*)mat->val_data,
                                                  (const I*)mat->ind_data,
                                                  (const T*)x->values,
                                                  (const T*)beta,
                                                  (T*)y->values);
    }

    // CSR
    if(mat->format == rocsparse_format_csr)
    {
        return rocsparse_csrmv_template<I, J, T>(handle,
                                                 trans,
                                                 (J)mat->rows,
                                                 (J)mat->cols,
                                                 (I)mat->nnz,
                                                 (const T*)alpha,
                                                 mat->descr,
                                                 (const T*)mat->val_data,
                                                 (const I*)mat->row_data,
                                                 (const J*)mat->col_data,
                                                 (alg == rocsparse_spmv_alg_csr_stream) ? nullptr
                                                                                        : mat->info,
                                                 (const T*)x->values,
                                                 (const T*)beta,
                                                 (T*)y->values);
    }

    // ELL
    if(mat->format == rocsparse_format_ell)
    {
        return rocsparse_ellmv_template<I, T>(handle,
                                              trans,
                                              (I)mat->rows,
                                              (I)mat->cols,
                                              (const T*)alpha,
                                              mat->descr,
                                              (const T*)mat->val_data,
                                              (const I*)mat->col_data,
                                              (I)mat->nnz,
                                              (const T*)x->values,
                                              (const T*)beta,
                                              (T*)y->values);
    }

    return rocsparse_status_not_implemented;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spmv(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr mat,
                                           const rocsparse_dnvec_descr x,
                                           const void*                 beta,
                                           const rocsparse_dnvec_descr y,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_spmv_alg          alg,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spmv",
              trans,
              (const void*&)alpha,
              (const void*&)mat,
              (const void*&)x,
              (const void*&)beta,
              (const void*&)y,
              compute_type,
              alg,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat);
    RETURN_IF_NULLPTR(x);
    RETURN_IF_NULLPTR(y);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        RETURN_IF_NULLPTR(buffer_size);
    }

    // Check if descriptors are initialized
    if(mat->init == false || x->init == false || y->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Check for matching types while we do not support mixed precision computation
    if(compute_type != mat->data_type || compute_type != x->data_type
       || compute_type != y->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    RETURN_SPMV(mat->row_type,
                mat->col_type,
                compute_type,
                handle,
                trans,
                alpha,
                mat,
                x,
                beta,
                y,
                alg,
                buffer_size,
                temp_buffer);

    return rocsparse_status_not_implemented;
}

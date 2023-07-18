/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_dense_to_sparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_dense2coo.hpp"
#include "rocsparse_dense2csx_impl.hpp"
#include "rocsparse_nnz_impl.hpp"

#define RETURN_DENSETOSPARSE(itype, jtype, ctype, ...)                                             \
    {                                                                                              \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_r)                                                   \
            return rocsparse_dense_to_sparse_template<int32_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_r)                                                   \
            return rocsparse_dense_to_sparse_template<int32_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_c)                                                   \
            return rocsparse_dense_to_sparse_template<int32_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_c)                                                   \
            return rocsparse_dense_to_sparse_template<int32_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_r)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_r)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_c)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_c)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f32_r)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int64_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f64_r)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int64_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f32_c)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int64_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f64_c)                                                   \
            return rocsparse_dense_to_sparse_template<int64_t, int64_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                                      \
    }

template <typename I, typename J, typename T>
rocsparse_status rocsparse_dense_to_sparse_template(rocsparse_handle              handle,
                                                    rocsparse_const_dnmat_descr   mat_A,
                                                    rocsparse_spmat_descr         mat_B,
                                                    rocsparse_dense_to_sparse_alg alg,
                                                    size_t*                       buffer_size,
                                                    void*                         temp_buffer)
{
    if(buffer_size == nullptr && temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // If temp_buffer is nullptr, return buffer_size
    if(temp_buffer == nullptr)
    {
        if(mat_B->format == rocsparse_format_coo)
        {
            *buffer_size = sizeof(I) * mat_A->rows;
        }
        else if(mat_B->format == rocsparse_format_csr)
        {
            *buffer_size = sizeof(I) * mat_A->rows;
        }
        else if(mat_B->format == rocsparse_format_csc)
        {
            *buffer_size = sizeof(I) * mat_A->cols;
        }

        return rocsparse_status_success;
    }

    // If buffer_size is nullptr, perform analysis
    if(buffer_size == nullptr)
    {
        if(mat_B->format == rocsparse_format_coo)
        {
            return rocsparse_nnz_impl(handle,
                                      rocsparse_direction_row,
                                      mat_A->order,
                                      (I)mat_A->rows,
                                      (I)mat_A->cols,
                                      mat_B->descr,
                                      (const T*)mat_A->const_values,
                                      (I)mat_A->ld,
                                      (I*)temp_buffer,
                                      (I*)&mat_B->nnz);
        }
        else if(mat_B->format == rocsparse_format_csr)
        {
            return rocsparse_nnz_impl(handle,
                                      rocsparse_direction_row,
                                      mat_A->order,
                                      (J)mat_A->rows,
                                      (J)mat_A->cols,
                                      mat_B->descr,
                                      (const T*)mat_A->const_values,
                                      (I)mat_A->ld,
                                      (I*)temp_buffer,
                                      (I*)&mat_B->nnz);
        }
        else if(mat_B->format == rocsparse_format_csc)
        {
            return rocsparse_nnz_impl(handle,
                                      rocsparse_direction_column,
                                      mat_A->order,
                                      (J)mat_A->rows,
                                      (J)mat_A->cols,
                                      mat_B->descr,
                                      (const T*)mat_A->const_values,
                                      (I)mat_A->ld,
                                      (I*)temp_buffer,
                                      (I*)&mat_B->nnz);
        }
    }

    // COO
    if(mat_B->format == rocsparse_format_coo)
    {
        return rocsparse_dense2coo_template(handle,
                                            mat_A->order,
                                            (I)mat_A->rows,
                                            (I)mat_A->cols,
                                            mat_B->descr,
                                            (const T*)mat_A->const_values,
                                            (I)mat_A->ld,
                                            (I*)temp_buffer,
                                            (T*)mat_B->val_data,
                                            (I*)mat_B->row_data,
                                            (I*)mat_B->col_data);
    }

    // CSR
    if(mat_B->format == rocsparse_format_csr)
    {
        return rocsparse_dense2csx_impl<rocsparse_direction_row>(handle,
                                                                 mat_A->order,
                                                                 (J)mat_A->rows,
                                                                 (J)mat_A->cols,
                                                                 mat_B->descr,
                                                                 (const T*)mat_A->const_values,
                                                                 (I)mat_A->ld,
                                                                 (I*)temp_buffer,
                                                                 (T*)mat_B->val_data,
                                                                 (I*)mat_B->row_data,
                                                                 (J*)mat_B->col_data);
    }

    // CSC
    if(mat_B->format == rocsparse_format_csc)
    {
        return rocsparse_dense2csx_impl<rocsparse_direction_column>(handle,
                                                                    mat_A->order,
                                                                    (J)mat_A->rows,
                                                                    (J)mat_A->cols,
                                                                    mat_B->descr,
                                                                    (const T*)mat_A->const_values,
                                                                    (I)mat_A->ld,
                                                                    (I*)temp_buffer,
                                                                    (T*)mat_B->val_data,
                                                                    (I*)mat_B->col_data,
                                                                    (J*)mat_B->row_data);
    }

    return rocsparse_status_not_implemented;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle              handle,
                                                      rocsparse_const_dnmat_descr   mat_A,
                                                      rocsparse_spmat_descr         mat_B,
                                                      rocsparse_dense_to_sparse_alg alg,
                                                      size_t*                       buffer_size,
                                                      void*                         temp_buffer)
try
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_dense_sparse",
              (const void*&)mat_A,
              (const void*&)mat_B,
              alg,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check alg
    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat_A);
    RETURN_IF_NULLPTR(mat_B);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        RETURN_IF_NULLPTR(buffer_size);
    }

    // Check if descriptors are initialized
    if(mat_A->init == false || mat_B->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    if(mat_B->format == rocsparse_format_csc)
    {
        RETURN_DENSETOSPARSE(mat_B->col_type,
                             mat_B->row_type,
                             mat_B->data_type,
                             handle,
                             mat_A,
                             mat_B,
                             alg,
                             buffer_size,
                             temp_buffer);
    }
    else
    {
        RETURN_DENSETOSPARSE(mat_B->row_type,
                             mat_B->col_type,
                             mat_B->data_type,
                             handle,
                             mat_A,
                             mat_B,
                             alg,
                             buffer_size,
                             temp_buffer);
    }

    return rocsparse_status_not_implemented;
}
catch(...)
{
    return exception_to_rocsparse_status();
}

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
#include "rocsparse/rocsparse.h"
#include "utility.h"

#include "rocsparse_coo2dense.hpp"
#include "rocsparse_csx2dense_impl.hpp"

#define RETURN_SPARSETODENSE(itype, jtype, ctype, ...)                                             \
    {                                                                                              \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_r)                                                   \
            return rocsparse_sparse_to_dense_template<int32_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_r)                                                   \
            return rocsparse_sparse_to_dense_template<int32_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_c)                                                   \
            return rocsparse_sparse_to_dense_template<int32_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_c)                                                   \
            return rocsparse_sparse_to_dense_template<int32_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_r)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int32_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_r)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int32_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f32_c)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int32_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32                    \
           && ctype == rocsparse_datatype_f64_c)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int32_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f32_r)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int64_t, float>(__VA_ARGS__);       \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f64_r)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int64_t, double>(__VA_ARGS__);      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f32_c)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int64_t, rocsparse_float_complex>(  \
                __VA_ARGS__);                                                                      \
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64                    \
           && ctype == rocsparse_datatype_f64_c)                                                   \
            return rocsparse_sparse_to_dense_template<int64_t, int64_t, rocsparse_double_complex>( \
                __VA_ARGS__);                                                                      \
    }

template <typename I, typename J, typename T>
rocsparse_status rocsparse_sparse_to_dense_template(rocsparse_handle              handle,
                                                    const rocsparse_spmat_descr   mat_A,
                                                    rocsparse_dnmat_descr         mat_B,
                                                    rocsparse_sparse_to_dense_alg alg,
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
        // We do not need a buffer
        *buffer_size = 4;

        return rocsparse_status_success;
    }

    // COO
    if(mat_A->format == rocsparse_format_coo)
    {
        return rocsparse_coo2dense_template(handle,
                                            (I)mat_A->rows,
                                            (I)mat_A->cols,
                                            (I)mat_A->nnz,
                                            mat_A->descr,
                                            (const T*)mat_A->val_data,
                                            (const I*)mat_A->row_data,
                                            (const I*)mat_A->col_data,
                                            (T*)mat_B->values,
                                            (I)mat_B->ld,
                                            mat_B->order);
    }

    // CSR
    if(mat_A->format == rocsparse_format_csr)
    {
        return rocsparse_csx2dense_impl<rocsparse_direction_row>(handle,
                                                                 (J)mat_A->rows,
                                                                 (J)mat_A->cols,
                                                                 mat_A->descr,
                                                                 (const T*)mat_A->val_data,
                                                                 (const I*)mat_A->row_data,
                                                                 (const J*)mat_A->col_data,
                                                                 (T*)mat_B->values,
                                                                 (I)mat_B->ld,
                                                                 mat_B->order);
    }

    // CSC
    if(mat_A->format == rocsparse_format_csc)
    {
        return rocsparse_csx2dense_impl<rocsparse_direction_column>(handle,
                                                                    (J)mat_A->rows,
                                                                    (J)mat_A->cols,
                                                                    mat_A->descr,
                                                                    (const T*)mat_A->val_data,
                                                                    (const I*)mat_A->col_data,
                                                                    (const J*)mat_A->row_data,
                                                                    (T*)mat_B->values,
                                                                    (I)mat_B->ld,
                                                                    mat_B->order);
    }

    return rocsparse_status_not_implemented;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle              handle,
                                                      const rocsparse_spmat_descr   mat_A,
                                                      rocsparse_dnmat_descr         mat_B,
                                                      rocsparse_sparse_to_dense_alg alg,
                                                      size_t*                       buffer_size,
                                                      void*                         temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_sparse_dense",
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

    if(mat_A->format == rocsparse_format_csc)
    {
        RETURN_SPARSETODENSE(mat_A->col_type,
                             mat_A->row_type,
                             mat_A->data_type,
                             handle,
                             mat_A,
                             mat_B,
                             alg,
                             buffer_size,
                             temp_buffer);
    }
    else
    {
        RETURN_SPARSETODENSE(mat_A->row_type,
                             mat_A->col_type,
                             mat_A->data_type,
                             handle,
                             mat_A,
                             mat_B,
                             alg,
                             buffer_size,
                             temp_buffer);
    }

    return rocsparse_status_not_implemented;
}

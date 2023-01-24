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

#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include "rocsparse_dotci.hpp"
#include "rocsparse_doti.hpp"

#define RETURN_SPVV(itype, ctype, ...)                                                      \
    {                                                                                       \
        if(itype == rocsparse_indextype_i32 && ctype == rocsparse_datatype_f32_r)           \
            return rocsparse_spvv_template<int32_t, float>(__VA_ARGS__);                    \
        if(itype == rocsparse_indextype_i32 && ctype == rocsparse_datatype_f64_r)           \
            return rocsparse_spvv_template<int32_t, double>(__VA_ARGS__);                   \
        if(itype == rocsparse_indextype_i32 && ctype == rocsparse_datatype_f32_c)           \
            return rocsparse_spvv_template<int32_t, rocsparse_float_complex>(__VA_ARGS__);  \
        if(itype == rocsparse_indextype_i32 && ctype == rocsparse_datatype_f64_c)           \
            return rocsparse_spvv_template<int32_t, rocsparse_double_complex>(__VA_ARGS__); \
        if(itype == rocsparse_indextype_i64 && ctype == rocsparse_datatype_f32_r)           \
            return rocsparse_spvv_template<int64_t, float>(__VA_ARGS__);                    \
        if(itype == rocsparse_indextype_i64 && ctype == rocsparse_datatype_f64_r)           \
            return rocsparse_spvv_template<int64_t, double>(__VA_ARGS__);                   \
        if(itype == rocsparse_indextype_i64 && ctype == rocsparse_datatype_f32_c)           \
            return rocsparse_spvv_template<int64_t, rocsparse_float_complex>(__VA_ARGS__);  \
        if(itype == rocsparse_indextype_i64 && ctype == rocsparse_datatype_f64_c)           \
            return rocsparse_spvv_template<int64_t, rocsparse_double_complex>(__VA_ARGS__); \
    }

template <typename I, typename T>
rocsparse_status rocsparse_spvv_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans,
                                         rocsparse_const_spvec_descr x,
                                         rocsparse_const_dnvec_descr y,
                                         void*                       result,
                                         rocsparse_datatype          compute_type,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    // If temp_buffer is nullptr, return buffer_size
    if(temp_buffer == nullptr)
    {
        // We do not need a buffer
        *buffer_size = 4;

        return rocsparse_status_success;
    }

    // real precision
    if(compute_type == rocsparse_datatype_f32_r || compute_type == rocsparse_datatype_f64_r)
    {
        return rocsparse_doti_template(handle,
                                       (I)x->nnz,
                                       (const T*)x->const_val_data,
                                       (const I*)x->const_idx_data,
                                       (const T*)y->const_values,
                                       (T*)result,
                                       x->idx_base);
    }

    // complex precision
    if(compute_type == rocsparse_datatype_f32_c || compute_type == rocsparse_datatype_f64_c)
    {
        // non transpose
        if(trans == rocsparse_operation_none)
        {
            return rocsparse_doti_template(handle,
                                           (I)x->nnz,
                                           (const T*)x->const_val_data,
                                           (const I*)x->const_idx_data,
                                           (const T*)y->const_values,
                                           (T*)result,
                                           x->idx_base);
        }

        // conjugate transpose
        if(trans == rocsparse_operation_conjugate_transpose)
        {
            return rocsparse_dotci_template(handle,
                                            (I)x->nnz,
                                            (const T*)x->const_val_data,
                                            (const I*)x->const_idx_data,
                                            (const T*)y->const_values,
                                            (T*)result,
                                            x->idx_base);
        }
    }

    return rocsparse_status_not_implemented;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spvv(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spvec_descr x,
                                           rocsparse_const_dnvec_descr y,
                                           void*                       result,
                                           rocsparse_datatype          compute_type,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spvv",
              trans,
              (const void*&)x,
              (const void*&)y,
              (const void*&)result,
              compute_type,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check operation
    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    // Check compute type
    if(rocsparse_enum_utils::is_invalid(compute_type))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(x);
    RETURN_IF_NULLPTR(y);

    // Check for valid pointers
    RETURN_IF_NULLPTR(result);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        RETURN_IF_NULLPTR(buffer_size);
    }

    // Check if descriptors are initialized
    if(x->init == false || y->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Check for matching types while we do not support mixed precision computation
    if(compute_type != x->data_type || compute_type != y->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    RETURN_SPVV(x->idx_type,
                compute_type,
                handle,
                trans,
                x,
                y,
                result,
                compute_type,
                buffer_size,
                temp_buffer);

    return rocsparse_status_not_implemented;
}

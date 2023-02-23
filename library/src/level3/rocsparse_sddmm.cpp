/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"
#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include "rocsparse_sddmm.hpp"

template <rocsparse_format FORMAT, typename I, typename J, typename T, typename... Ts>
rocsparse_status rocsparse_sddmm_buffer_size_dispatch_alg(rocsparse_sddmm_alg alg, Ts&&... ts)
{
    switch(alg)
    {
    case rocsparse_sddmm_alg_default:
    {
        return rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_default, I, J, T>::
            buffer_size_template(ts...);
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename J, typename T, typename... Ts>
rocsparse_status rocsparse_sddmm_buffer_size_dispatch_format(rocsparse_format    format,
                                                             rocsparse_sddmm_alg alg,
                                                             Ts&&... ts)
{
    switch(format)
    {
    case rocsparse_format_coo:
    {
        return rocsparse_sddmm_buffer_size_dispatch_alg<rocsparse_format_coo, I, I, T>(alg, ts...);
    }

    case rocsparse_format_csr:
    {
        return rocsparse_sddmm_buffer_size_dispatch_alg<rocsparse_format_csr, I, J, T>(alg, ts...);
    }

    case rocsparse_format_coo_aos:
    {

        return rocsparse_sddmm_buffer_size_dispatch_alg<rocsparse_format_coo_aos, I, I, T>(alg,
                                                                                           ts...);
    }

    case rocsparse_format_csc:
    {
        return rocsparse_sddmm_buffer_size_dispatch_alg<rocsparse_format_csc, I, J, T>(alg, ts...);
    }

    case rocsparse_format_ell:
    {
        return rocsparse_sddmm_buffer_size_dispatch_alg<rocsparse_format_ell, I, I, T>(alg, ts...);
    }
    case rocsparse_format_bell:
    {
        return rocsparse_status_not_implemented;
    }
    case rocsparse_format_bsr:
    {
        return rocsparse_status_not_implemented;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename... Ts>
rocsparse_status rocsparse_sddmm_buffer_size_dispatch(rocsparse_format    format,
                                                      rocsparse_indextype itype,
                                                      rocsparse_indextype jtype,
                                                      rocsparse_datatype  ctype,
                                                      rocsparse_sddmm_alg alg,
                                                      Ts&&... ts)
{
    switch(ctype)
    {

#define DATATYPE_CASE(ENUMVAL, TYPE)                                                        \
    case ENUMVAL:                                                                           \
    {                                                                                       \
        switch(itype)                                                                       \
        {                                                                                   \
        case rocsparse_indextype_u16:                                                       \
        {                                                                                   \
            return rocsparse_status_not_implemented;                                        \
        }                                                                                   \
        case rocsparse_indextype_i32:                                                       \
        {                                                                                   \
            switch(jtype)                                                                   \
            {                                                                               \
            case rocsparse_indextype_u16:                                                   \
            case rocsparse_indextype_i64:                                                   \
            {                                                                               \
                return rocsparse_status_not_implemented;                                    \
            }                                                                               \
            case rocsparse_indextype_i32:                                                   \
            {                                                                               \
                return rocsparse_sddmm_buffer_size_dispatch_format<int32_t, int32_t, TYPE>( \
                    format, alg, ts...);                                                    \
            }                                                                               \
            }                                                                               \
        }                                                                                   \
        case rocsparse_indextype_i64:                                                       \
        {                                                                                   \
            switch(jtype)                                                                   \
            {                                                                               \
            case rocsparse_indextype_u16:                                                   \
            {                                                                               \
                return rocsparse_status_not_implemented;                                    \
            }                                                                               \
            case rocsparse_indextype_i32:                                                   \
            {                                                                               \
                return rocsparse_sddmm_buffer_size_dispatch_format<int64_t, int32_t, TYPE>( \
                    format, alg, ts...);                                                    \
            }                                                                               \
            case rocsparse_indextype_i64:                                                   \
            {                                                                               \
                return rocsparse_sddmm_buffer_size_dispatch_format<int64_t, int64_t, TYPE>( \
                    format, alg, ts...);                                                    \
            }                                                                               \
            }                                                                               \
        }                                                                                   \
        }                                                                                   \
    }

        DATATYPE_CASE(rocsparse_datatype_f32_r, float);
        DATATYPE_CASE(rocsparse_datatype_f64_r, double);
        DATATYPE_CASE(rocsparse_datatype_f32_c, rocsparse_float_complex);
        DATATYPE_CASE(rocsparse_datatype_f64_c, rocsparse_double_complex);
        //DATATYPE_CASE(rocsparse_datatype_i8_r, int8_t);
        //DATATYPE_CASE(rocsparse_datatype_u8_r, uint8_t);
        //DATATYPE_CASE(rocsparse_datatype_i32_r, int32_t);
        //DATATYPE_CASE(rocsparse_datatype_u32_r, uint32_t);

    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return rocsparse_status_not_implemented;
    }

#undef DATATYPE_CASE
    }

    return rocsparse_status_invalid_value;
}

extern "C" rocsparse_status rocsparse_sddmm_buffer_size(rocsparse_handle            handle,
                                                        rocsparse_operation         trans_A,
                                                        rocsparse_operation         trans_B,
                                                        const void*                 alpha,
                                                        rocsparse_const_dnmat_descr mat_A,
                                                        rocsparse_const_dnmat_descr mat_B,
                                                        const void*                 beta,
                                                        const rocsparse_spmat_descr mat_C,
                                                        rocsparse_datatype          compute_type,
                                                        rocsparse_sddmm_alg         alg,
                                                        size_t*                     buffer_size)
try
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_sddmm_buffer_size",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)mat_A,
              (const void*&)mat_B,
              (const void*&)beta,
              (const void*&)mat_C,
              compute_type,
              alg,
              (const void*&)buffer_size);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(compute_type))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    RETURN_IF_NULLPTR(buffer_size);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat_A);
    RETURN_IF_NULLPTR(mat_B);
    RETURN_IF_NULLPTR(mat_C);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

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
    return rocsparse_sddmm_buffer_size_dispatch(
        mat_C->format,
        (mat_C->format == rocsparse_format_csc) ? mat_C->col_type : mat_C->row_type,
        (mat_C->format == rocsparse_format_csc) ? mat_C->row_type : mat_C->col_type,
        compute_type,
        alg,
        //
        handle,
        trans_A,
        trans_B,
        alpha,
        mat_A,
        mat_B,
        beta,
        mat_C,
        compute_type,
        alg,
        buffer_size);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

template <rocsparse_format FORMAT, typename I, typename J, typename T, typename... Ts>
rocsparse_status rocsparse_sddmm_preprocess_dispatch_alg(rocsparse_sddmm_alg alg, Ts&&... ts)
{
    switch(alg)
    {
    case rocsparse_sddmm_alg_default:
    {
        return rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_default, I, J, T>::
            preprocess_template(ts...);
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename J, typename T, typename... Ts>
rocsparse_status rocsparse_sddmm_preprocess_dispatch_format(rocsparse_format    format,
                                                            rocsparse_sddmm_alg alg,
                                                            Ts&&... ts)
{
    switch(format)
    {
    case rocsparse_format_coo:
    {
        return rocsparse_sddmm_preprocess_dispatch_alg<rocsparse_format_coo, I, I, T>(alg, ts...);
    }

    case rocsparse_format_csr:
    {
        return rocsparse_sddmm_preprocess_dispatch_alg<rocsparse_format_csr, I, J, T>(alg, ts...);
    }

    case rocsparse_format_coo_aos:
    {

        return rocsparse_sddmm_preprocess_dispatch_alg<rocsparse_format_coo_aos, I, I, T>(alg,
                                                                                          ts...);
    }

    case rocsparse_format_csc:
    {
        return rocsparse_sddmm_preprocess_dispatch_alg<rocsparse_format_csc, I, J, T>(alg, ts...);
    }

    case rocsparse_format_ell:
    {
        return rocsparse_sddmm_preprocess_dispatch_alg<rocsparse_format_ell, I, I, T>(alg, ts...);
    }

    case rocsparse_format_bell:
    {
        return rocsparse_status_not_implemented;
    }
    case rocsparse_format_bsr:
    {
        return rocsparse_status_not_implemented;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename... Ts>
rocsparse_status rocsparse_sddmm_preprocess_dispatch(rocsparse_format    format,
                                                     rocsparse_indextype itype,
                                                     rocsparse_indextype jtype,
                                                     rocsparse_datatype  ctype,
                                                     rocsparse_sddmm_alg alg,
                                                     Ts&&... ts)
{
    switch(ctype)
    {
#define DATATYPE_CASE(ENUMVAL, TYPE)                                                       \
    case ENUMVAL:                                                                          \
    {                                                                                      \
        switch(itype)                                                                      \
        {                                                                                  \
        case rocsparse_indextype_u16:                                                      \
        {                                                                                  \
            return rocsparse_status_not_implemented;                                       \
        }                                                                                  \
        case rocsparse_indextype_i32:                                                      \
        {                                                                                  \
            switch(jtype)                                                                  \
            {                                                                              \
            case rocsparse_indextype_u16:                                                  \
            case rocsparse_indextype_i64:                                                  \
            {                                                                              \
                return rocsparse_status_not_implemented;                                   \
            }                                                                              \
            case rocsparse_indextype_i32:                                                  \
            {                                                                              \
                return rocsparse_sddmm_preprocess_dispatch_format<int32_t, int32_t, TYPE>( \
                    format, alg, ts...);                                                   \
            }                                                                              \
            }                                                                              \
        }                                                                                  \
        case rocsparse_indextype_i64:                                                      \
        {                                                                                  \
            switch(jtype)                                                                  \
            {                                                                              \
            case rocsparse_indextype_u16:                                                  \
            {                                                                              \
                return rocsparse_status_not_implemented;                                   \
            }                                                                              \
            case rocsparse_indextype_i32:                                                  \
            {                                                                              \
                return rocsparse_sddmm_preprocess_dispatch_format<int64_t, int32_t, TYPE>( \
                    format, alg, ts...);                                                   \
            }                                                                              \
            case rocsparse_indextype_i64:                                                  \
            {                                                                              \
                return rocsparse_sddmm_preprocess_dispatch_format<int64_t, int64_t, TYPE>( \
                    format, alg, ts...);                                                   \
            }                                                                              \
            }                                                                              \
        }                                                                                  \
        }                                                                                  \
    }

        DATATYPE_CASE(rocsparse_datatype_f32_r, float);
        DATATYPE_CASE(rocsparse_datatype_f64_r, double);
        DATATYPE_CASE(rocsparse_datatype_f32_c, rocsparse_float_complex);
        DATATYPE_CASE(rocsparse_datatype_f64_c, rocsparse_double_complex);
        //DATATYPE_CASE(rocsparse_datatype_i8_r, int8_t);
        //DATATYPE_CASE(rocsparse_datatype_u8_r, uint8_t);
        //DATATYPE_CASE(rocsparse_datatype_i32_r, int32_t);
        //DATATYPE_CASE(rocsparse_datatype_u32_r, uint32_t);

    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return rocsparse_status_not_implemented;
    }

#undef DATATYPE_CASE
    }

    return rocsparse_status_invalid_value;
}

extern "C" rocsparse_status rocsparse_sddmm_preprocess(rocsparse_handle            handle,
                                                       rocsparse_operation         trans_A,
                                                       rocsparse_operation         trans_B,
                                                       const void*                 alpha,
                                                       rocsparse_const_dnmat_descr mat_A,
                                                       rocsparse_const_dnmat_descr mat_B,
                                                       const void*                 beta,
                                                       const rocsparse_spmat_descr mat_C,
                                                       rocsparse_datatype          compute_type,
                                                       rocsparse_sddmm_alg         alg,
                                                       void*                       temp_buffer)
try
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_sddmm_preprocess",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)mat_A,
              (const void*&)mat_B,
              (const void*&)beta,
              (const void*&)mat_C,
              compute_type,
              alg,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(compute_type))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat_A);
    RETURN_IF_NULLPTR(mat_B);
    RETURN_IF_NULLPTR(mat_C);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

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

    if(mat_C->nnz == 0)
    {
        return rocsparse_status_success;
    }

    return rocsparse_sddmm_preprocess_dispatch(
        mat_C->format,
        (mat_C->format == rocsparse_format_csc) ? mat_C->col_type : mat_C->row_type,
        (mat_C->format == rocsparse_format_csc) ? mat_C->row_type : mat_C->col_type,
        compute_type,
        alg,
        //
        handle,
        trans_A,
        trans_B,
        alpha,
        mat_A,
        mat_B,
        beta,
        mat_C,
        compute_type,
        alg,
        temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

template <rocsparse_format FORMAT, typename I, typename J, typename T, typename... Ts>
rocsparse_status rocsparse_sddmm_dispatch_alg(rocsparse_sddmm_alg alg, Ts&&... ts)
{
    switch(alg)
    {
    case rocsparse_sddmm_alg_default:
    {
        return rocsparse_sddmm_st<FORMAT, rocsparse_sddmm_alg_default, I, J, T>::compute_template(
            ts...);
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename J, typename T, typename... Ts>
rocsparse_status
    rocsparse_sddmm_dispatch_format(rocsparse_format format, rocsparse_sddmm_alg alg, Ts&&... ts)
{
    switch(format)
    {
    case rocsparse_format_coo:
    {
        return rocsparse_sddmm_dispatch_alg<rocsparse_format_coo, I, I, T>(alg, ts...);
    }

    case rocsparse_format_csr:
    {
        return rocsparse_sddmm_dispatch_alg<rocsparse_format_csr, I, J, T>(alg, ts...);
    }

    case rocsparse_format_coo_aos:
    {

        return rocsparse_sddmm_dispatch_alg<rocsparse_format_coo_aos, I, I, T>(alg, ts...);
    }

    case rocsparse_format_csc:
    {
        return rocsparse_sddmm_dispatch_alg<rocsparse_format_csc, I, J, T>(alg, ts...);
    }

    case rocsparse_format_ell:
    {
        return rocsparse_sddmm_dispatch_alg<rocsparse_format_ell, I, I, T>(alg, ts...);
    }

    case rocsparse_format_bell:
    {
        return rocsparse_status_not_implemented;
    }
    case rocsparse_format_bsr:
    {
        return rocsparse_status_not_implemented;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename... Ts>
rocsparse_status rocsparse_sddmm_dispatch(rocsparse_format    format,
                                          rocsparse_indextype itype,
                                          rocsparse_indextype jtype,
                                          rocsparse_datatype  ctype,
                                          rocsparse_sddmm_alg alg,
                                          Ts&&... ts)
{
    switch(ctype)
    {

#define DATATYPE_CASE(ENUMVAL, TYPE)                                            \
    case ENUMVAL:                                                               \
    {                                                                           \
        switch(itype)                                                           \
        {                                                                       \
        case rocsparse_indextype_u16:                                           \
        {                                                                       \
            return rocsparse_status_not_implemented;                            \
        }                                                                       \
        case rocsparse_indextype_i32:                                           \
        {                                                                       \
            switch(jtype)                                                       \
            {                                                                   \
            case rocsparse_indextype_u16:                                       \
            case rocsparse_indextype_i64:                                       \
            {                                                                   \
                return rocsparse_status_not_implemented;                        \
            }                                                                   \
            case rocsparse_indextype_i32:                                       \
            {                                                                   \
                return rocsparse_sddmm_dispatch_format<int32_t, int32_t, TYPE>( \
                    format, alg, ts...);                                        \
            }                                                                   \
            }                                                                   \
        }                                                                       \
        case rocsparse_indextype_i64:                                           \
        {                                                                       \
            switch(jtype)                                                       \
            {                                                                   \
            case rocsparse_indextype_u16:                                       \
            {                                                                   \
                return rocsparse_status_not_implemented;                        \
            }                                                                   \
            case rocsparse_indextype_i32:                                       \
            {                                                                   \
                return rocsparse_sddmm_dispatch_format<int64_t, int32_t, TYPE>( \
                    format, alg, ts...);                                        \
            }                                                                   \
            case rocsparse_indextype_i64:                                       \
            {                                                                   \
                return rocsparse_sddmm_dispatch_format<int64_t, int64_t, TYPE>( \
                    format, alg, ts...);                                        \
            }                                                                   \
            }                                                                   \
        }                                                                       \
        }                                                                       \
    }

        DATATYPE_CASE(rocsparse_datatype_f32_r, float);
        DATATYPE_CASE(rocsparse_datatype_f64_r, double);
        DATATYPE_CASE(rocsparse_datatype_f32_c, rocsparse_float_complex);
        DATATYPE_CASE(rocsparse_datatype_f64_c, rocsparse_double_complex);
        //DATATYPE_CASE(rocsparse_datatype_i8_r, int8_t);
        //DATATYPE_CASE(rocsparse_datatype_u8_r, uint8_t);
        //DATATYPE_CASE(rocsparse_datatype_i32_r, int32_t);
        //DATATYPE_CASE(rocsparse_datatype_u32_r, uint32_t);

    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return rocsparse_status_not_implemented;
    }

#undef DATATYPE_CASE
    }
    return rocsparse_status_invalid_value;
}

extern "C" rocsparse_status rocsparse_sddmm(rocsparse_handle            handle,
                                            rocsparse_operation         trans_A,
                                            rocsparse_operation         trans_B,
                                            const void*                 alpha,
                                            rocsparse_const_dnmat_descr mat_A,
                                            rocsparse_const_dnmat_descr mat_B,
                                            const void*                 beta,
                                            const rocsparse_spmat_descr mat_C,
                                            rocsparse_datatype          compute_type,
                                            rocsparse_sddmm_alg         alg,
                                            void*                       temp_buffer)
try
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_sddmm",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)mat_A,
              (const void*&)mat_B,
              (const void*&)beta,
              (const void*&)mat_C,
              compute_type,
              alg,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(compute_type))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(alg))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat_A);
    RETURN_IF_NULLPTR(mat_B);
    RETURN_IF_NULLPTR(mat_C);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

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
    if(mat_C->nnz == 0)
    {
        return rocsparse_status_success;
    }

    return rocsparse_sddmm_dispatch(
        mat_C->format,
        (mat_C->format == rocsparse_format_csc) ? mat_C->col_type : mat_C->row_type,
        (mat_C->format == rocsparse_format_csc) ? mat_C->row_type : mat_C->col_type,
        compute_type,
        alg,
        //
        handle,
        trans_A,
        trans_B,
        alpha,
        mat_A,
        mat_B,
        beta,
        mat_C,
        compute_type,
        alg,
        temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

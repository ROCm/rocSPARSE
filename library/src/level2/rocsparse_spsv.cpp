/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_coosv.hpp"
#include "rocsparse_csrsv.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spsv_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans,
                                         const void*                 alpha,
                                         const rocsparse_spmat_descr mat,
                                         const rocsparse_dnvec_descr x,
                                         rocsparse_dnvec_descr       y,
                                         rocsparse_spsv_alg          alg,
                                         rocsparse_spsv_stage        stage,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    // STAGE 1 - compute required buffer size of temp_buffer
    if(stage == rocsparse_spsv_stage_buffer_size
       || (stage == rocsparse_spsv_stage_auto && temp_buffer == nullptr))
    {
        if(mat->format == rocsparse_format_csr)
        {
            return rocsparse_csrsv_buffer_size_template(handle,
                                                        trans,
                                                        (J)mat->rows,
                                                        (I)mat->nnz,
                                                        mat->descr,
                                                        (const T*)mat->val_data,
                                                        (const I*)mat->row_data,
                                                        (const J*)mat->col_data,
                                                        mat->info,
                                                        buffer_size);
        }
        else if(mat->format == rocsparse_format_coo)
        {
            return rocsparse_coosv_buffer_size_template(handle,
                                                        trans,
                                                        (I)mat->rows,
                                                        mat->nnz,
                                                        mat->descr,
                                                        (const T*)mat->val_data,
                                                        (const I*)mat->row_data,
                                                        (const I*)mat->col_data,
                                                        mat->info,
                                                        buffer_size);
        }
        else
        {
            return rocsparse_status_not_implemented;
        }
    }

    // STAGE 2 - preprocess stage
    if(stage == rocsparse_spsv_stage_preprocess
       || (stage == rocsparse_spsv_stage_auto && buffer_size == nullptr))
    {
        if(mat->analysed == false)
        {
            if(mat->format == rocsparse_format_csr)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_csrsv_analysis_template(handle,
                                                       trans,
                                                       (J)mat->rows,
                                                       (I)mat->nnz,
                                                       mat->descr,
                                                       (const T*)mat->val_data,
                                                       (const I*)mat->row_data,
                                                       (const J*)mat->col_data,
                                                       mat->info,
                                                       rocsparse_analysis_policy_force,
                                                       rocsparse_solve_policy_auto,
                                                       temp_buffer)));
            }
            else if(mat->format == rocsparse_format_coo)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_coosv_analysis_template(handle,
                                                       trans,
                                                       (I)mat->rows,
                                                       mat->nnz,
                                                       mat->descr,
                                                       (const T*)mat->val_data,
                                                       (const I*)mat->row_data,
                                                       (const I*)mat->col_data,
                                                       mat->info,
                                                       rocsparse_analysis_policy_force,
                                                       rocsparse_solve_policy_auto,
                                                       temp_buffer)));
            }
            else
            {
                return rocsparse_status_not_implemented;
            }

            mat->analysed = true;
        }

        return rocsparse_status_success;
    }

    // STAGE 3 - perform SpSV computation
    if(stage == rocsparse_spsv_stage_compute || stage == rocsparse_spsv_stage_auto)
    {
        if(mat->format == rocsparse_format_csr)
        {
            return rocsparse_csrsv_solve_template(handle,
                                                  trans,
                                                  (J)mat->rows,
                                                  (I)mat->nnz,
                                                  (const T*)alpha,
                                                  mat->descr,
                                                  (const T*)mat->val_data,
                                                  (const I*)mat->row_data,
                                                  (const J*)mat->col_data,
                                                  mat->info,
                                                  (const T*)x->values,
                                                  (T*)y->values,
                                                  rocsparse_solve_policy_auto,
                                                  temp_buffer);
        }
        else if(mat->format == rocsparse_format_coo)
        {
            return rocsparse_coosv_solve_template(handle,
                                                  trans,
                                                  (I)mat->rows,
                                                  mat->nnz,
                                                  (const T*)alpha,
                                                  mat->descr,
                                                  (const T*)mat->val_data,
                                                  (const I*)mat->row_data,
                                                  (const I*)mat->col_data,
                                                  mat->info,
                                                  (const T*)x->values,
                                                  (T*)y->values,
                                                  rocsparse_solve_policy_auto,
                                                  temp_buffer);
        }
        else
        {
            return rocsparse_status_not_implemented;
        }
    }

    return rocsparse_status_not_implemented;
}

template <typename... Ts>
rocsparse_status rocsparse_spsv_dynamic_dispatch(rocsparse_indextype itype,
                                                 rocsparse_indextype jtype,
                                                 rocsparse_datatype  ctype,
                                                 Ts&&... ts)
{
    switch(ctype)
    {

#define DATATYPE_CASE(ENUMVAL, TYPE)                                           \
    case ENUMVAL:                                                              \
    {                                                                          \
        switch(itype)                                                          \
        {                                                                      \
        case rocsparse_indextype_u16:                                          \
        {                                                                      \
            return rocsparse_status_not_implemented;                           \
        }                                                                      \
        case rocsparse_indextype_i32:                                          \
        {                                                                      \
            switch(jtype)                                                      \
            {                                                                  \
            case rocsparse_indextype_u16:                                      \
            case rocsparse_indextype_i64:                                      \
            {                                                                  \
                return rocsparse_status_not_implemented;                       \
            }                                                                  \
            case rocsparse_indextype_i32:                                      \
            {                                                                  \
                return rocsparse_spsv_template<int32_t, int32_t, TYPE>(ts...); \
            }                                                                  \
            }                                                                  \
        }                                                                      \
        case rocsparse_indextype_i64:                                          \
        {                                                                      \
            switch(jtype)                                                      \
            {                                                                  \
            case rocsparse_indextype_u16:                                      \
            {                                                                  \
                return rocsparse_status_not_implemented;                       \
            }                                                                  \
            case rocsparse_indextype_i32:                                      \
            {                                                                  \
                return rocsparse_spsv_template<int64_t, int32_t, TYPE>(ts...); \
            }                                                                  \
            case rocsparse_indextype_i64:                                      \
            {                                                                  \
                return rocsparse_spsv_template<int64_t, int64_t, TYPE>(ts...); \
            }                                                                  \
            }                                                                  \
        }                                                                      \
        }                                                                      \
    }

        DATATYPE_CASE(rocsparse_datatype_f32_r, float);
        DATATYPE_CASE(rocsparse_datatype_f64_r, double);
        DATATYPE_CASE(rocsparse_datatype_f32_c, rocsparse_float_complex);
        DATATYPE_CASE(rocsparse_datatype_f64_c, rocsparse_double_complex);

#undef DATATYPE_CASE
    }
    // LCOV_EXCL_START
    return rocsparse_status_invalid_value;
    // LCOV_EXCL_STOP
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spsv(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr mat,
                                           const rocsparse_dnvec_descr x,
                                           const rocsparse_dnvec_descr y,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_spsv_alg          alg,
                                           rocsparse_spsv_stage        stage,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spsv",
              trans,
              (const void*&)alpha,
              (const void*&)mat,
              (const void*&)x,
              (const void*&)y,
              compute_type,
              alg,
              stage,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat);
    RETURN_IF_NULLPTR(x);
    RETURN_IF_NULLPTR(y);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);

    if(rocsparse_enum_utils::is_invalid(trans))
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

    if(rocsparse_enum_utils::is_invalid(stage))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        RETURN_IF_NULLPTR(buffer_size);
    }

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    if(mat->init == false || x->init == false || y->init == false)
    {
        return rocsparse_status_not_initialized;
    }
    // LCOV_EXCL_STOP

    // Check for matching types while we do not support mixed precision computation
    if(compute_type != mat->data_type || compute_type != x->data_type
       || compute_type != y->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    return rocsparse_spsv_dynamic_dispatch(mat->row_type,
                                           mat->col_type,
                                           compute_type,
                                           handle,
                                           trans,
                                           alpha,
                                           mat,
                                           x,
                                           y,
                                           alg,
                                           stage,
                                           buffer_size,
                                           temp_buffer);
}

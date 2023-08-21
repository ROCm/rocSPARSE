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

#include "internal/generic/rocsparse_spsm.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spsm_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans_A,
                                         rocsparse_operation         trans_B,
                                         const void*                 alpha,
                                         rocsparse_const_spmat_descr matA,
                                         rocsparse_const_dnmat_descr matB,
                                         const rocsparse_dnmat_descr matC,
                                         rocsparse_spsm_alg          alg,
                                         rocsparse_spsm_stage        stage,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    // STAGE 1 - compute required buffer size of temp_buffer
    if(stage == rocsparse_spsm_stage_buffer_size)
    {
        if(matA->format == rocsparse_format_csr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsm_buffer_size_template(
                handle,
                trans_A,
                trans_B,
                (J)matA->rows,
                (trans_B == rocsparse_operation_none ? (J)matB->cols : (J)matB->rows),
                (I)matA->nnz,
                (const T*)alpha,
                matA->descr,
                (const T*)matA->const_val_data,
                (const I*)matA->const_row_data,
                (const J*)matA->const_col_data,
                (const T*)matB->const_values,
                (J)matB->ld,
                matA->info,
                rocsparse_solve_policy_auto,
                buffer_size));

            *buffer_size = std::max(static_cast<size_t>(4), *buffer_size);
            return rocsparse_status_success;
        }
        else if(matA->format == rocsparse_format_coo)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_coosm_buffer_size_template(
                handle,
                trans_A,
                trans_B,
                (I)matA->rows,
                (trans_B == rocsparse_operation_none ? (I)matB->cols : (I)matB->rows),
                matA->nnz,
                (const T*)alpha,
                matA->descr,
                (const T*)matA->const_val_data,
                (const I*)matA->const_row_data,
                (const I*)matA->const_col_data,
                (const T*)matB->const_values,
                (I)matB->ld,
                matA->info,
                rocsparse_solve_policy_auto,
                buffer_size));

            *buffer_size = std::max(static_cast<size_t>(4), *buffer_size);
            return rocsparse_status_success;
        }
        else
        {
            return rocsparse_status_not_implemented;
        }
    }

    // STAGE 2 - preprocess stage
    if(stage == rocsparse_spsm_stage_preprocess)
    {
        if(matA->analysed == false)
        {
            if(matA->format == rocsparse_format_csr)
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_csrsm_analysis_template(
                    handle,
                    trans_A,
                    trans_B,
                    (J)matA->rows,
                    (trans_B == rocsparse_operation_none ? (J)matB->cols : (J)matB->rows),
                    (I)matA->nnz,
                    (const T*)alpha,
                    matA->descr,
                    (const T*)matA->const_val_data,
                    (const I*)matA->const_row_data,
                    (const J*)matA->const_col_data,
                    (const T*)matB->const_values,
                    (J)matB->ld,
                    matA->info,
                    rocsparse_analysis_policy_force,
                    rocsparse_solve_policy_auto,
                    temp_buffer)));
            }
            else if(matA->format == rocsparse_format_coo)
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_coosm_analysis_template(
                    handle,
                    trans_A,
                    trans_B,
                    (I)matA->rows,
                    (trans_B == rocsparse_operation_none ? (I)matB->cols : (I)matB->rows),
                    matA->nnz,
                    (const T*)alpha,
                    matA->descr,
                    (const T*)matA->const_val_data,
                    (const I*)matA->const_row_data,
                    (const I*)matA->const_col_data,
                    (const T*)matB->const_values,
                    (I)matB->ld,
                    matA->info,
                    rocsparse_analysis_policy_force,
                    rocsparse_solve_policy_auto,
                    temp_buffer)));
            }
            else
            {
                return rocsparse_status_not_implemented;
            }

            matA->analysed = true;
        }

        return rocsparse_status_success;
    }

    // STAGE 3 - perform SpSM computation
    if(stage == rocsparse_spsm_stage_compute)
    {
        // copy B to C and perform in-place using C
        if(matB->rows > 0 && matB->cols > 0)
        {
            hipMemcpy2DAsync(matC->values,
                             matC->ld * sizeof(T),
                             matB->values,
                             matB->ld * sizeof(T),
                             (J)matB->rows * sizeof(T),
                             (J)matB->cols,
                             hipMemcpyDeviceToDevice,
                             handle->stream);
        }

        if(matA->format == rocsparse_format_csr)
        {
            return rocsparse_csrsm_solve_template(
                handle,
                trans_A,
                trans_B,
                (J)matA->rows,
                (trans_B == rocsparse_operation_none ? (J)matB->cols : (J)matB->rows),
                (I)matA->nnz,
                (const T*)alpha,
                matA->descr,
                (const T*)matA->const_val_data,
                (const I*)matA->const_row_data,
                (const J*)matA->const_col_data,
                (T*)matC->values,
                (J)matC->ld,
                matA->info,
                rocsparse_solve_policy_auto,
                temp_buffer);
        }
        else if(matA->format == rocsparse_format_coo)
        {
            return rocsparse_coosm_solve_template(
                handle,
                trans_A,
                trans_B,
                (I)matA->rows,
                (trans_B == rocsparse_operation_none ? (I)matB->cols : (I)matB->rows),
                matA->nnz,
                (const T*)alpha,
                matA->descr,
                (const T*)matA->const_val_data,
                (const I*)matA->const_row_data,
                (const I*)matA->const_col_data,
                (T*)matC->values,
                (I)matC->ld,
                matA->info,
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
rocsparse_status rocsparse_spsm_dynamic_dispatch(rocsparse_indextype itype,
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
                return rocsparse_spsm_template<int32_t, int32_t, TYPE>(ts...); \
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
                return rocsparse_spsm_template<int64_t, int32_t, TYPE>(ts...); \
            }                                                                  \
            case rocsparse_indextype_i64:                                      \
            {                                                                  \
                return rocsparse_spsm_template<int64_t, int64_t, TYPE>(ts...); \
            }                                                                  \
            }                                                                  \
        }                                                                      \
        }                                                                      \
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
    // LCOV_EXCL_START
    return rocsparse_status_invalid_value;
    // LCOV_EXCL_STOP
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spsm(rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           rocsparse_const_spmat_descr matA,
                                           rocsparse_const_dnmat_descr matB,
                                           const rocsparse_dnmat_descr matC,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_spsm_alg          alg,
                                           rocsparse_spsm_stage        stage,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
try
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spsm",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)matA,
              (const void*&)matB,
              (const void*&)matC,
              compute_type,
              alg,
              stage,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(matA);
    RETURN_IF_NULLPTR(matB);
    RETURN_IF_NULLPTR(matC);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);

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
    if(matA->init == false || matB->init == false || matC->init == false)
    {
        return rocsparse_status_not_initialized;
    }
    // LCOV_EXCL_STOP

    // Check for matching types while we do not support mixed precision computation
    if(compute_type != matA->data_type || compute_type != matB->data_type
       || compute_type != matC->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    return rocsparse_spsm_dynamic_dispatch(matA->row_type,
                                           matA->col_type,
                                           compute_type,
                                           handle,
                                           trans_A,
                                           trans_B,
                                           alpha,
                                           matA,
                                           matB,
                                           matC,
                                           alg,
                                           stage,
                                           buffer_size,
                                           temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

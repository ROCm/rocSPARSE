/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status spsm_template(rocsparse_handle            handle,
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
        switch(stage)
        {
        case rocsparse_spsm_stage_buffer_size:
        {
            switch(matA->format)
            {

            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_buffer_size_template(
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
                    matB->ld,
                    matA->info,
                    rocsparse_solve_policy_auto,
                    buffer_size));

                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_buffer_size_template(
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
                    matB->ld,
                    matA->info,
                    rocsparse_solve_policy_auto,
                    buffer_size));

                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spsm_stage_preprocess:
        {
            switch(matA->format)
            {

            case rocsparse_format_csr:
            {
                if(matA->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsm_analysis_template(
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
                        matB->ld,
                        matA->info,
                        rocsparse_analysis_policy_force,
                        rocsparse_solve_policy_auto,
                        temp_buffer)));
                    matA->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {

                if(matA->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR((rocsparse::coosm_analysis_template(
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
                        matB->ld,
                        matA->info,
                        rocsparse_analysis_policy_force,
                        rocsparse_solve_policy_auto,
                        temp_buffer)));
                    matA->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spsm_stage_compute:
        {
            // copy B to C and perform in-place using C
            if(matB->rows > 0 && matB->cols > 0)
            {
                hipMemcpy2DAsync(matC->values,
                                 sizeof(T) * matC->ld,
                                 matB->values,
                                 sizeof(T) * matB->ld,
                                 (J)matB->rows * sizeof(T),
                                 (J)matB->cols,
                                 hipMemcpyDeviceToDevice,
                                 handle->stream);
            }

            switch(matA->format)
            {

            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve_template(
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
                    matC->ld,
                    matA->info,
                    rocsparse_solve_policy_auto,
                    temp_buffer));
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve_template(
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
                    matC->ld,
                    matA->info,
                    rocsparse_solve_policy_auto,
                    temp_buffer));
                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            case rocsparse_format_csc:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        }
    }

    template <typename... Ts>
    rocsparse_status spsm_dynamic_dispatch(rocsparse_indextype itype,
                                           rocsparse_indextype jtype,
                                           rocsparse_datatype  ctype,
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
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);        \
        }                                                                       \
        case rocsparse_indextype_i32:                                           \
        {                                                                       \
            switch(jtype)                                                       \
            {                                                                   \
            case rocsparse_indextype_u16:                                       \
            case rocsparse_indextype_i64:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);    \
                return rocsparse_status_success;                                \
            }                                                                   \
            case rocsparse_indextype_i32:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(                                      \
                    (rocsparse::spsm_template<int32_t, int32_t, TYPE>(ts...))); \
                return rocsparse_status_success;                                \
            }                                                                   \
            }                                                                   \
        }                                                                       \
        case rocsparse_indextype_i64:                                           \
        {                                                                       \
            switch(jtype)                                                       \
            {                                                                   \
            case rocsparse_indextype_u16:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);    \
            }                                                                   \
            case rocsparse_indextype_i32:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(                                      \
                    (rocsparse::spsm_template<int64_t, int32_t, TYPE>(ts...))); \
                return rocsparse_status_success;                                \
            }                                                                   \
            case rocsparse_indextype_i64:                                       \
            {                                                                   \
                RETURN_IF_ROCSPARSE_ERROR(                                      \
                    (rocsparse::spsm_template<int64_t, int64_t, TYPE>(ts...))); \
                return rocsparse_status_success;                                \
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
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

#undef DATATYPE_CASE
        }
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spsm(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans_A, //1
                                           rocsparse_operation         trans_B, //2
                                           const void*                 alpha, //3
                                           rocsparse_const_spmat_descr matA, //4
                                           rocsparse_const_dnmat_descr matB, //5
                                           const rocsparse_dnmat_descr matC, //6
                                           rocsparse_datatype          compute_type, //7
                                           rocsparse_spsm_alg          alg, //8
                                           rocsparse_spsm_stage        stage, //9
                                           size_t*                     buffer_size, //10
                                           void*                       temp_buffer) //11
try
{

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

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, matA);
    ROCSPARSE_CHECKARG(4, matA, matA->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(5, matB);
    ROCSPARSE_CHECKARG(5, matB, matB->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(6, matC);
    ROCSPARSE_CHECKARG(6, matC, matC->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_ENUM(7, compute_type);
    ROCSPARSE_CHECKARG(7,
                       compute_type,
                       (compute_type != matA->data_type || compute_type != matB->data_type
                        || compute_type != matC->data_type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ENUM(8, alg);
    ROCSPARSE_CHECKARG_ENUM(9, stage);

    switch(stage)
    {
    case rocsparse_spsm_stage_buffer_size:
    {
        ROCSPARSE_CHECKARG_POINTER(10, buffer_size);
        break;
    }
    case rocsparse_spsm_stage_preprocess:
    {
        break;
    }
    case rocsparse_spsm_stage_compute:
    {
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsm_dynamic_dispatch(matA->row_type,
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
                                                               temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

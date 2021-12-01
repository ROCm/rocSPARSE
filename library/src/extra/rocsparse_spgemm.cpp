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
#include "utility.h"

#include "rocsparse_csrgemm.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spgemm_template(rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr A,
                                           const rocsparse_spmat_descr B,
                                           const void*                 beta,
                                           const rocsparse_spmat_descr D,
                                           rocsparse_spmat_descr       C,
                                           rocsparse_spgemm_alg        alg,
                                           rocsparse_spgemm_stage      stage,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer);

template <typename... Ts>
rocsparse_status rocsparse_spgemm_template_dispatch(rocsparse_indextype itype,
                                                    rocsparse_indextype jtype,
                                                    rocsparse_datatype  ctype,
                                                    Ts&&... params)
{
    ROCSPARSE_DEBUG_VERBOSE("begin");

    switch(itype)
    {
    case rocsparse_indextype_u16:
    {
        return rocsparse_status_not_implemented;
    }
    case rocsparse_indextype_i32:
    {
        switch(jtype)
        {
        case rocsparse_indextype_i64:
        case rocsparse_indextype_u16:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_indextype_i32:
        {
            switch(ctype)
            {
            case rocsparse_datatype_f32_r:
            {
                return rocsparse_spgemm_template<int32_t, int32_t, float>(params...);
            }
            case rocsparse_datatype_f64_r:
            {
                return rocsparse_spgemm_template<int32_t, int32_t, double>(params...);
            }
            case rocsparse_datatype_f32_c:
            {
                return rocsparse_spgemm_template<int32_t, int32_t, rocsparse_float_complex>(
                    params...);
            }
            case rocsparse_datatype_f64_c:
            {
                return rocsparse_spgemm_template<int32_t, int32_t, rocsparse_double_complex>(
                    params...);
            }
            }
        }
        }
    }
    case rocsparse_indextype_i64:
    {
        switch(jtype)
        {
        case rocsparse_indextype_u16:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_indextype_i32:
        {
            switch(ctype)
            {
            case rocsparse_datatype_f32_r:
            {
                return rocsparse_spgemm_template<int64_t, int32_t, float>(params...);
            }
            case rocsparse_datatype_f64_r:
            {
                return rocsparse_spgemm_template<int64_t, int32_t, double>(params...);
            }
            case rocsparse_datatype_f32_c:
            {
                return rocsparse_spgemm_template<int64_t, int32_t, rocsparse_float_complex>(
                    params...);
            }
            case rocsparse_datatype_f64_c:
            {
                return rocsparse_spgemm_template<int64_t, int32_t, rocsparse_double_complex>(
                    params...);
            }
            }
        }
        case rocsparse_indextype_i64:
        {
            switch(ctype)
            {
            case rocsparse_datatype_f32_r:
            {
                return rocsparse_spgemm_template<int64_t, int64_t, float>(params...);
            }
            case rocsparse_datatype_f64_r:
            {
                return rocsparse_spgemm_template<int64_t, int64_t, double>(params...);
            }
            case rocsparse_datatype_f32_c:
            {
                return rocsparse_spgemm_template<int64_t, int64_t, rocsparse_float_complex>(
                    params...);
            }
            case rocsparse_datatype_f64_c:
            {
                return rocsparse_spgemm_template<int64_t, int64_t, rocsparse_double_complex>(
                    params...);
            }
            }
        }
        }
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spgemm_template(rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr A,
                                           const rocsparse_spmat_descr B,
                                           const void*                 beta,
                                           const rocsparse_spmat_descr D,
                                           rocsparse_spmat_descr       C,
                                           rocsparse_spgemm_alg        alg,
                                           rocsparse_spgemm_stage      stage,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    ROCSPARSE_DEBUG_VERBOSE("begin");

    switch(stage)
    {
    case rocsparse_spgemm_stage_auto:
    {
        if(temp_buffer == nullptr)
        {
            return rocsparse_spgemm_template<I, J, T>(handle,
                                                      trans_A,
                                                      trans_B,
                                                      alpha,
                                                      A,
                                                      B,
                                                      beta,
                                                      D,
                                                      C,
                                                      alg,
                                                      rocsparse_spgemm_stage_buffer_size,
                                                      buffer_size,
                                                      temp_buffer);
        }
        else if(C->nnz == 0)
        {
            return rocsparse_spgemm_template<I, J, T>(handle,
                                                      trans_A,
                                                      trans_B,
                                                      alpha,
                                                      A,
                                                      B,
                                                      beta,
                                                      D,
                                                      C,
                                                      alg,
                                                      rocsparse_spgemm_stage_nnz,
                                                      buffer_size,
                                                      temp_buffer);
        }
        else
        {
            return rocsparse_spgemm_template<I, J, T>(handle,
                                                      trans_A,
                                                      trans_B,
                                                      alpha,
                                                      A,
                                                      B,
                                                      beta,
                                                      D,
                                                      C,
                                                      alg,
                                                      rocsparse_spgemm_stage_compute,
                                                      buffer_size,
                                                      temp_buffer);
        }
        break;
    }

    case rocsparse_spgemm_stage_buffer_size:
    {
        switch(A->format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_format_csr:
        {
            return rocsparse_csrgemm_buffer_size_template(handle,
                                                          trans_A,
                                                          trans_B,
                                                          (J)A->rows,
                                                          (J)B->cols,
                                                          (J)A->cols,
                                                          (const T*)alpha,
                                                          A->descr,
                                                          (I)A->nnz,
                                                          (const I*)A->row_data,
                                                          (const J*)A->col_data,
                                                          B->descr,
                                                          (I)B->nnz,
                                                          (const I*)B->row_data,
                                                          (const J*)B->col_data,
                                                          (const T*)beta,
                                                          D->descr,
                                                          (I)D->nnz,
                                                          (const I*)D->row_data,
                                                          (const J*)D->col_data,
                                                          C->info,
                                                          buffer_size);
        }
        }
    }

    case rocsparse_spgemm_stage_nnz:
    {
        switch(A->format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_format_csr:
        {
            I nnz_C;
            // non-zeros of C need to be on host
            rocsparse_pointer_mode ptr_mode;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode(handle, &ptr_mode));
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
            rocsparse_status status = rocsparse_csrgemm_nnz_template(handle,
                                                                     trans_A,
                                                                     trans_B,
                                                                     (J)A->rows,
                                                                     (J)B->cols,
                                                                     (J)A->cols,
                                                                     A->descr,
                                                                     (I)A->nnz,
                                                                     (const I*)A->row_data,
                                                                     (const J*)A->col_data,
                                                                     B->descr,
                                                                     (I)B->nnz,
                                                                     (const I*)B->row_data,
                                                                     (const J*)B->col_data,
                                                                     D->descr,
                                                                     (I)D->nnz,
                                                                     (const I*)D->row_data,
                                                                     (const J*)D->col_data,
                                                                     C->descr,
                                                                     (I*)C->row_data,
                                                                     &nnz_C,
                                                                     C->info,
                                                                     temp_buffer);

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, ptr_mode));

            C->nnz = nnz_C;

            return status;
        }
        }
    }

    case rocsparse_spgemm_stage_compute:
    {
        switch(A->format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_format_csr:
        {
            // CSR format
            return rocsparse_csrgemm_template(handle,
                                              trans_A,
                                              trans_B,
                                              (J)A->rows,
                                              (J)B->cols,
                                              (J)A->cols,
                                              (const T*)alpha,
                                              A->descr,
                                              (I)A->nnz,
                                              (const T*)A->val_data,
                                              (const I*)A->row_data,
                                              (const J*)A->col_data,
                                              B->descr,
                                              (I)B->nnz,
                                              (const T*)B->val_data,
                                              (const I*)B->row_data,
                                              (const J*)B->col_data,
                                              (const T*)beta,
                                              D->descr,
                                              (I)D->nnz,
                                              (const T*)D->val_data,
                                              (const I*)D->row_data,
                                              (const J*)D->col_data,
                                              C->descr,
                                              (T*)C->val_data,
                                              (const I*)C->row_data,
                                              (J*)C->col_data,
                                              C->info,
                                              temp_buffer);
        }
        }
    }

    case rocsparse_spgemm_stage_symbolic:
    {
        switch(A->format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_format_csr:
        {
            return rocsparse_csrgemm_symbolic_template(handle,
                                                       trans_A,
                                                       trans_B,
                                                       (J)A->rows,
                                                       (J)B->cols,
                                                       (J)A->cols,
                                                       A->descr,
                                                       (I)A->nnz,
                                                       (const I*)A->row_data,
                                                       (const J*)A->col_data,
                                                       B->descr,
                                                       (I)B->nnz,
                                                       (const I*)B->row_data,
                                                       (const J*)B->col_data,
                                                       D->descr,
                                                       (I)D->nnz,
                                                       (const I*)D->row_data,
                                                       (const J*)D->col_data,
                                                       C->descr,
                                                       (I)C->nnz,
                                                       (const I*)C->row_data,
                                                       (J*)C->col_data,
                                                       C->info,
                                                       temp_buffer);
        }
        }
    }

    case rocsparse_spgemm_stage_numeric:
    {
        switch(A->format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_format_csr:
        {

            return rocsparse_csrgemm_numeric_template(handle,
                                                      trans_A,
                                                      trans_B,
                                                      (J)A->rows,
                                                      (J)B->cols,
                                                      (J)A->cols,
                                                      (const T*)alpha,
                                                      A->descr,
                                                      (I)A->nnz,
                                                      (const T*)A->val_data,
                                                      (const I*)A->row_data,
                                                      (const J*)A->col_data,
                                                      B->descr,
                                                      (I)B->nnz,
                                                      (const T*)B->val_data,
                                                      (const I*)B->row_data,
                                                      (const J*)B->col_data,
                                                      (const T*)beta,
                                                      D->descr,
                                                      (I)D->nnz,
                                                      (const T*)D->val_data,
                                                      (const I*)D->row_data,
                                                      (const J*)D->col_data,
                                                      C->descr,
                                                      (I)C->nnz,
                                                      (T*)C->val_data,
                                                      (const I*)C->row_data,
                                                      (J*)C->col_data,
                                                      C->info,
                                                      temp_buffer);
        }
        }
    }
    }

    return rocsparse_status_not_implemented;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spgemm(rocsparse_handle            handle,
                                             rocsparse_operation         trans_A,
                                             rocsparse_operation         trans_B,
                                             const void*                 alpha,
                                             const rocsparse_spmat_descr A,
                                             const rocsparse_spmat_descr B,
                                             const void*                 beta,
                                             const rocsparse_spmat_descr D,
                                             rocsparse_spmat_descr       C,
                                             rocsparse_datatype          compute_type,
                                             rocsparse_spgemm_alg        alg,
                                             rocsparse_spgemm_stage      stage,
                                             size_t*                     buffer_size,
                                             void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spgemm",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)A,
              (const void*&)B,
              (const void*&)beta,
              (const void*&)D,
              (const void*&)C,
              compute_type,
              alg,
              stage,
              (const void*&)buffer_size,
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

    if(rocsparse_enum_utils::is_invalid(stage))
    {
        return rocsparse_status_invalid_value;
    }

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(A);
    RETURN_IF_NULLPTR(B);
    RETURN_IF_NULLPTR(D);
    RETURN_IF_NULLPTR(C);

    // Check for valid scalars
    if(alpha == nullptr && beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        RETURN_IF_NULLPTR(buffer_size);
    }

    // Check if descriptors are initialized
    if(A->init == false || B->init == false || C->init == false || D->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Check if all sparse matrices are in the same format
    if(A->format != B->format || A->format != C->format || A->format != D->format)
    {
        return rocsparse_status_not_implemented;
    }

    // Check for matching data types while we do not support mixed precision computation
    if(compute_type != A->data_type || compute_type != B->data_type || compute_type != C->data_type
       || compute_type != D->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    // Check for matching index types
    if(A->row_type != B->row_type || A->row_type != C->row_type || A->row_type != D->row_type
       || A->col_type != B->col_type || A->col_type != C->col_type || A->col_type != D->col_type)
    {
        return rocsparse_status_type_mismatch;
    }

    return rocsparse_spgemm_template_dispatch(A->row_type,
                                              A->col_type,
                                              compute_type,
                                              handle,
                                              trans_A,
                                              trans_B,
                                              alpha,
                                              A,
                                              B,
                                              beta,
                                              D,
                                              C,
                                              alg,
                                              stage,
                                              buffer_size,
                                              temp_buffer);

    return rocsparse_status_not_implemented;
}

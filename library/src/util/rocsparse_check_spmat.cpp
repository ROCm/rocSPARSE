/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_check_spmat.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_check_matrix_coo.hpp"
#include "rocsparse_check_matrix_csc.hpp"
#include "rocsparse_check_matrix_csr.hpp"
#include "rocsparse_check_matrix_ell.hpp"
#include "rocsparse_check_matrix_gebsr.hpp"

namespace rocsparse
{
    rocsparse_indextype determine_I_index_type(rocsparse_const_spmat_descr mat);
    rocsparse_indextype determine_J_index_type(rocsparse_const_spmat_descr mat);
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_check_spmat_template(rocsparse_handle            handle,
                                                rocsparse_const_spmat_descr mat,
                                                rocsparse_data_status*      data_status,
                                                rocsparse_check_spmat_stage stage,
                                                size_t*                     buffer_size,
                                                void*                       temp_buffer)
{
    switch(mat->format)
    {
    case rocsparse_format_coo:
    {
        switch(stage)
        {
        case rocsparse_check_spmat_stage_buffer_size:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_coo_buffer_size_impl<T, I>(handle,
                                                                   (I)mat->rows,
                                                                   (I)mat->cols,
                                                                   mat->nnz,
                                                                   (const T*)mat->const_val_data,
                                                                   (const I*)mat->const_row_data,
                                                                   (const I*)mat->const_col_data,
                                                                   mat->idx_base,
                                                                   (mat->descr)->type,
                                                                   (mat->descr)->fill_mode,
                                                                   (mat->descr)->storage_mode,
                                                                   buffer_size)));
            return rocsparse_status_success;
        }
        case rocsparse_check_spmat_stage_compute:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_coo_impl<T, I>(handle,
                                                       (I)mat->rows,
                                                       (I)mat->cols,
                                                       mat->nnz,
                                                       (const T*)mat->const_val_data,
                                                       (const I*)mat->const_row_data,
                                                       (const I*)mat->const_col_data,
                                                       mat->idx_base,
                                                       (mat->descr)->type,
                                                       (mat->descr)->fill_mode,
                                                       (mat->descr)->storage_mode,
                                                       data_status,
                                                       temp_buffer)));
            return rocsparse_status_success;
        }
        }
    }

    case rocsparse_format_csr:
    {
        switch(stage)
        {
        case rocsparse_check_spmat_stage_buffer_size:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_csr_buffer_size_impl<T, I, J>(handle,
                                                                      (J)mat->rows,
                                                                      (J)mat->cols,
                                                                      (I)mat->nnz,
                                                                      (const T*)mat->const_val_data,
                                                                      (const I*)mat->const_row_data,
                                                                      (const J*)mat->const_col_data,
                                                                      mat->idx_base,
                                                                      (mat->descr)->type,
                                                                      (mat->descr)->fill_mode,
                                                                      (mat->descr)->storage_mode,
                                                                      buffer_size)));
            return rocsparse_status_success;
        }
        case rocsparse_check_spmat_stage_compute:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_csr_impl<T, I, J>(handle,
                                                          (J)mat->rows,
                                                          (J)mat->cols,
                                                          (I)mat->nnz,
                                                          (const T*)mat->const_val_data,
                                                          (const I*)mat->const_row_data,
                                                          (const J*)mat->const_col_data,
                                                          mat->idx_base,
                                                          (mat->descr)->type,
                                                          (mat->descr)->fill_mode,
                                                          (mat->descr)->storage_mode,
                                                          data_status,
                                                          temp_buffer)));
            return rocsparse_status_success;
        }
        }
    }

    case rocsparse_format_csc:
    {
        switch(stage)
        {
        case rocsparse_check_spmat_stage_buffer_size:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_csc_buffer_size_impl<T, I, J>(handle,
                                                                      (J)mat->rows,
                                                                      (J)mat->cols,
                                                                      (I)mat->nnz,
                                                                      (const T*)mat->const_val_data,
                                                                      (const I*)mat->const_col_data,
                                                                      (const J*)mat->const_row_data,
                                                                      mat->idx_base,
                                                                      (mat->descr)->type,
                                                                      (mat->descr)->fill_mode,
                                                                      (mat->descr)->storage_mode,
                                                                      buffer_size)));
            return rocsparse_status_success;
        }
        case rocsparse_check_spmat_stage_compute:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_csc_impl<T, I, J>(handle,
                                                          (J)mat->rows,
                                                          (J)mat->cols,
                                                          (I)mat->nnz,
                                                          (const T*)mat->const_val_data,
                                                          (const I*)mat->const_col_data,
                                                          (const J*)mat->const_row_data,
                                                          mat->idx_base,
                                                          (mat->descr)->type,
                                                          (mat->descr)->fill_mode,
                                                          (mat->descr)->storage_mode,
                                                          data_status,
                                                          temp_buffer)));
            return rocsparse_status_success;
        }
        }
    }

    case rocsparse_format_ell:
    {
        switch(stage)
        {
        case rocsparse_check_spmat_stage_buffer_size:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_ell_buffer_size_impl<T, I>(handle,
                                                                   (I)mat->rows,
                                                                   (I)mat->cols,
                                                                   (I)mat->ell_width,
                                                                   (const T*)mat->const_val_data,
                                                                   (const I*)mat->const_col_data,
                                                                   mat->idx_base,
                                                                   (mat->descr)->type,
                                                                   (mat->descr)->fill_mode,
                                                                   (mat->descr)->storage_mode,
                                                                   buffer_size)));
            return rocsparse_status_success;
        }
        case rocsparse_check_spmat_stage_compute:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_ell_impl<T, I>(handle,
                                                       (I)mat->rows,
                                                       (I)mat->cols,
                                                       (I)mat->ell_width,
                                                       (const T*)mat->const_val_data,
                                                       (const I*)mat->const_col_data,
                                                       mat->idx_base,
                                                       (mat->descr)->type,
                                                       (mat->descr)->fill_mode,
                                                       (mat->descr)->storage_mode,
                                                       data_status,
                                                       temp_buffer)));
            return rocsparse_status_success;
        }
        }
    }

    case rocsparse_format_bsr:
    {
        switch(stage)
        {
        case rocsparse_check_spmat_stage_buffer_size:
        {
            RETURN_IF_ROCSPARSE_ERROR((rocsparse_check_matrix_gebsr_buffer_size_impl<T, I, J>(
                handle,
                mat->block_dir,
                (J)mat->rows,
                (J)mat->cols,
                (I)mat->nnz,
                (J)mat->block_dim,
                (J)mat->block_dim,
                (const T*)mat->const_val_data,
                (const I*)mat->const_row_data,
                (const J*)mat->const_col_data,
                mat->idx_base,
                (mat->descr)->type,
                (mat->descr)->fill_mode,
                (mat->descr)->storage_mode,
                buffer_size)));
            return rocsparse_status_success;
        }
        case rocsparse_check_spmat_stage_compute:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_check_matrix_gebsr_impl<T, I, J>(handle,
                                                            mat->block_dir,
                                                            (J)mat->rows,
                                                            (J)mat->cols,
                                                            (I)mat->nnz,
                                                            (J)mat->block_dim,
                                                            (J)mat->block_dim,
                                                            (const T*)mat->const_val_data,
                                                            (const I*)mat->const_row_data,
                                                            (const J*)mat->const_col_data,
                                                            mat->idx_base,
                                                            (mat->descr)->type,
                                                            (mat->descr)->fill_mode,
                                                            (mat->descr)->storage_mode,
                                                            data_status,
                                                            temp_buffer)));
            return rocsparse_status_success;
        }
        }
    }

    case rocsparse_format_coo_aos:
    case rocsparse_format_bell:
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

template <typename... Ts>
rocsparse_status rocsparse_check_spmat_dynamic_dispatch(rocsparse_indextype itype,
                                                        rocsparse_indextype jtype,
                                                        rocsparse_datatype  ctype,
                                                        rocsparse_format    format,
                                                        Ts&&... ts)
{
#define DISPATCH_COMPUTE_TYPE(ITYPE, JTYPE, CTYPE)                                            \
    switch(CTYPE)                                                                             \
    {                                                                                         \
    case rocsparse_datatype_f32_r:                                                            \
    {                                                                                         \
        return rocsparse_check_spmat_template<ITYPE, JTYPE, float>(ts...);                    \
    }                                                                                         \
    case rocsparse_datatype_f64_r:                                                            \
    {                                                                                         \
        return rocsparse_check_spmat_template<ITYPE, JTYPE, double>(ts...);                   \
    }                                                                                         \
    case rocsparse_datatype_f32_c:                                                            \
    {                                                                                         \
        return rocsparse_check_spmat_template<ITYPE, JTYPE, rocsparse_float_complex>(ts...);  \
    }                                                                                         \
    case rocsparse_datatype_f64_c:                                                            \
    {                                                                                         \
        return rocsparse_check_spmat_template<ITYPE, JTYPE, rocsparse_double_complex>(ts...); \
    }                                                                                         \
    case rocsparse_datatype_i8_r:                                                             \
    case rocsparse_datatype_u8_r:                                                             \
    case rocsparse_datatype_i32_r:                                                            \
    case rocsparse_datatype_u32_r:                                                            \
    {                                                                                         \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                          \
    }                                                                                         \
    }

    switch(itype)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_indextype_i32:
    {
        switch(jtype)
        {
        case rocsparse_indextype_u16:
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_COMPUTE_TYPE(int32_t, int32_t, ctype);
        }
        }
    }
    case rocsparse_indextype_i64:
    {
        switch(jtype)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_COMPUTE_TYPE(int64_t, int32_t, ctype);
        }
        case rocsparse_indextype_i64:
        {
            DISPATCH_COMPUTE_TYPE(int64_t, int64_t, ctype);
        }
        }
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

extern "C" rocsparse_status rocsparse_check_spmat(rocsparse_handle            handle,
                                                  rocsparse_const_spmat_descr mat,
                                                  rocsparse_data_status*      data_status,
                                                  rocsparse_check_spmat_stage stage,
                                                  size_t*                     buffer_size,
                                                  void*                       temp_buffer)
try
{

    // Logging
    log_trace(handle,
              "rocsparse_check_spmat",
              (const void*&)mat,
              (const void*&)data_status,
              stage,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, mat);
    ROCSPARSE_CHECKARG_POINTER(2, data_status);
    ROCSPARSE_CHECKARG_ENUM(3, stage);

    ROCSPARSE_CHECKARG(4,
                       buffer_size,
                       ((temp_buffer == nullptr) && (buffer_size == nullptr)),
                       rocsparse_status_invalid_pointer);

    ROCSPARSE_CHECKARG(1, mat, (mat->init == false), rocsparse_status_not_initialized);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_check_spmat_dynamic_dispatch(rocsparse::determine_I_index_type(mat),
                                               rocsparse::determine_J_index_type(mat),
                                               mat->data_type,
                                               mat->format,
                                               handle,
                                               mat,
                                               data_status,
                                               stage,
                                               buffer_size,
                                               temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

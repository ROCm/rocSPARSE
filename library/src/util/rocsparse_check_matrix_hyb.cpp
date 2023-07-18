/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/util/rocsparse_check_matrix_hyb.h"
#include "definitions.h"
#include "rocsparse_check_matrix_coo.hpp"
#include "rocsparse_check_matrix_ell.hpp"
#include "utility.h"

template <typename I>
rocsparse_status
    rocsparse_check_matrix_ell_buffer_size_template_dispatch(rocsparse_datatype     type,
                                                             rocsparse_handle       handle,
                                                             I                      m,
                                                             I                      n,
                                                             I                      ell_width,
                                                             void*                  ell_val,
                                                             const I*               ell_col_ind,
                                                             rocsparse_index_base   idx_base,
                                                             rocsparse_matrix_type  matrix_type,
                                                             rocsparse_fill_mode    uplo,
                                                             rocsparse_storage_mode storage,
                                                             size_t*                buffer_size)
{
    switch(type)
    {
    case rocsparse_datatype_f32_r:
        return rocsparse_check_matrix_ell_buffer_size_template<float>(handle,
                                                                      m,
                                                                      n,
                                                                      ell_width,
                                                                      (float*)ell_val,
                                                                      ell_col_ind,
                                                                      idx_base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      buffer_size);
    case rocsparse_datatype_f64_r:
        return rocsparse_check_matrix_ell_buffer_size_template<double>(handle,
                                                                       m,
                                                                       n,
                                                                       ell_width,
                                                                       (double*)ell_val,
                                                                       ell_col_ind,
                                                                       idx_base,
                                                                       matrix_type,
                                                                       uplo,
                                                                       storage,
                                                                       buffer_size);
    case rocsparse_datatype_f32_c:
        return rocsparse_check_matrix_ell_buffer_size_template<rocsparse_float_complex>(
            handle,
            m,
            n,
            ell_width,
            (rocsparse_float_complex*)ell_val,
            ell_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            buffer_size);
    case rocsparse_datatype_f64_c:
        return rocsparse_check_matrix_ell_buffer_size_template<rocsparse_double_complex>(
            handle,
            m,
            n,
            ell_width,
            (rocsparse_double_complex*)ell_val,
            ell_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            buffer_size);
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return rocsparse_status_not_implemented;
    }
    }
}

template <typename I>
rocsparse_status
    rocsparse_check_matrix_coo_buffer_size_template_dispatch(rocsparse_datatype     type,
                                                             rocsparse_handle       handle,
                                                             I                      m,
                                                             I                      n,
                                                             I                      nnz,
                                                             void*                  coo_val,
                                                             const I*               coo_row_ind,
                                                             const I*               coo_col_ind,
                                                             rocsparse_index_base   idx_base,
                                                             rocsparse_matrix_type  matrix_type,
                                                             rocsparse_fill_mode    uplo,
                                                             rocsparse_storage_mode storage,
                                                             size_t*                buffer_size)
{
    switch(type)
    {
    case rocsparse_datatype_f32_r:
        return rocsparse_check_matrix_coo_buffer_size_template<float>(handle,
                                                                      m,
                                                                      n,
                                                                      nnz,
                                                                      (float*)coo_val,
                                                                      coo_row_ind,
                                                                      coo_col_ind,
                                                                      idx_base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      buffer_size);
    case rocsparse_datatype_f64_r:
        return rocsparse_check_matrix_coo_buffer_size_template<double>(handle,
                                                                       m,
                                                                       n,
                                                                       nnz,
                                                                       (double*)coo_val,
                                                                       coo_row_ind,
                                                                       coo_col_ind,
                                                                       idx_base,
                                                                       matrix_type,
                                                                       uplo,
                                                                       storage,
                                                                       buffer_size);
    case rocsparse_datatype_f32_c:
        return rocsparse_check_matrix_coo_buffer_size_template<rocsparse_float_complex>(
            handle,
            m,
            n,
            nnz,
            (rocsparse_float_complex*)coo_val,
            coo_row_ind,
            coo_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            buffer_size);
    case rocsparse_datatype_f64_c:
        return rocsparse_check_matrix_coo_buffer_size_template<rocsparse_double_complex>(
            handle,
            m,
            n,
            nnz,
            (rocsparse_double_complex*)coo_val,
            coo_row_ind,
            coo_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            buffer_size);
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return rocsparse_status_not_implemented;
    }
    }
}

template <typename I>
rocsparse_status rocsparse_check_matrix_ell_template_dispatch(rocsparse_datatype     type,
                                                              rocsparse_handle       handle,
                                                              I                      m,
                                                              I                      n,
                                                              I                      ell_width,
                                                              void*                  ell_val,
                                                              const I*               ell_col_ind,
                                                              rocsparse_index_base   idx_base,
                                                              rocsparse_matrix_type  matrix_type,
                                                              rocsparse_fill_mode    uplo,
                                                              rocsparse_storage_mode storage,
                                                              rocsparse_data_status* data_status,
                                                              void*                  temp_buffer)
{
    switch(type)
    {
    case rocsparse_datatype_f32_r:
        return rocsparse_check_matrix_ell_template<float>(handle,
                                                          m,
                                                          n,
                                                          ell_width,
                                                          (float*)ell_val,
                                                          ell_col_ind,
                                                          idx_base,
                                                          matrix_type,
                                                          uplo,
                                                          storage,
                                                          data_status,
                                                          temp_buffer);
    case rocsparse_datatype_f64_r:
        return rocsparse_check_matrix_ell_template<double>(handle,
                                                           m,
                                                           n,
                                                           ell_width,
                                                           (double*)ell_val,
                                                           ell_col_ind,
                                                           idx_base,
                                                           matrix_type,
                                                           uplo,
                                                           storage,
                                                           data_status,
                                                           temp_buffer);
    case rocsparse_datatype_f32_c:
        return rocsparse_check_matrix_ell_template<rocsparse_float_complex>(
            handle,
            m,
            n,
            ell_width,
            (rocsparse_float_complex*)ell_val,
            ell_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            data_status,
            temp_buffer);
    case rocsparse_datatype_f64_c:
        return rocsparse_check_matrix_ell_template<rocsparse_double_complex>(
            handle,
            m,
            n,
            ell_width,
            (rocsparse_double_complex*)ell_val,
            ell_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            data_status,
            temp_buffer);
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return rocsparse_status_not_implemented;
    }
    }
}

template <typename I>
rocsparse_status rocsparse_check_matrix_coo_template_dispatch(rocsparse_datatype     type,
                                                              rocsparse_handle       handle,
                                                              I                      m,
                                                              I                      n,
                                                              I                      nnz,
                                                              void*                  coo_val,
                                                              const I*               coo_row_ind,
                                                              const I*               coo_col_ind,
                                                              rocsparse_index_base   idx_base,
                                                              rocsparse_matrix_type  matrix_type,
                                                              rocsparse_fill_mode    uplo,
                                                              rocsparse_storage_mode storage,
                                                              rocsparse_data_status* data_status,
                                                              void*                  temp_buffer)
{
    switch(type)
    {
    case rocsparse_datatype_f32_r:
        return rocsparse_check_matrix_coo_template<float>(handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (float*)coo_val,
                                                          coo_row_ind,
                                                          coo_col_ind,
                                                          idx_base,
                                                          matrix_type,
                                                          uplo,
                                                          storage,
                                                          data_status,
                                                          temp_buffer);
    case rocsparse_datatype_f64_r:
        return rocsparse_check_matrix_coo_template<double>(handle,
                                                           m,
                                                           n,
                                                           nnz,
                                                           (double*)coo_val,
                                                           coo_row_ind,
                                                           coo_col_ind,
                                                           idx_base,
                                                           matrix_type,
                                                           uplo,
                                                           storage,
                                                           data_status,
                                                           temp_buffer);
    case rocsparse_datatype_f32_c:
        return rocsparse_check_matrix_coo_template<rocsparse_float_complex>(
            handle,
            m,
            n,
            nnz,
            (rocsparse_float_complex*)coo_val,
            coo_row_ind,
            coo_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            data_status,
            temp_buffer);
    case rocsparse_datatype_f64_c:
        return rocsparse_check_matrix_coo_template<rocsparse_double_complex>(
            handle,
            m,
            n,
            nnz,
            (rocsparse_double_complex*)coo_val,
            coo_row_ind,
            coo_col_ind,
            idx_base,
            matrix_type,
            uplo,
            storage,
            data_status,
            temp_buffer);
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return rocsparse_status_not_implemented;
    }
    }
}

rocsparse_status rocsparse_check_matrix_hyb_buffer_size(rocsparse_handle        handle,
                                                        const rocsparse_hyb_mat hyb,
                                                        rocsparse_index_base    idx_base,
                                                        rocsparse_matrix_type   matrix_type,
                                                        rocsparse_fill_mode     uplo,
                                                        rocsparse_storage_mode  storage,
                                                        size_t*                 buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(hyb == nullptr || buffer_size == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    size_t           ell_buffer_size;
    rocsparse_status status
        = rocsparse_check_matrix_ell_buffer_size_template_dispatch(hyb->data_type_T,
                                                                   handle,
                                                                   hyb->m,
                                                                   hyb->n,
                                                                   hyb->ell_width,
                                                                   hyb->ell_val,
                                                                   hyb->ell_col_ind,
                                                                   idx_base,
                                                                   matrix_type,
                                                                   uplo,
                                                                   storage,
                                                                   &ell_buffer_size);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    size_t coo_buffer_size;
    status = rocsparse_check_matrix_coo_buffer_size_template_dispatch(hyb->data_type_T,
                                                                      handle,
                                                                      hyb->m,
                                                                      hyb->n,
                                                                      hyb->coo_nnz,
                                                                      hyb->coo_val,
                                                                      hyb->coo_row_ind,
                                                                      hyb->coo_col_ind,
                                                                      idx_base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &coo_buffer_size);

    *buffer_size = std::max(ell_buffer_size, coo_buffer_size);

    return status;
}

rocsparse_status rocsparse_check_matrix_hyb(rocsparse_handle        handle,
                                            const rocsparse_hyb_mat hyb,
                                            rocsparse_index_base    idx_base,
                                            rocsparse_matrix_type   matrix_type,
                                            rocsparse_fill_mode     uplo,
                                            rocsparse_storage_mode  storage,
                                            rocsparse_data_status*  data_status,
                                            void*                   temp_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(hyb == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    rocsparse_status status = rocsparse_check_matrix_ell_template_dispatch(hyb->data_type_T,
                                                                           handle,
                                                                           hyb->m,
                                                                           hyb->n,
                                                                           hyb->ell_width,
                                                                           hyb->ell_val,
                                                                           hyb->ell_col_ind,
                                                                           idx_base,
                                                                           matrix_type,
                                                                           uplo,
                                                                           storage,
                                                                           data_status,
                                                                           temp_buffer);

    if(status != rocsparse_status_success)
    {
        return status;
    }

    return rocsparse_check_matrix_coo_template_dispatch(hyb->data_type_T,
                                                        handle,
                                                        hyb->m,
                                                        hyb->n,
                                                        hyb->coo_nnz,
                                                        hyb->coo_val,
                                                        hyb->coo_row_ind,
                                                        hyb->coo_col_ind,
                                                        idx_base,
                                                        matrix_type,
                                                        uplo,
                                                        storage,
                                                        data_status,
                                                        temp_buffer);
}

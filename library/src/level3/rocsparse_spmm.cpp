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

#include "rocsparse_bellmm.hpp"
#include "rocsparse_coomm.hpp"
#include "rocsparse_cscmm.hpp"
#include "rocsparse_csrmm.hpp"

rocsparse_status rocsparse_spmm_alg2bellmm_alg(rocsparse_spmm_alg    spmm_alg,
                                               rocsparse_bellmm_alg& bellmm_alg)
{
    switch(spmm_alg)
    {
    case rocsparse_spmm_alg_default:
    case rocsparse_spmm_alg_bell:
    {
        bellmm_alg = rocsparse_bellmm_alg_default;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_bsr:
    case rocsparse_spmm_alg_csr:
    case rocsparse_spmm_alg_csr_row_split:
    case rocsparse_spmm_alg_csr_merge:
    case rocsparse_spmm_alg_coo_segmented:
    case rocsparse_spmm_alg_coo_atomic:
    case rocsparse_spmm_alg_coo_segmented_atomic:
    {
        return rocsparse_status_invalid_value;
    }
    }
    return rocsparse_status_invalid_value;
}

rocsparse_status rocsparse_spmm_alg2csrmm_alg(rocsparse_spmm_alg   spmm_alg,
                                              rocsparse_csrmm_alg& csrmm_alg)
{
    switch(spmm_alg)
    {
    case rocsparse_spmm_alg_default:
    case rocsparse_spmm_alg_csr:
    {
        csrmm_alg = rocsparse_csrmm_alg_default;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_csr_row_split:
    {
        csrmm_alg = rocsparse_csrmm_alg_row_split;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_csr_merge:
    {
        csrmm_alg = rocsparse_csrmm_alg_merge;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_bell:
    case rocsparse_spmm_alg_bsr:
    case rocsparse_spmm_alg_coo_segmented:
    case rocsparse_spmm_alg_coo_atomic:
    case rocsparse_spmm_alg_coo_segmented_atomic:
    {
        return rocsparse_status_invalid_value;
    }
    }
    return rocsparse_status_invalid_value;
}

rocsparse_status rocsparse_spmm_alg2coomm_alg(rocsparse_spmm_alg   spmm_alg,
                                              rocsparse_coomm_alg& coomm_alg)
{
    switch(spmm_alg)
    {
    case rocsparse_spmm_alg_default:
    {
        coomm_alg = rocsparse_coomm_alg_default;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_coo_segmented:
    {
        coomm_alg = rocsparse_coomm_alg_segmented;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_coo_atomic:
    {
        coomm_alg = rocsparse_coomm_alg_atomic;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_coo_segmented_atomic:
    {
        coomm_alg = rocsparse_coomm_alg_segmented_atomic;
        return rocsparse_status_success;
    }

    case rocsparse_spmm_alg_bell:
    case rocsparse_spmm_alg_bsr:
    case rocsparse_spmm_alg_csr:
    case rocsparse_spmm_alg_csr_row_split:
    case rocsparse_spmm_alg_csr_merge:
    {
        return rocsparse_status_invalid_value;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spmm_template_auto(rocsparse_handle            handle,
                                              rocsparse_operation         trans_A,
                                              rocsparse_operation         trans_B,
                                              const void*                 alpha,
                                              const rocsparse_spmat_descr mat_A,
                                              const rocsparse_dnmat_descr mat_B,
                                              const void*                 beta,
                                              const rocsparse_dnmat_descr mat_C,
                                              rocsparse_spmm_alg          alg,
                                              size_t*                     buffer_size,
                                              void*                       temp_buffer);

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spmm_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans_A,
                                         rocsparse_operation         trans_B,
                                         const void*                 alpha,
                                         const rocsparse_spmat_descr mat_A,
                                         const rocsparse_dnmat_descr mat_B,
                                         const void*                 beta,
                                         const rocsparse_dnmat_descr mat_C,
                                         rocsparse_spmm_alg          alg,
                                         rocsparse_spmm_stage        stage,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    switch(mat_A->format)
    {
    case rocsparse_format_csr:
    {
        rocsparse_csrmm_alg csrmm_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmm_alg2csrmm_alg(alg, csrmm_alg)));

        const J m = (J)mat_A->rows;
        const J n = (J)mat_C->cols;
        const J k = (J)mat_A->cols;

        switch(stage)
        {
        case rocsparse_spmm_stage_buffer_size:
        {
            return rocsparse_csrmm_buffer_size_template(handle,
                                                        trans_A,
                                                        csrmm_alg,
                                                        m,
                                                        n,
                                                        k,
                                                        (I)mat_A->nnz,
                                                        mat_A->descr,
                                                        (const T*)mat_A->val_data,
                                                        (const I*)mat_A->row_data,
                                                        (const J*)mat_A->col_data,
                                                        buffer_size);
        }
        case rocsparse_spmm_stage_preprocess:
        {
            return rocsparse_csrmm_analysis_template(handle,
                                                     trans_A,
                                                     csrmm_alg,
                                                     m,
                                                     n,
                                                     k,
                                                     (I)mat_A->nnz,
                                                     mat_A->descr,
                                                     (const T*)mat_A->val_data,
                                                     (const I*)mat_A->row_data,
                                                     (const J*)mat_A->col_data,
                                                     temp_buffer);
        }
        case rocsparse_spmm_stage_compute:
        {
            return rocsparse_csrmm_template(handle,
                                            trans_A,
                                            trans_B,
                                            mat_B->order,
                                            mat_C->order,
                                            csrmm_alg,
                                            m,
                                            n,
                                            k,
                                            (I)mat_A->nnz,
                                            (J)mat_A->batch_count,
                                            (I)mat_A->offsets_batch_stride,
                                            (I)mat_A->columns_values_batch_stride,
                                            (const T*)alpha,
                                            mat_A->descr,
                                            (const T*)mat_A->val_data,
                                            (const I*)mat_A->row_data,
                                            (const J*)mat_A->col_data,
                                            (const T*)mat_B->values,
                                            (J)mat_B->ld,
                                            (J)mat_B->batch_count,
                                            (I)mat_B->batch_stride,
                                            (const T*)beta,
                                            (T*)mat_C->values,
                                            (J)mat_C->ld,
                                            (J)mat_C->batch_count,
                                            (I)mat_C->batch_stride,
                                            temp_buffer,
                                            false);
        }

        case rocsparse_spmm_stage_auto:
        {
            return rocsparse_spmm_template_auto<I, J, T>(handle,
                                                         trans_A,
                                                         trans_B,
                                                         alpha,
                                                         mat_A,
                                                         mat_B,
                                                         beta,
                                                         mat_C,
                                                         alg,
                                                         buffer_size,
                                                         temp_buffer);
        }
        }
    }

    case rocsparse_format_csc:
    {
        rocsparse_csrmm_alg csrmm_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmm_alg2csrmm_alg(alg, csrmm_alg)));

        const J m = (J)mat_A->rows;
        const J n = (J)mat_C->cols;
        const J k = (J)mat_A->cols;

        switch(stage)
        {
        case rocsparse_spmm_stage_buffer_size:
        {
            return rocsparse_cscmm_buffer_size_template(handle,
                                                        trans_A,
                                                        csrmm_alg,
                                                        m,
                                                        n,
                                                        k,
                                                        (I)mat_A->nnz,
                                                        mat_A->descr,
                                                        (const T*)mat_A->val_data,
                                                        (const I*)mat_A->col_data,
                                                        (const J*)mat_A->row_data,
                                                        buffer_size);
        }
        case rocsparse_spmm_stage_preprocess:
        {
            return rocsparse_cscmm_analysis_template(handle,
                                                     trans_A,
                                                     csrmm_alg,
                                                     m,
                                                     n,
                                                     k,
                                                     (I)mat_A->nnz,
                                                     mat_A->descr,
                                                     (const T*)mat_A->val_data,
                                                     (const I*)mat_A->col_data,
                                                     (const J*)mat_A->row_data,
                                                     temp_buffer);
        }
        case rocsparse_spmm_stage_compute:
        {
            return rocsparse_cscmm_template(handle,
                                            trans_A,
                                            trans_B,
                                            mat_B->order,
                                            mat_C->order,
                                            csrmm_alg,
                                            m,
                                            n,
                                            k,
                                            (I)mat_A->nnz,
                                            (J)mat_A->batch_count,
                                            (I)mat_A->offsets_batch_stride,
                                            (I)mat_A->columns_values_batch_stride,
                                            (const T*)alpha,
                                            mat_A->descr,
                                            (const T*)mat_A->val_data,
                                            (const I*)mat_A->col_data,
                                            (const J*)mat_A->row_data,
                                            (const T*)mat_B->values,
                                            (J)mat_B->ld,
                                            (J)mat_B->batch_count,
                                            (I)mat_B->batch_stride,
                                            (const T*)beta,
                                            (T*)mat_C->values,
                                            (J)mat_C->ld,
                                            (J)mat_C->batch_count,
                                            (I)mat_C->batch_stride,
                                            temp_buffer);
        }
        case rocsparse_spmm_stage_auto:
        {
            return rocsparse_spmm_template_auto<I, J, T>(handle,
                                                         trans_A,
                                                         trans_B,
                                                         alpha,
                                                         mat_A,
                                                         mat_B,
                                                         beta,
                                                         mat_C,
                                                         alg,
                                                         buffer_size,
                                                         temp_buffer);
        }
        }
    }

    case rocsparse_format_coo:
    {
        rocsparse_coomm_alg coomm_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmm_alg2coomm_alg(alg, coomm_alg)));

        const I m = (I)mat_A->rows;
        const I n = (I)mat_C->cols;
        const I k = (I)mat_A->cols;

        switch(stage)
        {
        case rocsparse_spmm_stage_buffer_size:
        {
            return rocsparse_coomm_buffer_size_template(handle,
                                                        trans_A,
                                                        coomm_alg,
                                                        m,
                                                        n,
                                                        k,
                                                        mat_A->nnz,
                                                        (I)mat_C->batch_count,
                                                        mat_A->descr,
                                                        (const T*)mat_A->val_data,
                                                        (const I*)mat_A->row_data,
                                                        (const I*)mat_A->col_data,
                                                        buffer_size);
        }

        case rocsparse_spmm_stage_preprocess:
        {
            return rocsparse_coomm_analysis_template(handle,
                                                     trans_A,
                                                     coomm_alg,
                                                     m,
                                                     n,
                                                     k,
                                                     mat_A->nnz,
                                                     mat_A->descr,
                                                     (const T*)mat_A->val_data,
                                                     (const I*)mat_A->row_data,
                                                     (const I*)mat_A->col_data,
                                                     temp_buffer);
        }

        case rocsparse_spmm_stage_compute:
        {
            return rocsparse_coomm_template(handle,
                                            trans_A,
                                            trans_B,
                                            mat_B->order,
                                            mat_C->order,
                                            coomm_alg,
                                            m,
                                            n,
                                            k,
                                            mat_A->nnz,
                                            (I)mat_A->batch_count,
                                            (I)mat_A->batch_stride,
                                            (const T*)alpha,
                                            mat_A->descr,
                                            (const T*)mat_A->val_data,
                                            (const I*)mat_A->row_data,
                                            (const I*)mat_A->col_data,
                                            (const T*)mat_B->values,
                                            (I)mat_B->ld,
                                            (I)mat_B->batch_count,
                                            (I)mat_B->batch_stride,
                                            (const T*)beta,
                                            (T*)mat_C->values,
                                            (I)mat_C->ld,
                                            (I)mat_C->batch_count,
                                            (I)mat_C->batch_stride,
                                            temp_buffer);
        }

        case rocsparse_spmm_stage_auto:
        {
            return rocsparse_spmm_template_auto<I, J, T>(handle,
                                                         trans_A,
                                                         trans_B,
                                                         alpha,
                                                         mat_A,
                                                         mat_B,
                                                         beta,
                                                         mat_C,
                                                         alg,
                                                         buffer_size,
                                                         temp_buffer);
        }
        }
    }

    case rocsparse_format_bell:
    {
        rocsparse_bellmm_alg bellmm_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmm_alg2bellmm_alg(alg, bellmm_alg)));

        switch(stage)
        {
            //
            // STAGE BUFFER SIZE
            //
        case rocsparse_spmm_stage_buffer_size:
        {
            RETURN_IF_NULLPTR(buffer_size);
            return rocsparse_bellmm_template_buffer_size<T, I>(
                handle,
                trans_A,
                trans_B,
                mat_B->order,
                mat_C->order,
                mat_A->block_dir,
                (I)(mat_C->rows / mat_A->block_dim),
                (I)mat_C->cols,

                (trans_A == rocsparse_operation_none) ? (I)(mat_A->cols / mat_A->block_dim)
                                                      : (I)(mat_A->rows / mat_A->block_dim),

                (I)mat_A->ell_cols,
                (I)mat_A->block_dim,
                (const T*)alpha,
                mat_A->descr,
                (const I*)mat_A->col_data,
                (const T*)mat_A->val_data,
                (const T*)mat_B->values,
                (I)mat_B->ld,
                (const T*)beta,
                (T*)mat_C->values,
                (I)mat_C->ld,
                buffer_size);
        }

            //
            // STAGE PREPROCESS
            //
        case rocsparse_spmm_stage_preprocess:
        {
            return rocsparse_bellmm_template_preprocess<T, I>(
                handle,
                trans_A,
                trans_B,
                mat_B->order,
                mat_C->order,
                mat_A->block_dir,
                (I)(mat_C->rows / mat_A->block_dim),
                (I)mat_C->cols,

                (trans_A == rocsparse_operation_none) ? (I)(mat_A->cols / mat_A->block_dim)
                                                      : (I)(mat_A->rows / mat_A->block_dim),

                (I)mat_A->ell_cols,
                (I)mat_A->block_dim,
                (const T*)alpha,
                mat_A->descr,
                (const I*)mat_A->col_data,
                (const T*)mat_A->val_data,
                (const T*)mat_B->values,
                (I)mat_B->ld,
                (const T*)beta,
                (T*)mat_C->values,
                (I)mat_C->ld,
                temp_buffer);
        }

            //
            // STAGE COMPUTE
            //
        case rocsparse_spmm_stage_compute:
        {
            return rocsparse_bellmm_template<T, I>(handle,
                                                   trans_A,
                                                   trans_B,
                                                   mat_B->order,
                                                   mat_C->order,
                                                   mat_A->block_dir,
                                                   (I)(mat_C->rows / mat_A->block_dim),
                                                   (I)mat_C->cols,

                                                   (trans_A == rocsparse_operation_none)
                                                       ? (I)(mat_A->cols / mat_A->block_dim)
                                                       : (I)(mat_A->rows / mat_A->block_dim),

                                                   (I)mat_A->ell_cols,
                                                   (I)mat_A->block_dim,
                                                   (I)mat_A->batch_count,
                                                   (I)mat_A->batch_stride,
                                                   (const T*)alpha,
                                                   mat_A->descr,
                                                   (const I*)mat_A->col_data,
                                                   (const T*)mat_A->val_data,
                                                   (const T*)mat_B->values,
                                                   (I)mat_B->ld,
                                                   (I)mat_B->batch_count,
                                                   (I)mat_B->batch_stride,
                                                   (const T*)beta,
                                                   (T*)mat_C->values,
                                                   (I)mat_C->ld,
                                                   (I)mat_C->batch_count,
                                                   (I)mat_C->batch_stride,
                                                   temp_buffer);
        }

        case rocsparse_spmm_stage_auto:
        {
            return rocsparse_spmm_template_auto<I, J, T>(handle,
                                                         trans_A,
                                                         trans_B,
                                                         alpha,
                                                         mat_A,
                                                         mat_B,
                                                         beta,
                                                         mat_C,
                                                         alg,
                                                         buffer_size,
                                                         temp_buffer);
        }
        }

        break;
    }

    case rocsparse_format_coo_aos:
    case rocsparse_format_ell:
    case rocsparse_format_bsr:
    {
        return rocsparse_status_not_implemented;
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spmm_template_auto(rocsparse_handle            handle,
                                              rocsparse_operation         trans_A,
                                              rocsparse_operation         trans_B,
                                              const void*                 alpha,
                                              const rocsparse_spmat_descr mat_A,
                                              const rocsparse_dnmat_descr mat_B,
                                              const void*                 beta,
                                              const rocsparse_dnmat_descr mat_C,
                                              rocsparse_spmm_alg          alg,
                                              size_t*                     buffer_size,
                                              void*                       temp_buffer)
{
    if(temp_buffer == nullptr)
    {
        return rocsparse_spmm_template<I, J, T>(handle,
                                                trans_A,
                                                trans_B,
                                                alpha,
                                                mat_A,
                                                mat_B,
                                                beta,
                                                mat_C,
                                                alg,
                                                rocsparse_spmm_stage_buffer_size,
                                                buffer_size,
                                                temp_buffer);
    }
    else
    {
        rocsparse_status status = rocsparse_spmm_template<I, J, T>(handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   alpha,
                                                                   mat_A,
                                                                   mat_B,
                                                                   beta,
                                                                   mat_C,
                                                                   alg,
                                                                   rocsparse_spmm_stage_preprocess,
                                                                   buffer_size,
                                                                   temp_buffer);
        if(status != rocsparse_status_success)
        {
            return status;
        }

        return rocsparse_spmm_template<I, J, T>(handle,
                                                trans_A,
                                                trans_B,
                                                alpha,
                                                mat_A,
                                                mat_B,
                                                beta,
                                                mat_C,
                                                alg,
                                                rocsparse_spmm_stage_compute,
                                                buffer_size,
                                                temp_buffer);
    }
}

template <typename... Ts>
static inline rocsparse_status rocsparse_spmm_dynamic_dispatch(rocsparse_indextype itype,
                                                               rocsparse_indextype jtype,
                                                               rocsparse_datatype  ctype,
                                                               Ts&&... ts)
{
    switch(ctype)
    {

    case rocsparse_datatype_f32_r:
    {
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
            case rocsparse_indextype_u16:
            case rocsparse_indextype_i64:
            {
                return rocsparse_status_not_implemented;
            }
            case rocsparse_indextype_i32:
            {
                return rocsparse_spmm_template<int32_t, int32_t, float>(ts...);
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
                return rocsparse_spmm_template<int64_t, int32_t, float>(ts...);
            }
            case rocsparse_indextype_i64:
            {
                return rocsparse_spmm_template<int64_t, int64_t, float>(ts...);
            }
            }
        }
        }
    }

    case rocsparse_datatype_f64_r:
    {
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
            case rocsparse_indextype_u16:
            case rocsparse_indextype_i64:
            {
                return rocsparse_status_not_implemented;
            }
            case rocsparse_indextype_i32:
            {
                return rocsparse_spmm_template<int32_t, int32_t, double>(ts...);
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
                return rocsparse_spmm_template<int64_t, int32_t, double>(ts...);
            }
            case rocsparse_indextype_i64:
            {
                return rocsparse_spmm_template<int64_t, int64_t, double>(ts...);
            }
            }
        }
        }
    }

    case rocsparse_datatype_f32_c:
    {
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
            case rocsparse_indextype_u16:
            case rocsparse_indextype_i64:
            {
                return rocsparse_status_not_implemented;
            }
            case rocsparse_indextype_i32:
            {
                return rocsparse_spmm_template<int32_t, int32_t, rocsparse_float_complex>(ts...);
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
                return rocsparse_spmm_template<int64_t, int32_t, rocsparse_float_complex>(ts...);
            }
            case rocsparse_indextype_i64:
            {
                return rocsparse_spmm_template<int64_t, int64_t, rocsparse_float_complex>(ts...);
            }
            }
        }
        }
    }

    case rocsparse_datatype_f64_c:
    {
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
            case rocsparse_indextype_u16:
            case rocsparse_indextype_i64:
            {
                return rocsparse_status_not_implemented;
            }
            case rocsparse_indextype_i32:
            {
                return rocsparse_spmm_template<int32_t, int32_t, rocsparse_double_complex>(ts...);
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
                return rocsparse_spmm_template<int64_t, int32_t, rocsparse_double_complex>(ts...);
            }
            case rocsparse_indextype_i64:
            {
                return rocsparse_spmm_template<int64_t, int64_t, rocsparse_double_complex>(ts...);
            }
            }
        }
        }
    }
    }
    return rocsparse_status_invalid_value;
}

static rocsparse_indextype determine_I_index_type(rocsparse_spmat_descr mat)
{
    switch(mat->format)
    {
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_csr:
    case rocsparse_format_ell:
    case rocsparse_format_bell:
    case rocsparse_format_bsr:
    {
        return mat->row_type;
    }
    case rocsparse_format_csc:
    {
        return mat->col_type;
    }
    }
}

static rocsparse_indextype determine_J_index_type(rocsparse_spmat_descr mat)
{
    switch(mat->format)
    {
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_csr:
    case rocsparse_format_ell:
    case rocsparse_format_bell:
    case rocsparse_format_bsr:
    {
        return mat->col_type;
    }
    case rocsparse_format_csc:
    {
        return mat->row_type;
    }
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr mat_A,
                                           const rocsparse_dnmat_descr mat_B,
                                           const void*                 beta,
                                           const rocsparse_dnmat_descr mat_C,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_spmm_alg          alg,
                                           rocsparse_spmm_stage        stage,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spmm",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)mat_A,
              (const void*&)mat_B,
              (const void*&)beta,
              (const void*&)mat_C,
              compute_type,
              alg,
              stage,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat_A);
    RETURN_IF_NULLPTR(mat_B);
    RETURN_IF_NULLPTR(mat_C);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
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

    if(rocsparse_enum_utils::is_invalid(compute_type))
    {
        return rocsparse_status_invalid_value;
    }

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

    return rocsparse_spmm_dynamic_dispatch(determine_I_index_type(mat_A),
                                           determine_J_index_type(mat_A),
                                           compute_type,
                                           handle,
                                           trans_A,
                                           trans_B,
                                           alpha,
                                           mat_A,
                                           mat_B,
                                           beta,
                                           mat_C,
                                           alg,
                                           stage,
                                           buffer_size,
                                           temp_buffer);
}

extern "C" rocsparse_status rocsparse_spmm_ex(rocsparse_handle            handle,
                                              rocsparse_operation         trans_A,
                                              rocsparse_operation         trans_B,
                                              const void*                 alpha,
                                              const rocsparse_spmat_descr mat_A,
                                              const rocsparse_dnmat_descr mat_B,
                                              const void*                 beta,
                                              const rocsparse_dnmat_descr mat_C,
                                              rocsparse_datatype          compute_type,
                                              rocsparse_spmm_alg          alg,
                                              rocsparse_spmm_stage        stage,
                                              size_t*                     buffer_size,
                                              void*                       temp_buffer)
{
    return rocsparse_spmm(handle,
                          trans_A,
                          trans_B,
                          alpha,
                          mat_A,
                          mat_B,
                          beta,
                          mat_C,
                          compute_type,
                          alg,
                          stage,
                          buffer_size,
                          temp_buffer);
}

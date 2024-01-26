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

#include "rocsparse_bellmm.hpp"
#include "rocsparse_coomm.hpp"
#include "rocsparse_cscmm.hpp"
#include "rocsparse_csrmm.hpp"

namespace rocsparse
{
    rocsparse_status spmm_alg2bellmm_alg(rocsparse_spmm_alg    spmm_alg,
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
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    rocsparse_status spmm_alg2csrmm_alg(rocsparse_spmm_alg spmm_alg, rocsparse_csrmm_alg& csrmm_alg)
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
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    rocsparse_status spmm_alg2coomm_alg(rocsparse_spmm_alg spmm_alg, rocsparse_coomm_alg& coomm_alg)
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
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status spmm_template(rocsparse_handle            handle,
                                   rocsparse_operation         trans_A,
                                   rocsparse_operation         trans_B,
                                   const void*                 alpha,
                                   rocsparse_const_spmat_descr mat_A,
                                   rocsparse_const_dnmat_descr mat_B,
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
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2csrmm_alg(alg, csrmm_alg)));

            const J m = (J)mat_A->rows;
            const J n = (J)mat_C->cols;
            const J k = (J)mat_A->cols;

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrmm_buffer_size_template<T>(handle,
                                                             trans_A,
                                                             csrmm_alg,
                                                             m,
                                                             n,
                                                             k,
                                                             (I)mat_A->nnz,
                                                             mat_A->descr,
                                                             (const A*)mat_A->const_val_data,
                                                             (const I*)mat_A->const_row_data,
                                                             (const J*)mat_A->const_col_data,
                                                             buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrmm_analysis_template<T>(handle,
                                                          trans_A,
                                                          csrmm_alg,
                                                          m,
                                                          n,
                                                          k,
                                                          (I)mat_A->nnz,
                                                          mat_A->descr,
                                                          (const A*)mat_A->const_val_data,
                                                          (const I*)mat_A->const_row_data,
                                                          (const J*)mat_A->const_col_data,
                                                          temp_buffer));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrmm_template(handle,
                                              trans_A,
                                              trans_B,
                                              csrmm_alg,
                                              m,
                                              n,
                                              k,
                                              (I)mat_A->nnz,
                                              (J)mat_A->batch_count,
                                              mat_A->offsets_batch_stride,
                                              mat_A->columns_values_batch_stride,
                                              (const T*)alpha,
                                              mat_A->descr,
                                              (const A*)mat_A->const_val_data,
                                              (const I*)mat_A->const_row_data,
                                              (const J*)mat_A->const_col_data,
                                              (const B*)mat_B->const_values,
                                              mat_B->ld,
                                              (J)mat_B->batch_count,
                                              mat_B->batch_stride,
                                              mat_B->order,
                                              (const T*)beta,
                                              (C*)mat_C->values,
                                              mat_C->ld,
                                              (J)mat_C->batch_count,
                                              mat_C->batch_stride,
                                              mat_C->order,
                                              temp_buffer,
                                              false));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csc:
        {
            rocsparse_csrmm_alg csrmm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2csrmm_alg(alg, csrmm_alg)));

            const J m = (J)mat_A->rows;
            const J n = (J)mat_C->cols;
            const J k = (J)mat_A->cols;

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::cscmm_buffer_size_template<T>(handle,
                                                             trans_A,
                                                             csrmm_alg,
                                                             m,
                                                             n,
                                                             k,
                                                             (I)mat_A->nnz,
                                                             mat_A->descr,
                                                             (const A*)mat_A->const_val_data,
                                                             (const I*)mat_A->const_col_data,
                                                             (const J*)mat_A->const_row_data,
                                                             buffer_size));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::cscmm_analysis_template<T>(handle,
                                                          trans_A,
                                                          csrmm_alg,
                                                          m,
                                                          n,
                                                          k,
                                                          (I)mat_A->nnz,
                                                          mat_A->descr,
                                                          (const A*)mat_A->const_val_data,
                                                          (const I*)mat_A->const_col_data,
                                                          (const J*)mat_A->const_row_data,
                                                          temp_buffer));
                return rocsparse_status_success;
            }
            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::cscmm_template(handle,
                                              trans_A,
                                              trans_B,
                                              csrmm_alg,
                                              m,
                                              n,
                                              k,
                                              (I)mat_A->nnz,
                                              (J)mat_A->batch_count,
                                              mat_A->offsets_batch_stride,
                                              mat_A->columns_values_batch_stride,
                                              (const T*)alpha,
                                              mat_A->descr,
                                              (const A*)mat_A->const_val_data,
                                              (const I*)mat_A->const_col_data,
                                              (const J*)mat_A->const_row_data,
                                              (const B*)mat_B->const_values,
                                              mat_B->ld,
                                              (J)mat_B->batch_count,
                                              mat_B->batch_stride,
                                              mat_B->order,
                                              (const T*)beta,
                                              (C*)mat_C->values,
                                              mat_C->ld,
                                              (J)mat_C->batch_count,
                                              mat_C->batch_stride,
                                              mat_C->order,
                                              temp_buffer));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_coo:
        {
            rocsparse_coomm_alg coomm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2coomm_alg(alg, coomm_alg)));

            const I m = (I)mat_A->rows;
            const I n = (I)mat_C->cols;
            const I k = (I)mat_A->cols;

            switch(stage)
            {
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::coomm_buffer_size_template<T>(handle,
                                                             trans_A,
                                                             coomm_alg,
                                                             m,
                                                             n,
                                                             k,
                                                             mat_A->nnz,
                                                             (I)mat_C->batch_count,
                                                             mat_A->descr,
                                                             (const A*)mat_A->const_val_data,
                                                             (const I*)mat_A->const_row_data,
                                                             (const I*)mat_A->const_col_data,
                                                             buffer_size));
                return rocsparse_status_success;
            }

            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::coomm_analysis_template<T>(handle,
                                                          trans_A,
                                                          coomm_alg,
                                                          m,
                                                          n,
                                                          k,
                                                          mat_A->nnz,
                                                          mat_A->descr,
                                                          (const A*)mat_A->const_val_data,
                                                          (const I*)mat_A->const_row_data,
                                                          (const I*)mat_A->const_col_data,
                                                          temp_buffer));
                return rocsparse_status_success;
            }

            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomm_template(handle,
                                                                    trans_A,
                                                                    trans_B,
                                                                    coomm_alg,
                                                                    m,
                                                                    n,
                                                                    k,
                                                                    mat_A->nnz,
                                                                    (I)mat_A->batch_count,
                                                                    mat_A->batch_stride,
                                                                    (const T*)alpha,
                                                                    mat_A->descr,
                                                                    (const A*)mat_A->const_val_data,
                                                                    (const I*)mat_A->const_row_data,
                                                                    (const I*)mat_A->const_col_data,
                                                                    (const B*)mat_B->const_values,
                                                                    mat_B->ld,
                                                                    (I)mat_B->batch_count,
                                                                    mat_B->batch_stride,
                                                                    mat_B->order,
                                                                    (const T*)beta,
                                                                    (C*)mat_C->values,
                                                                    mat_C->ld,
                                                                    (I)mat_C->batch_count,
                                                                    mat_C->batch_stride,
                                                                    mat_C->order,
                                                                    temp_buffer));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_bell:
        {
            rocsparse_bellmm_alg bellmm_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmm_alg2bellmm_alg(alg, bellmm_alg)));

            switch(stage)
            {
                //
                // STAGE BUFFER SIZE
                //
            case rocsparse_spmm_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bellmm_template_buffer_size<T, I>(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->block_dir,
                    (I)(mat_C->rows / mat_A->block_dim),
                    (I)mat_C->cols,

                    (trans_A == rocsparse_operation_none) ? (I)(mat_A->cols / mat_A->block_dim)
                                                          : (I)(mat_A->rows / mat_A->block_dim),

                    (I)mat_A->ell_cols,
                    (I)mat_A->block_dim,
                    (const T*)alpha,
                    mat_A->descr,
                    (const I*)mat_A->const_col_data,
                    (const T*)mat_A->const_val_data,
                    (const T*)mat_B->const_values,
                    mat_B->ld,
                    mat_B->order,
                    (const T*)beta,
                    (T*)mat_C->values,
                    mat_C->ld,
                    mat_C->order,
                    buffer_size)));
                return rocsparse_status_success;
            }

                //
                // STAGE PREPROCESS
                //
            case rocsparse_spmm_stage_preprocess:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bellmm_template_preprocess<T, I>(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->block_dir,
                    (I)(mat_C->rows / mat_A->block_dim),
                    (I)mat_C->cols,

                    (trans_A == rocsparse_operation_none) ? (I)(mat_A->cols / mat_A->block_dim)
                                                          : (I)(mat_A->rows / mat_A->block_dim),

                    (I)mat_A->ell_cols,
                    (I)mat_A->block_dim,
                    (const T*)alpha,
                    mat_A->descr,
                    (const I*)mat_A->const_col_data,
                    (const T*)mat_A->const_val_data,
                    (const T*)mat_B->const_values,
                    mat_B->ld,
                    mat_B->order,
                    (const T*)beta,
                    (T*)mat_C->values,
                    mat_C->ld,
                    mat_C->order,
                    temp_buffer)));
                return rocsparse_status_success;
            }

                //
                // STAGE COMPUTE
                //
            case rocsparse_spmm_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::bellmm_template<T, I>(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->block_dir,
                    (I)(mat_C->rows / mat_A->block_dim),
                    (I)mat_C->cols,

                    (trans_A == rocsparse_operation_none) ? (I)(mat_A->cols / mat_A->block_dim)
                                                          : (I)(mat_A->rows / mat_A->block_dim),

                    (I)mat_A->ell_cols,
                    (I)mat_A->block_dim,
                    (I)mat_A->batch_count,
                    mat_A->batch_stride,
                    (const T*)alpha,
                    mat_A->descr,
                    (const I*)mat_A->const_col_data,
                    (const T*)mat_A->const_val_data,
                    (const T*)mat_B->const_values,
                    mat_B->ld,
                    (I)mat_B->batch_count,
                    mat_B->batch_stride,
                    mat_B->order,
                    (const T*)beta,
                    (T*)mat_C->values,
                    mat_C->ld,
                    (I)mat_C->batch_count,
                    mat_C->batch_stride,
                    mat_C->order,
                    temp_buffer)));
                return rocsparse_status_success;
            }
            }

            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_ell:
        case rocsparse_format_bsr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    template <typename... Ts>
    static inline rocsparse_status spmm_dynamic_dispatch(rocsparse_indextype itype,
                                                         rocsparse_indextype jtype,
                                                         rocsparse_datatype  atype,
                                                         rocsparse_datatype  btype,
                                                         rocsparse_datatype  ctype,
                                                         rocsparse_datatype  compute_type,
                                                         Ts&&... ts)
    {
        assert(compute_type == atype);
        assert(compute_type == btype);
        assert(compute_type == ctype);

        switch(compute_type)
        {
        case rocsparse_datatype_f32_r:
        {
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<float, int32_t, int32_t, float, float, float>(
                            ts...)));
                    return rocsparse_status_success;
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<float, int64_t, int32_t, float, float, float>(
                            ts...)));
                    return rocsparse_status_success;
                }
                case rocsparse_indextype_i64:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<float, int64_t, int64_t, float, float, float>(
                            ts...)));
                    return rocsparse_status_success;
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<double, int32_t, int32_t, double, double, double>(
                            ts...)));
                    return rocsparse_status_success;
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<double, int64_t, int32_t, double, double, double>(
                            ts...)));
                    return rocsparse_status_success;
                }
                case rocsparse_indextype_i64:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<double, int64_t, int64_t, double, double, double>(
                            ts...)));
                    return rocsparse_status_success;
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<rocsparse_float_complex,
                                                  int32_t,
                                                  int32_t,
                                                  rocsparse_float_complex,
                                                  rocsparse_float_complex,
                                                  rocsparse_float_complex>(ts...)));
                    return rocsparse_status_success;
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<rocsparse_float_complex,
                                                  int64_t,
                                                  int32_t,
                                                  rocsparse_float_complex,
                                                  rocsparse_float_complex,
                                                  rocsparse_float_complex>(ts...)));
                    return rocsparse_status_success;
                }
                case rocsparse_indextype_i64:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<rocsparse_float_complex,
                                                  int64_t,
                                                  int64_t,
                                                  rocsparse_float_complex,
                                                  rocsparse_float_complex,
                                                  rocsparse_float_complex>(ts...)));
                    return rocsparse_status_success;
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<rocsparse_double_complex,
                                                  int32_t,
                                                  int32_t,
                                                  rocsparse_double_complex,
                                                  rocsparse_double_complex,
                                                  rocsparse_double_complex>(ts...)));
                    return rocsparse_status_success;
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
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<rocsparse_double_complex,
                                                  int64_t,
                                                  int32_t,
                                                  rocsparse_double_complex,
                                                  rocsparse_double_complex,
                                                  rocsparse_double_complex>(ts...)));
                    return rocsparse_status_success;
                }
                case rocsparse_indextype_i64:
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::spmm_template<rocsparse_double_complex,
                                                  int64_t,
                                                  int64_t,
                                                  rocsparse_double_complex,
                                                  rocsparse_double_complex,
                                                  rocsparse_double_complex>(ts...)));
                    return rocsparse_status_success;
                }
                }
            }
            }
        }
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    rocsparse_indextype determine_I_index_type(rocsparse_const_spmat_descr mat)
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

    rocsparse_indextype determine_J_index_type(rocsparse_const_spmat_descr mat)
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

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_spmm(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans_A, //1
                                           rocsparse_operation         trans_B, //2
                                           const void*                 alpha, //3
                                           rocsparse_const_spmat_descr mat_A, //4
                                           rocsparse_const_dnmat_descr mat_B, //5
                                           const void*                 beta, //6
                                           const rocsparse_dnmat_descr mat_C, //7
                                           rocsparse_datatype          compute_type, //8
                                           rocsparse_spmm_alg          alg, //9
                                           rocsparse_spmm_stage        stage, //10
                                           size_t*                     buffer_size, //11
                                           void*                       temp_buffer) //12
try
{
    log_trace(handle,
              "rocsparse_spmm",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)mat_A,
              (const void*&)mat_B,
              (const void*&)mat_C,
              compute_type,
              alg,
              stage,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, mat_A);
    ROCSPARSE_CHECKARG(4, mat_A, mat_A->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(5, mat_B);
    ROCSPARSE_CHECKARG_POINTER(6, beta);
    ROCSPARSE_CHECKARG(5, mat_B, mat_B->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(7, mat_C);
    ROCSPARSE_CHECKARG(7, mat_C, mat_C->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_ENUM(8, compute_type);
    ROCSPARSE_CHECKARG(8,
                       compute_type,
                       (compute_type != mat_A->data_type || compute_type != mat_B->data_type
                        || compute_type != mat_C->data_type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ENUM(9, alg);
    ROCSPARSE_CHECKARG_ENUM(10, stage);
    ROCSPARSE_CHECKARG(11,
                       buffer_size,
                       (temp_buffer == nullptr && buffer_size == nullptr),
                       rocsparse_status_invalid_pointer);

    switch(stage)
    {

    case rocsparse_spmm_stage_buffer_size:
    {
        ROCSPARSE_CHECKARG_POINTER(11, buffer_size);
        break;
    }
    case rocsparse_spmm_stage_preprocess:
    {
        break;
    }
    case rocsparse_spmm_stage_compute:
    {
        break;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spmm_dynamic_dispatch(rocsparse::determine_I_index_type(mat_A),
                                         rocsparse::determine_J_index_type(mat_A),
                                         mat_A->data_type,
                                         mat_B->data_type,
                                         mat_C->data_type,
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
                                         temp_buffer));

    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

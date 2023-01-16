/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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

#include "rocsparse_bsrmv.hpp"
#include "rocsparse_coomv.hpp"
#include "rocsparse_coomv_aos.hpp"
#include "rocsparse_cscmv.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_ellmv.hpp"

static rocsparse_status rocsparse_check_spmv_alg(rocsparse_format format, rocsparse_spmv_alg alg)
{
    switch(format)
    {
    case rocsparse_format_csr:
    case rocsparse_format_csc:
    {
        switch(alg)
        {
        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_csr_stream:
        case rocsparse_spmv_alg_csr_adaptive:
        {
            return rocsparse_status_success;
        }
        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_coo_atomic:
        {
            return rocsparse_status_invalid_value;
        }
        }

        return rocsparse_status_invalid_value;
    }
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    {
        switch(alg)
        {
        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_coo_atomic:
        {
            return rocsparse_status_success;
        }
        case rocsparse_spmv_alg_csr_stream:
        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_ell:
        {
            return rocsparse_status_invalid_value;
        }
        }

        return rocsparse_status_invalid_value;
    }
    case rocsparse_format_ell:
    {
        switch(alg)
        {
        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_ell:
        {
            return rocsparse_status_success;
        }
        case rocsparse_spmv_alg_csr_stream:
        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_coo_atomic:
        {
            return rocsparse_status_invalid_value;
        }
        }

        return rocsparse_status_invalid_value;
    }
    case rocsparse_format_bell:
    {
        switch(alg)
        {
        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_csr_stream:
        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_coo_atomic:
        {
            return rocsparse_status_invalid_value;
        }
        }

        return rocsparse_status_invalid_value;
    }

    case rocsparse_format_bsr:
    {
        switch(alg)
        {
        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_bsr:
        {
            return rocsparse_status_success;
        }
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_csr_stream:
        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_coo_atomic:
        {
            return rocsparse_status_invalid_value;
        }
        }

        return rocsparse_status_invalid_value;
    }
    }

    return rocsparse_status_invalid_value;
}

static rocsparse_status rocsparse_spmv_alg2coomv_alg(rocsparse_spmv_alg   spmv_alg,
                                                     rocsparse_coomv_alg& coomv_alg)
{
    switch(spmv_alg)
    {
    case rocsparse_spmv_alg_default:
    {
        coomv_alg = rocsparse_coomv_alg_default;
        return rocsparse_status_success;
    }

    case rocsparse_spmv_alg_coo:
    {
        coomv_alg = rocsparse_coomv_alg_segmented;
        return rocsparse_status_success;
    }

    case rocsparse_spmv_alg_coo_atomic:
    {
        coomv_alg = rocsparse_coomv_alg_atomic;
        return rocsparse_status_success;
    }

    case rocsparse_spmv_alg_csr_adaptive:
    case rocsparse_spmv_alg_csr_stream:
    case rocsparse_spmv_alg_bsr:
    case rocsparse_spmv_alg_ell:
    {
        return rocsparse_status_invalid_value;
    }
    }
    return rocsparse_status_invalid_value;
}

static rocsparse_status rocsparse_spmv_alg2coomv_aos_alg(rocsparse_spmv_alg       spmv_alg,
                                                         rocsparse_coomv_aos_alg& coomv_aos_alg)
{
    switch(spmv_alg)
    {
    case rocsparse_spmv_alg_default:
    {
        coomv_aos_alg = rocsparse_coomv_aos_alg_default;
        return rocsparse_status_success;
    }

    case rocsparse_spmv_alg_coo:
    {
        coomv_aos_alg = rocsparse_coomv_aos_alg_segmented;
        return rocsparse_status_success;
    }

    case rocsparse_spmv_alg_coo_atomic:
    {
        coomv_aos_alg = rocsparse_coomv_aos_alg_atomic;
        return rocsparse_status_success;
    }

    case rocsparse_spmv_alg_csr_adaptive:
    case rocsparse_spmv_alg_csr_stream:
    case rocsparse_spmv_alg_bsr:
    case rocsparse_spmv_alg_ell:
    {
        return rocsparse_status_invalid_value;
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
    case rocsparse_format_bsr:
    case rocsparse_format_ell:
    case rocsparse_format_bell:
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
    case rocsparse_format_bsr:
    case rocsparse_format_ell:
    case rocsparse_format_bell:
    {
        return mat->col_type;
    }
    case rocsparse_format_csc:
    {
        return mat->row_type;
    }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse_spmv_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans,
                                         const void*                 alpha,
                                         const rocsparse_spmat_descr mat,
                                         const rocsparse_dnvec_descr x,
                                         const void*                 beta,
                                         const rocsparse_dnvec_descr y,
                                         rocsparse_spmv_alg          alg,
                                         rocsparse_spmv_stage        stage,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer);

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse_spmv_template_auto(rocsparse_handle            handle,
                                              rocsparse_operation         trans,
                                              const void*                 alpha,
                                              const rocsparse_spmat_descr mat,
                                              const rocsparse_dnvec_descr x,
                                              const void*                 beta,
                                              const rocsparse_dnvec_descr y,
                                              rocsparse_spmv_alg          alg,
                                              size_t*                     buffer_size,
                                              void*                       temp_buffer)
{
    if(temp_buffer == nullptr)
    {
        rocsparse_status status
            = rocsparse_spmv_template<T, I, J, A, X, Y>(handle,
                                                        trans,
                                                        alpha,
                                                        mat,
                                                        x,
                                                        beta,
                                                        y,
                                                        alg,
                                                        rocsparse_spmv_stage_buffer_size,
                                                        buffer_size,
                                                        temp_buffer);
        if(status != rocsparse_status_success)
        {
            return status;
        }

        //
        // This is needed in auto mode, otherwise the 'allocated' temporary buffer of size 0 will be again a nullptr.
        //
        *buffer_size = std::max(static_cast<size_t>(4), *buffer_size);
        return status;
    }
    else
    {
        rocsparse_status status
            = rocsparse_spmv_template<T, I, J, A, X, Y>(handle,
                                                        trans,
                                                        alpha,
                                                        mat,
                                                        x,
                                                        beta,
                                                        y,
                                                        alg,
                                                        rocsparse_spmv_stage_preprocess,
                                                        buffer_size,
                                                        temp_buffer);
        if(status != rocsparse_status_success)
        {
            return status;
        }

        return rocsparse_spmv_template<T, I, J, A, X, Y>(handle,
                                                         trans,
                                                         alpha,
                                                         mat,
                                                         x,
                                                         beta,
                                                         y,
                                                         alg,
                                                         rocsparse_spmv_stage_compute,
                                                         buffer_size,
                                                         temp_buffer);
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse_spmv_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans,
                                         const void*                 alpha,
                                         const rocsparse_spmat_descr mat,
                                         const rocsparse_dnvec_descr x,
                                         const void*                 beta,
                                         const rocsparse_dnvec_descr y,
                                         rocsparse_spmv_alg          alg,
                                         rocsparse_spmv_stage        stage,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    RETURN_IF_ROCSPARSE_ERROR((rocsparse_check_spmv_alg(mat->format, alg)));

    switch(mat->format)
    {
    case rocsparse_format_coo:
    {
        rocsparse_coomv_alg coomv_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmv_alg2coomv_alg(alg, coomv_alg)));

        switch(stage)
        {
        case rocsparse_spmv_stage_buffer_size:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        case rocsparse_spmv_stage_preprocess:
        {
            if(alg == rocsparse_spmv_alg_coo_atomic && mat->analysed == false)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_coomv_analysis_template(handle,
                                                       trans,
                                                       coomv_alg,
                                                       (I)mat->rows,
                                                       (I)mat->cols,
                                                       mat->nnz,
                                                       mat->descr,
                                                       (const A*)mat->val_data,
                                                       (const I*)mat->row_data,
                                                       (const I*)mat->col_data)));

                mat->analysed = true;
            }
            return rocsparse_status_success;
        }
        case rocsparse_spmv_stage_compute:
        {
            return rocsparse_coomv_template(handle,
                                            trans,
                                            coomv_alg,
                                            (I)mat->rows,
                                            (I)mat->cols,
                                            mat->nnz,
                                            (const T*)alpha,
                                            mat->descr,
                                            (const A*)mat->val_data,
                                            (const I*)mat->row_data,
                                            (const I*)mat->col_data,
                                            (const X*)x->values,
                                            (const T*)beta,
                                            (Y*)y->values);
        }

        case rocsparse_spmv_stage_auto:
        {
            return rocsparse_spmv_template_auto<T, I, J, A, X, Y>(
                handle, trans, alpha, mat, x, beta, y, alg, buffer_size, temp_buffer);
        }
        }
    }

    case rocsparse_format_coo_aos:
    {
        rocsparse_coomv_aos_alg coomv_aos_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmv_alg2coomv_aos_alg(alg, coomv_aos_alg)));

        switch(stage)
        {
        case rocsparse_spmv_stage_buffer_size:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        case rocsparse_spmv_stage_preprocess:
        {
            return rocsparse_status_success;
        }
        case rocsparse_spmv_stage_compute:
        {
            return rocsparse_coomv_aos_template(handle,
                                                trans,
                                                coomv_aos_alg,
                                                (I)mat->rows,
                                                (I)mat->cols,
                                                mat->nnz,
                                                (const T*)alpha,
                                                mat->descr,
                                                (const A*)mat->val_data,
                                                (const I*)mat->ind_data,
                                                (const X*)x->values,
                                                (const T*)beta,
                                                (Y*)y->values);
        }

        case rocsparse_spmv_stage_auto:
        {
            return rocsparse_spmv_template_auto<T, I, J, A, X, Y>(
                handle, trans, alpha, mat, x, beta, y, alg, buffer_size, temp_buffer);
        }
        }
    }

    case rocsparse_format_bsr:
    {
        switch(stage)
        {
        case rocsparse_spmv_stage_buffer_size:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_stage_preprocess:
        {
            rocsparse_status status = rocsparse_status_success;
            //
            // If algorithm 1 or default is selected and analysis step is required
            //
            if(alg == rocsparse_spmv_alg_default && mat->analysed == false)
            {
                status = rocsparse_bsrmv_analysis_template(handle,
                                                           mat->block_dir,
                                                           trans,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->val_data,
                                                           (const I*)mat->row_data,
                                                           (const J*)mat->col_data,
                                                           (J)mat->block_dim,
                                                           mat->info);
                if(status != rocsparse_status_success)
                {
                    return status;
                }

                mat->analysed = true;
            }

            return status;
        }

        case rocsparse_spmv_stage_compute:
        {
            return rocsparse_bsrmv_template(handle,
                                            mat->block_dir,
                                            trans,
                                            (J)mat->rows,
                                            (J)mat->cols,
                                            (I)mat->nnz,
                                            (const T*)alpha,
                                            mat->descr,
                                            (const A*)mat->val_data,
                                            (const I*)mat->row_data,
                                            (const J*)mat->col_data,
                                            (J)mat->block_dim,
                                            mat->info,
                                            (const X*)x->values,
                                            (const T*)beta,
                                            (Y*)y->values);
        }

        case rocsparse_spmv_stage_auto:
        {
            return rocsparse_spmv_template_auto<T, I, J, A, X, Y>(
                handle, trans, alpha, mat, x, beta, y, alg, buffer_size, temp_buffer);
        }
        }
    }

    case rocsparse_format_csr:
    {
        switch(stage)
        {
        case rocsparse_spmv_stage_buffer_size:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_stage_preprocess:
        {
            rocsparse_status status = rocsparse_status_success;
            //
            // If algorithm 1 or default is selected and analysis step is required
            //
            if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive)
               && mat->analysed == false)
            {
                status = rocsparse_csrmv_analysis_template(handle,
                                                           trans,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->val_data,
                                                           (const I*)mat->row_data,
                                                           (const J*)mat->col_data,
                                                           mat->info);
                if(status != rocsparse_status_success)
                {
                    return status;
                }

                mat->analysed = true;
            }

            return status;
        }

        case rocsparse_spmv_stage_compute:
        {
            return rocsparse_csrmv_template(handle,
                                            trans,
                                            (J)mat->rows,
                                            (J)mat->cols,
                                            (I)mat->nnz,
                                            (const T*)alpha,
                                            mat->descr,
                                            (const A*)mat->val_data,
                                            (const I*)mat->row_data,
                                            ((const I*)mat->row_data) + 1,
                                            (const J*)mat->col_data,
                                            (alg == rocsparse_spmv_alg_csr_stream) ? nullptr
                                                                                   : mat->info,
                                            (const X*)x->values,
                                            (const T*)beta,
                                            (Y*)y->values,
                                            false);
        }

        case rocsparse_spmv_stage_auto:
        {
            return rocsparse_spmv_template_auto<T, I, J, A, X, Y>(
                handle, trans, alpha, mat, x, beta, y, alg, buffer_size, temp_buffer);
        }
        }
    }

    case rocsparse_format_csc:
    {
        switch(stage)
        {
        case rocsparse_spmv_stage_buffer_size:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_stage_preprocess:
        {
            rocsparse_status status = rocsparse_status_success;
            //
            // If algorithm 1 or default is selected and analysis step is required
            //
            if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive)
               && mat->analysed == false)
            {
                status = rocsparse_cscmv_analysis_template(handle,
                                                           trans,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->val_data,
                                                           (const I*)mat->col_data,
                                                           (const J*)mat->row_data,
                                                           mat->info);
                if(status != rocsparse_status_success)
                {
                    return status;
                }

                mat->analysed = true;
            }

            return status;
        }

        case rocsparse_spmv_stage_compute:
        {
            return rocsparse_cscmv_template(handle,
                                            trans,
                                            (J)mat->rows,
                                            (J)mat->cols,
                                            (I)mat->nnz,
                                            (const T*)alpha,
                                            mat->descr,
                                            (const A*)mat->val_data,
                                            (const I*)mat->col_data,
                                            (const J*)mat->row_data,
                                            (alg == rocsparse_spmv_alg_csr_stream) ? nullptr
                                                                                   : mat->info,
                                            (const X*)x->values,
                                            (const T*)beta,
                                            (Y*)y->values);
        }

        case rocsparse_spmv_stage_auto:
        {
            return rocsparse_spmv_template_auto<T, I, J, A, X, Y>(
                handle, trans, alpha, mat, x, beta, y, alg, buffer_size, temp_buffer);
        }
        }
    }

    case rocsparse_format_ell:
    {
        switch(stage)
        {
        case rocsparse_spmv_stage_buffer_size:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_stage_preprocess:
        {
            return rocsparse_status_success;
        }

        case rocsparse_spmv_stage_compute:
        {
            return rocsparse_ellmv_template(handle,
                                            trans,
                                            (I)mat->rows,
                                            (I)mat->cols,
                                            (const T*)alpha,
                                            mat->descr,
                                            (const A*)mat->val_data,
                                            (const I*)mat->col_data,
                                            (I)mat->ell_width,
                                            (const X*)x->values,
                                            (const T*)beta,
                                            (Y*)y->values);
        }

        case rocsparse_spmv_stage_auto:
        {
            return rocsparse_spmv_template_auto<T, I, J, A, X, Y>(
                handle, trans, alpha, mat, x, beta, y, alg, buffer_size, temp_buffer);
        }
        }
    }

    case rocsparse_format_bell:
    {
        // LCOV_EXCL_START
        return rocsparse_status_not_implemented;
        // LCOV_EXCL_STOP
    }
    }

    return rocsparse_status_invalid_value;
}

template <typename... Ts>
rocsparse_status rocsparse_spmv_dynamic_dispatch(rocsparse_indextype itype,
                                                 rocsparse_indextype jtype,
                                                 rocsparse_datatype  atype,
                                                 rocsparse_datatype  xtype,
                                                 rocsparse_datatype  ytype,
                                                 rocsparse_datatype  ctype,
                                                 Ts&&... ts)
{
#define DISPATCH_COMPUTE_TYPE_I32R(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)                 \
    if(atype == rocsparse_datatype_i8_r && xtype == rocsparse_datatype_i8_r                  \
       && ytype == rocsparse_datatype_i32_r)                                                 \
    {                                                                                        \
        return rocsparse_spmv_template<CTYPE, ITYPE, JTYPE, int8_t, int8_t, int32_t>(ts...); \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        return rocsparse_status_not_implemented;                                             \
    }

#define DISPATCH_COMPUTE_TYPE_F32R(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)               \
    if(atype == rocsparse_datatype_f32_r && atype == xtype && atype == ytype)              \
    {                                                                                      \
        return rocsparse_spmv_template<CTYPE, ITYPE, JTYPE, float, float, float>(ts...);   \
    }                                                                                      \
    else if(atype == rocsparse_datatype_i8_r && xtype == rocsparse_datatype_i8_r           \
            && ytype == rocsparse_datatype_f32_r)                                          \
    {                                                                                      \
        return rocsparse_spmv_template<CTYPE, ITYPE, JTYPE, int8_t, int8_t, float>(ts...); \
    }                                                                                      \
    else                                                                                   \
    {                                                                                      \
        return rocsparse_status_not_implemented;                                           \
    }

#define DISPATCH_COMPUTE_TYPE_F64R(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)                \
    if(atype == rocsparse_datatype_f64_r && atype == xtype && atype == ytype)               \
    {                                                                                       \
        return rocsparse_spmv_template<CTYPE, ITYPE, JTYPE, double, double, double>(ts...); \
    }                                                                                       \
    else                                                                                    \
    {                                                                                       \
        return rocsparse_status_not_implemented;                                            \
    }

#define DISPATCH_COMPUTE_TYPE_F32C(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)       \
    if(atype == rocsparse_datatype_f32_c && atype == xtype && atype == ytype)      \
    {                                                                              \
        return rocsparse_spmv_template<CTYPE,                                      \
                                       ITYPE,                                      \
                                       JTYPE,                                      \
                                       rocsparse_float_complex,                    \
                                       rocsparse_float_complex,                    \
                                       rocsparse_float_complex>(ts...);            \
    }                                                                              \
    else if(atype == rocsparse_datatype_f32_r && xtype == rocsparse_datatype_f32_c \
            && ytype == rocsparse_datatype_f32_c)                                  \
    {                                                                              \
        return rocsparse_spmv_template<CTYPE,                                      \
                                       ITYPE,                                      \
                                       JTYPE,                                      \
                                       float,                                      \
                                       rocsparse_float_complex,                    \
                                       rocsparse_float_complex>(ts...);            \
    }                                                                              \
    else                                                                           \
    {                                                                              \
        return rocsparse_status_not_implemented;                                   \
    }

#define DISPATCH_COMPUTE_TYPE_F64C(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)       \
    if(atype == rocsparse_datatype_f64_c && atype == xtype && atype == ytype)      \
    {                                                                              \
        return rocsparse_spmv_template<CTYPE,                                      \
                                       ITYPE,                                      \
                                       JTYPE,                                      \
                                       rocsparse_double_complex,                   \
                                       rocsparse_double_complex,                   \
                                       rocsparse_double_complex>(ts...);           \
    }                                                                              \
    else if(atype == rocsparse_datatype_f64_r && xtype == rocsparse_datatype_f64_c \
            && ytype == rocsparse_datatype_f64_c)                                  \
    {                                                                              \
        return rocsparse_spmv_template<CTYPE,                                      \
                                       ITYPE,                                      \
                                       JTYPE,                                      \
                                       double,                                     \
                                       rocsparse_double_complex,                   \
                                       rocsparse_double_complex>(ts...);           \
    }                                                                              \
    else                                                                           \
    {                                                                              \
        return rocsparse_status_not_implemented;                                   \
    }

#define DISPATCH_COMPUTE_TYPE(ITYPE, JTYPE, atype, xtype, ytype, ctype)                         \
    switch(ctype)                                                                               \
    {                                                                                           \
    case rocsparse_datatype_i32_r:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_I32R(ITYPE, JTYPE, int32_t, atype, xtype, ytype)                  \
    }                                                                                           \
    case rocsparse_datatype_f32_r:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F32R(ITYPE, JTYPE, float, atype, xtype, ytype)                    \
    }                                                                                           \
    case rocsparse_datatype_f64_r:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F64R(ITYPE, JTYPE, double, atype, xtype, ytype)                   \
    }                                                                                           \
    case rocsparse_datatype_f32_c:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F32C(ITYPE, JTYPE, rocsparse_float_complex, atype, xtype, ytype)  \
    }                                                                                           \
    case rocsparse_datatype_f64_c:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F64C(ITYPE, JTYPE, rocsparse_double_complex, atype, xtype, ytype) \
    }                                                                                           \
    case rocsparse_datatype_i8_r:                                                               \
    case rocsparse_datatype_u8_r:                                                               \
    case rocsparse_datatype_u32_r:                                                              \
    {                                                                                           \
        return rocsparse_status_not_implemented;                                                \
    }                                                                                           \
    }

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
            DISPATCH_COMPUTE_TYPE(int32_t, int32_t, atype, xtype, ytype, ctype);
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
            DISPATCH_COMPUTE_TYPE(int64_t, int32_t, atype, xtype, ytype, ctype);
        }
        case rocsparse_indextype_i64:
        {
            DISPATCH_COMPUTE_TYPE(int64_t, int64_t, atype, xtype, ytype, ctype);
        }
        }
    }
    }

    return rocsparse_status_invalid_value;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spmv(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr mat,
                                           const rocsparse_dnvec_descr x,
                                           const void*                 beta,
                                           const rocsparse_dnvec_descr y,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_spmv_alg          alg,
                                           rocsparse_spmv_stage        stage,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);

    // Logging
    log_trace(handle,
              "rocsparse_spmv",
              trans,
              (const void*&)alpha,
              (const void*&)mat,
              (const void*&)x,
              (const void*&)beta,
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
    RETURN_IF_NULLPTR(beta);

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

    return rocsparse_spmv_dynamic_dispatch(determine_I_index_type(mat),
                                           determine_J_index_type(mat),
                                           mat->data_type,
                                           x->data_type,
                                           y->data_type,
                                           compute_type,
                                           handle,
                                           trans,
                                           alpha,
                                           mat,
                                           x,
                                           beta,
                                           y,
                                           alg,
                                           stage,
                                           buffer_size,
                                           temp_buffer);
}

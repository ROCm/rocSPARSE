/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

rocsparse_status rocsparse_check_spmv_alg(rocsparse_format format, rocsparse_spmv_alg alg)
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

rocsparse_status rocsparse_spmv_alg2coomv_alg(rocsparse_spmv_alg   spmv_alg,
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

rocsparse_status rocsparse_spmv_alg2coomv_aos_alg(rocsparse_spmv_alg       spmv_alg,
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

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spmv_template(rocsparse_handle            handle,
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
    RETURN_IF_ROCSPARSE_ERROR((rocsparse_check_spmv_alg(mat->format, alg)));

    switch(mat->format)
    {
    case rocsparse_format_coo:
    {
        rocsparse_coomv_alg coomv_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmv_alg2coomv_alg(alg, coomv_alg)));

        if(temp_buffer == nullptr)
        {
            // We do not need a buffer and we can't return a zero buffer size here
            // otherwise the allocated buffer will be again a null pointer.
            // That's a flaw we need to fix with the use of a rocsparse_spmv_stage enum.
            // We need to wait the next major release where the C-API is allowed to be modified.
            *buffer_size = 4;
            return rocsparse_status_success;
        }

        return rocsparse_coomv_template(handle,
                                        trans,
                                        coomv_alg,
                                        (I)mat->rows,
                                        (I)mat->cols,
                                        (I)mat->nnz,
                                        (const T*)alpha,
                                        mat->descr,
                                        (const T*)mat->val_data,
                                        (const I*)mat->row_data,
                                        (const I*)mat->col_data,
                                        (const T*)x->values,
                                        (const T*)beta,
                                        (T*)y->values);
    }

    case rocsparse_format_coo_aos:
    {
        rocsparse_coomv_aos_alg coomv_aos_alg;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse_spmv_alg2coomv_aos_alg(alg, coomv_aos_alg)));

        if(temp_buffer == nullptr)
        {
            // We do not need a buffer and we can't return a zero buffer size here
            // otherwise the allocated buffer will be again a null pointer.
            // That's a flaw we need to fix with the use of a rocsparse_spmv_stage enum.
            // We need to wait the next major release where the C-API is allowed to be modified.
            *buffer_size = 4;
            return rocsparse_status_success;
        }

        return rocsparse_coomv_aos_template(handle,
                                            trans,
                                            coomv_aos_alg,
                                            (I)mat->rows,
                                            (I)mat->cols,
                                            (I)mat->nnz,
                                            (const T*)alpha,
                                            mat->descr,
                                            (const T*)mat->val_data,
                                            (const I*)mat->ind_data,
                                            (const T*)x->values,
                                            (const T*)beta,
                                            (T*)y->values);
    }

    case rocsparse_format_csr:
    {
        // If temp_buffer is nullptr, return buffer_size
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer and we can't return a zero buffer size here
            // otherwise the allocated buffer will be again a null pointer.
            // That's a flaw we need to fix with the use of a rocsparse_spmv_stage enum.
            // We need to wait the next major release where the C-API is allowed to be modified.
            *buffer_size = 4;

            // If algorithm 1 or default is selected and analysis step is required
            if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive)
               && mat->analysed == false)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_csrmv_analysis_template(handle,
                                                       trans,
                                                       (J)mat->rows,
                                                       (J)mat->cols,
                                                       (I)mat->nnz,
                                                       mat->descr,
                                                       (const T*)mat->val_data,
                                                       (const I*)mat->row_data,
                                                       (const J*)mat->col_data,
                                                       mat->info)));

                mat->analysed = true;
            }

            return rocsparse_status_success;
        }

        return rocsparse_csrmv_template(handle,
                                        trans,
                                        (J)mat->rows,
                                        (J)mat->cols,
                                        (I)mat->nnz,
                                        (const T*)alpha,
                                        mat->descr,
                                        (const T*)mat->val_data,
                                        (const I*)mat->row_data,
                                        (const J*)mat->col_data,
                                        (alg == rocsparse_spmv_alg_csr_stream) ? nullptr
                                                                               : mat->info,
                                        (const T*)x->values,
                                        (const T*)beta,
                                        (T*)y->values,
                                        false);
    }

    case rocsparse_format_csc:
    {
        // If temp_buffer is nullptr, return buffer_size
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer and we can't return a zero buffer size here
            // otherwise the allocated buffer will be again a null pointer.
            // That's a flaw we need to fix with the use of a rocsparse_spmv_stage enum.
            // We need to wait the next major release where the C-API is allowed to be modified.
            *buffer_size = 4;

            // If algorithm 1 or default is selected and analysis step is required
            if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive)
               && mat->analysed == false)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_cscmv_analysis_template(handle,
                                                       trans,
                                                       (J)mat->rows,
                                                       (J)mat->cols,
                                                       (I)mat->nnz,
                                                       mat->descr,
                                                       (const T*)mat->val_data,
                                                       (const I*)mat->col_data,
                                                       (const J*)mat->row_data,
                                                       mat->info)));

                mat->analysed = true;
            }

            return rocsparse_status_success;
        }

        return rocsparse_cscmv_template(handle,
                                        trans,
                                        (J)mat->rows,
                                        (J)mat->cols,
                                        (I)mat->nnz,
                                        (const T*)alpha,
                                        mat->descr,
                                        (const T*)mat->val_data,
                                        (const I*)mat->col_data,
                                        (const J*)mat->row_data,
                                        (alg == rocsparse_spmv_alg_csr_stream) ? nullptr
                                                                               : mat->info,
                                        (const T*)x->values,
                                        (const T*)beta,
                                        (T*)y->values);
    }

    case rocsparse_format_ell:
    {
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer and we can't return a zero buffer size here
            // otherwise the allocated buffer will be again a null pointer.
            // That's a flaw we need to fix with the use of a rocsparse_spmv_stage enum.
            // We need to wait the next major release where the C-API is allowed to be modified.
            *buffer_size = 4;
            return rocsparse_status_success;
        }

        return rocsparse_ellmv_template(handle,
                                        trans,
                                        (I)mat->rows,
                                        (I)mat->cols,
                                        (const T*)alpha,
                                        mat->descr,
                                        (const T*)mat->val_data,
                                        (const I*)mat->col_data,
                                        (I)mat->ell_width,
                                        (const T*)x->values,
                                        (const T*)beta,
                                        (T*)y->values);
    }

    case rocsparse_format_bell:
    {
        // LCOV_EXCL_START
        return rocsparse_status_not_implemented;
        // LCOV_EXCL_STOP
    }

    case rocsparse_format_bsr:
    {
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer and we can't return a zero buffer size here
            // otherwise the allocated buffer will be again a null pointer.
            // That's a flaw we need to fix with the use of a rocsparse_spmv_stage enum.
            // We need to wait the next major release where the C-API is allowed to be modified.
            *buffer_size = 4;
            return rocsparse_status_success;
        }

        return rocsparse_bsrmv_template<T>(handle,
                                           mat->block_dir,
                                           trans,
                                           (J)mat->rows,
                                           (J)mat->cols,
                                           (I)mat->nnz,
                                           (const T*)alpha,
                                           mat->descr,
                                           (const T*)mat->val_data,
                                           (const I*)mat->row_data,
                                           (const J*)mat->col_data,
                                           (J)mat->block_dim,
                                           (const T*)x->values,
                                           (const T*)beta,
                                           (T*)y->values);
    }
    }

    return rocsparse_status_invalid_value;
}

template <typename... Ts>
rocsparse_status rocsparse_spmv_dynamic_dispatch(rocsparse_indextype itype,
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
                return rocsparse_spmv_template<int32_t, int32_t, TYPE>(ts...); \
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
                return rocsparse_spmv_template<int64_t, int32_t, TYPE>(ts...); \
            }                                                                  \
            case rocsparse_indextype_i64:                                      \
            {                                                                  \
                return rocsparse_spmv_template<int64_t, int64_t, TYPE>(ts...); \
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

    return rocsparse_spmv_dynamic_dispatch(determine_I_index_type(mat),
                                           determine_J_index_type(mat),
                                           compute_type,
                                           handle,
                                           trans,
                                           alpha,
                                           mat,
                                           x,
                                           beta,
                                           y,
                                           alg,
                                           buffer_size,
                                           temp_buffer);
}

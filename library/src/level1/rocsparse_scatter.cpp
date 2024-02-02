/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_scatter.h"
#include "control.h"
#include "handle.h"
#include "internal/level1/rocsparse_sctr.h"
#include "utility.h"

#include "rocsparse_sctr.hpp"

namespace rocsparse
{
    template <typename I, typename T>
    rocsparse_status scatter_template(rocsparse_handle            handle,
                                      rocsparse_const_spvec_descr x,
                                      rocsparse_dnvec_descr       y)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::sctr_template<I, T>)(handle,
                                                                   (I)x->nnz,
                                                                   (const T*)x->const_val_data,
                                                                   (const I*)x->const_idx_data,
                                                                   (T*)y->values,
                                                                   x->idx_base));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scatter(rocsparse_handle            handle,
                                              rocsparse_const_spvec_descr x,
                                              rocsparse_dnvec_descr       y)
try
{
    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    log_trace(handle, "rocsparse_scatter", (const void*&)x, (const void*&)y);

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(1, x);
    ROCSPARSE_CHECKARG_POINTER(2, y);

    // Check if descriptors are initialized
    ROCSPARSE_CHECKARG(1, x, x->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(2, y, y->init == false, rocsparse_status_not_initialized);

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(2, y, (y->data_type != x->data_type), rocsparse_status_not_implemented);

    // int8 ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scatter_template<int32_t, int8_t>)(handle, x, y));
        return rocsparse_status_success;
    }
    // single real ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scatter_template<int32_t, float>)(handle, x, y));
        return rocsparse_status_success;
    }
    // double real ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scatter_template<int32_t, double>)(handle, x, y));
        return rocsparse_status_success;
    }
    // single complex ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::scatter_template<int32_t, rocsparse_float_complex>)(handle, x, y));
        return rocsparse_status_success;
    }
    // double complex ; i32
    if(x->idx_type == rocsparse_indextype_i32 && x->data_type == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::scatter_template<int32_t, rocsparse_double_complex>)(handle, x, y));
        return rocsparse_status_success;
    }
    // int8 ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_i8_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scatter_template<int64_t, int8_t>)(handle, x, y));
        return rocsparse_status_success;
    }
    // single real ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f32_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scatter_template<int64_t, float>)(handle, x, y));
        return rocsparse_status_success;
    }
    // double real ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f64_r)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scatter_template<int64_t, double>)(handle, x, y));
        return rocsparse_status_success;
    }
    // single complex ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f32_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::scatter_template<int64_t, rocsparse_float_complex>)(handle, x, y));
        return rocsparse_status_success;
    }
    // double complex ; i64
    if(x->idx_type == rocsparse_indextype_i64 && x->data_type == rocsparse_datatype_f64_c)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::scatter_template<int64_t, rocsparse_double_complex>)(handle, x, y));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

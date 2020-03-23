/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include "rocsparse_csx2dense_impl.hpp"

extern "C" {

//
// Check if the macro CAPI_IMPL already exists.
//
#ifdef CAPI_IMPL
#error macro CAPI_IMPL is already defined.
#endif

//
// Definition of the C-implementation.
//
#define CAPI_IMPL(name_, type_)                                                 \
    rocsparse_status name_(rocsparse_handle          handle,                    \
                           rocsparse_int             m,                         \
                           rocsparse_int             n,                         \
                           const rocsparse_mat_descr descr,                     \
                           const type_*              csc_val,                   \
                           const rocsparse_int*      csc_col_ptr,               \
                           const rocsparse_int*      csc_row_ind,               \
                           type_*                    A,                         \
                           rocsparse_int             ld)                        \
    {                                                                           \
        try                                                                     \
        {                                                                       \
            return rocsparse_csx2dense_impl<rocsparse_direction_column, type_>( \
                handle, m, n, descr, csc_val, csc_col_ptr, csc_row_ind, A, ld); \
        }                                                                       \
        catch(...)                                                              \
        {                                                                       \
            return exception_to_rocsparse_status();                             \
        }                                                                       \
    }

//
// C-implementations.
//
CAPI_IMPL(rocsparse_scsc2dense, float);
CAPI_IMPL(rocsparse_dcsc2dense, double);
CAPI_IMPL(rocsparse_ccsc2dense, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zcsc2dense, rocsparse_double_complex);

//
// Undefine the macro CAPI_IMPL.
//
#undef CAPI_IMPL
}

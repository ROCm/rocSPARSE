/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include "rocsparse.hpp"

#include <complex>
#include <rocsparse.h>

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
// axpyi
template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const float*         alpha,
                                 const float*         x_val,
                                 const rocsparse_int* x_ind,
                                 float*               y,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_saxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const double*        alpha,
                                 const double*        x_val,
                                 const rocsparse_int* x_ind,
                                 double*              y,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_daxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle           handle,
                                 rocsparse_int              nnz,
                                 const std::complex<float>* alpha,
                                 const std::complex<float>* x_val,
                                 const rocsparse_int*       x_ind,
                                 std::complex<float>*       y,
                                 rocsparse_index_base       idx_base)
{
    return rocsparse_caxpyi(handle,
                            nnz,
                            (const rocsparse_float_complex*)alpha,
                            (const rocsparse_float_complex*)x_val,
                            x_ind,
                            (rocsparse_float_complex*)y,
                            idx_base);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle            handle,
                                 rocsparse_int               nnz,
                                 const std::complex<double>* alpha,
                                 const std::complex<double>* x_val,
                                 const rocsparse_int*        x_ind,
                                 std::complex<double>*       y,
                                 rocsparse_index_base        idx_base)
{
    return rocsparse_zaxpyi(handle,
                            nnz,
                            (const rocsparse_double_complex*)alpha,
                            (const rocsparse_double_complex*)x_val,
                            x_ind,
                            (rocsparse_double_complex*)y,
                            idx_base);
}

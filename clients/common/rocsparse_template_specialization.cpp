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

// doti
template <>
rocsparse_status rocsparse_doti(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const float*         x_val,
                                const rocsparse_int* x_ind,
                                const float*         y,
                                float*               result,
                                rocsparse_index_base idx_base)
{
    return rocsparse_sdoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
rocsparse_status rocsparse_doti(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const double*        x_val,
                                const rocsparse_int* x_ind,
                                const double*        y,
                                double*              result,
                                rocsparse_index_base idx_base)
{
    return rocsparse_ddoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
rocsparse_status rocsparse_doti(rocsparse_handle           handle,
                                rocsparse_int              nnz,
                                const std::complex<float>* x_val,
                                const rocsparse_int*       x_ind,
                                const std::complex<float>* y,
                                std::complex<float>*       result,
                                rocsparse_index_base       idx_base)
{
    return rocsparse_cdoti(handle,
                           nnz,
                           (const rocsparse_float_complex*)x_val,
                           x_ind,
                           (const rocsparse_float_complex*)y,
                           (rocsparse_float_complex*)result,
                           idx_base);
}

template <>
rocsparse_status rocsparse_doti(rocsparse_handle            handle,
                                rocsparse_int               nnz,
                                const std::complex<double>* x_val,
                                const rocsparse_int*        x_ind,
                                const std::complex<double>* y,
                                std::complex<double>*       result,
                                rocsparse_index_base        idx_base)
{
    return rocsparse_zdoti(handle,
                           nnz,
                           (const rocsparse_double_complex*)x_val,
                           x_ind,
                           (const rocsparse_double_complex*)y,
                           (rocsparse_double_complex*)result,
                           idx_base);
}

// dotci
template <>
rocsparse_status rocsparse_dotci(rocsparse_handle           handle,
                                 rocsparse_int              nnz,
                                 const std::complex<float>* x_val,
                                 const rocsparse_int*       x_ind,
                                 const std::complex<float>* y,
                                 std::complex<float>*       result,
                                 rocsparse_index_base       idx_base)
{
    return rocsparse_cdotci(handle,
                            nnz,
                            (const rocsparse_float_complex*)x_val,
                            x_ind,
                            (const rocsparse_float_complex*)y,
                            (rocsparse_float_complex*)result,
                            idx_base);
}

template <>
rocsparse_status rocsparse_dotci(rocsparse_handle            handle,
                                 rocsparse_int               nnz,
                                 const std::complex<double>* x_val,
                                 const rocsparse_int*        x_ind,
                                 const std::complex<double>* y,
                                 std::complex<double>*       result,
                                 rocsparse_index_base        idx_base)
{
    return rocsparse_zdotci(handle,
                            nnz,
                            (const rocsparse_double_complex*)x_val,
                            x_ind,
                            (const rocsparse_double_complex*)y,
                            (rocsparse_double_complex*)result,
                            idx_base);
}

// gthr
template <>
rocsparse_status rocsparse_gthr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const float*         y,
                                float*               x_val,
                                const rocsparse_int* x_ind,
                                rocsparse_index_base idx_base)
{
    return rocsparse_sgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const double*        y,
                                double*              x_val,
                                const rocsparse_int* x_ind,
                                rocsparse_index_base idx_base)
{
    return rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthr(rocsparse_handle           handle,
                                rocsparse_int              nnz,
                                const std::complex<float>* y,
                                std::complex<float>*       x_val,
                                const rocsparse_int*       x_ind,
                                rocsparse_index_base       idx_base)
{
    return rocsparse_cgthr(handle,
                           nnz,
                           (const rocsparse_float_complex*)y,
                           (rocsparse_float_complex*)x_val,
                           x_ind,
                           idx_base);
}

template <>
rocsparse_status rocsparse_gthr(rocsparse_handle            handle,
                                rocsparse_int               nnz,
                                const std::complex<double>* y,
                                std::complex<double>*       x_val,
                                const rocsparse_int*        x_ind,
                                rocsparse_index_base        idx_base)
{
    return rocsparse_zgthr(handle,
                           nnz,
                           (const rocsparse_double_complex*)y,
                           (rocsparse_double_complex*)x_val,
                           x_ind,
                           idx_base);
}

// gthrz
template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 float*               y,
                                 float*               x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_sgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 double*              y,
                                 double*              x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_dgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 std::complex<float>* y,
                                 std::complex<float>* x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_cgthrz(
        handle, nnz, (rocsparse_float_complex*)y, (rocsparse_float_complex*)x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle      handle,
                                 rocsparse_int         nnz,
                                 std::complex<double>* y,
                                 std::complex<double>* x_val,
                                 const rocsparse_int*  x_ind,
                                 rocsparse_index_base  idx_base)
{
    return rocsparse_zgthrz(handle,
                            nnz,
                            (rocsparse_double_complex*)y,
                            (rocsparse_double_complex*)x_val,
                            x_ind,
                            idx_base);
}

// sctr
template <>
rocsparse_status rocsparse_sctr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const float*         x_val,
                                const rocsparse_int* x_ind,
                                float*               y,
                                rocsparse_index_base idx_base)
{
    return rocsparse_ssctr(handle, nnz, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_sctr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const double*        x_val,
                                const rocsparse_int* x_ind,
                                double*              y,
                                rocsparse_index_base idx_base)
{
    return rocsparse_dsctr(handle, nnz, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_sctr(rocsparse_handle           handle,
                                rocsparse_int              nnz,
                                const std::complex<float>* x_val,
                                const rocsparse_int*       x_ind,
                                std::complex<float>*       y,
                                rocsparse_index_base       idx_base)
{
    return rocsparse_csctr(handle,
                           nnz,
                           (const rocsparse_float_complex*)x_val,
                           x_ind,
                           (rocsparse_float_complex*)y,
                           idx_base);
}

template <>
rocsparse_status rocsparse_sctr(rocsparse_handle            handle,
                                rocsparse_int               nnz,
                                const std::complex<double>* x_val,
                                const rocsparse_int*        x_ind,
                                std::complex<double>*       y,
                                rocsparse_index_base        idx_base)
{
    return rocsparse_zsctr(handle,
                           nnz,
                           (const rocsparse_double_complex*)x_val,
                           x_ind,
                           (rocsparse_double_complex*)y,
                           idx_base);
}

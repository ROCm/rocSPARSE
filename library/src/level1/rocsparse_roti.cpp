/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_roti.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sroti(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            float*               x_val,
                                            const rocsparse_int* x_ind,
                                            float*               y,
                                            const float*         c,
                                            const float*         s,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_roti_template(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

extern "C" rocsparse_status rocsparse_droti(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            double*              x_val,
                                            const rocsparse_int* x_ind,
                                            double*              y,
                                            const double*        c,
                                            const double*        s,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_roti_template(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

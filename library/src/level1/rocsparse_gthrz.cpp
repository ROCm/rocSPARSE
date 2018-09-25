/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "rocsparse.h"
#include "rocsparse_gthrz.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sgthrz(rocsparse_handle handle,
                                             rocsparse_int nnz,
                                             float* y,
                                             float* x_val,
                                             const rocsparse_int* x_ind,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_gthrz_template<float>(handle, nnz, y, x_val, x_ind, idx_base);
}

extern "C" rocsparse_status rocsparse_dgthrz(rocsparse_handle handle,
                                             rocsparse_int nnz,
                                             double* y,
                                             double* x_val,
                                             const rocsparse_int* x_ind,
                                             rocsparse_index_base idx_base)
{
    return rocsparse_gthrz_template<double>(handle, nnz, y, x_val, x_ind, idx_base);
}

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

#include "rocsparse_sctr.hpp"
#include "rocsparse.h"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_ssctr(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            const float*         x_val,
                                            const rocsparse_int* x_ind,
                                            float*               y,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_sctr_template<float>(handle, nnz, x_val, x_ind, y, idx_base);
}

extern "C" rocsparse_status rocsparse_dsctr(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            const double*        x_val,
                                            const rocsparse_int* x_ind,
                                            double*              y,
                                            rocsparse_index_base idx_base)
{
    return rocsparse_sctr_template<double>(handle, nnz, x_val, x_ind, y, idx_base);
}

/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "handle.h"

extern "C" rocsparse_status rocsparse_cscsort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* csc_col_ptr,
                                                          const rocsparse_int* csc_row_ind,
                                                          size_t*              buffer_size)
{
    return rocsparse_csrsort_buffer_size(handle, n, m, nnz, csc_col_ptr, csc_row_ind, buffer_size);
}

extern "C" rocsparse_status rocsparse_cscsort(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             n,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const rocsparse_int*      csc_col_ptr,
                                              rocsparse_int*            csc_row_ind,
                                              rocsparse_int*            perm,
                                              void*                     temp_buffer)
{
    return rocsparse_csrsort(handle, n, m, nnz, descr, csc_col_ptr, csc_row_ind, perm, temp_buffer);
}

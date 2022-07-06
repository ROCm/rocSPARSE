/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef ROCSPARSE_COOMV_AOS_HPP
#define ROCSPARSE_COOMV_AOS_HPP

#include "utility.h"

typedef enum rocsparse_coomv_aos_alg_
{
    rocsparse_coomv_aos_alg_default = 0,
    rocsparse_coomv_aos_alg_segmented,
    rocsparse_coomv_aos_alg_atomic
} rocsparse_coomv_aos_alg;

template <typename I, typename T>
rocsparse_status rocsparse_coomv_aos_template(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              rocsparse_coomv_aos_alg   alg,
                                              I                         m,
                                              I                         n,
                                              I                         nnz,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const T*                  coo_val,
                                              const I*                  coo_ind,
                                              const T*                  x,
                                              const T*                  beta_device_host,
                                              T*                        y);

#endif // ROCSPARSE_COOMV_AOS_HPP

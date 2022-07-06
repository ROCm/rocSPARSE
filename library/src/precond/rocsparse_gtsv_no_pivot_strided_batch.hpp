/*! \file */
/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_GTSV_NO_PIVOT_STRIDED_BATCH_HPP
#define ROCSPARSE_GTSV_NO_PIVOT_STRIDED_BATCH_HPP

#include "definitions.h"
#include "utility.h"

template <typename T>
rocsparse_status
    rocsparse_gtsv_no_pivot_strided_batch_buffer_size_template(rocsparse_handle handle,
                                                               rocsparse_int    m,
                                                               const T*         dl,
                                                               const T*         d,
                                                               const T*         du,
                                                               const T*         x,
                                                               rocsparse_int    batch_count,
                                                               rocsparse_int    batch_stride,
                                                               size_t*          buffer_size);

template <typename T>
rocsparse_status rocsparse_gtsv_no_pivot_strided_batch_template(rocsparse_handle handle,
                                                                rocsparse_int    m,
                                                                const T*         dl,
                                                                const T*         d,
                                                                const T*         du,
                                                                T*               x,
                                                                rocsparse_int    batch_count,
                                                                rocsparse_int    batch_stride,
                                                                void*            temp_buffer);

#endif // ROCSPARSE_GTSV_NO_PIVOT_STRIDED_BATCH_HPP

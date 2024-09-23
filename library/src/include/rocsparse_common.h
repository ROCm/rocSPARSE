/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-types.h"

namespace rocsparse
{
    template <typename I, typename T, typename U>
    rocsparse_status dense_transpose(rocsparse_handle handle,
                                     I                m,
                                     I                n,
                                     U                alpha_device_host,
                                     const T*         A,
                                     int64_t          lda,
                                     T*               B,
                                     int64_t          ldb);

    template <typename I, typename T>
    rocsparse_status dense_transpose_back(
        rocsparse_handle handle, I m, I n, const T* A, int64_t lda, T* B, int64_t ldb);

    template <typename I, typename T>
    rocsparse_status conjugate(rocsparse_handle handle, I length, T* array);

    template <typename I, typename T>
    rocsparse_status valset(rocsparse_handle handle, I length, T value, T* array);

    template <typename I, typename T>
    rocsparse_status valset_2d(
        rocsparse_handle handle, I m, I n, int64_t ld, T value, T* array, rocsparse_order order);

    template <typename I, typename T, typename U>
    rocsparse_status scale_array(rocsparse_handle handle, I length, U scalar_device_host, T* array);

    template <typename I, typename T, typename U>
    rocsparse_status scale_2d_array(rocsparse_handle handle,
                                    I                m,
                                    I                n,
                                    int64_t          ld,
                                    int64_t          batch_count,
                                    int64_t          stride,
                                    U                scalar_device_host,
                                    T*               array,
                                    rocsparse_order  order);
}

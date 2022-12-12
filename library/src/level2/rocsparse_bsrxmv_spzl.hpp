/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "definitions.h"

#include "utility.h"

#include "common.h"

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_2x2(rocsparse_handle     handle,
                 rocsparse_direction  dir,
                 J                    mb,
                 I                    nnzb,
                 U                    alpha_device_host,
                 J                    size_of_mask,
                 const J*             bsr_mask_ptr,
                 const I*             bsr_row_ptr,
                 const I*             bsr_end_ptr,
                 const J*             bsr_col_ind,
                 const A*             bsr_val,
                 const X*             x,
                 U                    beta_device_host,
                 Y*                   y,
                 rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_3x3(rocsparse_handle     handle,
                 rocsparse_direction  dir,
                 J                    mb,
                 I                    nnzb,
                 U                    alpha_device_host,
                 J                    size_of_mask,
                 const J*             bsr_mask_ptr,
                 const I*             bsr_row_ptr,
                 const I*             bsr_end_ptr,
                 const J*             bsr_col_ind,
                 const A*             bsr_val,
                 const X*             x,
                 U                    beta_device_host,
                 Y*                   y,
                 rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_4x4(rocsparse_handle     handle,
                 rocsparse_direction  dir,
                 J                    mb,
                 I                    nnzb,
                 U                    alpha_device_host,
                 J                    size_of_mask,
                 const J*             bsr_mask_ptr,
                 const I*             bsr_row_ptr,
                 const I*             bsr_end_ptr,
                 const J*             bsr_col_ind,
                 const A*             bsr_val,
                 const X*             x,
                 U                    beta_device_host,
                 Y*                   y,
                 rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_5x5(rocsparse_handle     handle,
                 rocsparse_direction  dir,
                 J                    mb,
                 I                    nnzb,
                 U                    alpha_device_host,
                 J                    size_of_mask,
                 const J*             bsr_mask_ptr,
                 const I*             bsr_row_ptr,
                 const I*             bsr_end_ptr,
                 const J*             bsr_col_ind,
                 const A*             bsr_val,
                 const X*             x,
                 U                    beta_device_host,
                 Y*                   y,
                 rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_8x8(rocsparse_handle     handle,
                 rocsparse_direction  dir,
                 J                    mb,
                 I                    nnzb,
                 U                    alpha_device_host,
                 J                    size_of_mask,
                 const J*             bsr_mask_ptr,
                 const I*             bsr_row_ptr,
                 const I*             bsr_end_ptr,
                 const J*             bsr_col_ind,
                 const A*             bsr_val,
                 const X*             x,
                 U                    beta_device_host,
                 Y*                   y,
                 rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_16x16(rocsparse_handle     handle,
                   rocsparse_direction  dir,
                   J                    mb,
                   I                    nnzb,
                   U                    alpha_device_host,
                   J                    size_of_mask,
                   const J*             bsr_mask_ptr,
                   const I*             bsr_row_ptr,
                   const I*             bsr_end_ptr,
                   const J*             bsr_col_ind,
                   const A*             bsr_val,
                   const X*             x,
                   U                    beta_device_host,
                   Y*                   y,
                   rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_17_32(rocsparse_handle     handle,
                   rocsparse_direction  dir,
                   J                    mb,
                   I                    nnzb,
                   U                    alpha_device_host,
                   J                    size_of_mask,
                   const J*             bsr_mask_ptr,
                   const I*             bsr_row_ptr,
                   const I*             bsr_end_ptr,
                   const J*             bsr_col_ind,
                   const A*             bsr_val,
                   J                    bsr_dim,
                   const X*             x,
                   U                    beta_device_host,
                   Y*                   y,
                   rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y, typename U>
void bsrxmvn_general(rocsparse_handle     handle,
                     rocsparse_direction  dir,
                     J                    mb,
                     U                    alpha_device_host,
                     J                    size_of_mask,
                     const J*             bsr_mask_ptr,
                     const I*             bsr_row_ptr,
                     const I*             bsr_end_ptr,
                     const J*             bsr_col_ind,
                     const A*             bsr_val,
                     J                    bsr_dim,
                     const X*             x,
                     U                    beta_device_host,
                     Y*                   y,
                     rocsparse_index_base base);

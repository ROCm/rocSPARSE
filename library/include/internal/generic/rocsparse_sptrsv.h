/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_SPTRSV_H
#define ROCSPARSE_SPTRSV_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv_buffer_size(rocsparse_handle            handle,
                                              rocsparse_sptrsv_descr      sptrsv_descr,
                                              rocsparse_const_spmat_descr spmat_descr,
                                              rocsparse_sptrsv_stage      sptrsv_stage,
                                              size_t*                     buffer_size_in_bytes);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_sptrsv(rocsparse_handle            handle,
                                  rocsparse_sptrsv_descr      sptrsv_descr,
                                  const void*                 alpha,
                                  rocsparse_const_spmat_descr spmat_descr,
                                  rocsparse_const_dnvec_descr dnvec_descr_x,
                                  const rocsparse_dnvec_descr dnvec_descr_y,
                                  rocsparse_sptrsv_stage      sptrsv_stage,
                                  size_t                      buffer_size_in_bytes,
                                  void*                       buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPSV_H */

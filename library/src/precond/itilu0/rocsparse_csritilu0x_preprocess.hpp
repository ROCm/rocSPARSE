/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CSRITILU0X_PREPROCESS_HPP
#define ROCSPARSE_CSRITILU0X_PREPROCESS_HPP

#include "definitions.h"
#include "utility.h"

template <typename I, typename J>
rocsparse_status rocsparse_csritilu0x_preprocess_template(rocsparse_handle     handle_,
                                                          rocsparse_itilu0_alg alg_,
                                                          J                    options_,
                                                          J                    nsweeps_,
                                                          J                    m_,
                                                          I                    nnz_,
                                                          const I* __restrict__ ptr_begin_,
                                                          const I* __restrict__ ptr_end_,
                                                          const J* __restrict__ ind_,
                                                          rocsparse_index_base base_,
                                                          rocsparse_diag_type  ldiag_type_,
                                                          rocsparse_direction  ldir_,
                                                          I                    lnnz_,
                                                          const I* __restrict__ lptr_begin_,
                                                          const I* __restrict__ lptr_end_,
                                                          const J* __restrict__ lind_,
                                                          rocsparse_index_base lbase_,
                                                          rocsparse_diag_type  udiag_type_,
                                                          rocsparse_direction  udir_,
                                                          I                    unnz_,
                                                          const I* __restrict__ uptr_begin_,
                                                          const I* __restrict__ uptr_end_,
                                                          const J* __restrict__ uind_,

                                                          rocsparse_index_base ubase_,
                                                          rocsparse_datatype   datatype_,
                                                          size_t               buffer_size_,
                                                          void* __restrict__ buffer_);

template <typename I, typename J>
rocsparse_status rocsparse_csritilu0x_preprocess_impl(rocsparse_handle     handle_,
                                                      rocsparse_itilu0_alg alg_,
                                                      J                    options_,
                                                      J                    nsweeps_,
                                                      J                    m_,
                                                      I                    nnz_,
                                                      const I* __restrict__ ptr_begin_,
                                                      const I* __restrict__ ptr_end_,
                                                      const J* __restrict__ ind_,
                                                      rocsparse_index_base base_,
                                                      rocsparse_diag_type  ldiag_type_,
                                                      rocsparse_direction  ldir_,
                                                      I                    lnnz_,
                                                      const I* __restrict__ lptr_begin_,
                                                      const I* __restrict__ lptr_end_,
                                                      const J* __restrict__ lind_,
                                                      rocsparse_index_base lbase_,
                                                      rocsparse_diag_type  udiag_type_,
                                                      rocsparse_direction  udir_,
                                                      I                    unnz_,
                                                      const I* __restrict__ uptr_begin_,
                                                      const I* __restrict__ uptr_end_,
                                                      const J* __restrict__ uind_,

                                                      rocsparse_index_base ubase_,
                                                      rocsparse_datatype   datatype_,
                                                      size_t               buffer_size_,
                                                      void* __restrict__ buffer_);

#endif // ROCSPARSE_CSRITILU0X_PREPROCESS_HPP

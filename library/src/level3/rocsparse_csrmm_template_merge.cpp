/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

#include "utility.h"

#include "rocsparse_coomm.hpp"

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse_csrmm_template_merge(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_order           order,
                                                J                         m,
                                                J                         n,
                                                J                         k,
                                                I                         nnz,
                                                U                         alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const T*                  B,
                                                J                         ldb,
                                                U                         beta_device_host,
                                                T*                        C,
                                                J                         ldc,
                                                void*                     temp_buffer)
{
    // Temporary buffer entry points
    char* ptr         = reinterpret_cast<char*>(temp_buffer);
    J*    csr_row_ind = reinterpret_cast<J*>(ptr);
    // ptr += sizeof(J) * ((nnz - 1) / 256 + 1) * 256;

    return rocsparse_coomm_template_dispatch(handle,
                                             trans_A,
                                             trans_B,
                                             order,
                                             rocsparse_coomm_alg_segmented,
                                             m,
                                             n,
                                             k,
                                             (J)nnz,
                                             (J)1,
                                             (J)0,
                                             alpha_device_host,
                                             descr,
                                             csr_val,
                                             csr_row_ind,
                                             csr_col_ind,
                                             B,
                                             ldb,
                                             (J)1,
                                             (J)0,
                                             beta_device_host,
                                             C,
                                             ldc,
                                             (J)1,
                                             (J)0);
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE, UTYPE)                                                     \
    template rocsparse_status rocsparse_csrmm_template_merge(rocsparse_handle    handle,            \
                                                             rocsparse_operation trans_A,           \
                                                             rocsparse_operation trans_B,           \
                                                             rocsparse_order     order,             \
                                                             JTYPE               m,                 \
                                                             JTYPE               n,                 \
                                                             JTYPE               k,                 \
                                                             ITYPE               nnz,               \
                                                             UTYPE               alpha_device_host, \
                                                             const rocsparse_mat_descr descr,       \
                                                             const TTYPE*              csr_val,     \
                                                             const ITYPE*              csr_row_ptr, \
                                                             const JTYPE*              csr_col_ind, \
                                                             const TTYPE*              B,           \
                                                             JTYPE                     ldb,         \
                                                             UTYPE  beta_device_host,               \
                                                             TTYPE* C,                              \
                                                             JTYPE  ldc,                            \
                                                             void*  temp_buffer)

INSTANTIATE(int32_t, int32_t, float, float);
INSTANTIATE(int32_t, int32_t, double, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float, float);
INSTANTIATE(int64_t, int32_t, double, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float, float);
INSTANTIATE(int64_t, int64_t, double, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int32_t, int32_t, float, const float*);
INSTANTIATE(int32_t, int32_t, double, const double*);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex, const rocsparse_double_complex*);
INSTANTIATE(int64_t, int32_t, float, const float*);
INSTANTIATE(int64_t, int32_t, double, const double*);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex, const rocsparse_double_complex*);
INSTANTIATE(int64_t, int64_t, float, const float*);
INSTANTIATE(int64_t, int64_t, double, const double*);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex, const rocsparse_double_complex*);
#undef INSTANTIATE

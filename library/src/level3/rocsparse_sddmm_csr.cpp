/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_sddmm_csx_kernel.hpp"

template <typename I, typename J, typename T>
struct rocsparse_sddmm_st<rocsparse_format_csr, rocsparse_sddmm_alg_default, I, J, T>
{
    static rocsparse_status buffer_size(rocsparse_handle     handle,
                                        rocsparse_operation  trans_A,
                                        rocsparse_operation  trans_B,
                                        rocsparse_order      order_A,
                                        rocsparse_order      order_B,
                                        J                    m,
                                        J                    n,
                                        J                    k,
                                        I                    nnz,
                                        const T*             alpha,
                                        const T*             A_val,
                                        int64_t              A_ld,
                                        const T*             B_val,
                                        int64_t              B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        T*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_sddmm_alg  alg,
                                        size_t*              buffer_size)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    static rocsparse_status preprocess(rocsparse_handle     handle,
                                       rocsparse_operation  trans_A,
                                       rocsparse_operation  trans_B,
                                       rocsparse_order      order_A,
                                       rocsparse_order      order_B,
                                       J                    m,
                                       J                    n,
                                       J                    k,
                                       I                    nnz,
                                       const T*             alpha,
                                       const T*             A_val,
                                       int64_t              A_ld,
                                       const T*             B_val,
                                       int64_t              B_ld,
                                       const T*             beta,
                                       const I*             C_row_data,
                                       const J*             C_col_data,
                                       T*                   C_val_data,
                                       rocsparse_index_base C_base,
                                       rocsparse_sddmm_alg  alg,
                                       void*                buffer)
    {
        return rocsparse_status_success;
    }

    static rocsparse_status compute(rocsparse_handle     handle,
                                    rocsparse_operation  trans_A,
                                    rocsparse_operation  trans_B,
                                    rocsparse_order      order_A,
                                    rocsparse_order      order_B,
                                    J                    m,
                                    J                    n,
                                    J                    k,
                                    I                    nnz,
                                    const T*             alpha,
                                    const T*             A_val,
                                    int64_t              A_ld,
                                    const T*             B_val,
                                    int64_t              B_ld,
                                    const T*             beta,
                                    const I*             C_row_data,
                                    const J*             C_col_data,
                                    T*                   C_val_data,
                                    rocsparse_index_base C_base,
                                    rocsparse_sddmm_alg  alg,
                                    void*                buffer)
    {
        static constexpr int NB = 512;

#define HLAUNCH(NT_)                                                   \
    int64_t num_blocks_x = (m - 1) / (NB / NT_) + 1;                   \
    dim3    blocks(num_blocks_x);                                      \
    dim3    threads(NB);                                               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                \
        (sddmm_csx_kernel<NB, NT_, rocsparse_direction_row, I, J, T>), \
        blocks,                                                        \
        threads,                                                       \
        0,                                                             \
        handle->stream,                                                \
        trans_A,                                                       \
        trans_B,                                                       \
        order_A,                                                       \
        order_B,                                                       \
        m,                                                             \
        n,                                                             \
        k,                                                             \
        nnz,                                                           \
        *(const T*)alpha,                                              \
        A_val,                                                         \
        A_ld,                                                          \
        B_val,                                                         \
        B_ld,                                                          \
        *(const T*)beta,                                               \
        (T*)C_val_data,                                                \
        (const I*)C_row_data,                                          \
        (const J*)C_col_data,                                          \
        C_base,                                                        \
        (T*)buffer)

#define DLAUNCH(NT_)                                                   \
    int64_t num_blocks_x = (m - 1) / (NB / NT_) + 1;                   \
    dim3    blocks(num_blocks_x);                                      \
    dim3    threads(NB);                                               \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                \
        (sddmm_csx_kernel<NB, NT_, rocsparse_direction_row, I, J, T>), \
        blocks,                                                        \
        threads,                                                       \
        0,                                                             \
        handle->stream,                                                \
        trans_A,                                                       \
        trans_B,                                                       \
        order_A,                                                       \
        order_B,                                                       \
        m,                                                             \
        n,                                                             \
        k,                                                             \
        nnz,                                                           \
        alpha,                                                         \
        A_val,                                                         \
        A_ld,                                                          \
        B_val,                                                         \
        B_ld,                                                          \
        beta,                                                          \
        (T*)C_val_data,                                                \
        (const I*)C_row_data,                                          \
        (const J*)C_col_data,                                          \
        C_base,                                                        \
        (T*)buffer)

        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }
            if(k > 4)
            {
                HLAUNCH(8);
            }
            else if(k > 2)
            {
                HLAUNCH(4);
            }
            else if(k > 1)
            {
                HLAUNCH(2);
            }
            else
            {
                HLAUNCH(1);
            }
        }
        else
        {
            if(k > 4)
            {
                DLAUNCH(8);
            }
            else if(k > 2)
            {
                DLAUNCH(4);
            }
            else if(k > 1)
            {
                DLAUNCH(2);
            }
            else
            {
                DLAUNCH(1);
            }
        }
        return rocsparse_status_success;
    }
};

template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int32_t,
                                   int32_t,
                                   rocsparse_double_complex>;

template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int32_t,
                                   rocsparse_double_complex>;

template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   float>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   double>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_float_complex>;
template struct rocsparse_sddmm_st<rocsparse_format_csr,
                                   rocsparse_sddmm_alg_default,
                                   int64_t,
                                   int64_t,
                                   rocsparse_double_complex>;

/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../conversion/rocsparse_identity.hpp"
#include "definitions.h"
#include "rocsparse_csrgemm.hpp"
#include "utility.h"

#include <rocprim/rocprim.hpp>

template <typename I, typename J>
rocsparse_status rocsparse_bsrgemm_nnzb_template(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 J                         mb,
                                                 J                         nb,
                                                 J                         kb,
                                                 J                         block_dim,
                                                 const rocsparse_mat_descr descr_A,
                                                 I                         nnzb_A,
                                                 const I*                  bsr_row_ptr_A,
                                                 const J*                  bsr_col_ind_A,
                                                 const rocsparse_mat_descr descr_B,
                                                 I                         nnzb_B,
                                                 const I*                  bsr_row_ptr_B,
                                                 const J*                  bsr_col_ind_B,
                                                 const rocsparse_mat_descr descr_D,
                                                 I                         nnzb_D,
                                                 const I*                  bsr_row_ptr_D,
                                                 const J*                  bsr_col_ind_D,
                                                 const rocsparse_mat_descr descr_C,
                                                 I*                        bsr_row_ptr_C,
                                                 I*                        nnzb_C,
                                                 const rocsparse_mat_info  info_C,
                                                 void*                     temp_buffer)
{

    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              "rocsparse_bsrgemm_nnzb",
              dir,
              trans_A,
              trans_B,
              mb,
              nb,
              kb,
              block_dim,
              (const void*&)descr_A,
              nnzb_A,
              (const void*&)bsr_row_ptr_A,
              (const void*&)bsr_col_ind_A,
              (const void*&)descr_B,
              nnzb_B,
              (const void*&)bsr_row_ptr_B,
              (const void*&)bsr_col_ind_B,
              (const void*&)descr_D,
              nnzb_D,
              (const void*&)bsr_row_ptr_D,
              (const void*&)bsr_col_ind_D,
              (const void*&)descr_C,
              (const void*&)bsr_row_ptr_C,
              (const void*&)nnzb_C,
              (const void*&)info_C,
              (const void*&)temp_buffer);

    // Check direction
    if(rocsparse_enum_utils::is_invalid(dir))
    {
        return rocsparse_status_invalid_value;
    }

    // Check operation
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    // Check valid sizes
    if(mb < 0 || nb < 0 || kb < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid rocsparse_csrgemm_info
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnzb_A > 0 && temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_csrgemm_nnz_template(handle,
                                          trans_A,
                                          trans_B,
                                          mb,
                                          nb,
                                          kb,
                                          descr_A,
                                          nnzb_A,
                                          bsr_row_ptr_A,
                                          bsr_col_ind_A,
                                          descr_B,
                                          nnzb_B,
                                          bsr_row_ptr_B,
                                          bsr_col_ind_B,
                                          descr_D,
                                          nnzb_D,
                                          bsr_row_ptr_D,
                                          bsr_col_ind_D,
                                          descr_C,
                                          bsr_row_ptr_C,
                                          nnzb_C,
                                          info_C,
                                          temp_buffer);
}

#define INSTANTIATE(ITYPE, JTYPE)                                            \
    template rocsparse_status rocsparse_bsrgemm_nnzb_template<ITYPE, JTYPE>( \
        rocsparse_handle          handle,                                    \
        rocsparse_direction       dir,                                       \
        rocsparse_operation       trans_A,                                   \
        rocsparse_operation       trans_B,                                   \
        JTYPE                     mb,                                        \
        JTYPE                     nb,                                        \
        JTYPE                     kb,                                        \
        JTYPE                     block_dim,                                 \
        const rocsparse_mat_descr descr_A,                                   \
        ITYPE                     nnzb_A,                                    \
        const ITYPE*              bsr_row_ptr_A,                             \
        const JTYPE*              bsr_col_ind_A,                             \
        const rocsparse_mat_descr descr_B,                                   \
        ITYPE                     nnzb_B,                                    \
        const ITYPE*              bsr_row_ptr_B,                             \
        const JTYPE*              bsr_col_ind_B,                             \
        const rocsparse_mat_descr descr_D,                                   \
        ITYPE                     nnzb_D,                                    \
        const ITYPE*              bsr_row_ptr_D,                             \
        const JTYPE*              bsr_col_ind_D,                             \
        const rocsparse_mat_descr descr_C,                                   \
        ITYPE*                    bsr_row_ptr_C,                             \
        ITYPE*                    nnzb_C,                                    \
        const rocsparse_mat_info  info_C,                                    \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

//
// rocsparse_xbsrgemm_nnz
//
extern "C" rocsparse_status rocsparse_bsrgemm_nnzb(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             kb,
                                                   rocsparse_int             block_dim,
                                                   const rocsparse_mat_descr descr_A,
                                                   rocsparse_int             nnzb_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   rocsparse_int             nnzb_B,
                                                   const rocsparse_int*      bsr_row_ptr_B,
                                                   const rocsparse_int*      bsr_col_ind_B,
                                                   const rocsparse_mat_descr descr_D,
                                                   rocsparse_int             nnzb_D,
                                                   const rocsparse_int*      bsr_row_ptr_D,
                                                   const rocsparse_int*      bsr_col_ind_D,
                                                   const rocsparse_mat_descr descr_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            nnzb_C,
                                                   const rocsparse_mat_info  info_C,
                                                   void*                     temp_buffer)
try
{
    return rocsparse_bsrgemm_nnzb_template(handle,
                                           dir,
                                           trans_A,
                                           trans_B,
                                           mb,
                                           nb,
                                           kb,
                                           block_dim,
                                           descr_A,
                                           nnzb_A,
                                           bsr_row_ptr_A,
                                           bsr_col_ind_A,
                                           descr_B,
                                           nnzb_B,
                                           bsr_row_ptr_B,
                                           bsr_col_ind_B,
                                           descr_D,
                                           nnzb_D,
                                           bsr_row_ptr_D,
                                           bsr_col_ind_D,
                                           descr_C,
                                           bsr_row_ptr_C,
                                           nnzb_C,
                                           info_C,
                                           temp_buffer);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

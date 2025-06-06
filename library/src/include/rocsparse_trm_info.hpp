/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_mat_descr.hpp"

typedef struct _rocsparse_trm_info
{
    // maximum non-zero entries per row
    int64_t max_nnz{};

    // device array to hold row permutation
    void* row_map{};
    // device array to hold pointer to diagonal entry
    void* trm_diag_ind{};
    // device pointers to hold transposed data
    void* trmt_perm{};
    void* trmt_row_ptr{};
    void* trmt_col_ind{};

    // some data to verify correct execution
    int64_t                     m{};
    int64_t                     nnz{};
    const _rocsparse_mat_descr* descr{};
    const void*                 trm_row_ptr{};
    const void*                 trm_col_ind{};

    rocsparse_indextype index_type_I = rocsparse_indextype_u16;
    rocsparse_indextype index_type_J = rocsparse_indextype_u16;

} * rocsparse_trm_info;

namespace rocsparse
{
    /********************************************************************************
 * \brief rocsparse_trm_info is a structure holding the rocsparse bsrsv, csrsv,
 * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
 * csrilu0_analysis and csric0_analysis. It must be initialized using the
 * create_trm_info() routine. It should be destroyed at the end
 * using destroy_trm_info().
 *******************************************************************************/
    rocsparse_status create_trm_info(rocsparse_trm_info* info);

    /********************************************************************************
 * \brief Copy trm info.
 *******************************************************************************/
    rocsparse_status copy_trm_info(rocsparse_trm_info dest, const rocsparse_trm_info src);

    /********************************************************************************
 * \brief Destroy trm info.
 *******************************************************************************/
    rocsparse_status destroy_trm_info(rocsparse_trm_info info);

}

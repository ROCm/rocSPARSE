/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_gcoo_aos2coo.hpp"
#include "rocsparse_convert_array.hpp"
#include "utility.h"

rocsparse_status rocsparse_gcoo_aos2coo(rocsparse_handle     handle,
                                        int64_t              m,
                                        int64_t              nnz,
                                        rocsparse_indextype  source_ind_type,
                                        const void*          source_ind_data,
                                        rocsparse_datatype   source_data_type,
                                        const void*          source_val_data,
                                        rocsparse_index_base source_idx_base,
                                        rocsparse_indextype  target_row_type,
                                        void*                target_row_data,
                                        rocsparse_indextype  target_col_type,
                                        void*                target_col_data,
                                        rocsparse_datatype   target_data_type,
                                        void*                target_val_data,
                                        rocsparse_index_base target_idx_base)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_convert_array(
        handle, nnz, target_row_type, target_row_data, 1, source_ind_type, source_ind_data, 2));

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_convert_array(handle,
                                nnz,
                                target_col_type,
                                target_col_data,
                                1,
                                source_ind_type,
                                reinterpret_cast<const char*>(source_ind_data)
                                    + rocsparse_indextype_sizeof(source_ind_type),
                                2));

    if(source_val_data != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_convert_array(
            handle, nnz, target_data_type, target_val_data, source_data_type, source_val_data));
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse_spmat_coo_aos2coo_buffer_size(rocsparse_handle            handle,
                                                         const rocsparse_spmat_descr source_,
                                                         rocsparse_spmat_descr       target_,
                                                         size_t*                     buffer_size_)

{
    buffer_size_[0] = 0;
    return rocsparse_status_success;
}

rocsparse_status rocsparse_spmat_coo_aos2coo(rocsparse_handle            handle,
                                             const rocsparse_spmat_descr source_,
                                             rocsparse_spmat_descr       target_,
                                             size_t                      buffer_size_,
                                             void*                       buffer_)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_gcoo_aos2coo(handle,
                                                     source_->rows,
                                                     source_->nnz,
                                                     source_->row_type,
                                                     source_->ind_data,
                                                     source_->data_type,
                                                     source_->val_data,
                                                     source_->idx_base,
                                                     target_->row_type,
                                                     target_->row_data,
                                                     target_->col_type,
                                                     target_->col_data,
                                                     target_->data_type,
                                                     target_->val_data,
                                                     target_->idx_base));

    return rocsparse_status_success;
}

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

#include "rocsparse_gcsr2coo.hpp"
#include "definitions.h"
#include "rocsparse_convert_array.hpp"
#include "rocsparse_csr2coo.hpp"

rocsparse_status rocsparse_gcsr2coo(rocsparse_handle     handle_,
                                    rocsparse_indextype  source_row_type_,
                                    const void*          source_row_,
                                    int64_t              nnz_,
                                    int64_t              m_,
                                    rocsparse_indextype  target_row_type_,
                                    void*                target_row_,
                                    rocsparse_index_base idx_base_)
{

#define DO(SROW, TROW)                                                         \
    do                                                                         \
    {                                                                          \
        RETURN_IF_ROCSPARSE_ERROR(                                             \
            (rocsparse_csr2coo_template<SROW, TROW>)(handle_,                  \
                                                     (const SROW*)source_row_, \
                                                     nnz_,                     \
                                                     m_,                       \
                                                     (TROW*)target_row_,       \
                                                     idx_base_));              \
        return rocsparse_status_success;                                       \
    } while(false)

    switch(source_row_type_)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_indextype_i32:
    {
        switch(target_row_type_)
        {
        case rocsparse_indextype_u16:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        case rocsparse_indextype_i32:
            DO(int32_t, int32_t);
        case rocsparse_indextype_i64:
            DO(int32_t, int64_t);
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    case rocsparse_indextype_i64:
    {
        switch(target_row_type_)
        {
        case rocsparse_indextype_u16:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        case rocsparse_indextype_i32:
            DO(int64_t, int32_t);
        case rocsparse_indextype_i64:
            DO(int64_t, int64_t);
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
    }
#undef DO
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

rocsparse_status rocsparse_spmat_csr2coo_buffer_size(rocsparse_handle            handle,
                                                     const rocsparse_spmat_descr source_,
                                                     rocsparse_spmat_descr       target_,
                                                     size_t*                     buffer_size_)
{
    buffer_size_[0] = 0;
    return rocsparse_status_success;
}

rocsparse_status rocsparse_spmat_csr2coo(rocsparse_handle            handle,
                                         const rocsparse_spmat_descr source_,
                                         rocsparse_spmat_descr       target_,
                                         size_t                      buffer_size_,
                                         void*                       buffer_)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_gcsr2coo(handle,
                                                 source_->row_type,
                                                 source_->row_data,
                                                 source_->nnz,
                                                 source_->rows,
                                                 target_->row_type,
                                                 target_->row_data,
                                                 source_->idx_base));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_convert_array(handle,
                                                      source_->nnz,
                                                      target_->col_type,
                                                      target_->col_data,
                                                      source_->col_type,
                                                      source_->col_data));
    if(source_->val_data != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_convert_array(handle,
                                                          source_->nnz,
                                                          target_->data_type,
                                                          target_->val_data,
                                                          source_->data_type,
                                                          source_->val_data));
    }
#if 0
	if (source_->descr->storage_mode != rocsparse_storage_mode_sorted &&
	    target_->descr->storage_mode == rocsparse_storage_mode_sorted)
	  {
	    std::cout << "Sort with internal memory allocation..." << std::endl;
	    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_sort_buffer_size(handle,
								       rocsparse_direction_row,
								       target_,
								       buffer_size_));
	    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_sort(handle,
								       rocsparse_direction_row,
								       target_,
								       buffer_));
	  }
#endif
    return rocsparse_status_success;
}

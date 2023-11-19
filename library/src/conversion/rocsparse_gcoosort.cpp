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

#include "rocsparse_gcoosort.hpp"
#include "definitions.h"
#include "rocsparse_coosort.hpp"

rocsparse_status rocsparse_gcoosort_buffer_size(rocsparse_handle    handle_,
                                                int64_t             m,
                                                int64_t             n,
                                                int64_t             nnz,
                                                rocsparse_indextype idx_type,
                                                const void*         row_data,
                                                const void*         col_data,
                                                size_t*             buffer_size)
{
#define CALL_TEMPLATE(IDX_TYPE)                                                 \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_coosort_buffer_size_template<IDX_TYPE>( \
        handle_, m, n, nnz, (const IDX_TYPE*)row_data, (const IDX_TYPE*)col_data, buffer_size))
    switch(idx_type)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_indextype_i32:
    {
        CALL_TEMPLATE(int32_t);
        return rocsparse_status_success;
    }
    case rocsparse_indextype_i64:
    {
        CALL_TEMPLATE(int64_t);
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);

#undef CALL_TEMPLATE
}

rocsparse_status rocsparse_gcoosort_by_row(rocsparse_handle    handle_,
                                           int64_t             m,
                                           int64_t             n,
                                           int64_t             nnz,
                                           rocsparse_indextype idx_type,
                                           void*               row_data,
                                           void*               col_data,
                                           void*               perm,
                                           void*               buffer)
{
#define CALL_TEMPLATE(IDX_TYPE)                                            \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_coosort_by_row_template<IDX_TYPE>( \
        handle_, m, n, nnz, (IDX_TYPE*)row_data, (IDX_TYPE*)col_data, (IDX_TYPE*)perm, buffer))
    switch(idx_type)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_indextype_i32:
    {
        CALL_TEMPLATE(int32_t);
        return rocsparse_status_success;
    }
    case rocsparse_indextype_i64:
    {
        CALL_TEMPLATE(int64_t);
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);

#undef CALL_TEMPLATE
}

rocsparse_status rocsparse_gcoosort_by_column(rocsparse_handle    handle_,
                                              int64_t             m,
                                              int64_t             n,
                                              int64_t             nnz,
                                              rocsparse_indextype idx_type,
                                              void*               row_data,
                                              void*               col_data,
                                              void*               perm,
                                              void*               buffer)
{
#define CALL_TEMPLATE(IDX_TYPE)                                               \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_coosort_by_column_template<IDX_TYPE>( \
        handle_, m, n, nnz, (IDX_TYPE*)row_data, (IDX_TYPE*)col_data, (IDX_TYPE*)perm, buffer))
    switch(idx_type)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        return rocsparse_status_success;
    }
    case rocsparse_indextype_i32:
    {
        CALL_TEMPLATE(int32_t);
        return rocsparse_status_success;
    }
    case rocsparse_indextype_i64:
    {
        CALL_TEMPLATE(int64_t);
        return rocsparse_status_success;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);

#undef CALL_TEMPLATE
}

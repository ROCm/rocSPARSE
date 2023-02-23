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
#include "rocsparse_check_matrix_coo.hpp"
#include "definitions.h"
#include "utility.h"

#include "check_matrix_coo_device.h"

std::string rocsparse_matrixtype2string(rocsparse_matrix_type type);

const char* rocsparse_datastatus2string(rocsparse_data_status data_status);

template <typename T, typename I>
rocsparse_status rocsparse_check_matrix_coo_buffer_size_template(rocsparse_handle       handle,
                                                                 I                      m,
                                                                 I                      n,
                                                                 int64_t                nnz,
                                                                 const T*               coo_val,
                                                                 const I*               coo_row_ind,
                                                                 const I*               coo_col_ind,
                                                                 rocsparse_index_base   idx_base,
                                                                 rocsparse_matrix_type  matrix_type,
                                                                 rocsparse_fill_mode    uplo,
                                                                 rocsparse_storage_mode storage,
                                                                 size_t*                buffer_size)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        log_debug(handle, "Index base is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(matrix_type))
    {
        log_debug(handle, "Matrix type is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(uplo))
    {
        log_debug(handle, "Matrix fill mode is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(storage))
    {
        log_debug(handle, "Storage mode is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(m < 0 || n < 0 || nnz < 0)
    {
        log_debug(handle, "m, n, and nnz cannot be negative.");
        return rocsparse_status_invalid_size;
    }

    if(matrix_type != rocsparse_matrix_type_general)
    {
        if(m != n)
        {
            log_debug(handle,
                      ("Matrix was specified to be " + rocsparse_matrixtype2string(matrix_type)
                       + " but m != n"));
            return rocsparse_status_invalid_size;
        }
    }

    if(buffer_size == nullptr)
    {
        log_debug(handle, "buffer size pointer cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        log_debug(handle,
                  "COO values array, row indices, and column indices array must all be nullptr or "
                  "none nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // If all are null, ensure it is the zero matrix
    if(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
    {
        if(nnz != 0)
        {
            log_debug(handle,
                      "COO values array, row indices, and column indices array are all nullptr "
                      "indicating a zero matrix but this does not match nnz.");
            return rocsparse_status_invalid_pointer;
        }
        else
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
    }

    // data status
    *buffer_size = sizeof(rocsparse_data_status) * 256;

    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_check_matrix_coo_template(rocsparse_handle       handle,
                                                     I                      m,
                                                     I                      n,
                                                     int64_t                nnz,
                                                     const T*               coo_val,
                                                     const I*               coo_row_ind,
                                                     const I*               coo_col_ind,
                                                     rocsparse_index_base   idx_base,
                                                     rocsparse_matrix_type  matrix_type,
                                                     rocsparse_fill_mode    uplo,
                                                     rocsparse_storage_mode storage,
                                                     rocsparse_data_status* data_status,
                                                     void*                  temp_buffer)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(rocsparse_enum_utils::is_invalid(idx_base))
    {
        log_debug(handle, "Index base is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(matrix_type))
    {
        log_debug(handle, "Matrix type is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(uplo))
    {
        log_debug(handle, "Matrix fill mode is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(storage))
    {
        log_debug(handle, "Storage mode is invalid.");
        return rocsparse_status_invalid_value;
    }

    if(m < 0 || n < 0 || nnz < 0)
    {
        log_debug(handle, "m, n, and nnz cannot be negative.");
        return rocsparse_status_invalid_size;
    }

    if(matrix_type != rocsparse_matrix_type_general)
    {
        if(m != n)
        {
            log_debug(handle,
                      ("Matrix was specified to be " + rocsparse_matrixtype2string(matrix_type)
                       + " but m != n"));
            return rocsparse_status_invalid_size;
        }
    }

    if(data_status == nullptr)
    {
        log_debug(handle, "data status pointer cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // All must be null (zero matrix) or none null
    if(!(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
       && !(coo_val != nullptr && coo_row_ind != nullptr && coo_col_ind != nullptr))
    {
        log_debug(handle,
                  "COO values array, row indices, and column indices array must all be nullptr or "
                  "none nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // If all are null, ensure it is the zero matrix
    if(coo_val == nullptr && coo_row_ind == nullptr && coo_col_ind == nullptr)
    {
        if(nnz != 0)
        {
            log_debug(handle,
                      "COO values array, row indices, and column indices array are all nullptr "
                      "indicating a zero matrix but this does not match nnz.");
            return rocsparse_status_invalid_pointer;
        }
        else
        {
            // clear output status to success
            *data_status = rocsparse_data_status_success;
            return rocsparse_status_success;
        }
    }

    if(temp_buffer == nullptr)
    {
        log_debug(handle, "COO temp buffer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // clear output status to success
    *data_status = rocsparse_data_status_success;

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    rocsparse_data_status* d_data_status = reinterpret_cast<rocsparse_data_status*>(ptr);
    ptr += sizeof(rocsparse_data_status) * 256;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_data_status, 0, sizeof(int)));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    hipLaunchKernelGGL((check_matrix_coo_device<256>),
                       dim3((nnz - 1) / 256 + 1),
                       dim3(256),
                       0,
                       handle->stream,
                       m,
                       n,
                       nnz,
                       coo_val,
                       coo_row_ind,
                       coo_col_ind,
                       idx_base,
                       matrix_type,
                       uplo,
                       storage,
                       d_data_status);

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(data_status,
                                       d_data_status,
                                       sizeof(rocsparse_data_status),
                                       hipMemcpyDeviceToHost,
                                       handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(*data_status != rocsparse_data_status_success)
    {
        log_debug(handle, rocsparse_datastatus2string(*data_status));
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                                            \
    template rocsparse_status rocsparse_check_matrix_coo_buffer_size_template<TTYPE, ITYPE>( \
        rocsparse_handle       handle,                                                       \
        ITYPE                  m,                                                            \
        ITYPE                  n,                                                            \
        int64_t                nnz,                                                          \
        const TTYPE*           coo_val,                                                      \
        const ITYPE*           coo_row_ind,                                                  \
        const ITYPE*           coo_col_ind,                                                  \
        rocsparse_index_base   idx_base,                                                     \
        rocsparse_matrix_type  matrix_type,                                                  \
        rocsparse_fill_mode    uplo,                                                         \
        rocsparse_storage_mode storage,                                                      \
        size_t*                buffer_size);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                                \
    template rocsparse_status rocsparse_check_matrix_coo_template<TTYPE, ITYPE>( \
        rocsparse_handle       handle,                                           \
        ITYPE                  m,                                                \
        ITYPE                  n,                                                \
        int64_t                nnz,                                              \
        const TTYPE*           coo_val,                                          \
        const ITYPE*           coo_row_ind,                                      \
        const ITYPE*           coo_col_ind,                                      \
        rocsparse_index_base   idx_base,                                         \
        rocsparse_matrix_type  matrix_type,                                      \
        rocsparse_fill_mode    uplo,                                             \
        rocsparse_storage_mode storage,                                          \
        rocsparse_data_status* data_status,                                      \
        void*                  temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,          \
                                     rocsparse_int          m,               \
                                     rocsparse_int          n,               \
                                     rocsparse_int          nnz,             \
                                     const TYPE*            coo_val,         \
                                     const rocsparse_int*   coo_row_ind,     \
                                     const rocsparse_int*   coo_col_ind,     \
                                     rocsparse_index_base   idx_base,        \
                                     rocsparse_matrix_type  matrix_type,     \
                                     rocsparse_fill_mode    uplo,            \
                                     rocsparse_storage_mode storage,         \
                                     size_t*                buffer_size)     \
    try                                                                      \
    {                                                                        \
        return rocsparse_check_matrix_coo_buffer_size_template(handle,       \
                                                               m,            \
                                                               n,            \
                                                               nnz,          \
                                                               coo_val,      \
                                                               coo_row_ind,  \
                                                               coo_col_ind,  \
                                                               idx_base,     \
                                                               matrix_type,  \
                                                               uplo,         \
                                                               storage,      \
                                                               buffer_size); \
    }                                                                        \
    catch(...)                                                               \
    {                                                                        \
        return exception_to_rocsparse_status();                              \
    }

C_IMPL(rocsparse_scheck_matrix_coo_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_coo_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_coo_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_coo_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                               \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,      \
                                     rocsparse_int          m,           \
                                     rocsparse_int          n,           \
                                     rocsparse_int          nnz,         \
                                     const TYPE*            coo_val,     \
                                     const rocsparse_int*   coo_row_ind, \
                                     const rocsparse_int*   coo_col_ind, \
                                     rocsparse_index_base   idx_base,    \
                                     rocsparse_matrix_type  matrix_type, \
                                     rocsparse_fill_mode    uplo,        \
                                     rocsparse_storage_mode storage,     \
                                     rocsparse_data_status* data_status, \
                                     void*                  temp_buffer) \
    try                                                                  \
    {                                                                    \
        return rocsparse_check_matrix_coo_template(handle,               \
                                                   m,                    \
                                                   n,                    \
                                                   nnz,                  \
                                                   coo_val,              \
                                                   coo_row_ind,          \
                                                   coo_col_ind,          \
                                                   idx_base,             \
                                                   matrix_type,          \
                                                   uplo,                 \
                                                   storage,              \
                                                   data_status,          \
                                                   temp_buffer);         \
    }                                                                    \
    catch(...)                                                           \
    {                                                                    \
        return exception_to_rocsparse_status();                          \
    }

C_IMPL(rocsparse_scheck_matrix_coo, float);
C_IMPL(rocsparse_dcheck_matrix_coo, double);
C_IMPL(rocsparse_ccheck_matrix_coo, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_coo, rocsparse_double_complex);
#undef C_IMPL

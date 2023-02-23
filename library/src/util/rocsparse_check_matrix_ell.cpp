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
#include "rocsparse_check_matrix_ell.hpp"
#include "definitions.h"
#include "utility.h"

#include "check_matrix_ell_device.h"

const char* rocsparse_datastatus2string(rocsparse_data_status data_status);

template <typename T, typename I>
rocsparse_status rocsparse_check_matrix_ell_buffer_size_template(rocsparse_handle       handle,
                                                                 I                      m,
                                                                 I                      n,
                                                                 I                      ell_width,
                                                                 const T*               ell_val,
                                                                 const I*               ell_col_ind,
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

    if(m < 0 || n < 0 || ell_width < 0)
    {
        log_debug(handle, "m, n, and ell_width cannot be negative.");
        return rocsparse_status_invalid_size;
    }

    if(matrix_type != rocsparse_matrix_type_general)
    {
        log_debug(handle, "ELL format only supports general matrix type.");
        return rocsparse_status_invalid_value;
    }

    if(buffer_size == nullptr)
    {
        log_debug(handle, "buffer size pointer cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Both must be null (zero matrix) or both not null
    if((ell_val == nullptr && ell_col_ind != nullptr)
       || (ell_val != nullptr && ell_col_ind == nullptr))
    {
        log_debug(handle,
                  "ELL values array and column indices array must all be nullptr or none nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // If both are null, it must be zero matrix
    if(ell_val == nullptr && ell_col_ind == nullptr)
    {
        if(m * ell_width != 0)
        {
            log_debug(handle,
                      "ELL values and column indices array are both nullptr indicating zero matrix "
                      "but this does not match what is found in row pointer array.");
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
rocsparse_status rocsparse_check_matrix_ell_template(rocsparse_handle       handle,
                                                     I                      m,
                                                     I                      n,
                                                     I                      ell_width,
                                                     const T*               ell_val,
                                                     const I*               ell_col_ind,
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

    if(m < 0 || n < 0 || ell_width < 0)
    {
        log_debug(handle, "m, n, and ell_width cannot be negative.");
        return rocsparse_status_invalid_size;
    }

    if(matrix_type != rocsparse_matrix_type_general)
    {
        log_debug(handle, "ELL format only supports general matrix type.");
        return rocsparse_status_invalid_value;
    }

    if(data_status == nullptr)
    {
        log_debug(handle, "data status pointer cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Both must be null (zero matrix) or both not null
    if((ell_val == nullptr && ell_col_ind != nullptr)
       || (ell_val != nullptr && ell_col_ind == nullptr))
    {
        log_debug(handle,
                  "ELL values array and column indices array must all be nullptr or none nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // If both are null, it must be zero matrix
    if(ell_val == nullptr && ell_col_ind == nullptr)
    {
        if(m * ell_width != 0)
        {
            log_debug(handle,
                      "ELL values and column indices array are both nullptr indicating zero matrix "
                      "but this does not match what is found in row pointer array.");
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
        log_debug(handle, "ELL temp buffer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // clear output status to success
    *data_status = rocsparse_data_status_success;

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    rocsparse_data_status* d_data_status = reinterpret_cast<rocsparse_data_status*>(ptr);
    ptr += sizeof(rocsparse_data_status) * 256;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_data_status, 0, sizeof(rocsparse_data_status)));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    hipLaunchKernelGGL((check_matrix_ell_device<256>),
                       dim3((m - 1) / 256 + 1),
                       dim3(256),
                       0,
                       handle->stream,
                       m,
                       n,
                       ell_width,
                       ell_val,
                       ell_col_ind,
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
    template rocsparse_status rocsparse_check_matrix_ell_buffer_size_template<TTYPE, ITYPE>( \
        rocsparse_handle       handle,                                                       \
        ITYPE                  m,                                                            \
        ITYPE                  n,                                                            \
        ITYPE                  ell_width,                                                    \
        const TTYPE*           ell_val,                                                      \
        const ITYPE*           ell_col_ind,                                                  \
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
    template rocsparse_status rocsparse_check_matrix_ell_template<TTYPE, ITYPE>( \
        rocsparse_handle       handle,                                           \
        ITYPE                  m,                                                \
        ITYPE                  n,                                                \
        ITYPE                  ell_width,                                        \
        const TTYPE*           ell_val,                                          \
        const ITYPE*           ell_col_ind,                                      \
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
                                     rocsparse_int          ell_width,       \
                                     const TYPE*            ell_val,         \
                                     const rocsparse_int*   ell_col_ind,     \
                                     rocsparse_index_base   idx_base,        \
                                     rocsparse_matrix_type  matrix_type,     \
                                     rocsparse_fill_mode    uplo,            \
                                     rocsparse_storage_mode storage,         \
                                     size_t*                buffer_size)     \
    try                                                                      \
    {                                                                        \
        return rocsparse_check_matrix_ell_buffer_size_template(handle,       \
                                                               m,            \
                                                               n,            \
                                                               ell_width,    \
                                                               ell_val,      \
                                                               ell_col_ind,  \
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

C_IMPL(rocsparse_scheck_matrix_ell_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_ell_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_ell_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_ell_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                               \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,      \
                                     rocsparse_int          m,           \
                                     rocsparse_int          n,           \
                                     rocsparse_int          ell_width,   \
                                     const TYPE*            ell_val,     \
                                     const rocsparse_int*   ell_col_ind, \
                                     rocsparse_index_base   idx_base,    \
                                     rocsparse_matrix_type  matrix_type, \
                                     rocsparse_fill_mode    uplo,        \
                                     rocsparse_storage_mode storage,     \
                                     rocsparse_data_status* data_status, \
                                     void*                  temp_buffer) \
    try                                                                  \
    {                                                                    \
        return rocsparse_check_matrix_ell_template(handle,               \
                                                   m,                    \
                                                   n,                    \
                                                   ell_width,            \
                                                   ell_val,              \
                                                   ell_col_ind,          \
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

C_IMPL(rocsparse_scheck_matrix_ell, float);
C_IMPL(rocsparse_dcheck_matrix_ell, double);
C_IMPL(rocsparse_ccheck_matrix_ell, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_ell, rocsparse_double_complex);
#undef C_IMPL

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
#include "internal/util/rocsparse_check_matrix_csr.h"
#include "definitions.h"
#include "rocsparse_check_matrix_csr.hpp"
#include "utility.h"

#include "check_matrix_csr_device.h"

#include <rocprim/rocprim.hpp>

std::string rocsparse_matrixtype2string(rocsparse_matrix_type type)
{
    switch(type)
    {
    case rocsparse_matrix_type_general:
        return "general";
    case rocsparse_matrix_type_symmetric:
        return "symmetric";
    case rocsparse_matrix_type_hermitian:
        return "hermitian";
    case rocsparse_matrix_type_triangular:
        return "triangular";
    }
    return "invalid";
}

const char* rocsparse_datastatus2string(rocsparse_data_status data_status)
{
    switch(data_status)
    {
    case rocsparse_data_status_success:
        return "No errors in data detected";
    case rocsparse_data_status_inf:
        return "An inf value was found in the values array.";
    case rocsparse_data_status_nan:
        return "An nan value was found in the values array.";
    case rocsparse_data_status_invalid_offset_ptr:
        return "An invalid offset pointer was detected.";
    case rocsparse_data_status_invalid_index:
        return "An invalid index was detected.";
    case rocsparse_data_status_duplicate_entry:
        return "A duplicate entry was detected.";
    case rocsparse_data_status_invalid_sorting:
        return "Sorting mode was detected to be invalid.";
    case rocsparse_data_status_invalid_fill:
        return "Fill mode was detected to be invalid.";
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_csr_buffer_size_template(rocsparse_handle       handle,
                                                                 J                      m,
                                                                 J                      n,
                                                                 I                      nnz,
                                                                 const T*               csr_val,
                                                                 const I*               csr_row_ptr,
                                                                 const J*               csr_col_ind,
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

    // Check row pointer array
    if(csr_row_ptr == nullptr)
    {
        log_debug(handle, "CSR row pointer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        log_debug(
            handle,
            "CSR values array and column indices array must be both nullptr or both not nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Check if zero matrix
    if(csr_val == nullptr && csr_col_ind == nullptr)
    {
        if(nnz != 0)
        {
            log_debug(handle,
                      "CSR values and column indices array are both nullptr indicating zero matrix "
                      "but this does not match what is found in row pointer array.");
            return rocsparse_status_invalid_pointer;
        }
    }

    *buffer_size = 0;
    *buffer_size += ((sizeof(rocsparse_data_status) - 1) / 256 + 1) * 256; // data status

    if(storage == rocsparse_storage_mode_unsorted)
    {
        // Determine required rocprim buffer size
        size_t                    rocprim_buffer_size;
        rocprim::double_buffer<J> dummy(nullptr, nullptr);
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(nullptr,
                                                                rocprim_buffer_size,
                                                                dummy,
                                                                dummy,
                                                                nnz,
                                                                m,
                                                                csr_row_ptr,
                                                                csr_row_ptr + 1,
                                                                0,
                                                                rocsparse_clz(n),
                                                                handle->stream));
        *buffer_size += ((rocprim_buffer_size - 1) / 256 + 1) * 256;

        // offset buffer
        *buffer_size += ((sizeof(I) * m) / 256 + 1) * 256;

        // columns buffer
        *buffer_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
        *buffer_size += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;
    }

    return rocsparse_status_success;
}

#define LAUNCH_CHECK_MATRIX_CSR(block_size, wf_size)                   \
    hipLaunchKernelGGL((check_matrix_csr_device<block_size, wf_size>), \
                       dim3((wf_size * m - 1) / block_size + 1),       \
                       dim3(block_size),                               \
                       0,                                              \
                       handle->stream,                                 \
                       m,                                              \
                       n,                                              \
                       nnz,                                            \
                       csr_val,                                        \
                       csr_row_ptr,                                    \
                       csr_col_ind,                                    \
                       csr_col_ind_sorted,                             \
                       idx_base,                                       \
                       matrix_type,                                    \
                       uplo,                                           \
                       storage,                                        \
                       d_data_status);

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_csr_template(rocsparse_handle       handle,
                                                     J                      m,
                                                     J                      n,
                                                     I                      nnz,
                                                     const T*               csr_val,
                                                     const I*               csr_row_ptr,
                                                     const J*               csr_col_ind,
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

    if(temp_buffer == nullptr)
    {
        log_debug(handle, "CSR temp buffer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Check row pointer array
    if(csr_row_ptr == nullptr)
    {
        log_debug(handle, "CSR row pointer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val == nullptr && csr_col_ind != nullptr)
       || (csr_val != nullptr && csr_col_ind == nullptr))
    {
        log_debug(
            handle,
            "CSR values array and column indices array must be both nullptr or both not nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Check if zero matrix
    if(csr_val == nullptr && csr_col_ind == nullptr)
    {
        if(nnz != 0)
        {
            log_debug(handle,
                      "CSR values and column indices array are both nullptr indicating zero matrix "
                      "but this does not match what is found in row pointer array.");
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check that nnz matches row pointer array
    I start = 0;
    I end   = 0;

    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&end, &csr_row_ptr[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&start, &csr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(nnz != (end - start))
    {
        log_debug(handle, "CSR row pointer array does not match nnz.");
        return rocsparse_status_invalid_value;
    }

    // clear output status to success
    *data_status = rocsparse_data_status_success;

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    rocsparse_data_status* d_data_status = reinterpret_cast<rocsparse_data_status*>(ptr);
    ptr += ((sizeof(rocsparse_data_status) - 1) / 256 + 1) * 256;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_data_status, 0, sizeof(rocsparse_data_status)));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    hipLaunchKernelGGL((check_row_ptr_array<256>),
                       dim3((m - 1) / 256 + 1),
                       dim3(256),
                       0,
                       handle->stream,
                       m,
                       csr_row_ptr,
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

        return rocsparse_status_success;
    }

    J avg_row_nnz = nnz / m;

    I* tmp_offsets = nullptr;
    J* tmp_cols1   = nullptr;
    J* tmp_cols2   = nullptr;

    // If columns are unsorted, then sort them in temp buffer
    if(storage == rocsparse_storage_mode_unsorted)
    {
        unsigned int startbit = 0;
        unsigned int endbit   = rocsparse_clz(n);
        size_t       size;

        // offsets buffer
        tmp_offsets = reinterpret_cast<I*>(ptr);
        ptr += ((sizeof(I) * m) / 256 + 1) * 256;

        // columns 1 buffer
        tmp_cols1 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

        // columns 2 buffer
        tmp_cols2 = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * nnz - 1) / 256 + 1) * 256;

        hipLaunchKernelGGL((shift_offsets_kernel<512>),
                           dim3(m / 512 + 1),
                           dim3(512),
                           0,
                           handle->stream,
                           m + 1,
                           csr_row_ptr,
                           tmp_offsets);

        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(tmp_cols1, csr_col_ind, sizeof(J) * nnz, hipMemcpyDeviceToDevice));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // rocprim buffer
        void* tmp_rocprim = reinterpret_cast<void*>(ptr);

        // Compute buffer size
        rocprim::double_buffer<J> dummy(tmp_cols1, tmp_cols2);
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(nullptr,
                                                               size,
                                                               dummy,
                                                               nnz,
                                                               m,
                                                               tmp_offsets,
                                                               tmp_offsets + 1,
                                                               startbit,
                                                               endbit,
                                                               handle->stream));

        // Sort by keys
        rocprim::double_buffer<J> keys(tmp_cols1, tmp_cols2);

        if(avg_row_nnz < 64)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(tmp_rocprim,
                                                                           size,
                                                                           keys,
                                                                           nnz,
                                                                           m,
                                                                           tmp_offsets,
                                                                           tmp_offsets + 1,
                                                                           startbit,
                                                                           endbit,
                                                                           handle->stream));
        }
        else if(avg_row_nnz < 128)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(tmp_rocprim,
                                                                           size,
                                                                           keys,
                                                                           nnz,
                                                                           m,
                                                                           tmp_offsets,
                                                                           tmp_offsets + 1,
                                                                           startbit,
                                                                           endbit,
                                                                           handle->stream));
        }
        else if(avg_row_nnz < 256)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(tmp_rocprim,
                                                                           size,
                                                                           keys,
                                                                           nnz,
                                                                           m,
                                                                           tmp_offsets,
                                                                           tmp_offsets + 1,
                                                                           startbit,
                                                                           endbit,
                                                                           handle->stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(tmp_rocprim,
                                                                   size,
                                                                   keys,
                                                                   nnz,
                                                                   m,
                                                                   tmp_offsets,
                                                                   tmp_offsets + 1,
                                                                   startbit,
                                                                   endbit,
                                                                   handle->stream));
        }

        if(keys.current() != tmp_cols2)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(tmp_cols2,
                                               keys.current(),
                                               sizeof(J) * nnz,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }
    }

    const J* csr_col_ind_sorted
        = (storage == rocsparse_storage_mode_unsorted) ? tmp_cols2 : csr_col_ind;

    if(avg_row_nnz <= 4)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 4);
    }
    else if(avg_row_nnz <= 8)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 8);
    }
    else if(avg_row_nnz <= 16)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 16);
    }
    else if(avg_row_nnz <= 32)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 32);
    }
    else if(avg_row_nnz <= 64)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 64);
    }
    else if(avg_row_nnz <= 128)
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 128);
    }
    else
    {
        LAUNCH_CHECK_MATRIX_CSR(256, 256);
    }

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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                      \
    template rocsparse_status                                                 \
        rocsparse_check_matrix_csr_buffer_size_template<TTYPE, ITYPE, JTYPE>( \
            rocsparse_handle       handle,                                    \
            JTYPE                  m,                                         \
            JTYPE                  n,                                         \
            ITYPE                  nnz,                                       \
            const TTYPE*           csr_val,                                   \
            const ITYPE*           csr_row_ptr,                               \
            const JTYPE*           csr_col_ind,                               \
            rocsparse_index_base   idx_base,                                  \
            rocsparse_matrix_type  matrix_type,                               \
            rocsparse_fill_mode    uplo,                                      \
            rocsparse_storage_mode storage,                                   \
            size_t*                buffer_size);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                \
    template rocsparse_status rocsparse_check_matrix_csr_template<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle       handle,                                                  \
        JTYPE                  m,                                                       \
        JTYPE                  n,                                                       \
        ITYPE                  nnz,                                                     \
        const TTYPE*           csr_val,                                                 \
        const ITYPE*           csr_row_ptr,                                             \
        const JTYPE*           csr_col_ind,                                             \
        rocsparse_index_base   idx_base,                                                \
        rocsparse_matrix_type  matrix_type,                                             \
        rocsparse_fill_mode    uplo,                                                    \
        rocsparse_storage_mode storage,                                                 \
        rocsparse_data_status* data_status,                                             \
        void*                  temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
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
                                     const TYPE*            csr_val,         \
                                     const rocsparse_int*   csr_row_ptr,     \
                                     const rocsparse_int*   csr_col_ind,     \
                                     rocsparse_index_base   idx_base,        \
                                     rocsparse_matrix_type  matrix_type,     \
                                     rocsparse_fill_mode    uplo,            \
                                     rocsparse_storage_mode storage,         \
                                     size_t*                buffer_size)     \
    try                                                                      \
    {                                                                        \
        return rocsparse_check_matrix_csr_buffer_size_template(handle,       \
                                                               m,            \
                                                               n,            \
                                                               nnz,          \
                                                               csr_val,      \
                                                               csr_row_ptr,  \
                                                               csr_col_ind,  \
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

C_IMPL(rocsparse_scheck_matrix_csr_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_csr_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_csr_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_csr_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                               \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,      \
                                     rocsparse_int          m,           \
                                     rocsparse_int          n,           \
                                     rocsparse_int          nnz,         \
                                     const TYPE*            csr_val,     \
                                     const rocsparse_int*   csr_row_ptr, \
                                     const rocsparse_int*   csr_col_ind, \
                                     rocsparse_index_base   idx_base,    \
                                     rocsparse_matrix_type  matrix_type, \
                                     rocsparse_fill_mode    uplo,        \
                                     rocsparse_storage_mode storage,     \
                                     rocsparse_data_status* data_status, \
                                     void*                  temp_buffer) \
    try                                                                  \
    {                                                                    \
        return rocsparse_check_matrix_csr_template(handle,               \
                                                   m,                    \
                                                   n,                    \
                                                   nnz,                  \
                                                   csr_val,              \
                                                   csr_row_ptr,          \
                                                   csr_col_ind,          \
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

C_IMPL(rocsparse_scheck_matrix_csr, float);
C_IMPL(rocsparse_dcheck_matrix_csr, double);
C_IMPL(rocsparse_ccheck_matrix_csr, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_csr, rocsparse_double_complex);
#undef C_IMPL

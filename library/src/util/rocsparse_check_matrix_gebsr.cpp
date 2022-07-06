/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_check_matrix_gebsr.hpp"
#include "definitions.h"
#include "utility.h"

#include <rocprim/rocprim.hpp>

#include "check_matrix_gebsr_device.h"

const char* rocsparse_datastatus2string(rocsparse_data_status data_status);

template <typename T, typename I, typename J>
rocsparse_status
    rocsparse_check_matrix_gebsr_buffer_size_template(rocsparse_handle       handle,
                                                      rocsparse_direction    dir,
                                                      J                      mb,
                                                      J                      nb,
                                                      I                      nnzb,
                                                      J                      row_block_dim,
                                                      J                      col_block_dim,
                                                      const T*               bsr_val,
                                                      const I*               bsr_row_ptr,
                                                      const J*               bsr_col_ind,
                                                      rocsparse_index_base   idx_base,
                                                      rocsparse_matrix_type  matrix_type,
                                                      rocsparse_fill_mode    uplo,
                                                      rocsparse_storage_mode storage,
                                                      size_t*                buffer_size)
{
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(rocsparse_enum_utils::is_invalid(dir))
    {
        log_debug(handle, "Direction is invalid.");
        return rocsparse_status_invalid_value;
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

    if(mb < 0 || nb < 0 || nnzb < 0)
    {
        log_debug(handle, "mb, nb, and nnzb cannot be negative.");
        return rocsparse_status_invalid_size;
    }

    if(row_block_dim <= 0 || col_block_dim <= 0)
    {
        log_debug(handle, "Row and column block dimension must both be greater than zero.");
        return rocsparse_status_invalid_size;
    }

    if(matrix_type != rocsparse_matrix_type_general)
    {
        log_debug(handle, "GEBSR format only supports general matrix type.");
        return rocsparse_status_invalid_value;
    }

    if(buffer_size == nullptr)
    {
        log_debug(handle, "buffer size pointer cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Check row pointer array
    if(bsr_row_ptr == nullptr)
    {
        log_debug(handle, "GEBSR row pointer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        log_debug(handle,
                  "GEBSR values array and column indices array must be both nullptr or both not "
                  "nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Check if zero matrix
    if(bsr_val == nullptr && bsr_col_ind == nullptr)
    {
        if(nnzb != 0)
        {
            log_debug(
                handle,
                "GEBSR values and column indices array are both nullptr indicating zero matrix "
                "but this does not match what is found in row pointer array.");
            return rocsparse_status_invalid_pointer;
        }
    }

    *buffer_size = 0;
    *buffer_size += sizeof(rocsparse_data_status) * 256; // data status

    if(storage == rocsparse_storage_mode_unsorted)
    {
        // Determine required rocprim buffer size
        size_t                    rocprim_buffer_size;
        rocprim::double_buffer<J> dummy(nullptr, nullptr);
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs(nullptr,
                                                                rocprim_buffer_size,
                                                                dummy,
                                                                dummy,
                                                                nnzb,
                                                                mb,
                                                                bsr_row_ptr,
                                                                bsr_row_ptr + 1,
                                                                0,
                                                                rocsparse_clz(nb),
                                                                handle->stream));
        *buffer_size += ((rocprim_buffer_size - 1) / 256 + 1) * 256;

        // offset buffer
        *buffer_size += sizeof(I) * (mb / 256 + 1) * 256;

        // columns buffer
        *buffer_size += sizeof(J) * ((nnzb - 1) / 256 + 1) * 256;
        *buffer_size += sizeof(J) * ((nnzb - 1) / 256 + 1) * 256;
    }

    return rocsparse_status_success;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_check_matrix_gebsr_template(rocsparse_handle       handle,
                                                       rocsparse_direction    dir,
                                                       J                      mb,
                                                       J                      nb,
                                                       I                      nnzb,
                                                       J                      row_block_dim,
                                                       J                      col_block_dim,
                                                       const T*               bsr_val,
                                                       const I*               bsr_row_ptr,
                                                       const J*               bsr_col_ind,
                                                       rocsparse_index_base   idx_base,
                                                       rocsparse_matrix_type  matrix_type,
                                                       rocsparse_fill_mode    uplo,
                                                       rocsparse_storage_mode storage,
                                                       rocsparse_data_status* data_status,
                                                       void*                  temp_buffer)
{
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    if(rocsparse_enum_utils::is_invalid(dir))
    {
        log_debug(handle, "Direction is invalid.");
        return rocsparse_status_invalid_value;
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

    if(mb < 0 || nb < 0 || nnzb < 0)
    {
        log_debug(handle, "mb, nb, and nnzb cannot be negative.");
        return rocsparse_status_invalid_size;
    }

    if(row_block_dim <= 0 || col_block_dim <= 0)
    {
        log_debug(handle, "Row and column block dimension must both be greater than zero.");
        return rocsparse_status_invalid_size;
    }

    if(matrix_type != rocsparse_matrix_type_general)
    {
        log_debug(handle, "GEBSR format only supports general matrix type.");
        return rocsparse_status_invalid_value;
    }

    if(data_status == nullptr)
    {
        log_debug(handle, "data status pointer cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    if(temp_buffer == nullptr)
    {
        log_debug(handle, "GEBSR temp buffer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Check row pointer array
    if(bsr_row_ptr == nullptr)
    {
        log_debug(handle, "GEBSR row pointer array cannot be nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((bsr_val == nullptr && bsr_col_ind != nullptr)
       || (bsr_val != nullptr && bsr_col_ind == nullptr))
    {
        log_debug(handle,
                  "GEBSR values array and column indices array must be both nullptr or both not "
                  "nullptr.");
        return rocsparse_status_invalid_pointer;
    }

    // Check if zero matrix
    if(bsr_val == nullptr && bsr_col_ind == nullptr)
    {
        if(nnzb != 0)
        {
            log_debug(
                handle,
                "GEBSR values and column indices array are both nullptr indicating zero matrix "
                "but this does not match what is found in row pointer array.");
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check that nnzb matches row pointer array
    I start = 0;
    I end   = 0;

    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&end, &bsr_row_ptr[mb], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&start, &bsr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(nnzb != (end - start))
    {
        log_debug(handle, "GEBSR row pointer array does not match nnzb.");
        return rocsparse_status_invalid_value;
    }

    // clear output status to success
    *data_status = rocsparse_data_status_success;

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    rocsparse_data_status* d_data_status = reinterpret_cast<rocsparse_data_status*>(ptr);
    ptr += sizeof(rocsparse_data_status) * 256;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_data_status, 0, sizeof(rocsparse_data_status)));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    hipLaunchKernelGGL((check_row_ptr_array<256>),
                       dim3((mb - 1) / 256 + 1),
                       dim3(256),
                       0,
                       handle->stream,
                       mb,
                       bsr_row_ptr,
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

    I* tmp_offsets = nullptr;
    J* tmp_cols1   = nullptr;
    J* tmp_cols2   = nullptr;

    // If columns are unsorted, then sort them in temp buffer
    if(storage == rocsparse_storage_mode_unsorted)
    {
        unsigned int startbit = 0;
        unsigned int endbit   = rocsparse_clz(nb);
        size_t       size;

        // offsets buffer
        tmp_offsets = reinterpret_cast<I*>(ptr);
        ptr += sizeof(I) * (mb / 256 + 1) * 256;

        // columns 1 buffer
        tmp_cols1 = reinterpret_cast<J*>(ptr);
        ptr += sizeof(J) * ((nnzb - 1) / 256 + 1) * 256;

        // columns 2 buffer
        tmp_cols2 = reinterpret_cast<J*>(ptr);
        ptr += sizeof(J) * ((nnzb - 1) / 256 + 1) * 256;

        hipLaunchKernelGGL((shift_offsets_kernel<512>),
                           dim3(mb / 512 + 1),
                           dim3(512),
                           0,
                           handle->stream,
                           mb + 1,
                           bsr_row_ptr,
                           tmp_offsets);

        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(tmp_cols1, bsr_col_ind, sizeof(J) * nnzb, hipMemcpyDeviceToDevice));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // rocprim buffer
        void* tmp_rocprim = reinterpret_cast<void*>(ptr);

        // Compute buffer size
        rocprim::double_buffer<J> dummy(tmp_cols1, tmp_cols2);
        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(nullptr,
                                                               size,
                                                               dummy,
                                                               nnzb,
                                                               mb,
                                                               tmp_offsets,
                                                               tmp_offsets + 1,
                                                               startbit,
                                                               endbit,
                                                               handle->stream));

        // Sort by keys
        rocprim::double_buffer<J> keys(tmp_cols1, tmp_cols2);

        // Determine blocksize and items per thread depending on average nnz per row
        J avg_row_nnzb = nnzb / mb;

        if(avg_row_nnzb < 64)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 1>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(tmp_rocprim,
                                                                           size,
                                                                           keys,
                                                                           nnzb,
                                                                           mb,
                                                                           tmp_offsets,
                                                                           tmp_offsets + 1,
                                                                           startbit,
                                                                           endbit,
                                                                           handle->stream));
        }
        else if(avg_row_nnzb < 128)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 2>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(tmp_rocprim,
                                                                           size,
                                                                           keys,
                                                                           nnzb,
                                                                           mb,
                                                                           tmp_offsets,
                                                                           tmp_offsets + 1,
                                                                           startbit,
                                                                           endbit,
                                                                           handle->stream));
        }
        else if(avg_row_nnzb < 256)
        {
            using config
                = rocprim::segmented_radix_sort_config<6, 5, rocprim::kernel_config<64, 4>>;
            RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(tmp_rocprim,
                                                                           size,
                                                                           keys,
                                                                           nnzb,
                                                                           mb,
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
                                                                   nnzb,
                                                                   mb,
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
                                               sizeof(J) * nnzb,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }
    }

    const J* bsr_col_ind_sorted
        = (storage == rocsparse_storage_mode_unsorted) ? tmp_cols2 : bsr_col_ind;

    hipLaunchKernelGGL((check_matrix_gebsr_device<256>),
                       dim3((mb - 1) / 256 + 1),
                       dim3(256),
                       0,
                       handle->stream,
                       dir,
                       mb,
                       nb,
                       nnzb,
                       row_block_dim,
                       col_block_dim,
                       bsr_val,
                       bsr_row_ptr,
                       bsr_col_ind,
                       bsr_col_ind_sorted,
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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                        \
    template rocsparse_status                                                   \
        rocsparse_check_matrix_gebsr_buffer_size_template<TTYPE, ITYPE, JTYPE>( \
            rocsparse_handle       handle,                                      \
            rocsparse_direction    dir,                                         \
            JTYPE                  mb,                                          \
            JTYPE                  nb,                                          \
            ITYPE                  nnzb,                                        \
            JTYPE                  row_block_dim,                               \
            JTYPE                  col_block_dim,                               \
            const TTYPE*           bsr_val,                                     \
            const ITYPE*           bsr_row_ptr,                                 \
            const JTYPE*           bsr_col_ind,                                 \
            rocsparse_index_base   idx_base,                                    \
            rocsparse_matrix_type  matrix_type,                                 \
            rocsparse_fill_mode    uplo,                                        \
            rocsparse_storage_mode storage,                                     \
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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                  \
    template rocsparse_status rocsparse_check_matrix_gebsr_template<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle       handle,                                                    \
        rocsparse_direction    dir,                                                       \
        JTYPE                  mb,                                                        \
        JTYPE                  nb,                                                        \
        ITYPE                  nnzb,                                                      \
        JTYPE                  row_block_dim,                                             \
        JTYPE                  col_block_dim,                                             \
        const TTYPE*           bsr_val,                                                   \
        const ITYPE*           bsr_row_ptr,                                               \
        const JTYPE*           bsr_col_ind,                                               \
        rocsparse_index_base   idx_base,                                                  \
        rocsparse_matrix_type  matrix_type,                                               \
        rocsparse_fill_mode    uplo,                                                      \
        rocsparse_storage_mode storage,                                                   \
        rocsparse_data_status* data_status,                                               \
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
#define C_IMPL(NAME, TYPE)                                                      \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,             \
                                     rocsparse_direction    dir,                \
                                     rocsparse_int          mb,                 \
                                     rocsparse_int          nb,                 \
                                     rocsparse_int          nnzb,               \
                                     rocsparse_int          row_block_dim,      \
                                     rocsparse_int          col_block_dim,      \
                                     const TYPE*            bsr_val,            \
                                     const rocsparse_int*   bsr_row_ptr,        \
                                     const rocsparse_int*   bsr_col_ind,        \
                                     rocsparse_index_base   idx_base,           \
                                     rocsparse_matrix_type  matrix_type,        \
                                     rocsparse_fill_mode    uplo,               \
                                     rocsparse_storage_mode storage,            \
                                     size_t*                buffer_size)        \
    {                                                                           \
        return rocsparse_check_matrix_gebsr_buffer_size_template(handle,        \
                                                                 dir,           \
                                                                 mb,            \
                                                                 nb,            \
                                                                 nnzb,          \
                                                                 row_block_dim, \
                                                                 col_block_dim, \
                                                                 bsr_val,       \
                                                                 bsr_row_ptr,   \
                                                                 bsr_col_ind,   \
                                                                 idx_base,      \
                                                                 matrix_type,   \
                                                                 uplo,          \
                                                                 storage,       \
                                                                 buffer_size);  \
    }

C_IMPL(rocsparse_scheck_matrix_gebsr_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_gebsr_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_gebsr_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_gebsr_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                 \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,        \
                                     rocsparse_direction    dir,           \
                                     rocsparse_int          mb,            \
                                     rocsparse_int          nb,            \
                                     rocsparse_int          nnzb,          \
                                     rocsparse_int          row_block_dim, \
                                     rocsparse_int          col_block_dim, \
                                     const TYPE*            bsr_val,       \
                                     const rocsparse_int*   bsr_row_ptr,   \
                                     const rocsparse_int*   bsr_col_ind,   \
                                     rocsparse_index_base   idx_base,      \
                                     rocsparse_matrix_type  matrix_type,   \
                                     rocsparse_fill_mode    uplo,          \
                                     rocsparse_storage_mode storage,       \
                                     rocsparse_data_status* data_status,   \
                                     void*                  temp_buffer)   \
    {                                                                      \
        return rocsparse_check_matrix_gebsr_template(handle,               \
                                                     dir,                  \
                                                     mb,                   \
                                                     nb,                   \
                                                     nnzb,                 \
                                                     row_block_dim,        \
                                                     col_block_dim,        \
                                                     bsr_val,              \
                                                     bsr_row_ptr,          \
                                                     bsr_col_ind,          \
                                                     idx_base,             \
                                                     matrix_type,          \
                                                     uplo,                 \
                                                     storage,              \
                                                     data_status,          \
                                                     temp_buffer);         \
    }

C_IMPL(rocsparse_scheck_matrix_gebsr, float);
C_IMPL(rocsparse_dcheck_matrix_gebsr, double);
C_IMPL(rocsparse_ccheck_matrix_gebsr, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_gebsr, rocsparse_double_complex);
#undef C_IMPL

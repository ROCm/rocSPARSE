/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include <algorithm>
#include <exception>

// Return the leftmost significant bit position
#if defined(rocsparse_ILP64)
static inline rocsparse_int rocsparse_clz(rocsparse_int n)
{
    // __builtin_clzll is undefined for n == 0
    if(n == 0)
    {
        return 0;
    }
    return 64 - __builtin_clzll(n);
}
#else
static inline rocsparse_int rocsparse_clz(rocsparse_int n)
{
    // __builtin_clz is undefined for n == 0
    if(n == 0)
    {
        return 0;
    }
    return 32 - __builtin_clz(n);
}
#endif

// Return one on the device
static inline void rocsparse_one(const rocsparse_handle handle, float** one)
{
    *one = handle->sone;
}

static inline void rocsparse_one(const rocsparse_handle handle, double** one)
{
    *one = handle->done;
}

static inline void rocsparse_one(const rocsparse_handle handle, rocsparse_float_complex** one)
{
    *one = handle->cone;
}

static inline void rocsparse_one(const rocsparse_handle handle, rocsparse_double_complex** one)
{
    *one = handle->zone;
}

// if trace logging is turned on with
// (handle->layer_mode & rocsparse_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(rocsparse_handle handle, H head, Ts&&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_trace)
        {
            std::string comma_separator = ",";

            std::ostream* os = handle->log_trace_os;
            log_arguments(*os, comma_separator, head, std::forward<Ts>(xs)...);
        }
    }
}

// if bench logging is turned on with
// (handle->layer_mode & rocsparse_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocsparse-bench.
template <typename H, typename... Ts>
void log_bench(rocsparse_handle handle, H head, std::string precision, Ts&&... xs)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_bench)
        {
            std::string space_separator = " ";

            std::ostream* os = handle->log_bench_os;
            log_arguments(*os, space_separator, head, precision, std::forward<Ts>(xs)...);
        }
    }
}

// if debug logging is turned on with
// (handle->layer_mode & rocsparse_layer_mode_log_debug) == true
// then
// log_debug will call log_arguments to log a error message
// when a routine returns a status which is not rocsparse_status_success.
static inline void log_debug(rocsparse_handle handle, std::string message)
{
    if(nullptr != handle)
    {
        if(handle->layer_mode & rocsparse_layer_mode_log_debug)
        {
            std::string space_separator = " ";

            std::ostream* os = handle->log_debug_os;
            log_arguments(*os, space_separator, message);
        }
    }
}

// Trace log scalar values pointed to by pointer
template <typename T>
T log_trace_scalar_value(const T* value)
{
    return value ? *value : std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T log_trace_scalar_value(rocsparse_handle handle, const T* value)
{
    if(handle->layer_mode & rocsparse_layer_mode_log_trace)
    {
        T host;
        if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipStreamCaptureStatus capture_status;
            RETURN_IF_HIP_ERROR(hipStreamIsCapturing(handle->stream, &capture_status));

            if(capture_status == hipStreamCaptureStatusNone)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &host, value, sizeof(host), hipMemcpyDeviceToHost, handle->stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                value = &host;
            }
            else
            {
                value = nullptr;
            }
        }
        return log_trace_scalar_value(value);
    }
    return T{};
}

#define LOG_TRACE_SCALAR_VALUE(handle, value) log_trace_scalar_value(handle, value)

// Bench log scalar values pointed to by pointer
template <typename T>
T log_bench_scalar_value(const T* value)
{
    return (value ? *value : std::numeric_limits<T>::quiet_NaN());
}

template <typename T>
T log_bench_scalar_value(rocsparse_handle handle, const T* value)
{
    if(handle->layer_mode & rocsparse_layer_mode_log_bench)
    {
        T host;
        if(value && handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            hipStreamCaptureStatus capture_status;
            RETURN_IF_HIP_ERROR(hipStreamIsCapturing(handle->stream, &capture_status));

            if(capture_status == hipStreamCaptureStatusNone)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &host, value, sizeof(host), hipMemcpyDeviceToHost, handle->stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                value = &host;
            }
            else
            {
                value = nullptr;
            }
        }
        return log_bench_scalar_value(value);
    }
    return T{};
}

#define LOG_BENCH_SCALAR_VALUE(handle, name) log_bench_scalar_value(handle, name)

// replaces X in string with s, d, c, z or h depending on typename T
template <typename T>
std::string replaceX(std::string input_string)
{
    if(std::is_same<T, float>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 's');
    }
    else if(std::is_same<T, double>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'd');
    }
    /*
    else if(std::is_same<T, rocsparse_float_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'c');
    }
    else if(std::is_same<T, rocsparse_double_complex>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'z');
    }
    else if(std::is_same<T, rocsparse_half>::value)
    {
        std::replace(input_string.begin(), input_string.end(), 'X', 'h');
    }
    */
    return input_string;
}

//
// These macros can be redefined if the developer includes src/include/debug.h
//
#define ROCSPARSE_DEBUG_VERBOSE(msg__) (void)0
#define ROCSPARSE_RETURN_STATUS(token__) return rocsparse_status_##token__

// Convert the current C++ exception to rocsparse_status
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent rocsparse_status here
inline rocsparse_status exception_to_rocsparse_status(std::exception_ptr e
                                                      = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return rocsparse_status_success;
}
catch(const rocsparse_status& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return rocsparse_status_memory_error;
}
catch(...)
{
    return rocsparse_status_thrown_exception;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(const T* xp)
{
    return *xp;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(T x)
{
    return static_cast<T>(0);
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(const T* xp)
{
    return static_cast<T>(0);
}

//
// Provide some utility methods for enums.
//
struct rocsparse_enum_utils
{
    template <typename U>
    static inline bool is_invalid(U value_);
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_itilu0_alg value)
{
    switch(value)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    case rocsparse_itilu0_alg_async_split:
    case rocsparse_itilu0_alg_sync_split:
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_diag_type value)
{
    switch(value)
    {
    case rocsparse_diag_type_unit:
    case rocsparse_diag_type_non_unit:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_fill_mode value_)
{
    switch(value_)
    {
    case rocsparse_fill_mode_lower:
    case rocsparse_fill_mode_upper:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_storage_mode value_)
{
    switch(value_)
    {
    case rocsparse_storage_mode_sorted:
    case rocsparse_storage_mode_unsorted:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_index_base value_)
{
    switch(value_)
    {
    case rocsparse_index_base_zero:
    case rocsparse_index_base_one:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_matrix_type value_)
{
    switch(value_)
    {
    case rocsparse_matrix_type_general:
    case rocsparse_matrix_type_symmetric:
    case rocsparse_matrix_type_hermitian:
    case rocsparse_matrix_type_triangular:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_direction value_)
{
    switch(value_)
    {
    case rocsparse_direction_row:
    case rocsparse_direction_column:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_operation value_)
{
    switch(value_)
    {
    case rocsparse_operation_none:
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_indextype value_)
{
    switch(value_)
    {
    case rocsparse_indextype_u16:
    case rocsparse_indextype_i32:
    case rocsparse_indextype_i64:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_datatype value_)
{
    switch(value_)
    {
    case rocsparse_datatype_f32_r:
    case rocsparse_datatype_f64_r:
    case rocsparse_datatype_f32_c:
    case rocsparse_datatype_f64_c:
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_order value_)
{
    switch(value_)
    {
    case rocsparse_order_row:
    case rocsparse_order_column:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_action value)
{
    switch(value)
    {
    case rocsparse_action_numeric:
    case rocsparse_action_symbolic:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_hyb_partition value)
{
    switch(value)
    {
    case rocsparse_hyb_partition_auto:
    case rocsparse_hyb_partition_user:
    case rocsparse_hyb_partition_max:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_gtsv_interleaved_alg value_)
{
    switch(value_)
    {
    case rocsparse_gtsv_interleaved_alg_default:
    case rocsparse_gtsv_interleaved_alg_thomas:
    case rocsparse_gtsv_interleaved_alg_lu:
    case rocsparse_gtsv_interleaved_alg_qr:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_sparse_to_dense_alg value_)
{
    switch(value_)
    {
    case rocsparse_sparse_to_dense_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_dense_to_sparse_alg value_)
{
    switch(value_)
    {
    case rocsparse_dense_to_sparse_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spmv_alg value_)
{
    switch(value_)
    {
    case rocsparse_spmv_alg_default:
    case rocsparse_spmv_alg_coo:
    case rocsparse_spmv_alg_csr_adaptive:
    case rocsparse_spmv_alg_csr_stream:
    case rocsparse_spmv_alg_ell:
    case rocsparse_spmv_alg_coo_atomic:
    case rocsparse_spmv_alg_bsr:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spsv_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsv_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spitsv_alg value_)
{
    switch(value_)
    {
    case rocsparse_spitsv_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spmv_stage value_)
{
    switch(value_)
    {
    case rocsparse_spmv_stage_auto:
    case rocsparse_spmv_stage_buffer_size:
    case rocsparse_spmv_stage_preprocess:
    case rocsparse_spmv_stage_compute:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spsv_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsv_stage_auto:
    case rocsparse_spsv_stage_buffer_size:
    case rocsparse_spsv_stage_preprocess:
    case rocsparse_spsv_stage_compute:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spitsv_stage value_)
{
    switch(value_)
    {
    case rocsparse_spitsv_stage_auto:
    case rocsparse_spitsv_stage_buffer_size:
    case rocsparse_spitsv_stage_preprocess:
    case rocsparse_spitsv_stage_compute:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spsm_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsm_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spsm_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsm_stage_auto:
    case rocsparse_spsm_stage_buffer_size:
    case rocsparse_spsm_stage_preprocess:
    case rocsparse_spsm_stage_compute:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spmm_alg value_)
{
    switch(value_)
    {
    case rocsparse_spmm_alg_default:
    case rocsparse_spmm_alg_csr:
    case rocsparse_spmm_alg_coo_segmented:
    case rocsparse_spmm_alg_coo_atomic:
    case rocsparse_spmm_alg_csr_row_split:
    case rocsparse_spmm_alg_csr_merge:
    case rocsparse_spmm_alg_coo_segmented_atomic:
    case rocsparse_spmm_alg_bell:
    case rocsparse_spmm_alg_bsr:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spmm_stage value_)
{
    switch(value_)
    {
    case rocsparse_spmm_stage_auto:
    case rocsparse_spmm_stage_buffer_size:
    case rocsparse_spmm_stage_preprocess:
    case rocsparse_spmm_stage_compute:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_sddmm_alg value_)
{
    switch(value_)
    {
    case rocsparse_sddmm_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spgemm_alg value_)
{
    switch(value_)
    {
    case rocsparse_spgemm_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_spgemm_stage value_)
{
    switch(value_)
    {
    case rocsparse_spgemm_stage_auto:
    case rocsparse_spgemm_stage_buffer_size:
    case rocsparse_spgemm_stage_nnz:
    case rocsparse_spgemm_stage_compute:
    case rocsparse_spgemm_stage_symbolic:
    case rocsparse_spgemm_stage_numeric:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_solve_policy value_)
{
    switch(value_)
    {
    case rocsparse_solve_policy_auto:
    {
        return false;
    }
    }
    return true;
};

template <>
inline bool rocsparse_enum_utils::is_invalid(rocsparse_analysis_policy value_)
{
    switch(value_)
    {
    case rocsparse_analysis_policy_reuse:
    case rocsparse_analysis_policy_force:
    {
        return false;
    }
    }
    return true;
};

template <typename T>
struct floating_traits
{
    using data_t = T;
};

template <>
struct floating_traits<rocsparse_float_complex>
{
    using data_t = float;
};

template <>
struct floating_traits<rocsparse_double_complex>
{
    using data_t = double;
};

template <typename T>
using floating_data_t = typename floating_traits<T>::data_t;

#include "envariables.h"
#include "memstat.h"

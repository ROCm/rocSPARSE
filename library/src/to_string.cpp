/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "to_string.hpp"
#include "control.h"

const char* rocsparse::to_string(rocsparse_data_status data_status)
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
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

const char* rocsparse::to_string(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success:
        return "success";
    case rocsparse_status_invalid_handle:
        return "invalid handle";
    case rocsparse_status_not_implemented:
        return "not implemented";
    case rocsparse_status_invalid_pointer:
        return "invalid pointer";
    case rocsparse_status_invalid_size:
        return "invalid size";
    case rocsparse_status_memory_error:
        return "memory error";
    case rocsparse_status_internal_error:
        return "internal error";
    case rocsparse_status_invalid_value:
        return "invalid value";
    case rocsparse_status_arch_mismatch:
        return "arch mismatch";
    case rocsparse_status_zero_pivot:
        return "zero pivot";
    case rocsparse_status_not_initialized:
        return "not initialized";
    case rocsparse_status_type_mismatch:
        return "type mismatch";
    case rocsparse_status_requires_sorted_storage:
        return "requires sorted storage";
    case rocsparse_status_thrown_exception:
        return "thrown exception";
    case rocsparse_status_continue:
        return "continue";
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

#define CASE(C) \
    case C:     \
        return #C

const char* rocsparse::to_string(rocsparse_sparse_to_sparse_stage value)
{
    switch(value)
    {
        CASE(rocsparse_sparse_to_sparse_stage_analysis);
        CASE(rocsparse_sparse_to_sparse_stage_compute);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_sparse_to_sparse_alg value)
{
    switch(value)
    {
        CASE(rocsparse_sparse_to_sparse_alg_default);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_pointer_mode value)
{
    switch(value)
    {
        CASE(rocsparse_pointer_mode_device);
        CASE(rocsparse_pointer_mode_host);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spmat_attribute value)
{
    switch(value)
    {
        CASE(rocsparse_spmat_fill_mode);
        CASE(rocsparse_spmat_diag_type);
        CASE(rocsparse_spmat_matrix_type);
        CASE(rocsparse_spmat_storage_mode);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_itilu0_alg value)
{
    switch(value)
    {
        CASE(rocsparse_itilu0_alg_default);
        CASE(rocsparse_itilu0_alg_async_inplace);
        CASE(rocsparse_itilu0_alg_async_split);
        CASE(rocsparse_itilu0_alg_sync_split);
        CASE(rocsparse_itilu0_alg_sync_split_fusion);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_diag_type value)
{
    switch(value)
    {
        CASE(rocsparse_diag_type_unit);
        CASE(rocsparse_diag_type_non_unit);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_fill_mode value_)
{
    switch(value_)
    {
        CASE(rocsparse_fill_mode_lower);
        CASE(rocsparse_fill_mode_upper);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_storage_mode value_)
{
    switch(value_)
    {
        CASE(rocsparse_storage_mode_sorted);
        CASE(rocsparse_storage_mode_unsorted);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_index_base value_)
{
    switch(value_)
    {
        CASE(rocsparse_index_base_zero);
        CASE(rocsparse_index_base_one);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_matrix_type value_)
{
    switch(value_)
    {
        CASE(rocsparse_matrix_type_general);
        CASE(rocsparse_matrix_type_symmetric);
        CASE(rocsparse_matrix_type_hermitian);
        CASE(rocsparse_matrix_type_triangular);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_direction value_)
{
    switch(value_)
    {
        CASE(rocsparse_direction_row);
        CASE(rocsparse_direction_column);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_operation value_)
{
    switch(value_)
    {
        CASE(rocsparse_operation_none);
        CASE(rocsparse_operation_transpose);
        CASE(rocsparse_operation_conjugate_transpose);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_indextype value_)
{
    switch(value_)
    {
        CASE(rocsparse_indextype_u16);
        CASE(rocsparse_indextype_i32);
        CASE(rocsparse_indextype_i64);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_datatype value_)
{
    switch(value_)
    {
        CASE(rocsparse_datatype_f32_r);
        CASE(rocsparse_datatype_f64_r);
        CASE(rocsparse_datatype_f32_c);
        CASE(rocsparse_datatype_f64_c);
        CASE(rocsparse_datatype_i8_r);
        CASE(rocsparse_datatype_u8_r);
        CASE(rocsparse_datatype_i32_r);
        CASE(rocsparse_datatype_u32_r);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_order value_)
{
    switch(value_)
    {
        CASE(rocsparse_order_row);
        CASE(rocsparse_order_column);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_action value)
{
    switch(value)
    {
        CASE(rocsparse_action_numeric);
        CASE(rocsparse_action_symbolic);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_hyb_partition value)
{
    switch(value)
    {
        CASE(rocsparse_hyb_partition_auto);
        CASE(rocsparse_hyb_partition_user);
        CASE(rocsparse_hyb_partition_max);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_gtsv_interleaved_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_gtsv_interleaved_alg_default);
        CASE(rocsparse_gtsv_interleaved_alg_thomas);
        CASE(rocsparse_gtsv_interleaved_alg_lu);
        CASE(rocsparse_gtsv_interleaved_alg_qr);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_sparse_to_dense_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_sparse_to_dense_alg_default);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_dense_to_sparse_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_dense_to_sparse_alg_default);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spmv_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_spmv_alg_default);
        CASE(rocsparse_spmv_alg_coo);
        CASE(rocsparse_spmv_alg_csr_adaptive);
        CASE(rocsparse_spmv_alg_csr_stream);
        CASE(rocsparse_spmv_alg_ell);
        CASE(rocsparse_spmv_alg_coo_atomic);
        CASE(rocsparse_spmv_alg_bsr);
        CASE(rocsparse_spmv_alg_csr_lrb);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spsv_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_spsv_alg_default);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spitsv_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_spitsv_alg_default);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_check_spmat_stage value_)
{
    switch(value_)
    {
        CASE(rocsparse_check_spmat_stage_buffer_size);
        CASE(rocsparse_check_spmat_stage_compute);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spmv_stage value_)
{
    switch(value_)
    {
        CASE(rocsparse_spmv_stage_buffer_size);
        CASE(rocsparse_spmv_stage_preprocess);
        CASE(rocsparse_spmv_stage_compute);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spsv_stage value_)
{
    switch(value_)
    {
        CASE(rocsparse_spsv_stage_buffer_size);
        CASE(rocsparse_spsv_stage_preprocess);
        CASE(rocsparse_spsv_stage_compute);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spitsv_stage value_)
{
    switch(value_)
    {
        CASE(rocsparse_spitsv_stage_buffer_size);
        CASE(rocsparse_spitsv_stage_preprocess);
        CASE(rocsparse_spitsv_stage_compute);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spsm_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_spsm_alg_default);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spsm_stage value_)
{
    switch(value_)
    {
        CASE(rocsparse_spsm_stage_buffer_size);
        CASE(rocsparse_spsm_stage_preprocess);
        CASE(rocsparse_spsm_stage_compute);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spmm_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_spmm_alg_default);
        CASE(rocsparse_spmm_alg_csr);
        CASE(rocsparse_spmm_alg_coo_segmented);
        CASE(rocsparse_spmm_alg_coo_atomic);
        CASE(rocsparse_spmm_alg_csr_row_split);
        CASE(rocsparse_spmm_alg_csr_merge);
        CASE(rocsparse_spmm_alg_coo_segmented_atomic);
        CASE(rocsparse_spmm_alg_bell);
        CASE(rocsparse_spmm_alg_bsr);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spmm_stage value_)
{
    switch(value_)
    {
        CASE(rocsparse_spmm_stage_buffer_size);
        CASE(rocsparse_spmm_stage_preprocess);
        CASE(rocsparse_spmm_stage_compute);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_sddmm_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_sddmm_alg_default);
        CASE(rocsparse_sddmm_alg_dense);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spgemm_alg value_)
{
    switch(value_)
    {
        CASE(rocsparse_spgemm_alg_default);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_spgemm_stage value_)
{
    switch(value_)
    {
        CASE(rocsparse_spgemm_stage_buffer_size);
        CASE(rocsparse_spgemm_stage_nnz);
        CASE(rocsparse_spgemm_stage_compute);
        CASE(rocsparse_spgemm_stage_symbolic);
        CASE(rocsparse_spgemm_stage_numeric);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_solve_policy value_)
{
    switch(value_)
    {
        CASE(rocsparse_solve_policy_auto);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_analysis_policy value_)
{
    switch(value_)
    {
        CASE(rocsparse_analysis_policy_reuse);
        CASE(rocsparse_analysis_policy_force);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

const char* rocsparse::to_string(rocsparse_format value_)
{
    switch(value_)
    {
        CASE(rocsparse_format_coo);
        CASE(rocsparse_format_coo_aos);
        CASE(rocsparse_format_csr);
        CASE(rocsparse_format_csc);
        CASE(rocsparse_format_ell);
        CASE(rocsparse_format_bell);
        CASE(rocsparse_format_bsr);
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
};

#undef CASE

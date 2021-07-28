/*! \file */
/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_bellmm.hpp"
#include "definitions.h"
#include "utility.h"

template <typename T, typename U, typename I>
rocsparse_status rocsparse_bellmm_template_general(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   rocsparse_order           order_B,
                                                   rocsparse_order           order_C,
                                                   rocsparse_direction       dir_A,
                                                   I                         mb,
                                                   I                         n,
                                                   I                         kb,
                                                   I                         bell_cols,
                                                   I                         bell_block_dim,
                                                   U                         alpha,
                                                   const rocsparse_mat_descr descr,
                                                   const I*                  bell_col_ind,
                                                   const T*                  bell_val,
                                                   const T*                  B,
                                                   I                         ldb,
                                                   U                         beta,
                                                   T*                        C,
                                                   I                         ldc);

template <typename T, typename U, typename I>
rocsparse_status rocsparse_bellmm_template_dispatch(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    rocsparse_order           order_B,
                                                    rocsparse_order           order_C,
                                                    rocsparse_direction       dir_A,
                                                    I                         mb,
                                                    I                         n,
                                                    I                         kb,
                                                    I                         bell_cols,
                                                    I                         bell_block_dim,
                                                    U                         alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const I*                  bell_col_ind,
                                                    const T*                  bell_val,
                                                    const T*                  B,
                                                    I                         ldb,
                                                    U                         beta_device_host,
                                                    T*                        C,
                                                    I                         ldc)
{

    return rocsparse_bellmm_template_general(handle,
                                             trans_A,
                                             trans_B,
                                             order_B,
                                             order_C,
                                             dir_A,
                                             mb,
                                             n,
                                             kb,
                                             bell_cols,
                                             bell_block_dim,
                                             alpha_device_host,
                                             descr,
                                             bell_col_ind,
                                             bell_val,
                                             B,
                                             ldb,
                                             beta_device_host,
                                             C,
                                             ldc);
}

#if 0

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__ void ell2csr_index_base(rocsparse_int* __restrict__ nnz)
{
  --(*nnz);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
  void ell2csr_nnz_per_row(rocsparse_int m,
			   rocsparse_int n,
			   rocsparse_int ell_width,
			   const rocsparse_int* __restrict__ ell_col_ind,
			   rocsparse_index_base ell_base,
			   rocsparse_int* __restrict__ csr_row_ptr,
			   rocsparse_index_base csr_base)
{
  rocsparse_int ai = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;

  if(ai >= m)
    {
      return;
    }

  if(ai == 0)
    {
      csr_row_ptr[0] = csr_base;
    }

  rocsparse_int nnz = 0;

  for(rocsparse_int p = 0; p < ell_width; ++p)
    {
      rocsparse_int idx = ELL_IND(ai, p, m, ell_width);
      rocsparse_int col = ell_col_ind[idx] - ell_base;

      if(col >= 0 && col < n)
        {
	  ++nnz;
        }
      else
        {
	  break;
        }
    }

  csr_row_ptr[ai + 1] = nnz;
}

extern "C" rocsparse_status rocsparse_ell2csr_nnz(rocsparse_handle          handle,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr ell_descr,
                                                  rocsparse_int             ell_width,
                                                  const rocsparse_int*      ell_col_ind,
                                                  const rocsparse_mat_descr csr_descr,
                                                  rocsparse_int*            csr_row_ptr,
                                                  rocsparse_int*            csr_nnz)
{
  // Check for valid handle and matrix descriptor
  if(handle == nullptr)
    {
      return rocsparse_status_invalid_handle;
    }
  else if(ell_descr == nullptr)
    {
      return rocsparse_status_invalid_pointer;
    }
  else if(csr_descr == nullptr)
    {
      return rocsparse_status_invalid_pointer;
    }

  // Logging
  log_trace(handle,
	    "rocsparse_ell2csr_nnz",
	    m,
	    n,
	    (const void*&)ell_descr,
	    ell_width,
	    (const void*&)ell_col_ind,
	    (const void*&)csr_descr,
	    (const void*&)csr_row_ptr,
	    (const void*&)csr_nnz);

  // Check index base
  if(ell_descr->base != rocsparse_index_base_zero && ell_descr->base != rocsparse_index_base_one)
    {
      return rocsparse_status_invalid_value;
    }
  if(csr_descr->base != rocsparse_index_base_zero && csr_descr->base != rocsparse_index_base_one)
    {
      return rocsparse_status_invalid_value;
    }

  // Check matrix type
  if(ell_descr->type != rocsparse_matrix_type_general)
    {
      // TODO
      return rocsparse_status_not_implemented;
    }
  if(csr_descr->type != rocsparse_matrix_type_general)
    {
      // TODO
      return rocsparse_status_not_implemented;
    }

  // Check sizes
  if(m < 0 || n < 0 || ell_width < 0)
    {
      return rocsparse_status_invalid_size;
    }

  // Check csr_nnz pointer argument before setting
  if(csr_nnz == nullptr)
    {
      return rocsparse_status_invalid_pointer;
    }

  // Stream
  hipStream_t stream = handle->stream;

  // Quick return if possible
  if(m == 0 || n == 0 || ell_width == 0)
    {
      if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
	  RETURN_IF_HIP_ERROR(hipMemsetAsync(csr_nnz, 0, sizeof(rocsparse_int), stream));
        }
      else
        {
	  *csr_nnz = 0;
        }
      return rocsparse_status_success;
    }

  // Check pointer arguments
  if(ell_col_ind == nullptr)
    {
      return rocsparse_status_invalid_pointer;
    }
  else if(csr_row_ptr == nullptr)
    {
      return rocsparse_status_invalid_pointer;
    }

  // Count nnz per row
#define ELL2CSR_DIM 256
  dim3 ell2csr_blocks((m + 1) / ELL2CSR_DIM + 1);
  dim3 ell2csr_threads(ELL2CSR_DIM);

  hipLaunchKernelGGL((ell2csr_nnz_per_row<ELL2CSR_DIM>),
		     ell2csr_blocks,
		     ell2csr_threads,
		     0,
		     stream,
		     m,
		     n,
		     ell_width,
		     ell_col_ind,
		     ell_descr->base,
		     csr_row_ptr,
		     csr_descr->base);
#undef ELL2CSR_DIM

  // Exclusive sum to obtain csr_row_ptr array and number of non-zero elements
  size_t temp_storage_bytes = 0;

  // Obtain rocprim buffer size
  RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
					      temp_storage_bytes,
					      csr_row_ptr,
					      csr_row_ptr,
					      m + 1,
					      rocprim::plus<rocsparse_int>(),
					      stream));

  // Get rocprim buffer
  bool  d_temp_alloc;
  void* d_temp_storage;

  // Device buffer should be sufficient for rocprim in most cases
  if(handle->buffer_size >= temp_storage_bytes)
    {
      d_temp_storage = handle->buffer;
      d_temp_alloc   = false;
    }
  else
    {
      RETURN_IF_HIP_ERROR(hipMalloc(&d_temp_storage, temp_storage_bytes));
      d_temp_alloc = true;
    }

  // Perform actual inclusive sum
  RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(d_temp_storage,
					      temp_storage_bytes,
					      csr_row_ptr,
					      csr_row_ptr,
					      m + 1,
					      rocprim::plus<rocsparse_int>(),
					      stream));

  // Extract and adjust nnz
  if(csr_descr->base == rocsparse_index_base_one)
    {
      if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
	  RETURN_IF_HIP_ERROR(hipMemcpyAsync(
					     csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));

	  // Adjust nnz according to index base
	  hipLaunchKernelGGL((ell2csr_index_base<1>), dim3(1), dim3(1), 0, stream, csr_nnz);
        }
      else
        {
	  RETURN_IF_HIP_ERROR(
			      hipMemcpy(csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

	  // Adjust nnz according to index base
	  *csr_nnz -= csr_descr->base;
        }
    }
  else
    {
      if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
	  RETURN_IF_HIP_ERROR(hipMemcpyAsync(
					     csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));
        }
      else
        {
	  RETURN_IF_HIP_ERROR(
			      hipMemcpy(csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        }
    }

  // Free rocprim buffer, if allocated
  if(d_temp_alloc == true)
    {
      RETURN_IF_HIP_ERROR(hipFree(d_temp_storage));
    }

  return rocsparse_status_success;
}
#endif

template <unsigned int BLOCKSIZE, typename I>
__launch_bounds__(BLOCKSIZE) __global__ void count_nnz(I m,
                                                       I ell_width,
                                                       const I* __restrict__ ell_col_ind,
                                                       rocsparse_index_base ell_base,
                                                       I* __restrict__ nnz)
{
    I ai = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    if(ai >= m)
    {
        return;
    }

    if(ai == 0)
    {
        I local_nnz = 0;
        for(I idx = 0; idx < m * ell_width; ++idx)
        {
            //	  csr_row_ptr[0] = csr_base;
            I col = ell_col_ind[idx] - ell_base;

            if(col >= 0)
            {
                ++local_nnz;
            }
        }
        *nnz = local_nnz;
    }
}

template <typename T, typename I>
rocsparse_status rocsparse_bellmm_template_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_operation       trans_B,
                                                       rocsparse_order           order_B,
                                                       rocsparse_order           order_C,
                                                       rocsparse_direction       dir_A,
                                                       I                         mb,
                                                       I                         n,
                                                       I                         kb,
                                                       I                         bell_cols,
                                                       I                         bell_block_dim,
                                                       const T*                  alpha,
                                                       const rocsparse_mat_descr descr,
                                                       const I*                  bell_col_ind,
                                                       const T*                  bell_val,
                                                       const T*                  B,
                                                       I                         ldb,
                                                       const T*                  beta,
                                                       T*                        C,
                                                       I                         ldc,
                                                       size_t*                   buffer_size)
{
    *buffer_size = 4;
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_bellmm_template_preprocess(rocsparse_handle          handle,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      rocsparse_order           order_B,
                                                      rocsparse_order           order_C,
                                                      rocsparse_direction       dir_A,
                                                      I                         mb,
                                                      I                         n,
                                                      I                         kb,
                                                      I                         bell_cols,
                                                      I                         bell_block_dim,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const I*                  bell_col_ind,
                                                      const T*                  bell_val,
                                                      const T*                  B,
                                                      I                         ldb,
                                                      const T*                  beta,
                                                      T*                        C,
                                                      I                         ldc,
                                                      void*                     temp_buffer)
{
    return rocsparse_status_success;
}

template <typename T, typename I>
rocsparse_status rocsparse_bellmm_template(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           rocsparse_order           order_B,
                                           rocsparse_order           order_C,
                                           rocsparse_direction       dir_A,
                                           I                         mb,
                                           I                         n,
                                           I                         kb,
                                           I                         bell_cols,
                                           I                         block_dim,
                                           const T*                  alpha,
                                           const rocsparse_mat_descr descr,
                                           const I*                  bell_col_ind,
                                           const T*                  bell_val,
                                           const T*                  B,
                                           I                         ldb,
                                           const T*                  beta,
                                           T*                        C,
                                           I                         ldc,
                                           void*                     temp_buffer)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xbellmm"),
              trans_A,
              trans_B,
              order_B,
              order_C,
              dir_A,
              mb,
              n,
              kb,
              bell_cols,
              block_dim,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr,
              (const void*&)bell_col_ind,
              (const void*&)bell_val,
              (const void*&)B,
              ldb,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)C,
              ldc);

    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_B))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(order_C))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(dir_A))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(mb < 0 || n < 0 || kb < 0 || bell_cols < 0 || block_dim <= 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || n == 0 || kb == 0 || bell_cols == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(bell_val == nullptr || bell_col_ind == nullptr || B == nullptr || C == nullptr
       || alpha == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(trans_A != rocsparse_operation_none)
    {
        return rocsparse_status_not_implemented;
    }

    // Check leading dimension of B
    if((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column)
       || (trans_B != rocsparse_operation_none && order_B != rocsparse_order_column))
    {
        if(ldb < kb * block_dim)
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldb < n)
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(ldc < mb * block_dim && order_C == rocsparse_order_column)
    {
        return rocsparse_status_invalid_size;
    }
    else if(ldc < n && order_C == rocsparse_order_row)
    {
        return rocsparse_status_invalid_size;
    }

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        return rocsparse_bellmm_template_dispatch(handle,
                                                  trans_A,
                                                  trans_B,
                                                  order_B,
                                                  order_C,
                                                  dir_A,
                                                  mb,
                                                  n,
                                                  kb,
                                                  bell_cols,
                                                  block_dim,
                                                  alpha,
                                                  descr,
                                                  bell_col_ind,
                                                  bell_val,
                                                  B,
                                                  ldb,
                                                  beta,
                                                  C,
                                                  ldc);
    }
    else
    {
        return rocsparse_bellmm_template_dispatch(handle,
                                                  trans_A,
                                                  trans_B,
                                                  order_B,
                                                  order_C,
                                                  dir_A,
                                                  mb,
                                                  n,
                                                  kb,
                                                  bell_cols,
                                                  block_dim,
                                                  *alpha,
                                                  descr,
                                                  bell_col_ind,
                                                  bell_val,
                                                  B,
                                                  ldb,
                                                  *beta,
                                                  C,
                                                  ldc);
    }
}

#define INSTANTIATE(TTYPE, ITYPE)                                                                 \
    template rocsparse_status rocsparse_bellmm_template_buffer_size(                              \
        rocsparse_handle          handle,                                                         \
        rocsparse_operation       trans_A,                                                        \
        rocsparse_operation       trans_B,                                                        \
        rocsparse_order           order_B,                                                        \
        rocsparse_order           order_C,                                                        \
        rocsparse_direction       dir_A,                                                          \
        ITYPE                     mb,                                                             \
        ITYPE                     n,                                                              \
        ITYPE                     kb,                                                             \
        ITYPE                     bell_cols,                                                      \
        ITYPE                     bell_block_dim,                                                 \
        const TTYPE*              alpha,                                                          \
        const rocsparse_mat_descr descr,                                                          \
        const ITYPE*              bell_col_ind,                                                   \
        const TTYPE*              bell_val,                                                       \
        const TTYPE*              B,                                                              \
        ITYPE                     ldb,                                                            \
        const TTYPE*              beta,                                                           \
        TTYPE*                    C,                                                              \
        ITYPE                     ldc,                                                            \
        size_t*                   buffer_size);                                                                     \
    template rocsparse_status rocsparse_bellmm_template_preprocess(                               \
        rocsparse_handle          handle,                                                         \
        rocsparse_operation       trans_A,                                                        \
        rocsparse_operation       trans_B,                                                        \
        rocsparse_order           order_B,                                                        \
        rocsparse_order           order_C,                                                        \
        rocsparse_direction       dir_A,                                                          \
        ITYPE                     mb,                                                             \
        ITYPE                     n,                                                              \
        ITYPE                     kb,                                                             \
        ITYPE                     bell_cols,                                                      \
        ITYPE                     bell_block_dim,                                                 \
        const TTYPE*              alpha,                                                          \
        const rocsparse_mat_descr descr,                                                          \
        const ITYPE*              bell_col_ind,                                                   \
        const TTYPE*              bell_val,                                                       \
        const TTYPE*              B,                                                              \
        ITYPE                     ldb,                                                            \
        const TTYPE*              beta,                                                           \
        TTYPE*                    C,                                                              \
        ITYPE                     ldc,                                                            \
        void*                     temp_buffer);                                                                       \
    template rocsparse_status rocsparse_bellmm_template(rocsparse_handle          handle,         \
                                                        rocsparse_operation       trans_A,        \
                                                        rocsparse_operation       trans_B,        \
                                                        rocsparse_order           order_B,        \
                                                        rocsparse_order           order_C,        \
                                                        rocsparse_direction       dir_A,          \
                                                        ITYPE                     mb,             \
                                                        ITYPE                     n,              \
                                                        ITYPE                     kb,             \
                                                        ITYPE                     bell_cols,      \
                                                        ITYPE                     bell_block_dim, \
                                                        const TTYPE*              alpha,          \
                                                        const rocsparse_mat_descr descr,          \
                                                        const ITYPE*              bell_col_ind,   \
                                                        const TTYPE*              bell_val,       \
                                                        const TTYPE*              B,              \
                                                        ITYPE                     ldb,            \
                                                        const TTYPE*              beta,           \
                                                        TTYPE*                    C,              \
                                                        ITYPE                     ldc,            \
                                                        void*                     temp_buffer)

INSTANTIATE(float, int32_t);
INSTANTIATE(double, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t);

INSTANTIATE(float, int64_t);
INSTANTIATE(double, int64_t);
INSTANTIATE(rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, int64_t);

#undef INSTANTIATE

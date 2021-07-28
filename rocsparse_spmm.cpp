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

#include "definitions.h"
#include "handle.h"
#include "rocsparse.h"
#include "utility.h"

#include "rocsparse_bellmm.hpp"
#include "rocsparse_coomm.hpp"
#include "rocsparse_csrmm.hpp"

#if 0

#if 0

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // work1 buffer
    rocsparse_int* tmp_work1 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // work2 buffer
    rocsparse_int* tmp_work2 = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // perm buffer
    rocsparse_int* tmp_perm = reinterpret_cast<rocsparse_int*>(ptr);
    ptr += sizeof(rocsparse_int) * ((nnz - 1) / 256 + 1) * 256;

    // rocprim buffer
    void* tmp_rocprim = reinterpret_cast<void*>(ptr);

    // Load CSR column indices into work1 buffer
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        tmp_work1, csr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToDevice, stream));

    if(copy_values == rocsparse_action_symbolic)
    {
        // action symbolic

        // Create row indices
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_csr2coo(handle, csr_row_ptr, nnz, m, csc_row_ind, idx_base));
        // Stable sort COO by columns
        rocprim::double_buffer<rocsparse_int> keys(tmp_work1, tmp_perm);
        rocprim::double_buffer<rocsparse_int> vals(csc_row_ind, tmp_work2);

        size_t size = 0;

        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, size, keys, vals, nnz, startbit, endbit, stream));
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            tmp_rocprim, size, keys, vals, nnz, startbit, endbit, stream));

        // Create column pointers
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_coo2csr(handle, keys.current(), nnz, n, csc_col_ptr, idx_base));

        // Copy csc_row_ind if not current
        if(vals.current() != csc_row_ind)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(csc_row_ind,
                                               vals.current(),
                                               sizeof(rocsparse_int) * nnz,
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }

#endif

template <unsigned int BLOCKSIZE, typename I>
__launch_bounds__(BLOCKSIZE) __global__
  void transpose_ell_width_kernel_part1(I        m,
					I        ell_width,
					const I* ell_col_ind,
					I*       workspace,
					rocsparse_index_base idx_base)
{
  I tid = hipThreadIdx_x;
  I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

  __shared__ I sdata[BLOCKSIZE];
  sdata[tid] = 0;

  for(I idx = gid; idx < ell_width; idx += hipGridDim_x * BLOCKSIZE)
    {
      for (I i = 0;i < m;++i)
	{
	  I col_idx = ell_col_ind[i * ell_width + idx] - idx_base;

	}
      sdata[tid] = max(sdata[tid], csr_row_ptr[idx + 1] - csr_row_ptr[idx]);
    }

  __syncthreads();

  rocsparse_blockreduce_max<BLOCKSIZE>(tid, sdata);
  if(tid == 0)
    {
      workspace[hipBlockIdx_x] = sdata[0];
    }
}

// Compute non-zero entries per CSR row and do a block reduction over the maximum
// Store result in a workspace for final reduction on part2
template <unsigned int BLOCKSIZE, typename I>
__launch_bounds__(BLOCKSIZE) __global__
  void transpose_ell_width_kernel_part1(I        m,
					const I* csr_row_ptr,
					I*       workspace)
{
  I tid = hipThreadIdx_x;
  I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

  __shared__ I sdata[BLOCKSIZE];
  sdata[tid] = 0;

  for(I idx = gid; idx < m; idx += hipGridDim_x * BLOCKSIZE)
    {
      sdata[tid] = max(sdata[tid], csr_row_ptr[idx + 1] - csr_row_ptr[idx]);
    }

  __syncthreads();

  rocsparse_blockreduce_max<BLOCKSIZE>(tid, sdata);

  if(tid == 0)
    {
      workspace[hipBlockIdx_x] = sdata[0];
    }
}

// Part2 kernel for final reduction over the maximum CSR nnz row entries
template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
  void transpose_ell_width_kernel_part2(I m, I* workspace)
{
  I tid = hipThreadIdx_x;

  __shared__ I sdata[BLOCKSIZE];
  sdata[tid] = 0;

    for(I i = tid; i < m; i += BLOCKSIZE)
    {
        sdata[tid] = max(sdata[tid], workspace[i]);
    }

    __syncthreads();

    rocsparse_blockreduce_max<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[0] = sdata[0];
    }
}
#endif

template <typename I, typename J, typename T>
rocsparse_status rocsparse_spmm_template(rocsparse_handle            handle,
                                         rocsparse_operation         trans_A,
                                         rocsparse_operation         trans_B,
                                         const void*                 alpha,
                                         const rocsparse_spmat_descr mat_A,
                                         const rocsparse_dnmat_descr mat_B,
                                         const void*                 beta,
                                         const rocsparse_dnmat_descr mat_C,
                                         rocsparse_spmm_alg          alg,
                                         size_t*                     buffer_size,
                                         void*                       temp_buffer)
{
    rocsparse_spmm_alg algorithm = alg;
    if(algorithm == rocsparse_spmm_alg_default)
    {
        switch(mat_A->format)
        {
        case rocsparse_format_coo:
        {
            algorithm = rocsparse_spmm_alg_coo_atomic;
            break;
        }

        case rocsparse_format_csr:
        {
            algorithm = rocsparse_spmm_alg_csr;
            break;
        }

        case rocsparse_format_bell:
        {
            algorithm = rocsparse_spmm_alg_bell;
            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        {
            break;
        }
        }
    }

    // If temp_buffer is nullptr, return buffer_size
    if(temp_buffer == nullptr)
    {
        switch(mat_A->format)
        {
        case rocsparse_format_coo:
        {
            algorithm = rocsparse_spmm_alg_coo_atomic;
            // We do not need a buffer
            *buffer_size = 4;
            break;
        }

        case rocsparse_format_csr:
        {
            algorithm = rocsparse_spmm_alg_csr;
            // We do not need a buffer
            *buffer_size = 4;

            break;
        }

        case rocsparse_format_bell:
        {
            algorithm = rocsparse_spmm_alg_bell;
            if(trans_A == rocsparse_operation_none)
            {
                // We do not need a buffer
                *buffer_size = 4;
            }
            else
            {
                //
                // We need a buffer to transpose the symbolic part.
                //
                // - one array for storing block row index.
                // - one array for storing the mapping to the blocks in the original array.
                //
                // - transpose value of ell_width: 1 * sizeof(I)
                // - row_ind array: ell_width * mat_A->cols * sizeof(I)
                // - block_map array: ell_width * mat_A->cols * sizeof(I)
                //
                // total number of bytes : (1 + 2 * ell_width * mat_A->cols) * sizeof(I)
                //

                //
                //
                //
                rocsparse_int transposed_ell_width = 0;

                *buffer_size = (1 + transposed_ell_width * mat_A->cols);

#if 0

		   //
		   // 1/ For buffer size: compute the value of ell_width for the transpose.
		   //  1-a/ Getting the maximum of row indices per column.
		   //  2-b/ Store the value in first of the buffer size
		   //
		   // 2/ On analysis phase: compute the array of row indices + compute the array of the mapping to the current blocks.
		   // 3/ On compute phase:
		   //
		   //
		   {

		     //
		     // Count the number of values != -1 per column and take the maximum.
		     //
		     I ell_width;
		     I * count_nnzb_per_column;

		     RETURN_IF_HIP_ERROR(hipMalloc(&count_nnzb_per_column,sizeof(I) * mat_A->cols));
		     RETURN_IF_HIP_ERROR(hipMemsetAsync(count_nnzb_per_column, 0, sizeof(I) * mat_A->cols, handle->stream));


		     RETURN_IF_HIP_ERROR(hipFree(count_nnzb_per_column));

#define CSR2ELL_DIM 256
		     // Workspace size
		     I nblocks = CSR2ELL_DIM;
		     hipMalloc();
		     // Get workspace from handle device buffer
		     I* workspace = reinterpret_cast<I*>(handle->buffer);

		     dim3 csr2ell_blocks(nblocks);
		     dim3 csr2ell_threads(CSR2ELL_DIM);

		     // Compute maximum nnz per row
		     hipLaunchKernelGGL((ell_width_kernel_part1<CSR2ELL_DIM>),
					csr2ell_blocks,
					csr2ell_threads,
					0,
					stream,
					m,
					csr_row_ptr,
					workspace);

		     hipLaunchKernelGGL((ell_width_kernel_part2<CSR2ELL_DIM>),
					dim3(1),
					csr2ell_threads,
					0,
					stream,
					nblocks,
					workspace);

		     // Copy ELL width back to host, if handle says so
		     if(handle->pointer_mode == rocsparse_pointer_mode_device)
		       {
			 RETURN_IF_HIP_ERROR(hipMemcpyAsync(
							    ell_width, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToDevice, stream));
		       }
		     else
		       {
			 RETURN_IF_HIP_ERROR(
					     hipMemcpy(ell_width, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
		       }


		   }
#endif
            }

            break;
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        {
            break;
        }
        }

        return rocsparse_status_success;
    }

    switch(mat_A->format)
    {
    case rocsparse_format_coo:
    {
        return rocsparse_coomm_template(handle,
                                        trans_A,
                                        trans_B,
                                        mat_B->order,
                                        mat_C->order,
                                        algorithm,
                                        (I)mat_A->rows,
                                        (I)mat_C->cols,
                                        (I)mat_A->cols,
                                        (I)mat_A->nnz,
                                        (const T*)alpha,
                                        mat_A->descr,
                                        (const T*)mat_A->val_data,
                                        (const I*)mat_A->row_data,
                                        (const I*)mat_A->col_data,
                                        (const T*)mat_B->values,
                                        (I)mat_B->ld,
                                        (const T*)beta,
                                        (T*)mat_C->values,
                                        (I)mat_C->ld);
    }

    case rocsparse_format_csr:
    {
        J m = (J)mat_C->rows;
        J n = (J)mat_C->cols;
        J k = trans_A == rocsparse_operation_none ? (J)mat_A->cols : (J)mat_A->rows;

        return rocsparse_csrmm_template(handle,
                                        trans_A,
                                        trans_B,
                                        mat_B->order,
                                        mat_C->order,
                                        m,
                                        n,
                                        k,
                                        (I)mat_A->nnz,
                                        (const T*)alpha,
                                        mat_A->descr,
                                        (const T*)mat_A->val_data,
                                        (const I*)mat_A->row_data,
                                        (const J*)mat_A->col_data,
                                        (const T*)mat_B->values,
                                        (J)mat_B->ld,
                                        (const T*)beta,
                                        (T*)mat_C->values,
                                        (J)mat_C->ld);
    }

    case rocsparse_format_bell:
    {

        std::cout << "HHHH trans " << trans_A << " -> " << (trans_A == rocsparse_operation_none)
                  << std::endl;
        std::cout << "HHHH M     " << mat_A->rows << std::endl;
        std::cout << "HHHH N     " << mat_A->cols << std::endl;

        std::cout << "HHHH "
                  << ((trans_A == rocsparse_operation_none) ? (I)(mat_A->cols / mat_A->block_dim)
                                                            : (I)(mat_A->rows / mat_A->block_dim))
                  << std::endl;

        return rocsparse_bellmm_template<T, I>(handle,
                                               trans_A,
                                               trans_B,
                                               mat_B->order,
                                               mat_C->order,
                                               mat_A->block_dir,
                                               (I)(mat_C->rows / mat_A->block_dim),
                                               (I)mat_C->cols,

                                               (trans_A == rocsparse_operation_none)
                                                   ? (I)(mat_A->cols / mat_A->block_dim)
                                                   : (I)(mat_A->rows / mat_A->block_dim),

                                               (I)mat_A->ell_cols,
                                               (I)mat_A->block_dim,
                                               (const T*)alpha,
                                               mat_A->descr,
                                               (const I*)mat_A->col_data,
                                               (const T*)mat_A->val_data,
                                               (const T*)mat_B->values,
                                               (I)mat_B->ld,
                                               (const T*)beta,
                                               (T*)mat_C->values,
                                               (I)mat_C->ld);
    }

    case rocsparse_format_coo_aos:
    case rocsparse_format_csc:
    case rocsparse_format_ell:
    {
        return rocsparse_status_not_implemented;
    }
    }
}

template <typename... Ts>
rocsparse_status rocsparse_spmm_template_dynamic_dispatch(rocsparse_indextype itype,
                                                          rocsparse_indextype jtype,
                                                          rocsparse_datatype  ctype,
                                                          Ts&&... ts)
{
    switch(itype)
    {
    case rocsparse_indextype_u16:
    {
        return rocsparse_status_not_implemented;
    }
    case rocsparse_indextype_i32:
    {
        switch(jtype)
        {
        case rocsparse_indextype_u16:
        case rocsparse_indextype_i64:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_indextype_i32:
        {
            switch(ctype)
            {
            case rocsparse_datatype_f32_r:
            {
                return rocsparse_spmm_template<int32_t, int32_t, float>(ts...);
            }
            case rocsparse_datatype_f64_r:
            {
                return rocsparse_spmm_template<int32_t, int32_t, double>(ts...);
            }
            case rocsparse_datatype_f32_c:
            {
                return rocsparse_spmm_template<int32_t, int32_t, rocsparse_float_complex>(ts...);
            }
            case rocsparse_datatype_f64_c:
            {
                return rocsparse_spmm_template<int32_t, int32_t, rocsparse_double_complex>(ts...);
            }
            }
        }
        }
    }
    case rocsparse_indextype_i64:
    {
        switch(jtype)
        {
        case rocsparse_indextype_u16:
        {
            return rocsparse_status_not_implemented;
        }
        case rocsparse_indextype_i32:
        {
            switch(ctype)
            {
            case rocsparse_datatype_f32_r:
            {
                return rocsparse_spmm_template<int64_t, int32_t, float>(ts...);
            }
            case rocsparse_datatype_f64_r:
            {
                return rocsparse_spmm_template<int64_t, int32_t, double>(ts...);
            }
            case rocsparse_datatype_f32_c:
            {
                return rocsparse_spmm_template<int64_t, int32_t, rocsparse_float_complex>(ts...);
            }
            case rocsparse_datatype_f64_c:
            {
                return rocsparse_spmm_template<int64_t, int32_t, rocsparse_double_complex>(ts...);
            }
            }
        }
        case rocsparse_indextype_i64:
        {
            switch(ctype)
            {
            case rocsparse_datatype_f32_r:
            {
                return rocsparse_spmm_template<int64_t, int64_t, float>(ts...);
            }
            case rocsparse_datatype_f64_r:
            {
                return rocsparse_spmm_template<int64_t, int64_t, double>(ts...);
            }
            case rocsparse_datatype_f32_c:
            {
                return rocsparse_spmm_template<int64_t, int64_t, rocsparse_float_complex>(ts...);
            }
            case rocsparse_datatype_f64_c:
            {
                return rocsparse_spmm_template<int64_t, int64_t, rocsparse_double_complex>(ts...);
            }
            }
        }
        }
    }
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spmm(rocsparse_handle            handle,
                                           rocsparse_operation         trans_A,
                                           rocsparse_operation         trans_B,
                                           const void*                 alpha,
                                           const rocsparse_spmat_descr mat_A,
                                           const rocsparse_dnmat_descr mat_B,
                                           const void*                 beta,
                                           const rocsparse_dnmat_descr mat_C,
                                           rocsparse_datatype          compute_type,
                                           rocsparse_spmm_alg          alg,
                                           size_t*                     buffer_size,
                                           void*                       temp_buffer)
{
    // Check for invalid handle
    RETURN_IF_INVALID_HANDLE(handle);
    // Logging
    log_trace(handle,
              "rocsparse_spmm",
              trans_A,
              trans_B,
              (const void*&)alpha,
              (const void*&)mat_A,
              (const void*&)mat_B,
              (const void*&)beta,
              (const void*&)mat_C,
              compute_type,
              alg,
              (const void*&)buffer_size,
              (const void*&)temp_buffer);

    // Check for invalid descriptors
    RETURN_IF_NULLPTR(mat_A);
    RETURN_IF_NULLPTR(mat_B);
    RETURN_IF_NULLPTR(mat_C);

    // Check for valid pointers
    RETURN_IF_NULLPTR(alpha);
    RETURN_IF_NULLPTR(beta);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    if(temp_buffer == nullptr)
    {
        RETURN_IF_NULLPTR(buffer_size);
    }

    // Check if descriptors are initialized
    if(mat_A->init == false || mat_B->init == false || mat_C->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    // Check for matching types while we do not support mixed precision computation
    if(compute_type != mat_A->data_type || compute_type != mat_B->data_type
       || compute_type != mat_C->data_type)
    {
        return rocsparse_status_not_implemented;
    }

    return rocsparse_spmm_template_dynamic_dispatch(mat_A->row_type,
                                                    mat_A->col_type,
                                                    compute_type,
                                                    handle,
                                                    trans_A,
                                                    trans_B,
                                                    alpha,
                                                    mat_A,
                                                    mat_B,
                                                    beta,
                                                    mat_C,
                                                    alg,
                                                    buffer_size,
                                                    temp_buffer);
}

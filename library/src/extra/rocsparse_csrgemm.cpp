/*! \file */
/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse_csrgemm.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "definitions.h"
#include "utility.h"

#include <rocprim/rocprim.hpp>

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL,
          typename I,
          typename J,
          typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrgemm_fill_wf_per_row_host_pointer(J m,
                                              J nk,
                                              const J* __restrict__ offset,
                                              const J* __restrict__ perm,
                                              T alpha,
                                              const I* __restrict__ csr_row_ptr_A,
                                              const J* __restrict__ csr_col_ind_A,
                                              const T* __restrict__ csr_val_A,
                                              const I* __restrict__ csr_row_ptr_B,
                                              const J* __restrict__ csr_col_ind_B,
                                              const T* __restrict__ csr_val_B,
                                              T beta,
                                              const I* __restrict__ csr_row_ptr_D,
                                              const J* __restrict__ csr_col_ind_D,
                                              const T* __restrict__ csr_val_D,
                                              const I* __restrict__ csr_row_ptr_C,
                                              J* __restrict__ csr_col_ind_C,
                                              T* __restrict__ csr_val_C,
                                              rocsparse_index_base idx_base_A,
                                              rocsparse_index_base idx_base_B,
                                              rocsparse_index_base idx_base_C,
                                              rocsparse_index_base idx_base_D,
                                              bool                 mul,
                                              bool                 add)
{
    csrgemm_fill_wf_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(m,
                                                                         nk,
                                                                         offset,
                                                                         perm,
                                                                         alpha,
                                                                         csr_row_ptr_A,
                                                                         csr_col_ind_A,
                                                                         csr_val_A,
                                                                         csr_row_ptr_B,
                                                                         csr_col_ind_B,
                                                                         csr_val_B,
                                                                         beta,
                                                                         csr_row_ptr_D,
                                                                         csr_col_ind_D,
                                                                         csr_val_D,
                                                                         csr_row_ptr_C,
                                                                         csr_col_ind_C,
                                                                         csr_val_C,
                                                                         idx_base_A,
                                                                         idx_base_B,
                                                                         idx_base_C,
                                                                         idx_base_D,
                                                                         mul,
                                                                         add);
}

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL,
          typename I,
          typename J,
          typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrgemm_fill_wf_per_row_device_pointer(J m,
                                                J nk,
                                                const J* __restrict__ offset,
                                                const J* __restrict__ perm,
                                                const T* __restrict__ alpha,
                                                const I* __restrict__ csr_row_ptr_A,
                                                const J* __restrict__ csr_col_ind_A,
                                                const T* __restrict__ csr_val_A,
                                                const I* __restrict__ csr_row_ptr_B,
                                                const J* __restrict__ csr_col_ind_B,
                                                const T* __restrict__ csr_val_B,
                                                const T* __restrict__ beta,
                                                const I* __restrict__ csr_row_ptr_D,
                                                const J* __restrict__ csr_col_ind_D,
                                                const T* __restrict__ csr_val_D,
                                                const I* __restrict__ csr_row_ptr_C,
                                                J* __restrict__ csr_col_ind_C,
                                                T* __restrict__ csr_val_C,
                                                rocsparse_index_base idx_base_A,
                                                rocsparse_index_base idx_base_B,
                                                rocsparse_index_base idx_base_C,
                                                rocsparse_index_base idx_base_D,
                                                bool                 mul,
                                                bool                 add)
{
    csrgemm_fill_wf_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
        m,
        nk,
        offset,
        perm,
        (mul == true) ? *alpha : static_cast<T>(0),
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_val_A,
        csr_row_ptr_B,
        csr_col_ind_B,
        csr_val_B,
        (add == true) ? *beta : static_cast<T>(0),
        csr_row_ptr_D,
        csr_col_ind_D,
        csr_val_D,
        csr_row_ptr_C,
        csr_col_ind_C,
        csr_val_C,
        idx_base_A,
        idx_base_B,
        idx_base_C,
        idx_base_D,
        mul,
        add);
}

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL,
          typename I,
          typename J,
          typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrgemm_fill_block_per_row_host_pointer(J nk,
                                                 const J* __restrict__ offset,
                                                 const J* __restrict__ perm,
                                                 T alpha,
                                                 const I* __restrict__ csr_row_ptr_A,
                                                 const J* __restrict__ csr_col_ind_A,
                                                 const T* __restrict__ csr_val_A,
                                                 const I* __restrict__ csr_row_ptr_B,
                                                 const J* __restrict__ csr_col_ind_B,
                                                 const T* __restrict__ csr_val_B,
                                                 T beta,
                                                 const I* __restrict__ csr_row_ptr_D,
                                                 const J* __restrict__ csr_col_ind_D,
                                                 const T* __restrict__ csr_val_D,
                                                 const I* __restrict__ csr_row_ptr_C,
                                                 J* __restrict__ csr_col_ind_C,
                                                 T* __restrict__ csr_val_C,
                                                 rocsparse_index_base idx_base_A,
                                                 rocsparse_index_base idx_base_B,
                                                 rocsparse_index_base idx_base_C,
                                                 rocsparse_index_base idx_base_D,
                                                 bool                 mul,
                                                 bool                 add)
{
    csrgemm_fill_block_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(nk,
                                                                            offset,
                                                                            perm,
                                                                            alpha,
                                                                            csr_row_ptr_A,
                                                                            csr_col_ind_A,
                                                                            csr_val_A,
                                                                            csr_row_ptr_B,
                                                                            csr_col_ind_B,
                                                                            csr_val_B,
                                                                            beta,
                                                                            csr_row_ptr_D,
                                                                            csr_col_ind_D,
                                                                            csr_val_D,
                                                                            csr_row_ptr_C,
                                                                            csr_col_ind_C,
                                                                            csr_val_C,
                                                                            idx_base_A,
                                                                            idx_base_B,
                                                                            idx_base_C,
                                                                            idx_base_D,
                                                                            mul,
                                                                            add);
}

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int HASHSIZE,
          unsigned int HASHVAL,
          typename I,
          typename J,
          typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrgemm_fill_block_per_row_device_pointer(J nk,
                                                   const J* __restrict__ offset,
                                                   const J* __restrict__ perm,
                                                   const T* __restrict__ alpha,
                                                   const I* __restrict__ csr_row_ptr_A,
                                                   const J* __restrict__ csr_col_ind_A,
                                                   const T* __restrict__ csr_val_A,
                                                   const I* __restrict__ csr_row_ptr_B,
                                                   const J* __restrict__ csr_col_ind_B,
                                                   const T* __restrict__ csr_val_B,
                                                   const T* __restrict__ beta,
                                                   const I* __restrict__ csr_row_ptr_D,
                                                   const J* __restrict__ csr_col_ind_D,
                                                   const T* __restrict__ csr_val_D,
                                                   const I* __restrict__ csr_row_ptr_C,
                                                   J* __restrict__ csr_col_ind_C,
                                                   T* __restrict__ csr_val_C,
                                                   rocsparse_index_base idx_base_A,
                                                   rocsparse_index_base idx_base_B,
                                                   rocsparse_index_base idx_base_C,
                                                   rocsparse_index_base idx_base_D,
                                                   bool                 mul,
                                                   bool                 add)
{
    csrgemm_fill_block_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
        nk,
        offset,
        perm,
        (mul == true) ? *alpha : static_cast<T>(0),
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_val_A,
        csr_row_ptr_B,
        csr_col_ind_B,
        csr_val_B,
        (add == true) ? *beta : static_cast<T>(0),
        csr_row_ptr_D,
        csr_col_ind_D,
        csr_val_D,
        csr_row_ptr_C,
        csr_col_ind_C,
        csr_val_C,
        idx_base_A,
        idx_base_B,
        idx_base_C,
        idx_base_D,
        mul,
        add);
}

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int CHUNKSIZE,
          typename I,
          typename J,
          typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrgemm_fill_block_per_row_multipass_host_pointer(J n,
                                                           const J* __restrict__ offset,
                                                           const J* __restrict__ perm,
                                                           T alpha,
                                                           const I* __restrict__ csr_row_ptr_A,
                                                           const J* __restrict__ csr_col_ind_A,
                                                           const T* __restrict__ csr_val_A,
                                                           const I* __restrict__ csr_row_ptr_B,
                                                           const J* __restrict__ csr_col_ind_B,
                                                           const T* __restrict__ csr_val_B,
                                                           T beta,
                                                           const I* __restrict__ csr_row_ptr_D,
                                                           const J* __restrict__ csr_col_ind_D,
                                                           const T* __restrict__ csr_val_D,
                                                           const I* __restrict__ csr_row_ptr_C,
                                                           J* __restrict__ csr_col_ind_C,
                                                           T* __restrict__ csr_val_C,
                                                           I* __restrict__ workspace_B,
                                                           rocsparse_index_base idx_base_A,
                                                           rocsparse_index_base idx_base_B,
                                                           rocsparse_index_base idx_base_C,
                                                           rocsparse_index_base idx_base_D,
                                                           bool                 mul,
                                                           bool                 add)
{
    csrgemm_fill_block_per_row_multipass_device<BLOCKSIZE, WFSIZE, CHUNKSIZE>(n,
                                                                              offset,
                                                                              perm,
                                                                              alpha,
                                                                              csr_row_ptr_A,
                                                                              csr_col_ind_A,
                                                                              csr_val_A,
                                                                              csr_row_ptr_B,
                                                                              csr_col_ind_B,
                                                                              csr_val_B,
                                                                              beta,
                                                                              csr_row_ptr_D,
                                                                              csr_col_ind_D,
                                                                              csr_val_D,
                                                                              csr_row_ptr_C,
                                                                              csr_col_ind_C,
                                                                              csr_val_C,
                                                                              workspace_B,
                                                                              idx_base_A,
                                                                              idx_base_B,
                                                                              idx_base_C,
                                                                              idx_base_D,
                                                                              mul,
                                                                              add);
}

template <unsigned int BLOCKSIZE,
          unsigned int WFSIZE,
          unsigned int CHUNKSIZE,
          typename I,
          typename J,
          typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL
    void csrgemm_fill_block_per_row_multipass_device_pointer(J n,
                                                             const J* __restrict__ offset,
                                                             const J* __restrict__ perm,
                                                             const T* __restrict__ alpha,
                                                             const I* __restrict__ csr_row_ptr_A,
                                                             const J* __restrict__ csr_col_ind_A,
                                                             const T* __restrict__ csr_val_A,
                                                             const I* __restrict__ csr_row_ptr_B,
                                                             const J* __restrict__ csr_col_ind_B,
                                                             const T* __restrict__ csr_val_B,
                                                             const T* __restrict__ beta,
                                                             const I* __restrict__ csr_row_ptr_D,
                                                             const J* __restrict__ csr_col_ind_D,
                                                             const T* __restrict__ csr_val_D,
                                                             const I* __restrict__ csr_row_ptr_C,
                                                             J* __restrict__ csr_col_ind_C,
                                                             T* __restrict__ csr_val_C,
                                                             I* __restrict__ workspace_B,
                                                             rocsparse_index_base idx_base_A,
                                                             rocsparse_index_base idx_base_B,
                                                             rocsparse_index_base idx_base_C,
                                                             rocsparse_index_base idx_base_D,
                                                             bool                 mul,
                                                             bool                 add)
{
    csrgemm_fill_block_per_row_multipass_device<BLOCKSIZE, WFSIZE, CHUNKSIZE>(
        n,
        offset,
        perm,
        (mul == true) ? *alpha : static_cast<T>(0),
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_val_A,
        csr_row_ptr_B,
        csr_col_ind_B,
        csr_val_B,
        (add == true) ? *beta : static_cast<T>(0),
        csr_row_ptr_D,
        csr_col_ind_D,
        csr_val_D,
        csr_row_ptr_C,
        csr_col_ind_C,
        csr_val_C,
        workspace_B,
        idx_base_A,
        idx_base_B,
        idx_base_C,
        idx_base_D,
        mul,
        add);
}

// Disable for rocsparse_double_complex, as well as double and rocsparse_float_complex
// if I == J == int64_t, as required size would exceed available memory
template <typename I,
          typename J,
          typename T,
          typename std::enable_if<
              std::is_same<T, rocsparse_double_complex>::value
                  || (std::is_same<T, double>::value && std::is_same<I, int64_t>::value
                      && std::is_same<J, int64_t>::value)
                  || (std::is_same<T, rocsparse_float_complex>::value
                      && std::is_same<I, int64_t>::value && std::is_same<J, int64_t>::value),
              int>::type
          = 0>
static inline rocsparse_status csrgemm_launcher(rocsparse_handle     handle,
                                                J                    group_size,
                                                const J*             group_offset,
                                                const J*             perm,
                                                J                    m,
                                                J                    n,
                                                J                    k,
                                                const T*             alpha,
                                                const I*             csr_row_ptr_A,
                                                const J*             csr_col_ind_A,
                                                const T*             csr_val_A,
                                                const I*             csr_row_ptr_B,
                                                const J*             csr_col_ind_B,
                                                const T*             csr_val_B,
                                                const T*             beta,
                                                const I*             csr_row_ptr_D,
                                                const J*             csr_col_ind_D,
                                                const T*             csr_val_D,
                                                const I*             csr_row_ptr_C,
                                                J*                   csr_col_ind_C,
                                                T*                   csr_val_C,
                                                rocsparse_index_base base_A,
                                                rocsparse_index_base base_B,
                                                rocsparse_index_base base_C,
                                                rocsparse_index_base base_D,
                                                bool                 mul,
                                                bool                 add)
{
    return rocsparse_status_internal_error;
}

template <typename I,
          typename J,
          typename T,
          typename std::enable_if<
              std::is_same<T, float>::value
                  || (std::is_same<T, double>::value
                      && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value))
                  || (std::is_same<T, rocsparse_float_complex>::value
                      && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value)),
              int>::type
          = 0>
static inline rocsparse_status csrgemm_launcher(rocsparse_handle     handle,
                                                J                    group_size,
                                                const J*             group_offset,
                                                const J*             perm,
                                                J                    m,
                                                J                    n,
                                                J                    k,
                                                const T*             alpha,
                                                const I*             csr_row_ptr_A,
                                                const J*             csr_col_ind_A,
                                                const T*             csr_val_A,
                                                const I*             csr_row_ptr_B,
                                                const J*             csr_col_ind_B,
                                                const T*             csr_val_B,
                                                const T*             beta,
                                                const I*             csr_row_ptr_D,
                                                const J*             csr_col_ind_D,
                                                const T*             csr_val_D,
                                                const I*             csr_row_ptr_C,
                                                J*                   csr_col_ind_C,
                                                T*                   csr_val_C,
                                                rocsparse_index_base base_A,
                                                rocsparse_index_base base_B,
                                                rocsparse_index_base base_C,
                                                rocsparse_index_base base_D,
                                                bool                 mul,
                                                bool                 add)
{
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 64
#define CSRGEMM_HASHSIZE 4096
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<CSRGEMM_DIM,
                                                                      CSRGEMM_SUB,
                                                                      CSRGEMM_HASHSIZE,
                                                                      CSRGEMM_FLL_HASH>),
                           dim3(group_size),
                           dim3(CSRGEMM_DIM),
                           0,
                           handle->stream,
                           std::max(k, n),
                           group_offset,
                           perm,
                           alpha,
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_val_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_val_B,
                           beta,
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_val_D,
                           csr_row_ptr_C,
                           csr_col_ind_C,
                           csr_val_C,
                           base_A,
                           base_B,
                           base_C,
                           base_D,
                           mul,
                           add);
    }
    else
    {
        hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<CSRGEMM_DIM,
                                                                    CSRGEMM_SUB,
                                                                    CSRGEMM_HASHSIZE,
                                                                    CSRGEMM_FLL_HASH>),
                           dim3(group_size),
                           dim3(CSRGEMM_DIM),
                           0,
                           handle->stream,
                           std::max(k, n),
                           group_offset,
                           perm,
                           mul ? *alpha : static_cast<T>(0),
                           csr_row_ptr_A,
                           csr_col_ind_A,
                           csr_val_A,
                           csr_row_ptr_B,
                           csr_col_ind_B,
                           csr_val_B,
                           add ? *beta : static_cast<T>(0),
                           csr_row_ptr_D,
                           csr_col_ind_D,
                           csr_val_D,
                           csr_row_ptr_C,
                           csr_col_ind_C,
                           csr_val_C,
                           base_A,
                           base_B,
                           base_C,
                           base_D,
                           mul,
                           add);
    }
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
static inline rocsparse_status rocsparse_csrgemm_calc_template(rocsparse_handle          handle,
                                                               rocsparse_operation       trans_A,
                                                               rocsparse_operation       trans_B,
                                                               J                         m,
                                                               J                         n,
                                                               J                         k,
                                                               const T*                  alpha,
                                                               const rocsparse_mat_descr descr_A,
                                                               I                         nnz_A,
                                                               const T*                  csr_val_A,
                                                               const I* csr_row_ptr_A,
                                                               const J* csr_col_ind_A,
                                                               const rocsparse_mat_descr descr_B,
                                                               I                         nnz_B,
                                                               const T*                  csr_val_B,
                                                               const I* csr_row_ptr_B,
                                                               const J* csr_col_ind_B,
                                                               const T* beta,
                                                               const rocsparse_mat_descr descr_D,
                                                               I                         nnz_D,
                                                               const T*                  csr_val_D,
                                                               const I* csr_row_ptr_D,
                                                               const J* csr_col_ind_D,
                                                               const rocsparse_mat_descr descr_C,
                                                               T*                        csr_val_C,
                                                               const I* csr_row_ptr_C,
                                                               J*       csr_col_ind_C,
                                                               const rocsparse_mat_info info_C,
                                                               void*                    temp_buffer)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D
        = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

    // Flag for exceeding shared memory
    constexpr bool exceeding_smem
        = std::is_same<T, rocsparse_double_complex>::value
          || (std::is_same<T, double>::value && std::is_same<I, int64_t>::value
              && std::is_same<J, int64_t>::value)
          || (std::is_same<T, rocsparse_float_complex>::value && std::is_same<I, int64_t>::value
              && std::is_same<J, int64_t>::value);

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Determine maximum non-zero entries per row of all rows
    J* workspace = reinterpret_cast<J*>(buffer);

#define CSRGEMM_DIM 256
    hipLaunchKernelGGL((csrgemm_max_row_nnz_part1<CSRGEMM_DIM>),
                       dim3(CSRGEMM_DIM),
                       dim3(CSRGEMM_DIM),
                       0,
                       stream,
                       m,
                       csr_row_ptr_C,
                       workspace);

    hipLaunchKernelGGL(
        (csrgemm_max_row_nnz_part2<CSRGEMM_DIM>), dim3(1), dim3(CSRGEMM_DIM), 0, stream, workspace);
#undef CSRGEMM_DIM

    J nnz_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&nnz_max, workspace, sizeof(J), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    J* d_group_offset = reinterpret_cast<J*>(buffer);
    buffer += sizeof(J) * 256;

    // Group size buffer
    J h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(J) * CSRGEMM_MAXGROUPS);

    // Permutation array
    J* d_perm = nullptr;

    // If maximum of row nnz exceeds 16, we process the rows in groups of
    // similar sized row nnz
    if(nnz_max > 16)
    {
        // Group size buffer
        J* d_group_size = reinterpret_cast<J*>(buffer);
        buffer += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;

        // Permutation temporary arrays
        J* tmp_vals = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        J* tmp_perm = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        int* tmp_keys = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        int* tmp_groups = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        hipLaunchKernelGGL(
            (csrgemm_group_reduce_part2<CSRGEMM_DIM, CSRGEMM_MAXGROUPS, exceeding_smem>),
            dim3(CSRGEMM_DIM),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            m,
            csr_row_ptr_C,
            d_group_size,
            tmp_groups);

        hipLaunchKernelGGL((csrgemm_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
                           dim3(1),
                           dim3(CSRGEMM_DIM),
                           0,
                           stream,
                           d_group_size);
#undef CSRGEMM_DIM

        // Exclusive sum to obtain group offsets
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<J>(),
                                                    stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(rocprim_buffer,
                                                    rocprim_size,
                                                    d_group_size,
                                                    d_group_offset,
                                                    0,
                                                    CSRGEMM_MAXGROUPS,
                                                    rocprim::plus<J>(),
                                                    stream));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(J) * CSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_create_identity_permutation_template(handle, m, tmp_perm));

        rocprim::double_buffer<int> d_keys(tmp_groups, tmp_keys);
        rocprim::double_buffer<J>   d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        RETURN_IF_HIP_ERROR(
            rocprim::radix_sort_pairs(nullptr, rocprim_size, d_keys, d_vals, m, 0, 3, stream));
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(
            rocprim_buffer, rocprim_size, d_keys, d_vals, m, 0, 3, stream));

        d_perm = d_vals.current();

        // Release tmp_groups buffer
        buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Release tmp_keys buffer
        buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(J), stream));
    }

    // Compute columns and accumulate values for each group

    // pointer mode device
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // Group 0: 0 - 16 non-zeros per row
        if(h_group_size[0] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 16
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_device_pointer<CSRGEMM_DIM,
                                                                       CSRGEMM_SUB,
                                                                       CSRGEMM_HASHSIZE,
                                                                       CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[0],
                               std::max(k, n),
                               &d_group_offset[0],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 1: 17 - 32 non-zeros per row
        if(h_group_size[1] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 32
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_device_pointer<CSRGEMM_DIM,
                                                                       CSRGEMM_SUB,
                                                                       CSRGEMM_HASHSIZE,
                                                                       CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[1],
                               std::max(k, n),
                               &d_group_offset[1],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 2: 33 - 256 non-zeros per row
        if(h_group_size[2] > 0)
        {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 256
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[2]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[2],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 3: 257 - 512 non-zeros per row
        if(h_group_size[3] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 512
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[3]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[3],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 4: 513 - 1024 non-zeros per row
        if(h_group_size[4] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 1024
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[4]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[4],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 5: 1025 - 2048 non-zeros per row
        if(h_group_size[5] > 0)
        {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 2048
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_device_pointer<CSRGEMM_DIM,
                                                                          CSRGEMM_SUB,
                                                                          CSRGEMM_HASHSIZE,
                                                                          CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[5]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[5],
                               d_perm,
                               alpha,
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               beta,
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

#ifndef rocsparse_ILP64
        // Group 6: 2049 - 4096 non-zeros per row
        if(h_group_size[6] > 0 && !exceeding_smem)
        {
            RETURN_IF_ROCSPARSE_ERROR(csrgemm_launcher(handle,
                                                       h_group_size[6],
                                                       &d_group_offset[6],
                                                       d_perm,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       csr_row_ptr_A,
                                                       csr_col_ind_A,
                                                       csr_val_A,
                                                       csr_row_ptr_B,
                                                       csr_col_ind_B,
                                                       csr_val_B,
                                                       beta,
                                                       csr_row_ptr_D,
                                                       csr_col_ind_D,
                                                       csr_val_D,
                                                       csr_row_ptr_C,
                                                       csr_col_ind_C,
                                                       csr_val_C,
                                                       base_A,
                                                       base_B,
                                                       descr_C->base,
                                                       base_D,
                                                       info_C->csrgemm_info->mul,
                                                       info_C->csrgemm_info->add));
        }
#endif

        // Group 7: more than 4096 non-zeros per row
        if(h_group_size[7] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
            I* workspace_B = nullptr;

            if(info_C->csrgemm_info->mul == true)
            {
                // Allocate additional buffer for C = alpha * A * B
                RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace_B, sizeof(I) * nnz_A));
            }

            hipLaunchKernelGGL(
                (csrgemm_fill_block_per_row_multipass_device_pointer<CSRGEMM_DIM,
                                                                     CSRGEMM_SUB,
                                                                     CSRGEMM_CHUNKSIZE>),
                dim3(h_group_size[7]),
                dim3(CSRGEMM_DIM),
                0,
                stream,
                n,
                &d_group_offset[7],
                d_perm,
                alpha,
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                beta,
                csr_row_ptr_D,
                csr_col_ind_D,
                csr_val_D,
                csr_row_ptr_C,
                csr_col_ind_C,
                csr_val_C,
                workspace_B,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add);

            if(info_C->csrgemm_info->mul == true)
            {
                RETURN_IF_HIP_ERROR(hipFree(workspace_B));
            }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }
    }
    else
    {
        // Group 0: 0 - 16 non-zeros per row
        if(h_group_size[0] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 16
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_host_pointer<CSRGEMM_DIM,
                                                                     CSRGEMM_SUB,
                                                                     CSRGEMM_HASHSIZE,
                                                                     CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[0],
                               std::max(k, n),
                               &d_group_offset[0],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 1: 17 - 32 non-zeros per row
        if(h_group_size[1] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 32
            hipLaunchKernelGGL((csrgemm_fill_wf_per_row_host_pointer<CSRGEMM_DIM,
                                                                     CSRGEMM_SUB,
                                                                     CSRGEMM_HASHSIZE,
                                                                     CSRGEMM_FLL_HASH>),
                               dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               h_group_size[1],
                               std::max(k, n),
                               &d_group_offset[1],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 2: 33 - 256 non-zeros per row
        if(h_group_size[2] > 0)
        {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 256
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[2]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[2],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 3: 257 - 512 non-zeros per row
        if(h_group_size[3] > 0)
        {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 512
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[3]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[3],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 4: 513 - 1024 non-zeros per row
        if(h_group_size[4] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 1024
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[4]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[4],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

        // Group 5: 1025 - 2048 non-zeros per row
        if(h_group_size[5] > 0)
        {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 2048
            hipLaunchKernelGGL((csrgemm_fill_block_per_row_host_pointer<CSRGEMM_DIM,
                                                                        CSRGEMM_SUB,
                                                                        CSRGEMM_HASHSIZE,
                                                                        CSRGEMM_FLL_HASH>),
                               dim3(h_group_size[5]),
                               dim3(CSRGEMM_DIM),
                               0,
                               stream,
                               std::max(k, n),
                               &d_group_offset[5],
                               d_perm,
                               (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                               csr_row_ptr_A,
                               csr_col_ind_A,
                               csr_val_A,
                               csr_row_ptr_B,
                               csr_col_ind_B,
                               csr_val_B,
                               (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                               csr_row_ptr_D,
                               csr_col_ind_D,
                               csr_val_D,
                               csr_row_ptr_C,
                               csr_col_ind_C,
                               csr_val_C,
                               base_A,
                               base_B,
                               descr_C->base,
                               base_D,
                               info_C->csrgemm_info->mul,
                               info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }

#ifndef rocsparse_ILP64
        // Group 6: 2049 - 4096 non-zeros per row
        if(h_group_size[6] > 0 && !exceeding_smem)
        {
            RETURN_IF_ROCSPARSE_ERROR(csrgemm_launcher(handle,
                                                       h_group_size[6],
                                                       &d_group_offset[6],
                                                       d_perm,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       csr_row_ptr_A,
                                                       csr_col_ind_A,
                                                       csr_val_A,
                                                       csr_row_ptr_B,
                                                       csr_col_ind_B,
                                                       csr_val_B,
                                                       beta,
                                                       csr_row_ptr_D,
                                                       csr_col_ind_D,
                                                       csr_val_D,
                                                       csr_row_ptr_C,
                                                       csr_col_ind_C,
                                                       csr_val_C,
                                                       base_A,
                                                       base_B,
                                                       descr_C->base,
                                                       base_D,
                                                       info_C->csrgemm_info->mul,
                                                       info_C->csrgemm_info->add));
        }
#endif

        // Group 7: more than 4096 non-zeros per row
        if(h_group_size[7] > 0)
        {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
            I* workspace_B = nullptr;

            if(info_C->csrgemm_info->mul == true)
            {
                // Allocate additional buffer for C = alpha * A * B
                RETURN_IF_HIP_ERROR(hipMalloc((void**)&workspace_B, sizeof(I) * nnz_A));
            }

            hipLaunchKernelGGL(
                (csrgemm_fill_block_per_row_multipass_host_pointer<CSRGEMM_DIM,
                                                                   CSRGEMM_SUB,
                                                                   CSRGEMM_CHUNKSIZE>),
                dim3(h_group_size[7]),
                dim3(CSRGEMM_DIM),
                0,
                stream,
                n,
                &d_group_offset[7],
                d_perm,
                (info_C->csrgemm_info->mul == true) ? *alpha : static_cast<T>(0),
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                (info_C->csrgemm_info->add == true) ? *beta : static_cast<T>(0),
                csr_row_ptr_D,
                csr_col_ind_D,
                csr_val_D,
                csr_row_ptr_C,
                csr_col_ind_C,
                csr_val_C,
                workspace_B,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add);

            if(info_C->csrgemm_info->mul == true)
            {
                RETURN_IF_HIP_ERROR(hipFree(workspace_B));
            }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
        }
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
static inline rocsparse_status rocsparse_csrgemm_multadd_template(rocsparse_handle          handle,
                                                                  rocsparse_operation       trans_A,
                                                                  rocsparse_operation       trans_B,
                                                                  J                         m,
                                                                  J                         n,
                                                                  J                         k,
                                                                  const T*                  alpha,
                                                                  const rocsparse_mat_descr descr_A,
                                                                  I                         nnz_A,
                                                                  const T* csr_val_A,
                                                                  const I* csr_row_ptr_A,
                                                                  const J* csr_col_ind_A,
                                                                  const rocsparse_mat_descr descr_B,
                                                                  I                         nnz_B,
                                                                  const T* csr_val_B,
                                                                  const I* csr_row_ptr_B,
                                                                  const J* csr_col_ind_B,
                                                                  const T* beta,
                                                                  const rocsparse_mat_descr descr_D,
                                                                  I                         nnz_D,
                                                                  const T* csr_val_D,
                                                                  const I* csr_row_ptr_D,
                                                                  const J* csr_col_ind_D,
                                                                  const rocsparse_mat_descr descr_C,
                                                                  T*       csr_val_C,
                                                                  const I* csr_row_ptr_C,
                                                                  J*       csr_col_ind_C,
                                                                  const rocsparse_mat_info info_C,
                                                                  void* temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || csr_row_ptr_A == nullptr || descr_B == nullptr
       || csr_row_ptr_B == nullptr || descr_D == nullptr || csr_row_ptr_D == nullptr
       || descr_C == nullptr || csr_row_ptr_C == nullptr || temp_buffer == nullptr
       || alpha == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_A == nullptr && csr_col_ind_A != nullptr)
       || (csr_val_A != nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_B == nullptr && csr_col_ind_B != nullptr)
       || (csr_val_B != nullptr && csr_col_ind_B == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_C == nullptr && csr_col_ind_C != nullptr)
       || (csr_val_C != nullptr && csr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_D == nullptr && csr_col_ind_D != nullptr)
       || (csr_val_D != nullptr && csr_col_ind_D == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && (csr_col_ind_A == nullptr && csr_val_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && (csr_col_ind_B == nullptr && csr_val_B == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_D != 0 && (csr_col_ind_D == nullptr && csr_val_D == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_val_C == nullptr && csr_col_ind_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(
            hipMemcpy(&end, &csr_row_ptr_C[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&start, &csr_row_ptr_C[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        rocsparse_int nnz = (end - start);

        if(nnz != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Quick return if possible

    // m == 0 || n == 0 - do nothing
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // k == 0 || nnz_A == 0 || nnz_B == 0 - scale D with beta
    if(k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_csrgemm_scal_template(handle,
                                               m,
                                               n,
                                               beta,
                                               descr_D,
                                               nnz_D,
                                               csr_val_D,
                                               csr_row_ptr_D,
                                               csr_col_ind_D,
                                               descr_C,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               info_C,
                                               temp_buffer);
    }

    if((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none))
    {
        return rocsparse_status_not_implemented;
    }

    // nnz_D == 0 - compute alpha * A * B
    if(nnz_D == 0)
    {
        return rocsparse_csrgemm_mult_template(handle,
                                               trans_A,
                                               trans_B,
                                               m,
                                               n,
                                               k,
                                               alpha,
                                               descr_A,
                                               nnz_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               descr_B,
                                               nnz_B,
                                               csr_val_B,
                                               csr_row_ptr_B,
                                               csr_col_ind_B,
                                               descr_C,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               info_C,
                                               temp_buffer);
    }

    // Perform gemm calculation
    return rocsparse_csrgemm_calc_template(handle,
                                           trans_A,
                                           trans_B,
                                           m,
                                           n,
                                           k,
                                           alpha,
                                           descr_A,
                                           nnz_A,
                                           csr_val_A,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           descr_B,
                                           nnz_B,
                                           csr_val_B,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           beta,
                                           descr_D,
                                           nnz_D,
                                           csr_val_D,
                                           csr_row_ptr_D,
                                           csr_col_ind_D,
                                           descr_C,
                                           csr_val_C,
                                           csr_row_ptr_C,
                                           csr_col_ind_C,
                                           info_C,
                                           temp_buffer);
}

template <typename I, typename J, typename T>
static inline rocsparse_status rocsparse_csrgemm_mult_template(rocsparse_handle          handle,
                                                               rocsparse_operation       trans_A,
                                                               rocsparse_operation       trans_B,
                                                               J                         m,
                                                               J                         n,
                                                               J                         k,
                                                               const T*                  alpha,
                                                               const rocsparse_mat_descr descr_A,
                                                               I                         nnz_A,
                                                               const T*                  csr_val_A,
                                                               const I* csr_row_ptr_A,
                                                               const J* csr_col_ind_A,
                                                               const rocsparse_mat_descr descr_B,
                                                               I                         nnz_B,
                                                               const T*                  csr_val_B,
                                                               const I* csr_row_ptr_B,
                                                               const J* csr_col_ind_B,
                                                               const rocsparse_mat_descr descr_C,
                                                               T*                        csr_val_C,
                                                               const I* csr_row_ptr_C,
                                                               J*       csr_col_ind_C,
                                                               const rocsparse_mat_info info_C,
                                                               void*                    temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_A == nullptr || csr_row_ptr_A == nullptr || descr_B == nullptr
       || csr_row_ptr_B == nullptr || descr_C == nullptr || csr_row_ptr_C == nullptr
       || temp_buffer == nullptr || alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_A == nullptr && csr_col_ind_A != nullptr)
       || (csr_val_A != nullptr && csr_col_ind_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_B == nullptr && csr_col_ind_B != nullptr)
       || (csr_val_B != nullptr && csr_col_ind_B == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_C == nullptr && csr_col_ind_C != nullptr)
       || (csr_val_C != nullptr && csr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_A != 0 && (csr_col_ind_A == nullptr && csr_val_A == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_B != 0 && (csr_col_ind_B == nullptr && csr_val_B == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_val_C == nullptr && csr_col_ind_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(
            hipMemcpy(&end, &csr_row_ptr_C[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&start, &csr_row_ptr_C[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        rocsparse_int nnz = (end - start);

        if(nnz != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check matrix type
    if(descr_A->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_B->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_status_success;
    }

    if((trans_A != rocsparse_operation_none) || (trans_B != rocsparse_operation_none))
    {
        return rocsparse_status_not_implemented;
    }

    // Perform gemm calculation
    return rocsparse_csrgemm_calc_template(handle,
                                           trans_A,
                                           trans_B,
                                           m,
                                           n,
                                           k,
                                           alpha,
                                           descr_A,
                                           nnz_A,
                                           csr_val_A,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           descr_B,
                                           nnz_B,
                                           csr_val_B,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           (const T*)nullptr,
                                           nullptr,
                                           (I)0,
                                           (const T*)nullptr,
                                           (const I*)nullptr,
                                           (const J*)nullptr,
                                           descr_C,
                                           csr_val_C,
                                           csr_row_ptr_C,
                                           csr_col_ind_C,
                                           info_C,
                                           temp_buffer);
}

template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void csrgemm_copy_scale_host_pointer(
    I size, T alpha, const T* __restrict__ in, T* __restrict__ out)
{
    csrgemm_copy_scale_device<BLOCKSIZE>(size, alpha, in, out);
}

template <unsigned int BLOCKSIZE, typename I, typename T>
__launch_bounds__(BLOCKSIZE) ROCSPARSE_KERNEL void csrgemm_copy_scale_device_pointer(
    I size, const T* __restrict__ alpha, const T* __restrict__ in, T* __restrict__ out)
{
    csrgemm_copy_scale_device<BLOCKSIZE>(size, *alpha, in, out);
}

template <typename I, typename J, typename T>
static inline rocsparse_status rocsparse_csrgemm_scal_template(rocsparse_handle          handle,
                                                               J                         m,
                                                               J                         n,
                                                               const T*                  beta,
                                                               const rocsparse_mat_descr descr_D,
                                                               I                         nnz_D,
                                                               const T*                  csr_val_D,
                                                               const I* csr_row_ptr_D,
                                                               const J* csr_col_ind_D,
                                                               const rocsparse_mat_descr descr_C,
                                                               T*                        csr_val_C,
                                                               const I* csr_row_ptr_C,
                                                               J*       csr_col_ind_C,
                                                               const rocsparse_mat_info info_C,
                                                               void*                    temp_buffer)
{
    // Check for valid info structure
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || nnz_D < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check valid pointers
    if(descr_D == nullptr || csr_row_ptr_D == nullptr || descr_C == nullptr
       || csr_row_ptr_C == nullptr || temp_buffer == nullptr || beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_C == nullptr && csr_col_ind_C != nullptr)
       || (csr_val_C != nullptr && csr_col_ind_C == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    // value arrays and column indices arrays must both be null (zero matrix) or both not null
    if((csr_val_D == nullptr && csr_col_ind_D != nullptr)
       || (csr_val_D != nullptr && csr_col_ind_D == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_D != 0 && (csr_col_ind_D == nullptr && csr_val_D == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(csr_val_C == nullptr && csr_col_ind_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(
            hipMemcpy(&end, &csr_row_ptr_C[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(&start, &csr_row_ptr_C[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        rocsparse_int nnz = (end - start);

        if(nnz != 0)
        {
            return rocsparse_status_invalid_pointer;
        }
    }

    // Check matrix type
    if(descr_C->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }
    if(descr_D->type != rocsparse_matrix_type_general)
    {
        return rocsparse_status_not_implemented;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz_D == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define CSRGEMM_DIM 1024
    dim3 csrgemm_blocks((nnz_D - 1) / CSRGEMM_DIM + 1);
    dim3 csrgemm_threads(CSRGEMM_DIM);

    // Copy column entries, if D != C
    if(csr_col_ind_C != csr_col_ind_D)
    {
        hipLaunchKernelGGL((csrgemm_copy<CSRGEMM_DIM>),
                           csrgemm_blocks,
                           csrgemm_threads,
                           0,
                           stream,
                           nnz_D,
                           csr_col_ind_D,
                           csr_col_ind_C,
                           descr_D->base,
                           descr_C->base);
    }

    // Scale the matrix
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((csrgemm_copy_scale_device_pointer<CSRGEMM_DIM>),
                           csrgemm_blocks,
                           csrgemm_threads,
                           0,
                           stream,
                           nnz_D,
                           beta,
                           csr_val_D,
                           csr_val_C);
    }
    else
    {
        hipLaunchKernelGGL((csrgemm_copy_scale_host_pointer<CSRGEMM_DIM>),
                           csrgemm_blocks,
                           csrgemm_threads,
                           0,
                           stream,
                           nnz_D,
                           *beta,
                           csr_val_D,
                           csr_val_C);
    }
#undef CSRGEMM_DIM

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csrgemm_template(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         m,
                                            J                         n,
                                            J                         k,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            I                         nnz_A,
                                            const T*                  csr_val_A,
                                            const I*                  csr_row_ptr_A,
                                            const J*                  csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            I                         nnz_B,
                                            const T*                  csr_val_B,
                                            const I*                  csr_row_ptr_B,
                                            const J*                  csr_col_ind_B,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_D,
                                            I                         nnz_D,
                                            const T*                  csr_val_D,
                                            const I*                  csr_row_ptr_D,
                                            const J*                  csr_col_ind_D,
                                            const rocsparse_mat_descr descr_C,
                                            T*                        csr_val_C,
                                            const I*                  csr_row_ptr_C,
                                            J*                        csr_col_ind_C,
                                            const rocsparse_mat_info  info_C,
                                            void*                     temp_buffer)
{
    // Check for valid handle and info structure
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Check for valid rocsparse_mat_info
    if(info_C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsrgemm"),
              trans_A,
              trans_B,
              m,
              n,
              k,
              LOG_TRACE_SCALAR_VALUE(handle, alpha),
              (const void*&)descr_A,
              nnz_A,
              (const void*&)csr_val_A,
              (const void*&)csr_row_ptr_A,
              (const void*&)csr_col_ind_A,
              (const void*&)descr_B,
              nnz_B,
              (const void*&)csr_val_B,
              (const void*&)csr_row_ptr_B,
              (const void*&)csr_col_ind_B,
              LOG_TRACE_SCALAR_VALUE(handle, beta),
              (const void*&)descr_D,
              nnz_D,
              (const void*&)csr_val_D,
              (const void*&)csr_row_ptr_D,
              (const void*&)csr_col_ind_D,
              (const void*&)descr_C,
              (const void*&)csr_val_C,
              (const void*&)csr_row_ptr_C,
              (const void*&)csr_col_ind_C,
              (const void*&)info_C,
              (const void*&)temp_buffer);

    log_bench(handle,
              "./rocsparse-bench -f csrgemm -r",
              replaceX<T>("X"),
              "--mtx <matrix.mtx> ",
              "--alpha",
              LOG_BENCH_SCALAR_VALUE(handle, alpha),
              "--beta",
              LOG_BENCH_SCALAR_VALUE(handle, beta));

    // Check operation
    if(rocsparse_enum_utils::is_invalid(trans_A))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(trans_B))
    {
        return rocsparse_status_invalid_value;
    }

    // Check valid sizes
    if(m < 0 || n < 0 || k < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check for valid rocsparse_csrgemm_info
    if(info_C->csrgemm_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Either mult, add or multadd need to be performed
    if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == true)
    {
        // C = alpha * A * B + beta * D
        return rocsparse_csrgemm_multadd_template(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_val_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_val_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_val_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  descr_C,
                                                  csr_val_C,
                                                  csr_row_ptr_C,
                                                  csr_col_ind_C,
                                                  info_C,
                                                  temp_buffer);
    }
    else if(info_C->csrgemm_info->mul == true && info_C->csrgemm_info->add == false)
    {
        // C = alpha * A * B
        return rocsparse_csrgemm_mult_template(handle,
                                               trans_A,
                                               trans_B,
                                               m,
                                               n,
                                               k,
                                               alpha,
                                               descr_A,
                                               nnz_A,
                                               csr_val_A,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               descr_B,
                                               nnz_B,
                                               csr_val_B,
                                               csr_row_ptr_B,
                                               csr_col_ind_B,
                                               descr_C,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               info_C,
                                               temp_buffer);
    }
    else if(info_C->csrgemm_info->mul == false && info_C->csrgemm_info->add == true)
    {
        // C = beta * D
        return rocsparse_csrgemm_scal_template(handle,
                                               m,
                                               n,
                                               beta,
                                               descr_D,
                                               nnz_D,
                                               csr_val_D,
                                               csr_row_ptr_D,
                                               csr_col_ind_D,
                                               descr_C,
                                               csr_val_C,
                                               csr_row_ptr_C,
                                               csr_col_ind_C,
                                               info_C,
                                               temp_buffer);
    }
    else
    {
        // C = 0
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                       \
    template rocsparse_status rocsparse_csrgemm_template<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                                      \
        rocsparse_operation       trans_A,                                     \
        rocsparse_operation       trans_B,                                     \
        JTYPE                     m,                                           \
        JTYPE                     n,                                           \
        JTYPE                     k,                                           \
        const TTYPE*              alpha,                                       \
        const rocsparse_mat_descr descr_A,                                     \
        ITYPE                     nnz_A,                                       \
        const TTYPE*              csr_val_A,                                   \
        const ITYPE*              csr_row_ptr_A,                               \
        const JTYPE*              csr_col_ind_A,                               \
        const rocsparse_mat_descr descr_B,                                     \
        ITYPE                     nnz_B,                                       \
        const TTYPE*              csr_val_B,                                   \
        const ITYPE*              csr_row_ptr_B,                               \
        const JTYPE*              csr_col_ind_B,                               \
        const TTYPE*              beta,                                        \
        const rocsparse_mat_descr descr_D,                                     \
        ITYPE                     nnz_D,                                       \
        const TTYPE*              csr_val_D,                                   \
        const ITYPE*              csr_row_ptr_D,                               \
        const JTYPE*              csr_col_ind_D,                               \
        const rocsparse_mat_descr descr_C,                                     \
        TTYPE*                    csr_val_C,                                   \
        const ITYPE*              csr_row_ptr_C,                               \
        JTYPE*                    csr_col_ind_C,                               \
        const rocsparse_mat_info  info_C,                                      \
        void*                     temp_buffer);

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

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_operation       trans_A,       \
                                     rocsparse_operation       trans_B,       \
                                     rocsparse_int             m,             \
                                     rocsparse_int             n,             \
                                     rocsparse_int             k,             \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnz_A,         \
                                     const TYPE*               csr_val_A,     \
                                     const rocsparse_int*      csr_row_ptr_A, \
                                     const rocsparse_int*      csr_col_ind_A, \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnz_B,         \
                                     const TYPE*               csr_val_B,     \
                                     const rocsparse_int*      csr_row_ptr_B, \
                                     const rocsparse_int*      csr_col_ind_B, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_D,       \
                                     rocsparse_int             nnz_D,         \
                                     const TYPE*               csr_val_D,     \
                                     const rocsparse_int*      csr_row_ptr_D, \
                                     const rocsparse_int*      csr_col_ind_D, \
                                     const rocsparse_mat_descr descr_C,       \
                                     TYPE*                     csr_val_C,     \
                                     const rocsparse_int*      csr_row_ptr_C, \
                                     rocsparse_int*            csr_col_ind_C, \
                                     const rocsparse_mat_info  info_C,        \
                                     void*                     temp_buffer)   \
    {                                                                         \
        return rocsparse_csrgemm_template(handle,                             \
                                          trans_A,                            \
                                          trans_B,                            \
                                          m,                                  \
                                          n,                                  \
                                          k,                                  \
                                          alpha,                              \
                                          descr_A,                            \
                                          nnz_A,                              \
                                          csr_val_A,                          \
                                          csr_row_ptr_A,                      \
                                          csr_col_ind_A,                      \
                                          descr_B,                            \
                                          nnz_B,                              \
                                          csr_val_B,                          \
                                          csr_row_ptr_B,                      \
                                          csr_col_ind_B,                      \
                                          beta,                               \
                                          descr_D,                            \
                                          nnz_D,                              \
                                          csr_val_D,                          \
                                          csr_row_ptr_D,                      \
                                          csr_col_ind_D,                      \
                                          descr_C,                            \
                                          csr_val_C,                          \
                                          csr_row_ptr_C,                      \
                                          csr_col_ind_C,                      \
                                          info_C,                             \
                                          temp_buffer);                       \
    }

C_IMPL(rocsparse_scsrgemm, float);
C_IMPL(rocsparse_dcsrgemm, double);
C_IMPL(rocsparse_ccsrgemm, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrgemm, rocsparse_double_complex);
#undef C_IMPL

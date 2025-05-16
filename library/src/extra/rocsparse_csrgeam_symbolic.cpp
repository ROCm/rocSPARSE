/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <map>
#include <sstream>

#include "common.h"
#include "control.h"
#include "rocsparse_common.h"
#include "to_string.hpp"
#include "utility.h"

#include "rocsparse_csrgeam_symbolic.hpp"

#include "csrgeam_symbolic_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgeam_fill_symbolic_multipass_kernel(int64_t m,
                                                int64_t n,
                                                const I* __restrict__ csr_row_ptr_A,
                                                const J* __restrict__ csr_col_ind_A,
                                                const I* __restrict__ csr_row_ptr_B,
                                                const J* __restrict__ csr_col_ind_B,
                                                const I* __restrict__ csr_row_ptr_C,
                                                J* __restrict__ csr_col_ind_C,
                                                rocsparse_index_base idx_base_A,
                                                rocsparse_index_base idx_base_B,
                                                rocsparse_index_base idx_base_C)
    {
        rocsparse::csrgeam_fill_symbolic_multipass_device<BLOCKSIZE, WFSIZE>(m,
                                                                             n,
                                                                             csr_row_ptr_A,
                                                                             csr_col_ind_A,
                                                                             csr_row_ptr_B,
                                                                             csr_col_ind_B,
                                                                             csr_row_ptr_C,
                                                                             csr_col_ind_C,
                                                                             idx_base_A,
                                                                             idx_base_B,
                                                                             idx_base_C);
    }
}

namespace rocsparse
{
    static rocsparse_status csrgeam_symbolic_quickreturn(rocsparse_handle             handle,
                                                         rocsparse_operation          trans_A,
                                                         rocsparse_operation          trans_B,
                                                         int64_t                      m,
                                                         int64_t                      n,
                                                         const rocsparse_mat_descr    descr_A,
                                                         int64_t                      nnz_A,
                                                         const void*                  csr_row_ptr_A,
                                                         const void*                  csr_col_ind_A,
                                                         const rocsparse_mat_descr    descr_B,
                                                         int64_t                      nnz_B,
                                                         const void*                  csr_row_ptr_B,
                                                         const void*                  csr_col_ind_B,
                                                         const rocsparse_mat_descr    descr_C,
                                                         const void*                  csr_row_ptr_C,
                                                         void*                        csr_col_ind_C,
                                                         const rocsparse_spgeam_descr descr,
                                                         void*                        temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
        {
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    template <typename I, typename J>
    static rocsparse_status csrgeam_symbolic_dispatch(rocsparse_handle             handle,
                                                      rocsparse_operation          trans_A,
                                                      rocsparse_operation          trans_B,
                                                      int64_t                      m,
                                                      int64_t                      n,
                                                      const rocsparse_mat_descr    descr_A,
                                                      int64_t                      nnz_A,
                                                      const I*                     csr_row_ptr_A,
                                                      const J*                     csr_col_ind_A,
                                                      const rocsparse_mat_descr    descr_B,
                                                      int64_t                      nnz_B,
                                                      const I*                     csr_row_ptr_B,
                                                      const J*                     csr_col_ind_B,
                                                      const rocsparse_mat_descr    descr_C,
                                                      const I*                     csr_row_ptr_C,
                                                      J*                           csr_col_ind_C,
                                                      const rocsparse_spgeam_descr descr)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        if(descr == nullptr || (descr->multiplying_by_alpha() && descr->multiplying_by_beta()))
        {
            // Pointer mode device
#define CSRGEAM_DIM 256
            if(handle->wavefront_size == 32)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_fill_symbolic_multipass_kernel<CSRGEAM_DIM, 32>),
                    dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    csr_row_ptr_C,
                    csr_col_ind_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_fill_symbolic_multipass_kernel<CSRGEAM_DIM, 64>),
                    dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    csr_row_ptr_C,
                    csr_col_ind_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base);
            }
#undef CSRGEAM_DIM
        }
        else if(descr->multiplying_by_alpha() && !descr->multiplying_by_beta())
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::copy(
                handle, nnz_A, csr_col_ind_A, csr_col_ind_C, descr_A->base, descr_C->base));
        }
        else if(!descr->multiplying_by_alpha() && descr->multiplying_by_beta())
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::copy(
                handle, nnz_B, csr_col_ind_B, csr_col_ind_C, descr_B->base, descr_C->base));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J>
    static rocsparse_status csrgeam_symbolic_core(rocsparse_handle             handle,
                                                  rocsparse_operation          trans_A,
                                                  rocsparse_operation          trans_B,
                                                  int64_t                      m,
                                                  int64_t                      n,
                                                  const rocsparse_mat_descr    descr_A,
                                                  int64_t                      nnz_A,
                                                  const I*                     csr_row_ptr_A,
                                                  const J*                     csr_col_ind_A,
                                                  const rocsparse_mat_descr    descr_B,
                                                  int64_t                      nnz_B,
                                                  const I*                     csr_row_ptr_B,
                                                  const J*                     csr_col_ind_B,
                                                  const rocsparse_mat_descr    descr_C,
                                                  const I*                     csr_row_ptr_C,
                                                  J*                           csr_col_ind_C,
                                                  const rocsparse_spgeam_descr descr,
                                                  void*                        temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_symbolic_dispatch(handle,
                                                                       trans_A,
                                                                       trans_B,
                                                                       m,
                                                                       n,
                                                                       descr_A,
                                                                       nnz_A,
                                                                       csr_row_ptr_A,
                                                                       csr_col_ind_A,
                                                                       descr_B,
                                                                       nnz_B,
                                                                       csr_row_ptr_B,
                                                                       csr_col_ind_B,
                                                                       descr_C,
                                                                       csr_row_ptr_C,
                                                                       csr_col_ind_C,
                                                                       descr));
        return rocsparse_status_success;
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgeam_symbolic_template(rocsparse_handle             handle,
                                                      rocsparse_operation          trans_A,
                                                      rocsparse_operation          trans_B,
                                                      int64_t                      m,
                                                      int64_t                      n,
                                                      const rocsparse_mat_descr    descr_A,
                                                      int64_t                      nnz_A,
                                                      const void*                  csr_row_ptr_A,
                                                      const void*                  csr_col_ind_A,
                                                      const rocsparse_mat_descr    descr_B,
                                                      int64_t                      nnz_B,
                                                      const void*                  csr_row_ptr_B,
                                                      const void*                  csr_col_ind_B,
                                                      const rocsparse_mat_descr    descr_C,
                                                      const void*                  csr_row_ptr_C,
                                                      void*                        csr_col_ind_C,
                                                      const rocsparse_spgeam_descr descr,
                                                      void*                        temp_buffer)
{
    const rocsparse_status status = rocsparse::csrgeam_symbolic_quickreturn(handle,
                                                                            trans_A,
                                                                            trans_B,
                                                                            m,
                                                                            n,
                                                                            descr_A,
                                                                            nnz_A,
                                                                            (const I*)csr_row_ptr_A,
                                                                            (const J*)csr_col_ind_A,
                                                                            descr_B,
                                                                            nnz_B,
                                                                            (const I*)csr_row_ptr_B,
                                                                            (const J*)csr_col_ind_B,
                                                                            descr_C,
                                                                            (const I*)csr_row_ptr_C,
                                                                            (J*)csr_col_ind_C,
                                                                            descr,
                                                                            temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_symbolic_core(handle,
                                                               trans_A,
                                                               trans_B,
                                                               m,
                                                               n,
                                                               descr_A,
                                                               nnz_A,
                                                               (const I*)csr_row_ptr_A,
                                                               (const J*)csr_col_ind_A,
                                                               descr_B,
                                                               nnz_B,
                                                               (const I*)csr_row_ptr_B,
                                                               (const J*)csr_col_ind_B,
                                                               descr_C,
                                                               (const I*)csr_row_ptr_C,
                                                               (J*)csr_col_ind_C,
                                                               descr,
                                                               temp_buffer));
    return rocsparse_status_success;
}

namespace rocsparse
{
    typedef rocsparse_status (*csrgeam_symbolic_t)(rocsparse_handle             handle,
                                                   rocsparse_operation          trans_A,
                                                   rocsparse_operation          trans_B,
                                                   int64_t                      m,
                                                   int64_t                      n,
                                                   const rocsparse_mat_descr    descr_A,
                                                   int64_t                      nnz_A,
                                                   const void*                  csr_row_ptr_A,
                                                   const void*                  csr_col_ind_A,
                                                   const rocsparse_mat_descr    descr_B,
                                                   int64_t                      nnz_B,
                                                   const void*                  csr_row_ptr_B,
                                                   const void*                  csr_col_ind_B,
                                                   const rocsparse_mat_descr    descr_C,
                                                   const void*                  csr_row_ptr_C,
                                                   void*                        csr_col_ind_C,
                                                   const rocsparse_spgeam_descr descr,
                                                   void*                        temp_buffer);

    using csrgeam_symbolic_tuple = std::tuple<rocsparse_indextype, rocsparse_indextype>;

    // clang-format off
#define CSRGEAM_SYMBOLIC_CONFIG(I, J)                                           \
    {csrgeam_symbolic_tuple(I, J),                                              \
     csrgeam_symbolic_template<typename rocsparse::indextype_traits<I>::type_t, \
                               typename rocsparse::indextype_traits<J>::type_t>}
    // clang-format on

    static const std::map<csrgeam_symbolic_tuple, csrgeam_symbolic_t> s_csrgeam_symbolic_dispatch{
        {CSRGEAM_SYMBOLIC_CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i32),
         CSRGEAM_SYMBOLIC_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i32),
         CSRGEAM_SYMBOLIC_CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i64)}};

    static rocsparse_status csrgeam_symbolic_find(csrgeam_symbolic_t* function_,
                                                  rocsparse_indextype i_type_,
                                                  rocsparse_indextype j_type_)
    {

        const auto& it = rocsparse::s_csrgeam_symbolic_dispatch.find(
            rocsparse::csrgeam_symbolic_tuple(i_type_, j_type_));

        if(it != rocsparse::s_csrgeam_symbolic_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << ", i_type: " << rocsparse::to_string(i_type_) << std::endl
                      << ", j_type: " << rocsparse::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_csrgeam_symbolic_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                std::cout << std::endl
                          << std::endl
                          << ", i_type: " << rocsparse::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::to_string(j_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << ", i_type: " << rocsparse::to_string(i_type_)
                 << ", j_type: " << rocsparse::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }

}

rocsparse_status rocsparse::csrgeam_symbolic(rocsparse_handle             handle,
                                             rocsparse_operation          trans_A,
                                             rocsparse_operation          trans_B,
                                             int64_t                      m,
                                             int64_t                      n,
                                             const rocsparse_mat_descr    descr_A,
                                             int64_t                      nnz_A,
                                             rocsparse_indextype          csr_row_ptr_A_indextype,
                                             const void*                  csr_row_ptr_A,
                                             rocsparse_indextype          csr_col_ind_A_indextype,
                                             const void*                  csr_col_ind_A,
                                             const rocsparse_mat_descr    descr_B,
                                             int64_t                      nnz_B,
                                             rocsparse_indextype          csr_row_ptr_B_indextype,
                                             const void*                  csr_row_ptr_B,
                                             rocsparse_indextype          csr_col_ind_B_indextype,
                                             const void*                  csr_col_ind_B,
                                             const rocsparse_mat_descr    descr_C,
                                             rocsparse_indextype          csr_row_ptr_C_indextype,
                                             const void*                  csr_row_ptr_C,
                                             rocsparse_indextype          csr_col_ind_C_indextype,
                                             void*                        csr_col_ind_C,
                                             const rocsparse_spgeam_descr descr,
                                             void*                        temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::csrgeam_symbolic_t f;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrgeam_symbolic_find(&f, csr_row_ptr_C_indextype, csr_col_ind_C_indextype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                trans_A,
                                trans_B,
                                m,
                                n,
                                descr_A,
                                nnz_A,
                                csr_row_ptr_A,
                                csr_col_ind_A,
                                descr_B,
                                nnz_B,
                                csr_row_ptr_B,
                                csr_col_ind_B,
                                descr_C,
                                csr_row_ptr_C,
                                csr_col_ind_C,
                                descr,
                                temp_buffer));

    return rocsparse_status_success;
}

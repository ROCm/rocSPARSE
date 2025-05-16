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

#include "rocsparse_csrgeam_numeric.hpp"

#include "csrgeam_numeric_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgeam_fill_numeric_multipass_kernel(int64_t m,
                                               int64_t n,
                                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                               const I* __restrict__ csr_row_ptr_A,
                                               const J* __restrict__ csr_col_ind_A,
                                               const T* __restrict__ csr_val_A,
                                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                               const I* __restrict__ csr_row_ptr_B,
                                               const J* __restrict__ csr_col_ind_B,
                                               const T* __restrict__ csr_val_B,
                                               const I* __restrict__ csr_row_ptr_C,
                                               T* __restrict__ csr_val_C,
                                               rocsparse_index_base idx_base_A,
                                               rocsparse_index_base idx_base_B,
                                               rocsparse_index_base idx_base_C,
                                               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        rocsparse::csrgeam_fill_numeric_multipass_device<BLOCKSIZE, WFSIZE>(m,
                                                                            n,
                                                                            alpha,
                                                                            csr_row_ptr_A,
                                                                            csr_col_ind_A,
                                                                            csr_val_A,
                                                                            beta,
                                                                            csr_row_ptr_B,
                                                                            csr_col_ind_B,
                                                                            csr_val_B,
                                                                            csr_row_ptr_C,
                                                                            csr_val_C,
                                                                            idx_base_A,
                                                                            idx_base_B,
                                                                            idx_base_C);
    }
}

namespace rocsparse
{
    static rocsparse_status csrgeam_numeric_quickreturn(rocsparse_handle          handle,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        int64_t                   m,
                                                        int64_t                   n,
                                                        const void*               alpha_device_host,
                                                        const rocsparse_mat_descr descr_A,
                                                        int64_t                   nnz_A,
                                                        const void*               csr_val_A,
                                                        const void*               csr_row_ptr_A,
                                                        const void*               csr_col_ind_A,
                                                        const void*               beta_device_host,
                                                        const rocsparse_mat_descr descr_B,
                                                        int64_t                   nnz_B,
                                                        const void*               csr_val_B,
                                                        const void*               csr_row_ptr_B,
                                                        const void*               csr_col_ind_B,
                                                        const rocsparse_mat_descr descr_C,
                                                        void*                     csr_val_C,
                                                        const void*               csr_row_ptr_C,
                                                        const void*               csr_col_ind_C,
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

    template <typename I, typename J, typename T>
    static rocsparse_status csrgeam_numeric_dispatch(rocsparse_handle             handle,
                                                     rocsparse_operation          trans_A,
                                                     rocsparse_operation          trans_B,
                                                     int64_t                      m,
                                                     int64_t                      n,
                                                     const T*                     alpha_device_host,
                                                     const rocsparse_mat_descr    descr_A,
                                                     int64_t                      nnz_A,
                                                     const T*                     csr_val_A,
                                                     const I*                     csr_row_ptr_A,
                                                     const J*                     csr_col_ind_A,
                                                     const T*                     beta_device_host,
                                                     const rocsparse_mat_descr    descr_B,
                                                     int64_t                      nnz_B,
                                                     const T*                     csr_val_B,
                                                     const I*                     csr_row_ptr_B,
                                                     const J*                     csr_col_ind_B,
                                                     const rocsparse_mat_descr    descr_C,
                                                     T*                           csr_val_C,
                                                     const I*                     csr_row_ptr_C,
                                                     const J*                     csr_col_ind_C,
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
                    (rocsparse::csrgeam_fill_numeric_multipass_kernel<CSRGEAM_DIM, 32>),
                    dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_val_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    csr_val_B,
                    csr_row_ptr_C,
                    csr_val_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_fill_numeric_multipass_kernel<CSRGEAM_DIM, 64>),
                    dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_val_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    csr_val_B,
                    csr_row_ptr_C,
                    csr_val_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
#undef CSRGEAM_DIM
        }
        else if(descr->multiplying_by_alpha() && !descr->multiplying_by_beta())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::copy_and_scale(handle, nnz_A, csr_val_A, csr_val_C, alpha_device_host));
        }
        else if(!descr->multiplying_by_alpha() && descr->multiplying_by_beta())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::copy_and_scale(handle, nnz_B, csr_val_B, csr_val_C, beta_device_host));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static rocsparse_status csrgeam_numeric_core(rocsparse_handle             handle,
                                                 rocsparse_operation          trans_A,
                                                 rocsparse_operation          trans_B,
                                                 int64_t                      m,
                                                 int64_t                      n,
                                                 const T*                     alpha_device_host,
                                                 const rocsparse_mat_descr    descr_A,
                                                 int64_t                      nnz_A,
                                                 const T*                     csr_val_A,
                                                 const I*                     csr_row_ptr_A,
                                                 const J*                     csr_col_ind_A,
                                                 const T*                     beta_device_host,
                                                 const rocsparse_mat_descr    descr_B,
                                                 int64_t                      nnz_B,
                                                 const T*                     csr_val_B,
                                                 const I*                     csr_row_ptr_B,
                                                 const J*                     csr_col_ind_B,
                                                 const rocsparse_mat_descr    descr_C,
                                                 T*                           csr_val_C,
                                                 const I*                     csr_row_ptr_C,
                                                 const J*                     csr_col_ind_C,
                                                 const rocsparse_spgeam_descr descr,
                                                 void*                        temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_numeric_dispatch(handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      m,
                                                                      n,
                                                                      alpha_device_host,
                                                                      descr_A,
                                                                      nnz_A,
                                                                      csr_val_A,
                                                                      csr_row_ptr_A,
                                                                      csr_col_ind_A,
                                                                      beta_device_host,
                                                                      descr_B,
                                                                      nnz_B,
                                                                      csr_val_B,
                                                                      csr_row_ptr_B,
                                                                      csr_col_ind_B,
                                                                      descr_C,
                                                                      csr_val_C,
                                                                      csr_row_ptr_C,
                                                                      csr_col_ind_C,
                                                                      descr));
        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csrgeam_numeric_template(rocsparse_handle             handle,
                                                     rocsparse_operation          trans_A,
                                                     rocsparse_operation          trans_B,
                                                     int64_t                      m,
                                                     int64_t                      n,
                                                     const void*                  alpha_device_host,
                                                     const rocsparse_mat_descr    descr_A,
                                                     int64_t                      nnz_A,
                                                     const void*                  csr_val_A,
                                                     const void*                  csr_row_ptr_A,
                                                     const void*                  csr_col_ind_A,
                                                     const void*                  beta_device_host,
                                                     const rocsparse_mat_descr    descr_B,
                                                     int64_t                      nnz_B,
                                                     const void*                  csr_val_B,
                                                     const void*                  csr_row_ptr_B,
                                                     const void*                  csr_col_ind_B,
                                                     const rocsparse_mat_descr    descr_C,
                                                     void*                        csr_val_C,
                                                     const void*                  csr_row_ptr_C,
                                                     const void*                  csr_col_ind_C,
                                                     const rocsparse_spgeam_descr descr,
                                                     void*                        temp_buffer)
{
    const rocsparse_status status
        = rocsparse::csrgeam_numeric_quickreturn(handle,
                                                 trans_A,
                                                 trans_B,
                                                 m,
                                                 n,
                                                 (const T*)alpha_device_host,
                                                 descr_A,
                                                 nnz_A,
                                                 (const T*)csr_val_A,
                                                 (const I*)csr_row_ptr_A,
                                                 (const J*)csr_col_ind_A,
                                                 (const T*)beta_device_host,
                                                 descr_B,
                                                 nnz_B,
                                                 (const T*)csr_val_B,
                                                 (const I*)csr_row_ptr_B,
                                                 (const J*)csr_col_ind_B,
                                                 descr_C,
                                                 (T*)csr_val_C,
                                                 (const I*)csr_row_ptr_C,
                                                 (const J*)csr_col_ind_C,
                                                 descr,
                                                 temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_numeric_core(handle,
                                                              trans_A,
                                                              trans_B,
                                                              m,
                                                              n,
                                                              (const T*)alpha_device_host,
                                                              descr_A,
                                                              nnz_A,
                                                              (const T*)csr_val_A,
                                                              (const I*)csr_row_ptr_A,
                                                              (const J*)csr_col_ind_A,
                                                              (const T*)beta_device_host,
                                                              descr_B,
                                                              nnz_B,
                                                              (const T*)csr_val_B,
                                                              (const I*)csr_row_ptr_B,
                                                              (const J*)csr_col_ind_B,
                                                              descr_C,
                                                              (T*)csr_val_C,
                                                              (const I*)csr_row_ptr_C,
                                                              (const J*)csr_col_ind_C,
                                                              descr,
                                                              temp_buffer));
    return rocsparse_status_success;
}

namespace rocsparse
{
    typedef rocsparse_status (*csrgeam_numeric_t)(rocsparse_handle             handle,
                                                  rocsparse_operation          trans_A,
                                                  rocsparse_operation          trans_B,
                                                  int64_t                      m,
                                                  int64_t                      n,
                                                  const void*                  alpha,
                                                  const rocsparse_mat_descr    descr_A,
                                                  int64_t                      nnz_A,
                                                  const void*                  csr_val_A,
                                                  const void*                  csr_row_ptr_A,
                                                  const void*                  csr_col_ind_A,
                                                  const void*                  beta,
                                                  const rocsparse_mat_descr    descr_B,
                                                  int64_t                      nnz_B,
                                                  const void*                  csr_val_B,
                                                  const void*                  csr_row_ptr_B,
                                                  const void*                  csr_col_ind_B,
                                                  const rocsparse_mat_descr    descr_C,
                                                  void*                        csr_val_C,
                                                  const void*                  csr_row_ptr_C,
                                                  const void*                  csr_col_ind_C,
                                                  const rocsparse_spgeam_descr descr,
                                                  void*                        temp_buffer);

    using csrgeam_numeric_tuple
        = std::tuple<rocsparse_datatype, rocsparse_indextype, rocsparse_indextype>;

#define CSRGEAM_NUMERIC_CONFIG(T, I, J)                                        \
    {csrgeam_numeric_tuple(T, I, J),                                           \
     csrgeam_numeric_template<typename rocsparse::datatype_traits<T>::type_t,  \
                              typename rocsparse::indextype_traits<I>::type_t, \
                              typename rocsparse::indextype_traits<J>::type_t>}

    static const std::map<csrgeam_numeric_tuple, csrgeam_numeric_t> s_csrgeam_numeric_dispatch{
        {CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f32_r, rocsparse_indextype_i32, rocsparse_indextype_i32),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f64_r, rocsparse_indextype_i32, rocsparse_indextype_i32),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f32_c, rocsparse_indextype_i32, rocsparse_indextype_i32),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f64_c, rocsparse_indextype_i32, rocsparse_indextype_i32),

         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f32_r, rocsparse_indextype_i64, rocsparse_indextype_i32),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f64_r, rocsparse_indextype_i64, rocsparse_indextype_i32),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f32_c, rocsparse_indextype_i64, rocsparse_indextype_i32),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f64_c, rocsparse_indextype_i64, rocsparse_indextype_i32),

         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f32_r, rocsparse_indextype_i64, rocsparse_indextype_i64),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f64_r, rocsparse_indextype_i64, rocsparse_indextype_i64),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f32_c, rocsparse_indextype_i64, rocsparse_indextype_i64),
         CSRGEAM_NUMERIC_CONFIG(
             rocsparse_datatype_f64_c, rocsparse_indextype_i64, rocsparse_indextype_i64)}};

    static rocsparse_status csrgeam_numeric_find(csrgeam_numeric_t*  function_,
                                                 rocsparse_datatype  t_type_,
                                                 rocsparse_indextype i_type_,
                                                 rocsparse_indextype j_type_)
    {

        const auto& it = rocsparse::s_csrgeam_numeric_dispatch.find(
            rocsparse::csrgeam_numeric_tuple(t_type_, i_type_, j_type_));

        if(it != rocsparse::s_csrgeam_numeric_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << "t_type: " << rocsparse::to_string(t_type_) << std::endl
                      << ", i_type: " << rocsparse::to_string(i_type_) << std::endl
                      << ", j_type: " << rocsparse::to_string(j_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_csrgeam_numeric_dispatch)
            {
                const auto& t      = p.first;
                const auto  t_type = std::get<0>(t);
                const auto  i_type = std::get<1>(t);
                const auto  j_type = std::get<2>(t);
                std::cout << std::endl
                          << std::endl
                          << "t_type: " << rocsparse::to_string(t_type) << std::endl
                          << ", i_type: " << rocsparse::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::to_string(j_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << "t_type: " << rocsparse::to_string(t_type_)
                 << ", i_type: " << rocsparse::to_string(i_type_)
                 << ", j_type: " << rocsparse::to_string(j_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::csrgeam_numeric(rocsparse_handle             handle,
                                            rocsparse_operation          trans_A,
                                            rocsparse_operation          trans_B,
                                            int64_t                      m,
                                            int64_t                      n,
                                            rocsparse_datatype           alpha_device_host_datatype,
                                            const void*                  alpha,
                                            const rocsparse_mat_descr    descr_A,
                                            int64_t                      nnz_A,
                                            rocsparse_datatype           csr_val_A_datatype,
                                            const void*                  csr_val_A,
                                            rocsparse_indextype          csr_row_ptr_A_indextype,
                                            const void*                  csr_row_ptr_A,
                                            rocsparse_indextype          csr_col_ind_A_indextype,
                                            const void*                  csr_col_ind_A,
                                            rocsparse_datatype           beta_device_host_datatype,
                                            const void*                  beta,
                                            const rocsparse_mat_descr    descr_B,
                                            int64_t                      nnz_B,
                                            rocsparse_datatype           csr_val_B_datatype,
                                            const void*                  csr_val_B,
                                            rocsparse_indextype          csr_row_ptr_B_indextype,
                                            const void*                  csr_row_ptr_B,
                                            rocsparse_indextype          csr_col_ind_B_indextype,
                                            const void*                  csr_col_ind_B,
                                            const rocsparse_mat_descr    descr_C,
                                            rocsparse_datatype           csr_val_C_datatype,
                                            void*                        csr_val_C,
                                            rocsparse_indextype          csr_row_ptr_C_indextype,
                                            const void*                  csr_row_ptr_C,
                                            rocsparse_indextype          csr_col_ind_C_indextype,
                                            const void*                  csr_col_ind_C,
                                            const rocsparse_spgeam_descr descr,
                                            void*                        temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::csrgeam_numeric_t f;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_numeric_find(
        &f, alpha_device_host_datatype, csr_row_ptr_C_indextype, csr_col_ind_C_indextype));

    RETURN_IF_ROCSPARSE_ERROR(f(handle,
                                trans_A,
                                trans_B,
                                m,
                                n,
                                alpha,
                                descr_A,
                                nnz_A,
                                csr_val_A,
                                csr_row_ptr_A,
                                csr_col_ind_A,
                                beta,
                                descr_B,
                                nnz_B,
                                csr_val_B,
                                csr_row_ptr_B,
                                csr_col_ind_B,
                                descr_C,
                                csr_val_C,
                                csr_row_ptr_C,
                                csr_col_ind_C,
                                descr,
                                temp_buffer));

    return rocsparse_status_success;
}

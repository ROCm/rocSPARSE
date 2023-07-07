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

#include "common.h"
#include "definitions.h"
#include "rocsparse_csritsv.hpp"
#include "utility.h"

#include "rocsparse_csrmv.hpp"

template <unsigned int BLOCKSIZE, typename T>
rocsparse_status rocsparse_nrminf(rocsparse_handle          handle_,
                                  size_t                    nitems_,
                                  const T*                  x_,
                                  floating_data_t<T>*       nrm_,
                                  const floating_data_t<T>* nrm0_,
                                  bool                      MX);

template <unsigned int BLOCKSIZE, typename T>
rocsparse_status rocsparse_nrminf_diff(rocsparse_handle          handle_,
                                       size_t                    nitems_,
                                       const T*                  x_,
                                       const T*                  y_,
                                       floating_data_t<T>*       nrm_,
                                       const floating_data_t<T>* nrm0_,
                                       bool                      MX);

namespace
{
    template <typename T, typename I, typename J>
    struct calculator_inverse_diagonal_t
    {

        template <unsigned int BLOCKSIZE, bool CONJ>
        ROCSPARSE_KERNEL(BLOCKSIZE)
        void kernel_inverse_diagonal(J m,
                                     const J* __restrict__ ind,
                                     const T* __restrict__ val,
                                     rocsparse_index_base base,
                                     T* __restrict__ invdiag,
                                     const I* __restrict__ ptr_diag,
                                     const J              ptr_shift,
                                     rocsparse_index_base ptr_diag_base,
                                     rocsparse_int* __restrict__ zero_pivot)
        {
            const J tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
            if(tid < m)
            {
                const I k = ptr_diag[tid] - ptr_diag_base + ptr_shift;
                const J j = ind[k] - base;
                if(j == tid)
                {
                    const T local_val = (!CONJ) ? val[k] : rocsparse_conj(val[k]);
                    if(local_val != static_cast<T>(0))
                    {
                        invdiag[tid] = static_cast<T>(1) / local_val;
                    }
                    else
                    {
                        atomicMin(zero_pivot, tid + base);
                        invdiag[tid] = static_cast<T>(1);
                    }
                }
                else
                {
                    atomicMin(zero_pivot, tid + base);
                    invdiag[tid] = static_cast<T>(1);
                }
            }
        }

        static rocsparse_status calculate(rocsparse_handle    handle,
                                          rocsparse_operation trans,
                                          J                   m,
                                          I                   nnz,
                                          const J* __restrict__ csr_ind,
                                          const T* __restrict__ csr_val,
                                          rocsparse_index_base csr_base,
                                          T*                   invdiag,
                                          const I* __restrict__ csr_diag_ind,
                                          J                    ptr_shift,
                                          rocsparse_index_base csr_diag_ind_base,
                                          rocsparse_int* __restrict__ zero_pivot)
        {
            //
            // Compute inverse of the diagonal.
            //
            static constexpr unsigned int BLOCKSIZE = 1024;
            dim3                          blocks((m - 1) / BLOCKSIZE + 1);
            dim3                          threads(BLOCKSIZE);
            switch(trans)
            {
            case rocsparse_operation_transpose:
            case rocsparse_operation_none:
            {
                hipLaunchKernelGGL((kernel_inverse_diagonal<BLOCKSIZE, false>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->stream,
                                   m,
                                   csr_ind,
                                   csr_val,
                                   csr_base,
                                   invdiag,
                                   csr_diag_ind,
                                   ptr_shift,
                                   csr_diag_ind_base,
                                   zero_pivot);
                break;
            }
            case rocsparse_operation_conjugate_transpose:
            {
                hipLaunchKernelGGL((kernel_inverse_diagonal<BLOCKSIZE, true>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->stream,
                                   m,
                                   csr_ind,
                                   csr_val,
                                   csr_base,
                                   invdiag,
                                   csr_diag_ind,
                                   ptr_shift,
                                   csr_diag_ind_base,
                                   zero_pivot);
                break;
            }
            }

            return rocsparse_status_success;
        }
    };
}

template <unsigned int BLOCKSIZE, typename J, typename T>
ROCSPARSE_KERNEL(BLOCKSIZE)
void kernel_add_scaled_residual(J m,
                                const T* __restrict__ r_,
                                T* __restrict__ y_,
                                const T* __restrict__ invdiag)
{
    const unsigned int tid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;
    if(tid < m)
    {
        y_[tid] = y_[tid] + invdiag[tid] * r_[tid];
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csritsv_solve_template(rocsparse_handle          handle,
                                                  rocsparse_int*            host_nmaxiter,
                                                  const floating_data_t<T>* host_tol,
                                                  floating_data_t<T>*       host_history,
                                                  rocsparse_operation       trans,
                                                  J                         m,
                                                  I                         nnz,
                                                  const T*                  alpha_device_host,
                                                  const rocsparse_mat_descr descr,
                                                  const T*                  csr_val,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  const T*                  x,
                                                  T*                        y,
                                                  rocsparse_solve_policy    policy,
                                                  void*                     temp_buffer)
{

    static constexpr bool verbose = false;
    if(m == 0)
    {
        return rocsparse_status_success;
    }

    rocsparse_csritsv_info        csritsv_info = info->csritsv_info;
    static constexpr unsigned int BLOCKSIZE    = 1024;
    dim3                          blocks((m - 1) / BLOCKSIZE + 1);
    dim3                          threads(BLOCKSIZE);

    //
    // reinitialize zero pivot.
    //
    if(descr->diag_type == rocsparse_diag_type_unit)
    {
        rocsparse_int max = std::numeric_limits<rocsparse_int>::max();
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            info->zero_pivot, &max, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));

        // Wait for device transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    }

    const rocsparse_fill_mode fill_mode = descr->fill_mode;
    const rocsparse_diag_type diag_type = descr->diag_type;
    if(nnz == 0)
    {

        //
        //
        //
        if(diag_type == rocsparse_diag_type_unit)
        {

            //
            // Copy.
            //
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(y, x, sizeof(T) * m, hipMemcpyDeviceToDevice, handle->stream));

            //
            // Scale.
            //

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {

                hipLaunchKernelGGL((scale_array<BLOCKSIZE>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->stream,
                                   m,
                                   y,
                                   alpha_device_host);
            }
            else
            {
                hipLaunchKernelGGL((scale_array<BLOCKSIZE>),
                                   blocks,
                                   threads,
                                   0,
                                   handle->stream,
                                   m,
                                   y,
                                   *alpha_device_host);
            }

            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_assign_async(static_cast<rocsparse_int*>(info->zero_pivot),
                                       (rocsparse_int)descr->base,
                                       handle->stream));
            return rocsparse_status_success;
        }
    }

    hipStream_t         stream      = handle->stream;
    const rocsparse_int nmaxiter    = host_nmaxiter[0];
    const bool          breakable   = (host_tol != nullptr);
    const bool          recordable  = (host_history != nullptr);
    const bool          compute_nrm = (recordable || breakable);

    if(false == csritsv_info->is_submatrix)
    {
        //
        // y_{k+1} = y_{k} + inv(D) * (alpha * x - A * y_{k} )
        // y_{k+1} = y_{k} + inv(D) * (alpha * x - A * y_{k} )
        //
        // if (r_k)
    }
    else
    {
    }

    floating_data_t<T> host_nrm[1];

    //
    // Use buffer as a vector of size 2xm.
    //
    T* y_p                = (T*)temp_buffer;
    T* invdiag            = (rocsparse_diag_type_non_unit == diag_type) ? &y_p[m] : nullptr;
    T* csrmv_alpha_device = (rocsparse_diag_type_non_unit == diag_type) ? &y_p[m * 2] : &y_p[m];
    floating_data_t<T>* device_nrm = (floating_data_t<T>*)(csrmv_alpha_device + 1);

    //
    // Check if we need to store csrmv_alpha on host or on device.
    //
    const T*       csrmv_alpha_device_host{};
    static const T s_minus_one = static_cast<T>(-1);
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        csrmv_alpha_device_host = csrmv_alpha_device;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            csrmv_alpha_device, &s_minus_one, sizeof(T), hipMemcpyHostToDevice, handle->stream));
    }
    else
    {
        csrmv_alpha_device_host = &s_minus_one;
    }

    const I* ptr_begin;
    const I* ptr_end;
    const I* ptr_diag;
    J        ptr_diag_shift;
    switch(diag_type)
    {
    case rocsparse_diag_type_non_unit:
    {
        if(csritsv_info->is_submatrix)
        {
            switch(fill_mode)
            {
            case rocsparse_fill_mode_lower:
            {
                ptr_begin      = csr_row_ptr;
                ptr_end        = (const I*)csritsv_info->ptr_end;
                ptr_diag       = ptr_end;
                ptr_diag_shift = -1;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                ptr_begin      = (const I*)csritsv_info->ptr_end;
                ptr_end        = csr_row_ptr + 1;
                ptr_diag       = ptr_begin;
                ptr_diag_shift = 0;
                break;
            }
            }
        }
        else
        {
            ptr_begin = csr_row_ptr;
            ptr_end   = csr_row_ptr + 1;
            switch(fill_mode)
            {
            case rocsparse_fill_mode_lower:
            {
                ptr_diag_shift = -1;
                ptr_diag       = ptr_end;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                ptr_diag_shift = 0;
                ptr_diag       = ptr_begin;
                break;
            }
            }
        }
        break;
    }

    case rocsparse_diag_type_unit:
    {
        //
        // We can simplify since D is identity and we expect that only T is stored.
        // yk+1 = yk + inv(D) * ( alpha * x - (D + T) yk )
        // yk+1 = alpha * x - T yk
        //
        // yk+1 = alpha * x - T yk
        // rk = alpha * x - (Id+T) yk = yk+1 - yk
        // rk = yk+1 - yk
        if(csritsv_info->is_submatrix)
        {
            switch(fill_mode)
            {
            case rocsparse_fill_mode_lower:
            {
                ptr_begin = csr_row_ptr;
                ptr_end   = (const I*)csritsv_info->ptr_end;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                ptr_begin = (const I*)csritsv_info->ptr_end;
                ptr_end   = csr_row_ptr + 1;
                break;
            }
            }
        }
        else
        {
            ptr_begin = csr_row_ptr;
            ptr_end   = csr_row_ptr + 1;
        }
        break;
    }
    }

    //
    // Compute norm of the matrix.
    //

    rocsparse_int ch = 1;
    switch(diag_type)
    {
    case rocsparse_diag_type_non_unit:
    {
        //
        // Compute the inverse of the diagonal.
        //
        RETURN_IF_ROCSPARSE_ERROR(
            (calculator_inverse_diagonal_t<T, I, J>::calculate)(handle,
                                                                trans,
                                                                m,
                                                                nnz,
                                                                csr_col_ind,
                                                                csr_val,
                                                                descr->base,
                                                                invdiag,
                                                                ptr_diag,
                                                                ptr_diag_shift,
                                                                descr->base,
                                                                (rocsparse_int*)info->zero_pivot));

        rocsparse_int zero_pivot;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&zero_pivot,
                                           info->zero_pivot,
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        if(zero_pivot != std::numeric_limits<rocsparse_int>::max())
        {
            return rocsparse_status_success;
        }

        //
        // in y out y
        //

        //
        // y_p = y
        // y = r_k
        // y = y_p + invD * (r_k)

        //
        // tmp = (alpha*x - A * y)
        // tmp *= invdiag(tmp)
        // y = y + tmp;
        //
        //

        for(rocsparse_int iter = 0; iter < nmaxiter; ++iter)
        {
            // Compute r_k
            //
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(y_p, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));

            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_csrmv_template<T, I, J, T, T, T>)(handle,
                                                             trans,
                                                             m,
                                                             m,
                                                             nnz,
                                                             csrmv_alpha_device_host,
                                                             descr,
                                                             csr_val,
                                                             ptr_begin,
                                                             ptr_end,
                                                             csr_col_ind,
                                                             info,
                                                             y,
                                                             alpha_device_host,
                                                             y_p,
                                                             false));
            bool break_loop = false;
            if(compute_nrm)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse_nrminf<1024>(handle, m, y_p, device_nrm, nullptr, false));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(host_nrm,
                                                   device_nrm,
                                                   sizeof(floating_data_t<T>),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                if(verbose)
                {
                    std::cout << "device iter " << iter << ", nrm " << host_nrm[0] << std::endl;
                }
                break_loop = (breakable) ? (host_nrm[0] <= host_tol[0]) : false;
                if(recordable)
                {
                    host_history[iter] = host_nrm[0];
                }
            }

            //
            // Add scale the residual
            //
            hipLaunchKernelGGL((kernel_add_scaled_residual<BLOCKSIZE>),
                               blocks,
                               threads,
                               0,
                               handle->stream,
                               m,
                               y_p,
                               y,
                               invdiag);

            if(break_loop)
            {
                host_nmaxiter[0] = iter + 1;
                break;
            }
        }

        break;
    }

    case rocsparse_diag_type_unit:
    {

        for(rocsparse_int iter = 0; iter < nmaxiter; ++iter)
        {

            //
            // swap pointers.
            //
            {
                auto tmp = y_p;
                y_p      = y;
                y        = tmp;
                ch *= -1;
            }

            //
            // We can simplify since D is identity and we expect that only T is stored.
            //  yk+1 = yk + inv(D) * ( alpha * x - (D + T) yk )
            // becomes
            //  yk+1 = alpha * x - T yk
            //

            //
            // Copy x to y_{k+1}
            //
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(y, x, sizeof(T) * m, hipMemcpyDeviceToDevice, stream));

            //
            // y_{k+1} = -T yk + alpha * y_{k+1}
            //
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse_csrmv_template<T, I, J, T, T, T>)(handle,
                                                             trans,
                                                             m,
                                                             m,
                                                             nnz,
                                                             csrmv_alpha_device_host,
                                                             descr,
                                                             csr_val,
                                                             ptr_begin,
                                                             ptr_end,
                                                             csr_col_ind,
                                                             info,
                                                             y_p,
                                                             alpha_device_host,
                                                             y,
                                                             false));

            bool break_loop = false;
            if(compute_nrm)
            {
                //
                // nrm.
                //
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse_nrminf_diff<1024>(handle, m, y_p, y, device_nrm, nullptr, false));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(host_nrm,
                                                   device_nrm,
                                                   sizeof(floating_data_t<T>),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                if(verbose)
                {
                    std::cout << "device iter " << iter << ", nrm " << host_nrm[0] << std::endl;
                }
                break_loop = (breakable) ? (host_nrm[0] <= host_tol[0]) : false;
                if(recordable)
                {
                    host_history[iter] = host_nrm[0];
                }
            }

            if(break_loop)
            {
                host_nmaxiter[0] = iter + 1;
                break;
            }
        }

        if(ch < 0)
        {

            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(y_p, y, sizeof(T) * m, hipMemcpyDeviceToDevice, handle->stream));
        }

        break;
    }
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse_csritsv_solve_impl(rocsparse_handle          handle,
                                              rocsparse_int*            host_nmaxiter,
                                              const floating_data_t<T>* host_tol,
                                              floating_data_t<T>*       host_history,
                                              rocsparse_operation       trans,
                                              J                         m,
                                              I                         nnz,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const T*                  csr_val,
                                              const I*                  csr_row_ptr,
                                              const J*                  csr_col_ind,
                                              rocsparse_mat_info        info,
                                              const T*                  x,
                                              T*                        y,
                                              rocsparse_solve_policy    policy,
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
    else if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging
    log_trace(handle,
              replaceX<T>("rocsparse_Xcsritsv_solve"),
              (const void*&)host_nmaxiter,
              (const void*&)host_tol,
              (const void*&)host_history,
              trans,
              m,
              nnz,
              LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
              (const void*&)descr,
              (const void*&)csr_val,
              (const void*&)csr_row_ptr,
              (const void*&)csr_col_ind,
              (const void*&)info,
              (const void*&)x,
              (const void*&)y,
              policy,
              (const void*&)temp_buffer);

    if(rocsparse_enum_utils::is_invalid(trans))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(policy))
    {
        return rocsparse_status_invalid_value;
    }

    // Check matrix type
    if(descr->type != rocsparse_matrix_type_general
       && descr->type != rocsparse_matrix_type_triangular)
    {
        return rocsparse_status_not_implemented;
    }

    // Check matrix sorting mode
    if(descr->storage_mode != rocsparse_storage_mode_sorted)
    {
        return rocsparse_status_requires_sorted_storage;
    }

    // Check sizes
    if(m < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(m > 0 && csr_row_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz > 0 && csr_col_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz > 0 && csr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(m > 0 && nnz > 0 && temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(host_nmaxiter == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(m > 0 && info->csritsv_info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_csritsv_solve_template(handle,
                                            host_nmaxiter,
                                            host_tol,
                                            host_history,
                                            trans,
                                            m,
                                            nnz,
                                            alpha_device_host,
                                            descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            x,
                                            y,
                                            policy,
                                            temp_buffer);
}

#define INSTANTIATE(I, J, T)                                             \
    template rocsparse_status rocsparse_csritsv_solve_template<I, J, T>( \
        rocsparse_handle handle,                                         \
        rocsparse_int * host_nmaxiter,                                   \
        const floating_data_t<T>* host_tol,                              \
        floating_data_t<T>*       host_history,                          \
        rocsparse_operation       trans,                                 \
        J                         m,                                     \
        I                         nnz,                                   \
        const T*                  alpha_device_host,                     \
        const rocsparse_mat_descr descr,                                 \
        const T*                  csr_val,                               \
        const I*                  csr_row_ptr,                           \
        const J*                  csr_col_ind,                           \
        rocsparse_mat_info        info,                                  \
        const T*                  x,                                     \
        T*                        y,                                     \
        rocsparse_solve_policy    policy,                                \
        void*                     temp_buffer);                                              \
    template rocsparse_status rocsparse_csritsv_solve_impl<I, J, T>(     \
        rocsparse_handle handle,                                         \
        rocsparse_int * host_nmaxiter,                                   \
        const floating_data_t<T>* host_tol,                              \
        floating_data_t<T>*       host_history,                          \
        rocsparse_operation       trans,                                 \
        J                         m,                                     \
        I                         nnz,                                   \
        const T*                  alpha_device_host,                     \
        const rocsparse_mat_descr descr,                                 \
        const T*                  csr_val,                               \
        const I*                  csr_row_ptr,                           \
        const J*                  csr_col_ind,                           \
        rocsparse_mat_info        info,                                  \
        const T*                  x,                                     \
        T*                        y,                                     \
        rocsparse_solve_policy    policy,                                \
        void*                     temp_buffer)

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

#define C_IMPL(NAME, T)                                                           \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,            \
                                     rocsparse_int*            host_nmaxiter,     \
                                     const floating_data_t<T>* host_tol,          \
                                     floating_data_t<T>*       host_history,      \
                                     rocsparse_operation       trans,             \
                                     rocsparse_int             m,                 \
                                     rocsparse_int             nnz,               \
                                     const T*                  alpha_device_host, \
                                     const rocsparse_mat_descr descr,             \
                                     const T*                  csr_val,           \
                                     const rocsparse_int*      csr_row_ptr,       \
                                     const rocsparse_int*      csr_col_ind,       \
                                     rocsparse_mat_info        info,              \
                                     const T*                  x,                 \
                                     T*                        y,                 \
                                     rocsparse_solve_policy    policy,            \
                                     void*                     temp_buffer)       \
    try                                                                           \
    {                                                                             \
        return rocsparse_csritsv_solve_impl(handle,                               \
                                            host_nmaxiter,                        \
                                            host_tol,                             \
                                            host_history,                         \
                                            trans,                                \
                                            m,                                    \
                                            nnz,                                  \
                                            alpha_device_host,                    \
                                            descr,                                \
                                            csr_val,                              \
                                            csr_row_ptr,                          \
                                            csr_col_ind,                          \
                                            info,                                 \
                                            x,                                    \
                                            y,                                    \
                                            policy,                               \
                                            temp_buffer);                         \
    }                                                                             \
    catch(...)                                                                    \
    {                                                                             \
        return exception_to_rocsparse_status();                                   \
    }

C_IMPL(rocsparse_scsritsv_solve, float);
C_IMPL(rocsparse_dcsritsv_solve, double);
C_IMPL(rocsparse_ccsritsv_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zcsritsv_solve, rocsparse_double_complex);

#undef C_IMPL

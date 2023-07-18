/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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
#include "internal/precond/rocsparse_csritilu0.h"
#include "rocsparse_csritilu0_driver.hpp"

template <typename T, typename I, typename J, typename... P>
static rocsparse_status compute_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
{
    switch(alg_)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_inplace>::compute<T, I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_async_split:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_split>::compute<T, I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split:
    {
        return rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::compute<T, I, J>::run(
            parameters...);
    }
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_sync_split_fusion>::compute<T, I, J>::run(parameters...);
    }
    }

    return rocsparse_status_invalid_value;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_csritilu0_compute_template(rocsparse_handle     handle_,
                                                      rocsparse_itilu0_alg alg_,
                                                      J                    options_,
                                                      J*                   nmaxiter_,
                                                      floating_data_t<T>   tol_,
                                                      J                    m_,
                                                      I                    nnz_,
                                                      const I* __restrict__ ptr_,
                                                      const J* __restrict__ ind_,
                                                      const T* __restrict__ val_,
                                                      T* __restrict__ ilu0_,
                                                      rocsparse_index_base base_,
                                                      size_t               buffer_size_,
                                                      void* __restrict__ buffer_)
{

    // Quick return if possible
    if(m_ == 0)
    {
        nmaxiter_[0] = 0;
        return rocsparse_status_success;
    }
    if(nnz_ == 0)
    {
        nmaxiter_[0] = 0;
        return rocsparse_status_zero_pivot;
    }

    return compute_dispatch<T, I, J>(alg_,
                                     handle_,
                                     alg_,
                                     options_,
                                     nmaxiter_,
                                     tol_,
                                     m_,
                                     nnz_,
                                     ptr_,
                                     ind_,
                                     val_,
                                     ilu0_,
                                     base_,
                                     buffer_size_,
                                     buffer_);
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_csritilu0_compute_impl(rocsparse_handle     handle_,
                                                  rocsparse_itilu0_alg alg_,
                                                  J                    options_,
                                                  J*                   nmaxiter_,
                                                  floating_data_t<T>   tol_,
                                                  J                    m_,
                                                  I                    nnz_,
                                                  const I* __restrict__ ptr_,
                                                  const J* __restrict__ ind_,
                                                  const T* __restrict__ val_,
                                                  T* __restrict__ ilu0_,
                                                  rocsparse_index_base base_,
                                                  size_t               buffer_size_,
                                                  void* __restrict__ buffer_)
{
    // Check for valid handle and matrix descriptor
    if(handle_ == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle_,
              "rocsparse_csritilu0_compute",
              alg_,
              options_,
              (const void*&)nmaxiter_,
              tol_,
              m_,
              nnz_,
              (const void*&)ptr_,
              (const void*&)ind_,
              (const void*&)val_,
              base_,
              buffer_size_,
              (const void*&)buffer_);

    if(rocsparse_enum_utils::is_invalid(alg_))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(base_))
    {
        return rocsparse_status_invalid_value;
    }

    if(options_ < 0)
    {
        return rocsparse_status_invalid_value;
    }

    if(tol_ < 0)
    {
        return rocsparse_status_invalid_value;
    }

    if(m_ < 0 || nnz_ < 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(nmaxiter_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check pointer arguments
    if(nnz_ > 0 && ptr_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_ > 0 && ind_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(buffer_size_ > 0 && buffer_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_ > 0 && val_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_ > 0 && ilu0_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_csritilu0_compute_template(handle_,
                                                alg_,
                                                options_,
                                                nmaxiter_,
                                                tol_,
                                                m_,
                                                nnz_,
                                                ptr_,
                                                ind_,
                                                val_,
                                                ilu0_,
                                                base_,
                                                buffer_size_,
                                                buffer_);
}

#define IMPL(NAME, T, I, J)                                             \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle_,      \
                                     rocsparse_itilu0_alg alg_,         \
                                     J                    options_,     \
                                     J*                   nmaxiter_,    \
                                     floating_data_t<T>   tol_,         \
                                     J                    m_,           \
                                     I                    nnz_,         \
                                     const I*             ptr_,         \
                                     const J*             ind_,         \
                                     const T*             val_,         \
                                     T*                   ilu0_,        \
                                     rocsparse_index_base base_,        \
                                     size_t               buffer_size_, \
                                     void*                buffer_)      \
    try                                                                 \
    {                                                                   \
        return rocsparse_csritilu0_compute_impl<T, I, J>(handle_,       \
                                                         alg_,          \
                                                         options_,      \
                                                         nmaxiter_,     \
                                                         tol_,          \
                                                         m_,            \
                                                         nnz_,          \
                                                         ptr_,          \
                                                         ind_,          \
                                                         val_,          \
                                                         ilu0_,         \
                                                         base_,         \
                                                         buffer_size_,  \
                                                         buffer_);      \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocsparse_status();                         \
    }

IMPL(rocsparse_scsritilu0_compute, float, rocsparse_int, rocsparse_int);
IMPL(rocsparse_dcsritilu0_compute, double, rocsparse_int, rocsparse_int);
IMPL(rocsparse_ccsritilu0_compute, rocsparse_float_complex, rocsparse_int, rocsparse_int);
IMPL(rocsparse_zcsritilu0_compute, rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef IMPL

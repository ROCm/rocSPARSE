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
#include "rocsparse_csritilu0_driver.hpp"

template <typename T, typename J, typename... P>
static rocsparse_status history_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
{
    switch(alg_)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_inplace>::history<floating_data_t<T>, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_async_split:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_split>::history<floating_data_t<T>, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_sync_split>::history<floating_data_t<T>, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        return rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split_fusion>::
            history<floating_data_t<T>, J>::run(parameters...);
    }
    }

    return rocsparse_status_invalid_value;
}

template <typename T, typename J>
rocsparse_status rocsparse_csritilu0_history_template(rocsparse_handle     handle_,
                                                      rocsparse_itilu0_alg alg_,
                                                      J* __restrict__ niter_,
                                                      floating_data_t<T>* __restrict__ nrms_,
                                                      size_t buffer_size_,
                                                      void* __restrict__ buffer_)
{
    return history_dispatch<T, J>(alg_, handle_, alg_, niter_, nrms_, buffer_size_, buffer_);
}

template <typename T, typename J>
rocsparse_status rocsparse_csritilu0_history_impl(rocsparse_handle     handle_,
                                                  rocsparse_itilu0_alg alg_,
                                                  J* __restrict__ niter_,
                                                  floating_data_t<T>* __restrict__ nrms_,
                                                  size_t buffer_size_,
                                                  void* __restrict__ buffer_)
{
    // Check for valid handle and matrix descriptor
    if(handle_ == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle_,
              "rocsparse_csritilu0_history",
              alg_,
              (const void*&)niter_,
              (const void*&)nrms_,
              buffer_size_,
              (const void*&)buffer_);
    // Check pointer arguments
    if(rocsparse_enum_utils::is_invalid(alg_))
    {
        return rocsparse_status_invalid_value;
    }

    if(niter_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nrms_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(buffer_size_ > 0 && buffer_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(rocsparse_enum_utils::is_invalid(alg_))
    {
        return rocsparse_status_invalid_value;
    }

    return rocsparse_csritilu0_history_template<T, J>(
        handle_, alg_, niter_, nrms_, buffer_size_, buffer_);
}

#define IMPL(NAME, T, J)                                                \
    extern "C" rocsparse_status NAME(rocsparse_handle     handle_,      \
                                     rocsparse_itilu0_alg alg_,         \
                                     J*                   niter_,       \
                                     floating_data_t<T>*  nrms_,        \
                                     size_t               buffer_size_, \
                                     void*                buffer_)      \
    try                                                                 \
    {                                                                   \
        return rocsparse_csritilu0_history_impl<T, J>(                  \
            handle_, alg_, niter_, nrms_, buffer_size_, buffer_);       \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocsparse_status();                         \
    }

IMPL(rocsparse_scsritilu0_history, float, rocsparse_int);
IMPL(rocsparse_dcsritilu0_history, double, rocsparse_int);
IMPL(rocsparse_ccsritilu0_history, rocsparse_float_complex, rocsparse_int);
IMPL(rocsparse_zcsritilu0_history, rocsparse_double_complex, rocsparse_int);

#undef IMPL

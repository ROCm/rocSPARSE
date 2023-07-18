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

#include "rocsparse_csritilu0_preprocess.hpp"
#include "common.h"
#include "internal/precond/rocsparse_csritilu0.h"
#include "rocsparse_csritilu0_driver.hpp"

template <typename I, typename J, typename... P>
static rocsparse_status preprocess_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
{
    switch(alg_)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_inplace>::preprocess<I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_async_split:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_split>::preprocess<I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split:
    {
        return rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::preprocess<I, J>::run(
            parameters...);
    }
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_sync_split_fusion>::preprocess<I, J>::run(parameters...);
    }
    }

    return rocsparse_status_invalid_value;
}

template <typename I, typename J>
rocsparse_status rocsparse_csritilu0_preprocess_template(rocsparse_handle     handle_,
                                                         rocsparse_itilu0_alg alg_,
                                                         J                    options_,
                                                         J                    nmaxiter_,
                                                         J                    m_,
                                                         I                    nnz_,
                                                         const I* __restrict__ ptr_,
                                                         const J* __restrict__ ind_,
                                                         rocsparse_index_base base_,
                                                         rocsparse_datatype   datatype_,
                                                         size_t               buffer_size_,
                                                         void* __restrict__ buffer_)
{
    // Quick return if possible
    if(m_ == 0)
    {
        return rocsparse_status_success;
    }
    if(nnz_ == 0)
    {
        return rocsparse_status_zero_pivot;
    }

    return preprocess_dispatch<I, J>(alg_,
                                     handle_,
                                     alg_,
                                     options_,
                                     nmaxiter_,
                                     m_,
                                     nnz_,
                                     ptr_,
                                     ind_,
                                     base_,
                                     datatype_,
                                     buffer_size_,
                                     buffer_);
}

template <typename I, typename J>
rocsparse_status rocsparse_csritilu0_preprocess_impl(rocsparse_handle     handle_,
                                                     rocsparse_itilu0_alg alg_,
                                                     J                    options_,
                                                     J                    nmaxiter_,
                                                     J                    m_,
                                                     I                    nnz_,
                                                     const I* __restrict__ ptr_,
                                                     const J* __restrict__ ind_,
                                                     rocsparse_index_base base_,
                                                     rocsparse_datatype   datatype_,
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
              "rocsparse_csritilu0_preprocess",
              alg_,
              options_,
              nmaxiter_,
              m_,
              nnz_,
              (const void*&)ptr_,
              (const void*&)ind_,
              base_,
              datatype_,
              (const void*&)buffer_size_);

    if(rocsparse_enum_utils::is_invalid(base_))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(datatype_))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(alg_))
    {
        return rocsparse_status_invalid_value;
    }
    if(options_ < 0)
    {
        return rocsparse_status_invalid_value;
    }
    if(nmaxiter_ < 0)
    {
        return rocsparse_status_invalid_value;
    }

    // Check sizes
    if(m_ < 0 || nnz_ < 0)
    {
        return rocsparse_status_invalid_size;
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

    return rocsparse_csritilu0_preprocess_template(handle_,
                                                   alg_,
                                                   options_,
                                                   nmaxiter_,
                                                   m_,
                                                   nnz_,
                                                   ptr_,
                                                   ind_,
                                                   base_,
                                                   datatype_,
                                                   buffer_size_,
                                                   buffer_);
}

extern "C" rocsparse_status rocsparse_csritilu0_preprocess(rocsparse_handle     handle_,
                                                           rocsparse_itilu0_alg alg_,
                                                           rocsparse_int        options_,
                                                           rocsparse_int        nmaxiter_,
                                                           rocsparse_int        m_,
                                                           rocsparse_int        nnz_,
                                                           const rocsparse_int* ptr_,
                                                           const rocsparse_int* ind_,
                                                           rocsparse_index_base base_,
                                                           rocsparse_datatype   datatype_,
                                                           size_t               buffer_size_,
                                                           void*                buffer_)
try
{
    return rocsparse_csritilu0_preprocess_impl<rocsparse_int, rocsparse_int>(handle_,
                                                                             alg_,
                                                                             options_,
                                                                             nmaxiter_,
                                                                             m_,
                                                                             nnz_,
                                                                             ptr_,
                                                                             ind_,
                                                                             base_,
                                                                             datatype_,
                                                                             buffer_size_,
                                                                             buffer_);
}
catch(...)
{
    return exception_to_rocsparse_status();
}

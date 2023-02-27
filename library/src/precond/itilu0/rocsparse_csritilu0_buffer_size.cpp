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

#include "rocsparse_csritilu0_buffer_size.hpp"
#include "common.h"
#include "rocsparse_csritilu0_driver.hpp"

template <typename I, typename J, typename... P>
static rocsparse_status buffer_size_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
{
    switch(alg_)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_inplace>::buffer_size<I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_async_split:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_async_split>::buffer_size<I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split:
    {
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_sync_split>::buffer_size<I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        // Fall back to the sync split algorithm.
        return rocsparse_csritilu0_driver_t<
            rocsparse_itilu0_alg_sync_split>::buffer_size<I, J>::run(parameters...);
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename I, typename J>
rocsparse_status rocsparse_csritilu0_buffer_size_template(rocsparse_handle     handle_,
                                                          rocsparse_itilu0_alg alg_,
                                                          J                    options_,
                                                          J                    nmaxiter_,
                                                          J                    m_,
                                                          I                    nnz_,
                                                          const I* __restrict__ ptr_,
                                                          const J* __restrict__ ind_,
                                                          rocsparse_index_base base_,
                                                          rocsparse_datatype   datatype_,
                                                          size_t* __restrict__ buffer_size_)
{
    // Quick return if possible
    if(m_ == 0)
    {
        *buffer_size_ = 0;
        return rocsparse_status_success;
    }
    if(nnz_ == 0)
    {
        return rocsparse_status_zero_pivot;
    }

    return buffer_size_dispatch<I, J>(alg_,
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
                                      buffer_size_);
}

template <typename I, typename J>
rocsparse_status rocsparse_csritilu0_buffer_size_impl(rocsparse_handle     handle_,
                                                      rocsparse_itilu0_alg alg_,
                                                      J                    options_,
                                                      J                    nmaxiter_,
                                                      J                    m_,
                                                      I                    nnz_,
                                                      const I* __restrict__ ptr_,
                                                      const J* __restrict__ ind_,
                                                      rocsparse_index_base base_,
                                                      rocsparse_datatype   datatype_,
                                                      size_t* __restrict__ buffer_size_)
{
    // Check for valid handle and matrix descriptor
    if(handle_ == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    log_trace(handle_,
              "rocsparse_csritilu0_buffer_size",
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

    if(m_ < 0 || nnz_ < 0)
    {
        return rocsparse_status_invalid_size;
    }

    if(nnz_ > 0 && ptr_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(nnz_ > 0 && ind_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(buffer_size_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    return rocsparse_csritilu0_buffer_size_template(
        handle_, alg_, options_, nmaxiter_, m_, nnz_, ptr_, ind_, base_, datatype_, buffer_size_);
}

extern "C" rocsparse_status rocsparse_csritilu0_buffer_size(rocsparse_handle     handle_,
                                                            rocsparse_itilu0_alg alg_,
                                                            rocsparse_int        options_,
                                                            rocsparse_int        nmaxiter_,
                                                            rocsparse_int        m_,
                                                            rocsparse_int        nnz_,
                                                            const rocsparse_int* __restrict__ ptr_,
                                                            const rocsparse_int* __restrict__ ind_,
                                                            rocsparse_index_base base_,
                                                            rocsparse_datatype   datatype_,
                                                            size_t* __restrict__ buffer_size_)
{
    return rocsparse_csritilu0_buffer_size_impl<rocsparse_int, rocsparse_int>(
        handle_, alg_, options_, nmaxiter_, m_, nnz_, ptr_, ind_, base_, datatype_, buffer_size_);
}

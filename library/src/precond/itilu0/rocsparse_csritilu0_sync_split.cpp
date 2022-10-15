/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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

template <>
struct rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>
{
    //
    // History, same as algorithm 1.
    //
    template <typename T, typename J>
    struct history
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J*                   niter_,
                                    T*                   data_,
                                    size_t               buffer_size_,
                                    void*                buffer_)
        {
            return rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split_fusion>::
                history<T, J>::run(handle_, alg_, niter_, data_, buffer_size_, buffer_);
        }
    };

    //
    // Buffer size, same as algorithm 1.
    //
    template <typename I, typename J>
    struct buffer_size
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J                    nsweeps_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_datatype   datatype_,
                                    size_t* __restrict__ buffer_size_)
        {
            return rocsparse_csritilu0_driver_t<
                rocsparse_itilu0_alg_sync_split_fusion>::buffer_size<I, J>::run(handle_,
                                                                                alg_,
                                                                                options_,
                                                                                nsweeps_,
                                                                                m_,
                                                                                nnz_,
                                                                                ptr_,
                                                                                ind_,
                                                                                base_,
                                                                                datatype_,
                                                                                buffer_size_);
        }
    };

    //
    // Preprocess, same as algorithm 1.
    //
    template <typename I, typename J>
    struct preprocess
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J                    nsweeps_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    rocsparse_index_base base_,
                                    rocsparse_datatype   datatype_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer_)

        {
            return rocsparse_csritilu0_driver_t<
                rocsparse_itilu0_alg_sync_split_fusion>::preprocess<I, J>::run(handle_,
                                                                               alg_,
                                                                               options_,
                                                                               nsweeps_,
                                                                               m_,
                                                                               nnz_,
                                                                               ptr_,
                                                                               ind_,
                                                                               base_,
                                                                               datatype_,
                                                                               buffer_size_,
                                                                               buffer_);
        }
    };

    //
    // Compute, same as algorithm 1 but with algorithm 2 in the back end.
    //
    template <typename T, typename I, typename J>
    struct compute
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J                    options_,
                                    J*                   nsweeps_,
                                    floating_data_t<T>   tol_,
                                    J                    m_,
                                    I                    nnz_,
                                    const I* __restrict__ ptr_,
                                    const J* __restrict__ ind_,
                                    const T* __restrict__ val_,
                                    T* __restrict__ x_,
                                    rocsparse_index_base base_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer_)
        {
            return rocsparse_csritilu0_driver_t<
                rocsparse_itilu0_alg_sync_split_fusion>::compute<T, I, J>::run(handle_,
                                                                               alg_,
                                                                               options_,
                                                                               nsweeps_,
                                                                               tol_,
                                                                               m_,
                                                                               nnz_,
                                                                               ptr_,
                                                                               ind_,
                                                                               val_,
                                                                               x_,
                                                                               base_,
                                                                               buffer_size_,
                                                                               buffer_);
        }
    };
};

#define INSTANTIATE(T, I, J) \
    template struct rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::compute<T, I, J>

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);

#undef INSTANTIATE

#define INSTANTIATE(T, J) \
    template struct rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::history<T, J>

INSTANTIATE(float, int32_t);
INSTANTIATE(double, int32_t);

#undef INSTANTIATE

#define INSTANTIATE(I, J)                                                                          \
    template struct rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::buffer_size<I,  \
                                                                                               J>; \
    template struct rocsparse_csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::preprocess<I, J>;

INSTANTIATE(int32_t, int32_t);

#undef INSTANTIATE

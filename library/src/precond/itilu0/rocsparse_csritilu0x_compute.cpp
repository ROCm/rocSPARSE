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
#include "rocsparse_csritilu0x_driver.hpp"

template <typename T, typename I, typename J, typename... P>
static rocsparse_status compute_dispatch(rocsparse_itilu0_alg alg_, P&&... parameters)
{
    switch(alg_)
    {
    case rocsparse_itilu0_alg_default:
    case rocsparse_itilu0_alg_async_inplace:
    {
        return rocsparse_status_internal_error;
    }
    case rocsparse_itilu0_alg_async_split:
    {
        return rocsparse_csritilu0x_driver_t<
            rocsparse_itilu0_alg_async_split>::compute<T, I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split:
    {
        return rocsparse_csritilu0x_driver_t<
            rocsparse_itilu0_alg_sync_split>::compute<T, I, J>::run(parameters...);
    }
    case rocsparse_itilu0_alg_sync_split_fusion:
    {
        return rocsparse_csritilu0x_driver_t<
            rocsparse_itilu0_alg_sync_split_fusion>::compute<T, I, J>::run(parameters...);
    }
    }
    return rocsparse_status_invalid_value;
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_csritilu0x_compute_template(rocsparse_handle     handle_,
                                                       rocsparse_itilu0_alg alg_,
                                                       rocsparse_int        options_,
                                                       J* __restrict__ nsweeps_,
                                                       floating_data_t<T> tol_,
                                                       J                  m_,
                                                       I                  nnz_,
                                                       const I* __restrict__ ptr_begin_,
                                                       const I* __restrict__ ptr_end_,
                                                       const J* __restrict__ ind_,
                                                       const T* __restrict__ val_,
                                                       rocsparse_index_base base_,
                                                       rocsparse_diag_type  ldiag_type_,
                                                       rocsparse_direction  ldir_,
                                                       rocsparse_int        lnnz_,
                                                       const I* __restrict__ lptr_begin_,
                                                       const I* __restrict__ lptr_end_,
                                                       const J* __restrict__ lind_,
                                                       T* __restrict__ lval_,
                                                       rocsparse_index_base lbase_,
                                                       rocsparse_diag_type  udiag_type_,
                                                       rocsparse_direction  udir_,
                                                       I                    unnz_,
                                                       const I* __restrict__ uptr_begin_,
                                                       const I* __restrict__ uptr_end_,
                                                       const J* __restrict__ uind_,
                                                       T* __restrict__ uval_,
                                                       rocsparse_index_base ubase_,
                                                       T* __restrict__ dval_,
                                                       size_t buffer_size_,
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

    if(ldir_ != rocsparse_direction_row)
    {
        return rocsparse_status_not_implemented;
    }

    if(udir_ != rocsparse_direction_column)
    {
        return rocsparse_status_not_implemented;
    }

    return compute_dispatch<T, I, J>(alg_,
                                     handle_,
                                     options_,
                                     nsweeps_,
                                     tol_,
                                     m_,
                                     nnz_,
                                     ptr_begin_,
                                     ptr_end_,
                                     ind_,
                                     val_,
                                     base_,
                                     ldiag_type_,
                                     ldir_,
                                     lnnz_,
                                     lptr_begin_,
                                     lptr_end_,
                                     lind_,
                                     lval_,
                                     lbase_,
                                     udiag_type_,
                                     udir_,
                                     unnz_,
                                     uptr_begin_,
                                     uptr_end_,
                                     uind_,
                                     uval_,
                                     ubase_,
                                     dval_,
                                     buffer_size_,
                                     buffer_);
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse_csritilu0x_compute_impl(rocsparse_handle     handle_,
                                                   rocsparse_itilu0_alg alg_,
                                                   rocsparse_int        options_,
                                                   J* __restrict__ nsweeps_,
                                                   floating_data_t<T> tol_,
                                                   J                  m_,
                                                   I                  nnz_,
                                                   const I* __restrict__ ptr_begin_,
                                                   const I* __restrict__ ptr_end_,
                                                   const J* __restrict__ ind_,
                                                   const T* __restrict__ val_,
                                                   rocsparse_index_base base_,
                                                   rocsparse_diag_type  ldiag_type_,
                                                   rocsparse_direction  ldir_,
                                                   rocsparse_int        lnnz_,
                                                   const I* __restrict__ lptr_begin_,
                                                   const I* __restrict__ lptr_end_,
                                                   const J* __restrict__ lind_,
                                                   T* __restrict__ lval_,
                                                   rocsparse_index_base lbase_,
                                                   rocsparse_diag_type  udiag_type_,
                                                   rocsparse_direction  udir_,
                                                   I                    unnz_,
                                                   const I* __restrict__ uptr_begin_,
                                                   const I* __restrict__ uptr_end_,
                                                   const J* __restrict__ uind_,
                                                   T* __restrict__ uval_,
                                                   rocsparse_index_base ubase_,
                                                   T* __restrict__ dval_,
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
              replaceX<T>("rocsparse_Xcsritilu0x"),
              options_,
              (const void*&)nsweeps_,
              tol_,
              m_,
              nnz_,
              (const void*&)ptr_begin_,
              (const void*&)ptr_end_,
              (const void*&)ind_,
              (const void*&)val_,
              base_,
              ldiag_type_,
              ldir_,
              lnnz_,
              (const void*&)lptr_begin_,
              (const void*&)lptr_end_,
              (const void*&)lind_,
              (const void*&)lval_,
              lbase_,
              udiag_type_,
              udir_,
              unnz_,
              (const void*&)uptr_begin_,
              (const void*&)uptr_end_,
              (const void*&)uind_,

              (const void*&)uval_,
              ubase_,

              (const void*&)dval_,
              buffer_size_,
              (const void*&)buffer_);

    // Check sizes
    if(m_ < 0 || nnz_ < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m_ == 0)
    {
        return rocsparse_status_success;
    }

    if(nnz_ == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(ptr_begin_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    if(ptr_end_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(ind_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(val_ == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    if(rocsparse_enum_utils::is_invalid(base_))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(ubase_))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(lbase_))
    {
        return rocsparse_status_invalid_value;
    }

    if(rocsparse_enum_utils::is_invalid(udiag_type_))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(udir_))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(ldiag_type_))
    {
        return rocsparse_status_invalid_value;
    }
    if(rocsparse_enum_utils::is_invalid(ldir_))
    {
        return rocsparse_status_invalid_value;
    }

    if((m_ > 0)
       && (uptr_begin_ == nullptr || lptr_begin_ == nullptr || uptr_end_ == nullptr
           || lptr_end_ == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if((unnz_ > 0) && (uind_ == nullptr || uval_ == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if((lnnz_ > 0) && (lind_ == nullptr || lval_ == nullptr))
    {
        return rocsparse_status_invalid_pointer;
    }

    if((ldiag_type_ == rocsparse_diag_type_unit && udiag_type_ == rocsparse_diag_type_unit)
       && ((m_ > 0) && (dval_ == nullptr)))
    {
        return rocsparse_status_invalid_pointer;
    }

    if(ldir_ != rocsparse_direction_row)
    {
        return rocsparse_status_not_implemented;
    }
    if(udir_ != rocsparse_direction_column)
    {
        return rocsparse_status_not_implemented;
    }

    return rocsparse_csritilu0x_compute_template(handle_,
                                                 alg_,
                                                 options_,
                                                 nsweeps_,
                                                 tol_,
                                                 m_,
                                                 nnz_,
                                                 ptr_begin_,
                                                 ptr_end_,
                                                 ind_,
                                                 val_,
                                                 base_,
                                                 ldiag_type_,
                                                 ldir_,
                                                 lnnz_,
                                                 lptr_begin_,
                                                 lptr_end_,
                                                 lind_,
                                                 lval_,
                                                 lbase_,
                                                 udiag_type_,
                                                 udir_,
                                                 unnz_,
                                                 uptr_begin_,
                                                 uptr_end_,
                                                 uind_,
                                                 uval_,
                                                 ubase_,
                                                 dval_,
                                                 buffer_size_,
                                                 buffer_);
}

#define INSTANTIATE(TOK, T, I, J)                                          \
    template rocsparse_status rocsparse_csritilu0x_compute_##TOK<T, I, J>( \
        rocsparse_handle     handle_,                                      \
        rocsparse_itilu0_alg alg_,                                         \
        J                    options_,                                     \
        J* __restrict__ nsweeps_,                                          \
        floating_data_t<T> tol_,                                           \
        J                  m_,                                             \
        I                  nnz_,                                           \
        const I* __restrict__ ptr_begin_,                                  \
        const I* __restrict__ ptr_end_,                                    \
        const J* __restrict__ ind_,                                        \
        const T* __restrict__ val_,                                        \
        rocsparse_index_base base_,                                        \
        rocsparse_diag_type  ldiag_type_,                                  \
        rocsparse_direction  ldir_,                                        \
        I                    lnnz_,                                        \
        const I* __restrict__ lptr_begin_,                                 \
        const I* __restrict__ lptr_end_,                                   \
        const J* __restrict__ lind_,                                       \
        T* __restrict__ lval_,                                             \
        rocsparse_index_base lbase_,                                       \
        rocsparse_diag_type  udiag_type_,                                  \
        rocsparse_direction  udir_,                                        \
        I                    unnz_,                                        \
        const I* __restrict__ uptr_begin_,                                 \
        const I* __restrict__ uptr_end_,                                   \
        const J* __restrict__ uind_,                                       \
        T* __restrict__ uval_,                                             \
        rocsparse_index_base ubase_,                                       \
        T* __restrict__ dval_,                                             \
        size_t buffer_size_,                                               \
        void* __restrict__ buffer_)

INSTANTIATE(template, float, int32_t, int32_t);
INSTANTIATE(template, double, int32_t, int32_t);
INSTANTIATE(template, rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(template, rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(impl, float, int32_t, int32_t);
INSTANTIATE(impl, double, int32_t, int32_t);
INSTANTIATE(impl, rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(impl, rocsparse_double_complex, int32_t, int32_t);

#undef INSTANTIATE

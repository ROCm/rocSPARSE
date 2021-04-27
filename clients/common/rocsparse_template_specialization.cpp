/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include "rocsparse.hpp"

#include "rocsparse_enum.hpp"
constexpr rocsparse_matrix_type_t::value_t
    rocsparse_matrix_type_t::values[rocsparse_matrix_type_t::nvalues];

constexpr rocsparse_operation_t::value_t
    rocsparse_operation_t::values[rocsparse_operation_t::nvalues];

std::ostream& operator<<(std::ostream& out, const rocsparse_operation& v)
{
    out << rocsparse_operation2string(v);
    return out;
}

std::ostream& operator<<(std::ostream& out, const rocsparse_direction& v)
{
    out << rocsparse_direction2string(v);
    return out;
}

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
// axpyi
template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const float*         alpha,
                                 const float*         x_val,
                                 const rocsparse_int* x_ind,
                                 float*               y,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_saxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 const double*        alpha,
                                 const double*        x_val,
                                 const rocsparse_int* x_ind,
                                 double*              y,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_daxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle               handle,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_float_complex* x_val,
                                 const rocsparse_int*           x_ind,
                                 rocsparse_float_complex*       y,
                                 rocsparse_index_base           idx_base)
{
    return rocsparse_caxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_axpyi(rocsparse_handle                handle,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_double_complex* x_val,
                                 const rocsparse_int*            x_ind,
                                 rocsparse_double_complex*       y,
                                 rocsparse_index_base            idx_base)
{
    return rocsparse_zaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

// doti
template <>
rocsparse_status rocsparse_doti(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const float*         x_val,
                                const rocsparse_int* x_ind,
                                const float*         y,
                                float*               result,
                                rocsparse_index_base idx_base)
{
    return rocsparse_sdoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
rocsparse_status rocsparse_doti(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const double*        x_val,
                                const rocsparse_int* x_ind,
                                const double*        y,
                                double*              result,
                                rocsparse_index_base idx_base)
{
    return rocsparse_ddoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
rocsparse_status rocsparse_doti(rocsparse_handle               handle,
                                rocsparse_int                  nnz,
                                const rocsparse_float_complex* x_val,
                                const rocsparse_int*           x_ind,
                                const rocsparse_float_complex* y,
                                rocsparse_float_complex*       result,
                                rocsparse_index_base           idx_base)
{
    return rocsparse_cdoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
rocsparse_status rocsparse_doti(rocsparse_handle                handle,
                                rocsparse_int                   nnz,
                                const rocsparse_double_complex* x_val,
                                const rocsparse_int*            x_ind,
                                const rocsparse_double_complex* y,
                                rocsparse_double_complex*       result,
                                rocsparse_index_base            idx_base)
{
    return rocsparse_zdoti(handle, nnz, x_val, x_ind, y, result, idx_base);
}

// dotci
template <>
rocsparse_status rocsparse_dotci(rocsparse_handle               handle,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* x_val,
                                 const rocsparse_int*           x_ind,
                                 const rocsparse_float_complex* y,
                                 rocsparse_float_complex*       result,
                                 rocsparse_index_base           idx_base)
{
    return rocsparse_cdotci(handle, nnz, x_val, x_ind, y, result, idx_base);
}

template <>
rocsparse_status rocsparse_dotci(rocsparse_handle                handle,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* x_val,
                                 const rocsparse_int*            x_ind,
                                 const rocsparse_double_complex* y,
                                 rocsparse_double_complex*       result,
                                 rocsparse_index_base            idx_base)
{
    return rocsparse_zdotci(handle, nnz, x_val, x_ind, y, result, idx_base);
}

// gthr
template <>
rocsparse_status rocsparse_gthr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const float*         y,
                                float*               x_val,
                                const rocsparse_int* x_ind,
                                rocsparse_index_base idx_base)
{
    return rocsparse_sgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const double*        y,
                                double*              x_val,
                                const rocsparse_int* x_ind,
                                rocsparse_index_base idx_base)
{
    return rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthr(rocsparse_handle               handle,
                                rocsparse_int                  nnz,
                                const rocsparse_float_complex* y,
                                rocsparse_float_complex*       x_val,
                                const rocsparse_int*           x_ind,
                                rocsparse_index_base           idx_base)
{
    return rocsparse_cgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthr(rocsparse_handle                handle,
                                rocsparse_int                   nnz,
                                const rocsparse_double_complex* y,
                                rocsparse_double_complex*       x_val,
                                const rocsparse_int*            x_ind,
                                rocsparse_index_base            idx_base)
{
    return rocsparse_zgthr(handle, nnz, y, x_val, x_ind, idx_base);
}

// gthrz
template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 float*               y,
                                 float*               x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_sgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle     handle,
                                 rocsparse_int        nnz,
                                 double*              y,
                                 double*              x_val,
                                 const rocsparse_int* x_ind,
                                 rocsparse_index_base idx_base)
{
    return rocsparse_dgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle         handle,
                                 rocsparse_int            nnz,
                                 rocsparse_float_complex* y,
                                 rocsparse_float_complex* x_val,
                                 const rocsparse_int*     x_ind,
                                 rocsparse_index_base     idx_base)
{
    return rocsparse_cgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

template <>
rocsparse_status rocsparse_gthrz(rocsparse_handle          handle,
                                 rocsparse_int             nnz,
                                 rocsparse_double_complex* y,
                                 rocsparse_double_complex* x_val,
                                 const rocsparse_int*      x_ind,
                                 rocsparse_index_base      idx_base)
{
    return rocsparse_zgthrz(handle, nnz, y, x_val, x_ind, idx_base);
}

// sctr
template <>
rocsparse_status rocsparse_sctr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const float*         x_val,
                                const rocsparse_int* x_ind,
                                float*               y,
                                rocsparse_index_base idx_base)
{
    return rocsparse_ssctr(handle, nnz, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_sctr(rocsparse_handle     handle,
                                rocsparse_int        nnz,
                                const double*        x_val,
                                const rocsparse_int* x_ind,
                                double*              y,
                                rocsparse_index_base idx_base)
{
    return rocsparse_dsctr(handle, nnz, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_sctr(rocsparse_handle               handle,
                                rocsparse_int                  nnz,
                                const rocsparse_float_complex* x_val,
                                const rocsparse_int*           x_ind,
                                rocsparse_float_complex*       y,
                                rocsparse_index_base           idx_base)
{
    return rocsparse_csctr(handle, nnz, x_val, x_ind, y, idx_base);
}

template <>
rocsparse_status rocsparse_sctr(rocsparse_handle                handle,
                                rocsparse_int                   nnz,
                                const rocsparse_double_complex* x_val,
                                const rocsparse_int*            x_ind,
                                rocsparse_double_complex*       y,
                                rocsparse_index_base            idx_base)
{
    return rocsparse_zsctr(handle, nnz, x_val, x_ind, y, idx_base);
}

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
// bsrmv
template <>
rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans,
                                 rocsparse_int             mb,
                                 rocsparse_int             nb,
                                 rocsparse_int             nnzb,
                                 const float*              alpha,
                                 const rocsparse_mat_descr descr,
                                 const float*              bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             bsr_dim,
                                 const float*              x,
                                 const float*              beta,
                                 float*                    y)
{
    return rocsparse_sbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

template <>
rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans,
                                 rocsparse_int             mb,
                                 rocsparse_int             nb,
                                 rocsparse_int             nnzb,
                                 const double*             alpha,
                                 const rocsparse_mat_descr descr,
                                 const double*             bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             bsr_dim,
                                 const double*             x,
                                 const double*             beta,
                                 double*                   y)
{
    return rocsparse_dbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

template <>
rocsparse_status rocsparse_bsrmv(rocsparse_handle               handle,
                                 rocsparse_direction            dir,
                                 rocsparse_operation            trans,
                                 rocsparse_int                  mb,
                                 rocsparse_int                  nb,
                                 rocsparse_int                  nnzb,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* bsr_val,
                                 const rocsparse_int*           bsr_row_ptr,
                                 const rocsparse_int*           bsr_col_ind,
                                 rocsparse_int                  bsr_dim,
                                 const rocsparse_float_complex* x,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       y)
{
    return rocsparse_cbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

template <>
rocsparse_status rocsparse_bsrmv(rocsparse_handle                handle,
                                 rocsparse_direction             dir,
                                 rocsparse_operation             trans,
                                 rocsparse_int                   mb,
                                 rocsparse_int                   nb,
                                 rocsparse_int                   nnzb,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* bsr_val,
                                 const rocsparse_int*            bsr_row_ptr,
                                 const rocsparse_int*            bsr_col_ind,
                                 rocsparse_int                   bsr_dim,
                                 const rocsparse_double_complex* x,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       y)
{
    return rocsparse_zbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            x,
                            beta,
                            y);
}

// bsrsv
template <>
rocsparse_status rocsparse_bsrsv_buffer_size(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans,
                                             rocsparse_int             mb,
                                             rocsparse_int             nnzb,
                                             const rocsparse_mat_descr descr,
                                             const float*              bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             bsr_dim,
                                             rocsparse_mat_info        info,
                                             size_t*                   buffer_size)
{
    return rocsparse_sbsrsv_buffer_size(handle,
                                        dir,
                                        trans,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        bsr_dim,
                                        info,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_bsrsv_buffer_size(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans,
                                             rocsparse_int             mb,
                                             rocsparse_int             nnzb,
                                             const rocsparse_mat_descr descr,
                                             const double*             bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             bsr_dim,
                                             rocsparse_mat_info        info,
                                             size_t*                   buffer_size)
{
    return rocsparse_dbsrsv_buffer_size(handle,
                                        dir,
                                        trans,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        bsr_dim,
                                        info,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_bsrsv_buffer_size(rocsparse_handle               handle,
                                             rocsparse_direction            dir,
                                             rocsparse_operation            trans,
                                             rocsparse_int                  mb,
                                             rocsparse_int                  nnzb,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* bsr_val,
                                             const rocsparse_int*           bsr_row_ptr,
                                             const rocsparse_int*           bsr_col_ind,
                                             rocsparse_int                  bsr_dim,
                                             rocsparse_mat_info             info,
                                             size_t*                        buffer_size)
{
    return rocsparse_cbsrsv_buffer_size(handle,
                                        dir,
                                        trans,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        bsr_dim,
                                        info,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_bsrsv_buffer_size(rocsparse_handle                handle,
                                             rocsparse_direction             dir,
                                             rocsparse_operation             trans,
                                             rocsparse_int                   mb,
                                             rocsparse_int                   nnzb,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* bsr_val,
                                             const rocsparse_int*            bsr_row_ptr,
                                             const rocsparse_int*            bsr_col_ind,
                                             rocsparse_int                   bsr_dim,
                                             rocsparse_mat_info              info,
                                             size_t*                         buffer_size)
{
    return rocsparse_zbsrsv_buffer_size(handle,
                                        dir,
                                        trans,
                                        mb,
                                        nnzb,
                                        descr,
                                        bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        bsr_dim,
                                        info,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_bsrsv_analysis(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          rocsparse_int             mb,
                                          rocsparse_int             nnzb,
                                          const rocsparse_mat_descr descr,
                                          const float*              bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             bsr_dim,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer)
{
    return rocsparse_sbsrsv_analysis(handle,
                                     dir,
                                     trans,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     bsr_dim,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrsv_analysis(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          rocsparse_int             mb,
                                          rocsparse_int             nnzb,
                                          const rocsparse_mat_descr descr,
                                          const double*             bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             bsr_dim,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer)
{
    return rocsparse_dbsrsv_analysis(handle,
                                     dir,
                                     trans,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     bsr_dim,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrsv_analysis(rocsparse_handle               handle,
                                          rocsparse_direction            dir,
                                          rocsparse_operation            trans,
                                          rocsparse_int                  mb,
                                          rocsparse_int                  nnzb,
                                          const rocsparse_mat_descr      descr,
                                          const rocsparse_float_complex* bsr_val,
                                          const rocsparse_int*           bsr_row_ptr,
                                          const rocsparse_int*           bsr_col_ind,
                                          rocsparse_int                  bsr_dim,
                                          rocsparse_mat_info             info,
                                          rocsparse_analysis_policy      analysis,
                                          rocsparse_solve_policy         solve,
                                          void*                          temp_buffer)
{
    return rocsparse_cbsrsv_analysis(handle,
                                     dir,
                                     trans,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     bsr_dim,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrsv_analysis(rocsparse_handle                handle,
                                          rocsparse_direction             dir,
                                          rocsparse_operation             trans,
                                          rocsparse_int                   mb,
                                          rocsparse_int                   nnzb,
                                          const rocsparse_mat_descr       descr,
                                          const rocsparse_double_complex* bsr_val,
                                          const rocsparse_int*            bsr_row_ptr,
                                          const rocsparse_int*            bsr_col_ind,
                                          rocsparse_int                   bsr_dim,
                                          rocsparse_mat_info              info,
                                          rocsparse_analysis_policy       analysis,
                                          rocsparse_solve_policy          solve,
                                          void*                           temp_buffer)
{
    return rocsparse_zbsrsv_analysis(handle,
                                     dir,
                                     trans,
                                     mb,
                                     nnzb,
                                     descr,
                                     bsr_val,
                                     bsr_row_ptr,
                                     bsr_col_ind,
                                     bsr_dim,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrsv_solve(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_operation       trans,
                                       rocsparse_int             mb,
                                       rocsparse_int             nnzb,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr,
                                       const float*              bsr_val,
                                       const rocsparse_int*      bsr_row_ptr,
                                       const rocsparse_int*      bsr_col_ind,
                                       rocsparse_int             bsr_dim,
                                       rocsparse_mat_info        info,
                                       const float*              x,
                                       float*                    y,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
{
    return rocsparse_sbsrsv_solve(handle,
                                  dir,
                                  trans,
                                  mb,
                                  nnzb,
                                  alpha,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  x,
                                  y,
                                  policy,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrsv_solve(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_operation       trans,
                                       rocsparse_int             mb,
                                       rocsparse_int             nnzb,
                                       const double*             alpha,
                                       const rocsparse_mat_descr descr,
                                       const double*             bsr_val,
                                       const rocsparse_int*      bsr_row_ptr,
                                       const rocsparse_int*      bsr_col_ind,
                                       rocsparse_int             bsr_dim,
                                       rocsparse_mat_info        info,
                                       const double*             x,
                                       double*                   y,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
{
    return rocsparse_dbsrsv_solve(handle,
                                  dir,
                                  trans,
                                  mb,
                                  nnzb,
                                  alpha,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  x,
                                  y,
                                  policy,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrsv_solve(rocsparse_handle               handle,
                                       rocsparse_direction            dir,
                                       rocsparse_operation            trans,
                                       rocsparse_int                  mb,
                                       rocsparse_int                  nnzb,
                                       const rocsparse_float_complex* alpha,
                                       const rocsparse_mat_descr      descr,
                                       const rocsparse_float_complex* bsr_val,
                                       const rocsparse_int*           bsr_row_ptr,
                                       const rocsparse_int*           bsr_col_ind,
                                       rocsparse_int                  bsr_dim,
                                       rocsparse_mat_info             info,
                                       const rocsparse_float_complex* x,
                                       rocsparse_float_complex*       y,
                                       rocsparse_solve_policy         policy,
                                       void*                          temp_buffer)
{
    return rocsparse_cbsrsv_solve(handle,
                                  dir,
                                  trans,
                                  mb,
                                  nnzb,
                                  alpha,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  x,
                                  y,
                                  policy,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrsv_solve(rocsparse_handle                handle,
                                       rocsparse_direction             dir,
                                       rocsparse_operation             trans,
                                       rocsparse_int                   mb,
                                       rocsparse_int                   nnzb,
                                       const rocsparse_double_complex* alpha,
                                       const rocsparse_mat_descr       descr,
                                       const rocsparse_double_complex* bsr_val,
                                       const rocsparse_int*            bsr_row_ptr,
                                       const rocsparse_int*            bsr_col_ind,
                                       rocsparse_int                   bsr_dim,
                                       rocsparse_mat_info              info,
                                       const rocsparse_double_complex* x,
                                       rocsparse_double_complex*       y,
                                       rocsparse_solve_policy          policy,
                                       void*                           temp_buffer)
{
    return rocsparse_zbsrsv_solve(handle,
                                  dir,
                                  trans,
                                  mb,
                                  nnzb,
                                  alpha,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  x,
                                  y,
                                  policy,
                                  temp_buffer);
}

// coomv
template <>
rocsparse_status rocsparse_coomv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             nnz,
                                 const float*              alpha,
                                 const rocsparse_mat_descr descr,
                                 const float*              coo_val,
                                 const rocsparse_int*      coo_row_ind,
                                 const rocsparse_int*      coo_col_ind,
                                 const float*              x,
                                 const float*              beta,
                                 float*                    y)
{
    return rocsparse_scoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

template <>
rocsparse_status rocsparse_coomv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             nnz,
                                 const double*             alpha,
                                 const rocsparse_mat_descr descr,
                                 const double*             coo_val,
                                 const rocsparse_int*      coo_row_ind,
                                 const rocsparse_int*      coo_col_ind,
                                 const double*             x,
                                 const double*             beta,
                                 double*                   y)
{
    return rocsparse_dcoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

template <>
rocsparse_status rocsparse_coomv(rocsparse_handle               handle,
                                 rocsparse_operation            trans,
                                 rocsparse_int                  m,
                                 rocsparse_int                  n,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* coo_val,
                                 const rocsparse_int*           coo_row_ind,
                                 const rocsparse_int*           coo_col_ind,
                                 const rocsparse_float_complex* x,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       y)
{
    return rocsparse_ccoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

template <>
rocsparse_status rocsparse_coomv(rocsparse_handle                handle,
                                 rocsparse_operation             trans,
                                 rocsparse_int                   m,
                                 rocsparse_int                   n,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* coo_val,
                                 const rocsparse_int*            coo_row_ind,
                                 const rocsparse_int*            coo_col_ind,
                                 const rocsparse_double_complex* x,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       y)
{
    return rocsparse_zcoomv(
        handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
}

// csrmv
template <>
rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             nnz,
                                          const rocsparse_mat_descr descr,
                                          const float*              csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          rocsparse_mat_info        info)
{
    return rocsparse_scsrmv_analysis(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

template <>
rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_int             m,
                                          rocsparse_int             n,
                                          rocsparse_int             nnz,
                                          const rocsparse_mat_descr descr,
                                          const double*             csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          rocsparse_mat_info        info)
{
    return rocsparse_dcsrmv_analysis(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

template <>
rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle               handle,
                                          rocsparse_operation            trans,
                                          rocsparse_int                  m,
                                          rocsparse_int                  n,
                                          rocsparse_int                  nnz,
                                          const rocsparse_mat_descr      descr,
                                          const rocsparse_float_complex* csr_val,
                                          const rocsparse_int*           csr_row_ptr,
                                          const rocsparse_int*           csr_col_ind,
                                          rocsparse_mat_info             info)
{
    return rocsparse_ccsrmv_analysis(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

template <>
rocsparse_status rocsparse_csrmv_analysis(rocsparse_handle                handle,
                                          rocsparse_operation             trans,
                                          rocsparse_int                   m,
                                          rocsparse_int                   n,
                                          rocsparse_int                   nnz,
                                          const rocsparse_mat_descr       descr,
                                          const rocsparse_double_complex* csr_val,
                                          const rocsparse_int*            csr_row_ptr,
                                          const rocsparse_int*            csr_col_ind,
                                          rocsparse_mat_info              info)
{
    return rocsparse_zcsrmv_analysis(
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             nnz,
                                 const float*              alpha,
                                 const rocsparse_mat_descr descr,
                                 const float*              csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 rocsparse_mat_info        info,
                                 const float*              x,
                                 const float*              beta,
                                 float*                    y)
{
    return rocsparse_scsrmv(handle,
                            trans,
                            m,
                            n,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            info,
                            x,
                            beta,
                            y);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             nnz,
                                 const double*             alpha,
                                 const rocsparse_mat_descr descr,
                                 const double*             csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 rocsparse_mat_info        info,
                                 const double*             x,
                                 const double*             beta,
                                 double*                   y)
{
    return rocsparse_dcsrmv(handle,
                            trans,
                            m,
                            n,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            info,
                            x,
                            beta,
                            y);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle               handle,
                                 rocsparse_operation            trans,
                                 rocsparse_int                  m,
                                 rocsparse_int                  n,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* csr_val,
                                 const rocsparse_int*           csr_row_ptr,
                                 const rocsparse_int*           csr_col_ind,
                                 rocsparse_mat_info             info,
                                 const rocsparse_float_complex* x,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       y)
{
    return rocsparse_ccsrmv(handle,
                            trans,
                            m,
                            n,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            info,
                            x,
                            beta,
                            y);
}

template <>
rocsparse_status rocsparse_csrmv(rocsparse_handle                handle,
                                 rocsparse_operation             trans,
                                 rocsparse_int                   m,
                                 rocsparse_int                   n,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* csr_val,
                                 const rocsparse_int*            csr_row_ptr,
                                 const rocsparse_int*            csr_col_ind,
                                 rocsparse_mat_info              info,
                                 const rocsparse_double_complex* x,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       y)
{
    return rocsparse_zcsrmv(handle,
                            trans,
                            m,
                            n,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            info,
                            x,
                            beta,
                            y);
}

// csrsv
template <>
rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const float*              csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             size_t*                   buffer_size)
{
    return rocsparse_scsrsv_buffer_size(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             const double*             csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             size_t*                   buffer_size)
{
    return rocsparse_dcsrsv_buffer_size(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle               handle,
                                             rocsparse_operation            trans,
                                             rocsparse_int                  m,
                                             rocsparse_int                  nnz,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* csr_val,
                                             const rocsparse_int*           csr_row_ptr,
                                             const rocsparse_int*           csr_col_ind,
                                             rocsparse_mat_info             info,
                                             size_t*                        buffer_size)
{
    return rocsparse_ccsrsv_buffer_size(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle                handle,
                                             rocsparse_operation             trans,
                                             rocsparse_int                   m,
                                             rocsparse_int                   nnz,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* csr_val,
                                             const rocsparse_int*            csr_row_ptr,
                                             const rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info              info,
                                             size_t*                         buffer_size)
{
    return rocsparse_zcsrsv_buffer_size(
        handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_int             m,
                                          rocsparse_int             nnz,
                                          const rocsparse_mat_descr descr,
                                          const float*              csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer)
{
    return rocsparse_scsrsv_analysis(handle,
                                     trans,
                                     m,
                                     nnz,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          rocsparse_int             m,
                                          rocsparse_int             nnz,
                                          const rocsparse_mat_descr descr,
                                          const double*             csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer)
{
    return rocsparse_dcsrsv_analysis(handle,
                                     trans,
                                     m,
                                     nnz,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle               handle,
                                          rocsparse_operation            trans,
                                          rocsparse_int                  m,
                                          rocsparse_int                  nnz,
                                          const rocsparse_mat_descr      descr,
                                          const rocsparse_float_complex* csr_val,
                                          const rocsparse_int*           csr_row_ptr,
                                          const rocsparse_int*           csr_col_ind,
                                          rocsparse_mat_info             info,
                                          rocsparse_analysis_policy      analysis,
                                          rocsparse_solve_policy         solve,
                                          void*                          temp_buffer)
{
    return rocsparse_ccsrsv_analysis(handle,
                                     trans,
                                     m,
                                     nnz,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsv_analysis(rocsparse_handle                handle,
                                          rocsparse_operation             trans,
                                          rocsparse_int                   m,
                                          rocsparse_int                   nnz,
                                          const rocsparse_mat_descr       descr,
                                          const rocsparse_double_complex* csr_val,
                                          const rocsparse_int*            csr_row_ptr,
                                          const rocsparse_int*            csr_col_ind,
                                          rocsparse_mat_info              info,
                                          rocsparse_analysis_policy       analysis,
                                          rocsparse_solve_policy          solve,
                                          void*                           temp_buffer)
{
    return rocsparse_zcsrsv_analysis(handle,
                                     trans,
                                     m,
                                     nnz,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsv_solve(rocsparse_handle          handle,
                                       rocsparse_operation       trans,
                                       rocsparse_int             m,
                                       rocsparse_int             nnz,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr,
                                       const float*              csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_mat_info        info,
                                       const float*              x,
                                       float*                    y,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
{
    return rocsparse_scsrsv_solve(handle,
                                  trans,
                                  m,
                                  nnz,
                                  alpha,
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

template <>
rocsparse_status rocsparse_csrsv_solve(rocsparse_handle          handle,
                                       rocsparse_operation       trans,
                                       rocsparse_int             m,
                                       rocsparse_int             nnz,
                                       const double*             alpha,
                                       const rocsparse_mat_descr descr,
                                       const double*             csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_mat_info        info,
                                       const double*             x,
                                       double*                   y,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
{
    return rocsparse_dcsrsv_solve(handle,
                                  trans,
                                  m,
                                  nnz,
                                  alpha,
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

template <>
rocsparse_status rocsparse_csrsv_solve(rocsparse_handle               handle,
                                       rocsparse_operation            trans,
                                       rocsparse_int                  m,
                                       rocsparse_int                  nnz,
                                       const rocsparse_float_complex* alpha,
                                       const rocsparse_mat_descr      descr,
                                       const rocsparse_float_complex* csr_val,
                                       const rocsparse_int*           csr_row_ptr,
                                       const rocsparse_int*           csr_col_ind,
                                       rocsparse_mat_info             info,
                                       const rocsparse_float_complex* x,
                                       rocsparse_float_complex*       y,
                                       rocsparse_solve_policy         policy,
                                       void*                          temp_buffer)
{
    return rocsparse_ccsrsv_solve(handle,
                                  trans,
                                  m,
                                  nnz,
                                  alpha,
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

template <>
rocsparse_status rocsparse_csrsv_solve(rocsparse_handle                handle,
                                       rocsparse_operation             trans,
                                       rocsparse_int                   m,
                                       rocsparse_int                   nnz,
                                       const rocsparse_double_complex* alpha,
                                       const rocsparse_mat_descr       descr,
                                       const rocsparse_double_complex* csr_val,
                                       const rocsparse_int*            csr_row_ptr,
                                       const rocsparse_int*            csr_col_ind,
                                       rocsparse_mat_info              info,
                                       const rocsparse_double_complex* x,
                                       rocsparse_double_complex*       y,
                                       rocsparse_solve_policy          policy,
                                       void*                           temp_buffer)
{
    return rocsparse_zcsrsv_solve(handle,
                                  trans,
                                  m,
                                  nnz,
                                  alpha,
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

// ellmv
template <>
rocsparse_status rocsparse_ellmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 const float*              alpha,
                                 const rocsparse_mat_descr descr,
                                 const float*              ell_val,
                                 const rocsparse_int*      ell_col_ind,
                                 rocsparse_int             ell_width,
                                 const float*              x,
                                 const float*              beta,
                                 float*                    y)
{
    return rocsparse_sellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

template <>
rocsparse_status rocsparse_ellmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 const double*             alpha,
                                 const rocsparse_mat_descr descr,
                                 const double*             ell_val,
                                 const rocsparse_int*      ell_col_ind,
                                 rocsparse_int             ell_width,
                                 const double*             x,
                                 const double*             beta,
                                 double*                   y)
{
    return rocsparse_dellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

template <>
rocsparse_status rocsparse_ellmv(rocsparse_handle               handle,
                                 rocsparse_operation            trans,
                                 rocsparse_int                  m,
                                 rocsparse_int                  n,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* ell_val,
                                 const rocsparse_int*           ell_col_ind,
                                 rocsparse_int                  ell_width,
                                 const rocsparse_float_complex* x,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       y)
{
    return rocsparse_cellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

template <>
rocsparse_status rocsparse_ellmv(rocsparse_handle                handle,
                                 rocsparse_operation             trans,
                                 rocsparse_int                   m,
                                 rocsparse_int                   n,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* ell_val,
                                 const rocsparse_int*            ell_col_ind,
                                 rocsparse_int                   ell_width,
                                 const rocsparse_double_complex* x,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       y)
{
    return rocsparse_zellmv(
        handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
}

// hybmv
template <>
rocsparse_status rocsparse_hybmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 const float*              alpha,
                                 const rocsparse_mat_descr descr,
                                 const rocsparse_hyb_mat   hyb,
                                 const float*              x,
                                 const float*              beta,
                                 float*                    y)
{
    return rocsparse_shybmv(handle, trans, alpha, descr, hyb, x, beta, y);
}

template <>
rocsparse_status rocsparse_hybmv(rocsparse_handle          handle,
                                 rocsparse_operation       trans,
                                 const double*             alpha,
                                 const rocsparse_mat_descr descr,
                                 const rocsparse_hyb_mat   hyb,
                                 const double*             x,
                                 const double*             beta,
                                 double*                   y)
{
    return rocsparse_dhybmv(handle, trans, alpha, descr, hyb, x, beta, y);
}

template <>
rocsparse_status rocsparse_hybmv(rocsparse_handle               handle,
                                 rocsparse_operation            trans,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_hyb_mat        hyb,
                                 const rocsparse_float_complex* x,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       y)
{
    return rocsparse_chybmv(handle, trans, alpha, descr, hyb, x, beta, y);
}

template <>
rocsparse_status rocsparse_hybmv(rocsparse_handle                handle,
                                 rocsparse_operation             trans,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_hyb_mat         hyb,
                                 const rocsparse_double_complex* x,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       y)
{
    return rocsparse_zhybmv(handle, trans, alpha, descr, hyb, x, beta, y);
}

// bsrmv
template <>
rocsparse_status rocsparse_gebsrmv(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_operation       trans,
                                   rocsparse_int             mb,
                                   rocsparse_int             nb,
                                   rocsparse_int             nnzb,
                                   const float*              alpha,
                                   const rocsparse_mat_descr descr,
                                   const float*              bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             row_block_dim,
                                   rocsparse_int             col_block_dim,
                                   const float*              x,
                                   const float*              beta,
                                   float*                    y)
{
    return rocsparse_sgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

template <>
rocsparse_status rocsparse_gebsrmv(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_operation       trans,
                                   rocsparse_int             mb,
                                   rocsparse_int             nb,
                                   rocsparse_int             nnzb,
                                   const double*             alpha,
                                   const rocsparse_mat_descr descr,
                                   const double*             bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             row_block_dim,
                                   rocsparse_int             col_block_dim,
                                   const double*             x,
                                   const double*             beta,
                                   double*                   y)
{
    return rocsparse_dgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

template <>
rocsparse_status rocsparse_gebsrmv(rocsparse_handle               handle,
                                   rocsparse_direction            dir,
                                   rocsparse_operation            trans,
                                   rocsparse_int                  mb,
                                   rocsparse_int                  nb,
                                   rocsparse_int                  nnzb,
                                   const rocsparse_float_complex* alpha,
                                   const rocsparse_mat_descr      descr,
                                   const rocsparse_float_complex* bsr_val,
                                   const rocsparse_int*           bsr_row_ptr,
                                   const rocsparse_int*           bsr_col_ind,
                                   rocsparse_int                  row_block_dim,
                                   rocsparse_int                  col_block_dim,
                                   const rocsparse_float_complex* x,
                                   const rocsparse_float_complex* beta,
                                   rocsparse_float_complex*       y)
{
    return rocsparse_cgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

template <>
rocsparse_status rocsparse_gebsrmv(rocsparse_handle                handle,
                                   rocsparse_direction             dir,
                                   rocsparse_operation             trans,
                                   rocsparse_int                   mb,
                                   rocsparse_int                   nb,
                                   rocsparse_int                   nnzb,
                                   const rocsparse_double_complex* alpha,
                                   const rocsparse_mat_descr       descr,
                                   const rocsparse_double_complex* bsr_val,
                                   const rocsparse_int*            bsr_row_ptr,
                                   const rocsparse_int*            bsr_col_ind,
                                   rocsparse_int                   row_block_dim,
                                   rocsparse_int                   col_block_dim,
                                   const rocsparse_double_complex* x,
                                   const rocsparse_double_complex* beta,
                                   rocsparse_double_complex*       y)
{
    return rocsparse_zgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

// gemvi
template <>
rocsparse_status rocsparse_gemvi_buffer_size<float>(rocsparse_handle    handle,
                                                    rocsparse_operation trans,
                                                    rocsparse_int       m,
                                                    rocsparse_int       n,
                                                    rocsparse_int       nnz,
                                                    size_t*             buffer_size)
{
    return rocsparse_sgemvi_buffer_size(handle, trans, m, n, nnz, buffer_size);
}

template <>
rocsparse_status rocsparse_gemvi_buffer_size<double>(rocsparse_handle    handle,
                                                     rocsparse_operation trans,
                                                     rocsparse_int       m,
                                                     rocsparse_int       n,
                                                     rocsparse_int       nnz,
                                                     size_t*             buffer_size)
{
    return rocsparse_dgemvi_buffer_size(handle, trans, m, n, nnz, buffer_size);
}

template <>
rocsparse_status rocsparse_gemvi_buffer_size<rocsparse_float_complex>(rocsparse_handle    handle,
                                                                      rocsparse_operation trans,
                                                                      rocsparse_int       m,
                                                                      rocsparse_int       n,
                                                                      rocsparse_int       nnz,
                                                                      size_t* buffer_size)
{
    return rocsparse_cgemvi_buffer_size(handle, trans, m, n, nnz, buffer_size);
}

template <>
rocsparse_status rocsparse_gemvi_buffer_size<rocsparse_double_complex>(rocsparse_handle    handle,
                                                                       rocsparse_operation trans,
                                                                       rocsparse_int       m,
                                                                       rocsparse_int       n,
                                                                       rocsparse_int       nnz,
                                                                       size_t* buffer_size)
{
    return rocsparse_zgemvi_buffer_size(handle, trans, m, n, nnz, buffer_size);
}

template <>
rocsparse_status rocsparse_gemvi(rocsparse_handle     handle,
                                 rocsparse_operation  trans,
                                 rocsparse_int        m,
                                 rocsparse_int        n,
                                 const float*         alpha,
                                 const float*         A,
                                 rocsparse_int        lda,
                                 rocsparse_int        nnz,
                                 const float*         x_val,
                                 const rocsparse_int* x_ind,
                                 const float*         beta,
                                 float*               y,
                                 rocsparse_index_base idx_base,
                                 void*                temp_buffer)
{
    return rocsparse_sgemvi(
        handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

template <>
rocsparse_status rocsparse_gemvi(rocsparse_handle     handle,
                                 rocsparse_operation  trans,
                                 rocsparse_int        m,
                                 rocsparse_int        n,
                                 const double*        alpha,
                                 const double*        A,
                                 rocsparse_int        lda,
                                 rocsparse_int        nnz,
                                 const double*        x_val,
                                 const rocsparse_int* x_ind,
                                 const double*        beta,
                                 double*              y,
                                 rocsparse_index_base idx_base,
                                 void*                temp_buffer)
{
    return rocsparse_dgemvi(
        handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

template <>
rocsparse_status rocsparse_gemvi(rocsparse_handle               handle,
                                 rocsparse_operation            trans,
                                 rocsparse_int                  m,
                                 rocsparse_int                  n,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_float_complex* A,
                                 rocsparse_int                  lda,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* x_val,
                                 const rocsparse_int*           x_ind,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       y,
                                 rocsparse_index_base           idx_base,
                                 void*                          temp_buffer)
{
    return rocsparse_cgemvi(
        handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

template <>
rocsparse_status rocsparse_gemvi(rocsparse_handle                handle,
                                 rocsparse_operation             trans,
                                 rocsparse_int                   m,
                                 rocsparse_int                   n,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_double_complex* A,
                                 rocsparse_int                   lda,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* x_val,
                                 const rocsparse_int*            x_ind,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       y,
                                 rocsparse_index_base            idx_base,
                                 void*                           temp_buffer)
{
    return rocsparse_zgemvi(
        handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
// bsrmm
template <>
rocsparse_status rocsparse_bsrmm(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             mb,
                                 rocsparse_int             n,
                                 rocsparse_int             kb,
                                 rocsparse_int             nnzb,
                                 const float*              alpha,
                                 const rocsparse_mat_descr descr,
                                 const float*              bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             block_dim,
                                 const float*              B,
                                 rocsparse_int             ldb,
                                 const float*              beta,
                                 float*                    C,
                                 rocsparse_int             ldc)
{
    return rocsparse_sbsrmm(handle,
                            dir,
                            trans_A,
                            trans_B,
                            mb,
                            n,
                            kb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            block_dim,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_bsrmm(rocsparse_handle          handle,
                                 rocsparse_direction       dir,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             mb,
                                 rocsparse_int             n,
                                 rocsparse_int             kb,
                                 rocsparse_int             nnzb,
                                 const double*             alpha,
                                 const rocsparse_mat_descr descr,
                                 const double*             bsr_val,
                                 const rocsparse_int*      bsr_row_ptr,
                                 const rocsparse_int*      bsr_col_ind,
                                 rocsparse_int             block_dim,
                                 const double*             B,
                                 rocsparse_int             ldb,
                                 const double*             beta,
                                 double*                   C,
                                 rocsparse_int             ldc)
{
    return rocsparse_dbsrmm(handle,
                            dir,
                            trans_A,
                            trans_B,
                            mb,
                            n,
                            kb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            block_dim,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_bsrmm(rocsparse_handle               handle,
                                 rocsparse_direction            dir,
                                 rocsparse_operation            trans_A,
                                 rocsparse_operation            trans_B,
                                 rocsparse_int                  mb,
                                 rocsparse_int                  n,
                                 rocsparse_int                  kb,
                                 rocsparse_int                  nnzb,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* bsr_val,
                                 const rocsparse_int*           bsr_row_ptr,
                                 const rocsparse_int*           bsr_col_ind,
                                 rocsparse_int                  block_dim,
                                 const rocsparse_float_complex* B,
                                 rocsparse_int                  ldb,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       C,
                                 rocsparse_int                  ldc)
{
    return rocsparse_cbsrmm(handle,
                            dir,
                            trans_A,
                            trans_B,
                            mb,
                            n,
                            kb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            block_dim,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_bsrmm(rocsparse_handle                handle,
                                 rocsparse_direction             dir,
                                 rocsparse_operation             trans_A,
                                 rocsparse_operation             trans_B,
                                 rocsparse_int                   mb,
                                 rocsparse_int                   n,
                                 rocsparse_int                   kb,
                                 rocsparse_int                   nnzb,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* bsr_val,
                                 const rocsparse_int*            bsr_row_ptr,
                                 const rocsparse_int*            bsr_col_ind,
                                 rocsparse_int                   block_dim,
                                 const rocsparse_double_complex* B,
                                 rocsparse_int                   ldb,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       C,
                                 rocsparse_int                   ldc)
{
    return rocsparse_zbsrmm(handle,
                            dir,
                            trans_A,
                            trans_B,
                            mb,
                            n,
                            kb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            block_dim,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

// gebsrmm
template <>
rocsparse_status rocsparse_gebsrmm(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_operation       trans_A,
                                   rocsparse_operation       trans_B,
                                   rocsparse_int             mb,
                                   rocsparse_int             n,
                                   rocsparse_int             kb,
                                   rocsparse_int             nnzb,
                                   const float*              alpha,
                                   const rocsparse_mat_descr descr,
                                   const float*              bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             row_block_dim,
                                   rocsparse_int             col_block_dim,
                                   const float*              B,
                                   rocsparse_int             ldb,
                                   const float*              beta,
                                   float*                    C,
                                   rocsparse_int             ldc)
{
    return rocsparse_sgebsrmm(handle,
                              dir,
                              trans_A,
                              trans_B,
                              mb,
                              n,
                              kb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              B,
                              ldb,
                              beta,
                              C,
                              ldc);
}

template <>
rocsparse_status rocsparse_gebsrmm(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_operation       trans_A,
                                   rocsparse_operation       trans_B,
                                   rocsparse_int             mb,
                                   rocsparse_int             n,
                                   rocsparse_int             kb,
                                   rocsparse_int             nnzb,
                                   const double*             alpha,
                                   const rocsparse_mat_descr descr,
                                   const double*             bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             row_block_dim,
                                   rocsparse_int             col_block_dim,
                                   const double*             B,
                                   rocsparse_int             ldb,
                                   const double*             beta,
                                   double*                   C,
                                   rocsparse_int             ldc)
{
    return rocsparse_dgebsrmm(handle,
                              dir,
                              trans_A,
                              trans_B,
                              mb,
                              n,
                              kb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              B,
                              ldb,
                              beta,
                              C,
                              ldc);
}

template <>
rocsparse_status rocsparse_gebsrmm(rocsparse_handle               handle,
                                   rocsparse_direction            dir,
                                   rocsparse_operation            trans_A,
                                   rocsparse_operation            trans_B,
                                   rocsparse_int                  mb,
                                   rocsparse_int                  n,
                                   rocsparse_int                  kb,
                                   rocsparse_int                  nnzb,
                                   const rocsparse_float_complex* alpha,
                                   const rocsparse_mat_descr      descr,
                                   const rocsparse_float_complex* bsr_val,
                                   const rocsparse_int*           bsr_row_ptr,
                                   const rocsparse_int*           bsr_col_ind,
                                   rocsparse_int                  row_block_dim,
                                   rocsparse_int                  col_block_dim,
                                   const rocsparse_float_complex* B,
                                   rocsparse_int                  ldb,
                                   const rocsparse_float_complex* beta,
                                   rocsparse_float_complex*       C,
                                   rocsparse_int                  ldc)
{
    return rocsparse_cgebsrmm(handle,
                              dir,
                              trans_A,
                              trans_B,
                              mb,
                              n,
                              kb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              B,
                              ldb,
                              beta,
                              C,
                              ldc);
}

template <>
rocsparse_status rocsparse_gebsrmm(rocsparse_handle                handle,
                                   rocsparse_direction             dir,
                                   rocsparse_operation             trans_A,
                                   rocsparse_operation             trans_B,
                                   rocsparse_int                   mb,
                                   rocsparse_int                   n,
                                   rocsparse_int                   kb,
                                   rocsparse_int                   nnzb,
                                   const rocsparse_double_complex* alpha,
                                   const rocsparse_mat_descr       descr,
                                   const rocsparse_double_complex* bsr_val,
                                   const rocsparse_int*            bsr_row_ptr,
                                   const rocsparse_int*            bsr_col_ind,
                                   rocsparse_int                   row_block_dim,
                                   rocsparse_int                   col_block_dim,
                                   const rocsparse_double_complex* B,
                                   rocsparse_int                   ldb,
                                   const rocsparse_double_complex* beta,
                                   rocsparse_double_complex*       C,
                                   rocsparse_int                   ldc)
{
    return rocsparse_zgebsrmm(handle,
                              dir,
                              trans_A,
                              trans_B,
                              mb,
                              n,
                              kb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              B,
                              ldb,
                              beta,
                              C,
                              ldc);
}

// csrmm
template <>
rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             k,
                                 rocsparse_int             nnz,
                                 const float*              alpha,
                                 const rocsparse_mat_descr descr,
                                 const float*              csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 const float*              B,
                                 rocsparse_int             ldb,
                                 const float*              beta,
                                 float*                    C,
                                 rocsparse_int             ldc)
{
    return rocsparse_scsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             k,
                                 rocsparse_int             nnz,
                                 const double*             alpha,
                                 const rocsparse_mat_descr descr,
                                 const double*             csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 const double*             B,
                                 rocsparse_int             ldb,
                                 const double*             beta,
                                 double*                   C,
                                 rocsparse_int             ldc)
{
    return rocsparse_dcsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_csrmm(rocsparse_handle               handle,
                                 rocsparse_operation            trans_A,
                                 rocsparse_operation            trans_B,
                                 rocsparse_int                  m,
                                 rocsparse_int                  n,
                                 rocsparse_int                  k,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* csr_val,
                                 const rocsparse_int*           csr_row_ptr,
                                 const rocsparse_int*           csr_col_ind,
                                 const rocsparse_float_complex* B,
                                 rocsparse_int                  ldb,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       C,
                                 rocsparse_int                  ldc)
{
    return rocsparse_ccsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_csrmm(rocsparse_handle                handle,
                                 rocsparse_operation             trans_A,
                                 rocsparse_operation             trans_B,
                                 rocsparse_int                   m,
                                 rocsparse_int                   n,
                                 rocsparse_int                   k,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* csr_val,
                                 const rocsparse_int*            csr_row_ptr,
                                 const rocsparse_int*            csr_col_ind,
                                 const rocsparse_double_complex* B,
                                 rocsparse_int                   ldb,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       C,
                                 rocsparse_int                   ldc)
{
    return rocsparse_zcsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

// csrsm
template <>
rocsparse_status rocsparse_csrsm_buffer_size(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_int             m,
                                             rocsparse_int             nrhs,
                                             rocsparse_int             nnz,
                                             const float*              alpha,
                                             const rocsparse_mat_descr descr,
                                             const float*              csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             const float*              B,
                                             rocsparse_int             ldb,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             size_t*                   buffer_size)
{
    return rocsparse_scsrsm_buffer_size(handle,
                                        trans_A,
                                        trans_B,
                                        m,
                                        nrhs,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        B,
                                        ldb,
                                        info,
                                        policy,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_csrsm_buffer_size(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_int             m,
                                             rocsparse_int             nrhs,
                                             rocsparse_int             nnz,
                                             const double*             alpha,
                                             const rocsparse_mat_descr descr,
                                             const double*             csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             const double*             B,
                                             rocsparse_int             ldb,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             size_t*                   buffer_size)
{
    return rocsparse_dcsrsm_buffer_size(handle,
                                        trans_A,
                                        trans_B,
                                        m,
                                        nrhs,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        B,
                                        ldb,
                                        info,
                                        policy,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_csrsm_buffer_size(rocsparse_handle               handle,
                                             rocsparse_operation            trans_A,
                                             rocsparse_operation            trans_B,
                                             rocsparse_int                  m,
                                             rocsparse_int                  nrhs,
                                             rocsparse_int                  nnz,
                                             const rocsparse_float_complex* alpha,
                                             const rocsparse_mat_descr      descr,
                                             const rocsparse_float_complex* csr_val,
                                             const rocsparse_int*           csr_row_ptr,
                                             const rocsparse_int*           csr_col_ind,
                                             const rocsparse_float_complex* B,
                                             rocsparse_int                  ldb,
                                             rocsparse_mat_info             info,
                                             rocsparse_solve_policy         policy,
                                             size_t*                        buffer_size)
{
    return rocsparse_ccsrsm_buffer_size(handle,
                                        trans_A,
                                        trans_B,
                                        m,
                                        nrhs,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        B,
                                        ldb,
                                        info,
                                        policy,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_csrsm_buffer_size(rocsparse_handle                handle,
                                             rocsparse_operation             trans_A,
                                             rocsparse_operation             trans_B,
                                             rocsparse_int                   m,
                                             rocsparse_int                   nrhs,
                                             rocsparse_int                   nnz,
                                             const rocsparse_double_complex* alpha,
                                             const rocsparse_mat_descr       descr,
                                             const rocsparse_double_complex* csr_val,
                                             const rocsparse_int*            csr_row_ptr,
                                             const rocsparse_int*            csr_col_ind,
                                             const rocsparse_double_complex* B,
                                             rocsparse_int                   ldb,
                                             rocsparse_mat_info              info,
                                             rocsparse_solve_policy          policy,
                                             size_t*                         buffer_size)
{
    return rocsparse_zcsrsm_buffer_size(handle,
                                        trans_A,
                                        trans_B,
                                        m,
                                        nrhs,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        B,
                                        ldb,
                                        info,
                                        policy,
                                        buffer_size);
}

template <>
rocsparse_status rocsparse_csrsm_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_int             m,
                                          rocsparse_int             nrhs,
                                          rocsparse_int             nnz,
                                          const float*              alpha,
                                          const rocsparse_mat_descr descr,
                                          const float*              csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          const float*              B,
                                          rocsparse_int             ldb,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer)
{
    return rocsparse_scsrsm_analysis(handle,
                                     trans_A,
                                     trans_B,
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     B,
                                     ldb,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsm_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          rocsparse_int             m,
                                          rocsparse_int             nrhs,
                                          rocsparse_int             nnz,
                                          const double*             alpha,
                                          const rocsparse_mat_descr descr,
                                          const double*             csr_val,
                                          const rocsparse_int*      csr_row_ptr,
                                          const rocsparse_int*      csr_col_ind,
                                          const double*             B,
                                          rocsparse_int             ldb,
                                          rocsparse_mat_info        info,
                                          rocsparse_analysis_policy analysis,
                                          rocsparse_solve_policy    solve,
                                          void*                     temp_buffer)
{
    return rocsparse_dcsrsm_analysis(handle,
                                     trans_A,
                                     trans_B,
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     B,
                                     ldb,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsm_analysis(rocsparse_handle               handle,
                                          rocsparse_operation            trans_A,
                                          rocsparse_operation            trans_B,
                                          rocsparse_int                  m,
                                          rocsparse_int                  nrhs,
                                          rocsparse_int                  nnz,
                                          const rocsparse_float_complex* alpha,
                                          const rocsparse_mat_descr      descr,
                                          const rocsparse_float_complex* csr_val,
                                          const rocsparse_int*           csr_row_ptr,
                                          const rocsparse_int*           csr_col_ind,
                                          const rocsparse_float_complex* B,
                                          rocsparse_int                  ldb,
                                          rocsparse_mat_info             info,
                                          rocsparse_analysis_policy      analysis,
                                          rocsparse_solve_policy         solve,
                                          void*                          temp_buffer)
{
    return rocsparse_ccsrsm_analysis(handle,
                                     trans_A,
                                     trans_B,
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     B,
                                     ldb,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsm_analysis(rocsparse_handle                handle,
                                          rocsparse_operation             trans_A,
                                          rocsparse_operation             trans_B,
                                          rocsparse_int                   m,
                                          rocsparse_int                   nrhs,
                                          rocsparse_int                   nnz,
                                          const rocsparse_double_complex* alpha,
                                          const rocsparse_mat_descr       descr,
                                          const rocsparse_double_complex* csr_val,
                                          const rocsparse_int*            csr_row_ptr,
                                          const rocsparse_int*            csr_col_ind,
                                          const rocsparse_double_complex* B,
                                          rocsparse_int                   ldb,
                                          rocsparse_mat_info              info,
                                          rocsparse_analysis_policy       analysis,
                                          rocsparse_solve_policy          solve,
                                          void*                           temp_buffer)
{
    return rocsparse_zcsrsm_analysis(handle,
                                     trans_A,
                                     trans_B,
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     descr,
                                     csr_val,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     B,
                                     ldb,
                                     info,
                                     analysis,
                                     solve,
                                     temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsm_solve(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             nrhs,
                                       rocsparse_int             nnz,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr,
                                       const float*              csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       float*                    B,
                                       rocsparse_int             ldb,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
{
    return rocsparse_scsrsm_solve(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  nrhs,
                                  nnz,
                                  alpha,
                                  descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  B,
                                  ldb,
                                  info,
                                  policy,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsm_solve(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       rocsparse_int             m,
                                       rocsparse_int             nrhs,
                                       rocsparse_int             nnz,
                                       const double*             alpha,
                                       const rocsparse_mat_descr descr,
                                       const double*             csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       double*                   B,
                                       rocsparse_int             ldb,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
{
    return rocsparse_dcsrsm_solve(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  nrhs,
                                  nnz,
                                  alpha,
                                  descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  B,
                                  ldb,
                                  info,
                                  policy,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsm_solve(rocsparse_handle               handle,
                                       rocsparse_operation            trans_A,
                                       rocsparse_operation            trans_B,
                                       rocsparse_int                  m,
                                       rocsparse_int                  nrhs,
                                       rocsparse_int                  nnz,
                                       const rocsparse_float_complex* alpha,
                                       const rocsparse_mat_descr      descr,
                                       const rocsparse_float_complex* csr_val,
                                       const rocsparse_int*           csr_row_ptr,
                                       const rocsparse_int*           csr_col_ind,
                                       rocsparse_float_complex*       B,
                                       rocsparse_int                  ldb,
                                       rocsparse_mat_info             info,
                                       rocsparse_solve_policy         policy,
                                       void*                          temp_buffer)
{
    return rocsparse_ccsrsm_solve(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  nrhs,
                                  nnz,
                                  alpha,
                                  descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  B,
                                  ldb,
                                  info,
                                  policy,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_csrsm_solve(rocsparse_handle                handle,
                                       rocsparse_operation             trans_A,
                                       rocsparse_operation             trans_B,
                                       rocsparse_int                   m,
                                       rocsparse_int                   nrhs,
                                       rocsparse_int                   nnz,
                                       const rocsparse_double_complex* alpha,
                                       const rocsparse_mat_descr       descr,
                                       const rocsparse_double_complex* csr_val,
                                       const rocsparse_int*            csr_row_ptr,
                                       const rocsparse_int*            csr_col_ind,
                                       rocsparse_double_complex*       B,
                                       rocsparse_int                   ldb,
                                       rocsparse_mat_info              info,
                                       rocsparse_solve_policy          policy,
                                       void*                           temp_buffer)
{
    return rocsparse_zcsrsm_solve(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  nrhs,
                                  nnz,
                                  alpha,
                                  descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  B,
                                  ldb,
                                  info,
                                  policy,
                                  temp_buffer);
}

// gemmi
template <>
rocsparse_status rocsparse_gemmi(rocsparse_handle          handle,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             k,
                                 rocsparse_int             nnz,
                                 const float*              alpha,
                                 const float*              A,
                                 rocsparse_int             lda,
                                 const rocsparse_mat_descr descr,
                                 const float*              csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 const float*              beta,
                                 float*                    C,
                                 rocsparse_int             ldc)
{
    return rocsparse_sgemmi(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            A,
                            lda,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_gemmi(rocsparse_handle          handle,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 rocsparse_int             m,
                                 rocsparse_int             n,
                                 rocsparse_int             k,
                                 rocsparse_int             nnz,
                                 const double*             alpha,
                                 const double*             A,
                                 rocsparse_int             lda,
                                 const rocsparse_mat_descr descr,
                                 const double*             csr_val,
                                 const rocsparse_int*      csr_row_ptr,
                                 const rocsparse_int*      csr_col_ind,
                                 const double*             beta,
                                 double*                   C,
                                 rocsparse_int             ldc)
{
    return rocsparse_dgemmi(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            A,
                            lda,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_gemmi(rocsparse_handle               handle,
                                 rocsparse_operation            trans_A,
                                 rocsparse_operation            trans_B,
                                 rocsparse_int                  m,
                                 rocsparse_int                  n,
                                 rocsparse_int                  k,
                                 rocsparse_int                  nnz,
                                 const rocsparse_float_complex* alpha,
                                 const rocsparse_float_complex* A,
                                 rocsparse_int                  lda,
                                 const rocsparse_mat_descr      descr,
                                 const rocsparse_float_complex* csr_val,
                                 const rocsparse_int*           csr_row_ptr,
                                 const rocsparse_int*           csr_col_ind,
                                 const rocsparse_float_complex* beta,
                                 rocsparse_float_complex*       C,
                                 rocsparse_int                  ldc)
{
    return rocsparse_cgemmi(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            A,
                            lda,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            beta,
                            C,
                            ldc);
}

template <>
rocsparse_status rocsparse_gemmi(rocsparse_handle                handle,
                                 rocsparse_operation             trans_A,
                                 rocsparse_operation             trans_B,
                                 rocsparse_int                   m,
                                 rocsparse_int                   n,
                                 rocsparse_int                   k,
                                 rocsparse_int                   nnz,
                                 const rocsparse_double_complex* alpha,
                                 const rocsparse_double_complex* A,
                                 rocsparse_int                   lda,
                                 const rocsparse_mat_descr       descr,
                                 const rocsparse_double_complex* csr_val,
                                 const rocsparse_int*            csr_row_ptr,
                                 const rocsparse_int*            csr_col_ind,
                                 const rocsparse_double_complex* beta,
                                 rocsparse_double_complex*       C,
                                 rocsparse_int                   ldc)
{
    return rocsparse_zgemmi(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            A,
                            lda,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            beta,
                            C,
                            ldc);
}

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
// csrgeam
template <>
rocsparse_status rocsparse_csrgeam(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const float*              alpha,
                                   const rocsparse_mat_descr descr_A,
                                   rocsparse_int             nnz_A,
                                   const float*              csr_val_A,
                                   const rocsparse_int*      csr_row_ptr_A,
                                   const rocsparse_int*      csr_col_ind_A,
                                   const float*              beta,
                                   const rocsparse_mat_descr descr_B,
                                   rocsparse_int             nnz_B,
                                   const float*              csr_val_B,
                                   const rocsparse_int*      csr_row_ptr_B,
                                   const rocsparse_int*      csr_col_ind_B,
                                   const rocsparse_mat_descr descr_C,
                                   float*                    csr_val_C,
                                   const rocsparse_int*      csr_row_ptr_C,
                                   rocsparse_int*            csr_col_ind_C)
{
    return rocsparse_scsrgeam(handle,
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
                              csr_col_ind_C);
}

template <>
rocsparse_status rocsparse_csrgeam(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const double*             alpha,
                                   const rocsparse_mat_descr descr_A,
                                   rocsparse_int             nnz_A,
                                   const double*             csr_val_A,
                                   const rocsparse_int*      csr_row_ptr_A,
                                   const rocsparse_int*      csr_col_ind_A,
                                   const double*             beta,
                                   const rocsparse_mat_descr descr_B,
                                   rocsparse_int             nnz_B,
                                   const double*             csr_val_B,
                                   const rocsparse_int*      csr_row_ptr_B,
                                   const rocsparse_int*      csr_col_ind_B,
                                   const rocsparse_mat_descr descr_C,
                                   double*                   csr_val_C,
                                   const rocsparse_int*      csr_row_ptr_C,
                                   rocsparse_int*            csr_col_ind_C)
{
    return rocsparse_dcsrgeam(handle,
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
                              csr_col_ind_C);
}

template <>
rocsparse_status rocsparse_csrgeam(rocsparse_handle               handle,
                                   rocsparse_int                  m,
                                   rocsparse_int                  n,
                                   const rocsparse_float_complex* alpha,
                                   const rocsparse_mat_descr      descr_A,
                                   rocsparse_int                  nnz_A,
                                   const rocsparse_float_complex* csr_val_A,
                                   const rocsparse_int*           csr_row_ptr_A,
                                   const rocsparse_int*           csr_col_ind_A,
                                   const rocsparse_float_complex* beta,
                                   const rocsparse_mat_descr      descr_B,
                                   rocsparse_int                  nnz_B,
                                   const rocsparse_float_complex* csr_val_B,
                                   const rocsparse_int*           csr_row_ptr_B,
                                   const rocsparse_int*           csr_col_ind_B,
                                   const rocsparse_mat_descr      descr_C,
                                   rocsparse_float_complex*       csr_val_C,
                                   const rocsparse_int*           csr_row_ptr_C,
                                   rocsparse_int*                 csr_col_ind_C)
{
    return rocsparse_ccsrgeam(handle,
                              m,
                              n,
                              (const rocsparse_float_complex*)alpha,
                              descr_A,
                              nnz_A,
                              (const rocsparse_float_complex*)csr_val_A,
                              csr_row_ptr_A,
                              csr_col_ind_A,
                              (const rocsparse_float_complex*)beta,
                              descr_B,
                              nnz_B,
                              (const rocsparse_float_complex*)csr_val_B,
                              csr_row_ptr_B,
                              csr_col_ind_B,
                              descr_C,
                              (rocsparse_float_complex*)csr_val_C,
                              csr_row_ptr_C,
                              csr_col_ind_C);
}

template <>
rocsparse_status rocsparse_csrgeam(rocsparse_handle                handle,
                                   rocsparse_int                   m,
                                   rocsparse_int                   n,
                                   const rocsparse_double_complex* alpha,
                                   const rocsparse_mat_descr       descr_A,
                                   rocsparse_int                   nnz_A,
                                   const rocsparse_double_complex* csr_val_A,
                                   const rocsparse_int*            csr_row_ptr_A,
                                   const rocsparse_int*            csr_col_ind_A,
                                   const rocsparse_double_complex* beta,
                                   const rocsparse_mat_descr       descr_B,
                                   rocsparse_int                   nnz_B,
                                   const rocsparse_double_complex* csr_val_B,
                                   const rocsparse_int*            csr_row_ptr_B,
                                   const rocsparse_int*            csr_col_ind_B,
                                   const rocsparse_mat_descr       descr_C,
                                   rocsparse_double_complex*       csr_val_C,
                                   const rocsparse_int*            csr_row_ptr_C,
                                   rocsparse_int*                  csr_col_ind_C)
{
    return rocsparse_zcsrgeam(handle,
                              m,
                              n,
                              (const rocsparse_double_complex*)alpha,
                              descr_A,
                              nnz_A,
                              (const rocsparse_double_complex*)csr_val_A,
                              csr_row_ptr_A,
                              csr_col_ind_A,
                              (const rocsparse_double_complex*)beta,
                              descr_B,
                              nnz_B,
                              (const rocsparse_double_complex*)csr_val_B,
                              csr_row_ptr_B,
                              csr_col_ind_B,
                              descr_C,
                              (rocsparse_double_complex*)csr_val_C,
                              csr_row_ptr_C,
                              csr_col_ind_C);
}

// csrgemm
template <>
rocsparse_status rocsparse_csrgemm_buffer_size(rocsparse_handle          handle,
                                               rocsparse_operation       trans_A,
                                               rocsparse_operation       trans_B,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               rocsparse_int             k,
                                               const float*              alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int             nnz_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int             nnz_B,
                                               const rocsparse_int*      csr_row_ptr_B,
                                               const rocsparse_int*      csr_col_ind_B,
                                               const float*              beta,
                                               const rocsparse_mat_descr descr_D,
                                               rocsparse_int             nnz_D,
                                               const rocsparse_int*      csr_row_ptr_D,
                                               const rocsparse_int*      csr_col_ind_D,
                                               rocsparse_mat_info        info_C,
                                               size_t*                   buffer_size)
{
    return rocsparse_scsrgemm_buffer_size(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          beta,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          info_C,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_csrgemm_buffer_size(rocsparse_handle          handle,
                                               rocsparse_operation       trans_A,
                                               rocsparse_operation       trans_B,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               rocsparse_int             k,
                                               const double*             alpha,
                                               const rocsparse_mat_descr descr_A,
                                               rocsparse_int             nnz_A,
                                               const rocsparse_int*      csr_row_ptr_A,
                                               const rocsparse_int*      csr_col_ind_A,
                                               const rocsparse_mat_descr descr_B,
                                               rocsparse_int             nnz_B,
                                               const rocsparse_int*      csr_row_ptr_B,
                                               const rocsparse_int*      csr_col_ind_B,
                                               const double*             beta,
                                               const rocsparse_mat_descr descr_D,
                                               rocsparse_int             nnz_D,
                                               const rocsparse_int*      csr_row_ptr_D,
                                               const rocsparse_int*      csr_col_ind_D,
                                               rocsparse_mat_info        info_C,
                                               size_t*                   buffer_size)
{
    return rocsparse_dcsrgemm_buffer_size(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          beta,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          info_C,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_csrgemm_buffer_size(rocsparse_handle               handle,
                                               rocsparse_operation            trans_A,
                                               rocsparse_operation            trans_B,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               rocsparse_int                  k,
                                               const rocsparse_float_complex* alpha,
                                               const rocsparse_mat_descr      descr_A,
                                               rocsparse_int                  nnz_A,
                                               const rocsparse_int*           csr_row_ptr_A,
                                               const rocsparse_int*           csr_col_ind_A,
                                               const rocsparse_mat_descr      descr_B,
                                               rocsparse_int                  nnz_B,
                                               const rocsparse_int*           csr_row_ptr_B,
                                               const rocsparse_int*           csr_col_ind_B,
                                               const rocsparse_float_complex* beta,
                                               const rocsparse_mat_descr      descr_D,
                                               rocsparse_int                  nnz_D,
                                               const rocsparse_int*           csr_row_ptr_D,
                                               const rocsparse_int*           csr_col_ind_D,
                                               rocsparse_mat_info             info_C,
                                               size_t*                        buffer_size)
{
    return rocsparse_ccsrgemm_buffer_size(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          beta,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          info_C,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_csrgemm_buffer_size(rocsparse_handle                handle,
                                               rocsparse_operation             trans_A,
                                               rocsparse_operation             trans_B,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               rocsparse_int                   k,
                                               const rocsparse_double_complex* alpha,
                                               const rocsparse_mat_descr       descr_A,
                                               rocsparse_int                   nnz_A,
                                               const rocsparse_int*            csr_row_ptr_A,
                                               const rocsparse_int*            csr_col_ind_A,
                                               const rocsparse_mat_descr       descr_B,
                                               rocsparse_int                   nnz_B,
                                               const rocsparse_int*            csr_row_ptr_B,
                                               const rocsparse_int*            csr_col_ind_B,
                                               const rocsparse_double_complex* beta,
                                               const rocsparse_mat_descr       descr_D,
                                               rocsparse_int                   nnz_D,
                                               const rocsparse_int*            csr_row_ptr_D,
                                               const rocsparse_int*            csr_col_ind_D,
                                               rocsparse_mat_info              info_C,
                                               size_t*                         buffer_size)
{
    return rocsparse_zcsrgemm_buffer_size(handle,
                                          trans_A,
                                          trans_B,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          descr_A,
                                          nnz_A,
                                          csr_row_ptr_A,
                                          csr_col_ind_A,
                                          descr_B,
                                          nnz_B,
                                          csr_row_ptr_B,
                                          csr_col_ind_B,
                                          beta,
                                          descr_D,
                                          nnz_D,
                                          csr_row_ptr_D,
                                          csr_col_ind_D,
                                          info_C,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_csrgemm(rocsparse_handle          handle,
                                   rocsparse_operation       trans_A,
                                   rocsparse_operation       trans_B,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   rocsparse_int             k,
                                   const float*              alpha,
                                   const rocsparse_mat_descr descr_A,
                                   rocsparse_int             nnz_A,
                                   const float*              csr_val_A,
                                   const rocsparse_int*      csr_row_ptr_A,
                                   const rocsparse_int*      csr_col_ind_A,
                                   const rocsparse_mat_descr descr_B,
                                   rocsparse_int             nnz_B,
                                   const float*              csr_val_B,
                                   const rocsparse_int*      csr_row_ptr_B,
                                   const rocsparse_int*      csr_col_ind_B,
                                   const float*              beta,
                                   const rocsparse_mat_descr descr_D,
                                   rocsparse_int             nnz_D,
                                   const float*              csr_val_D,
                                   const rocsparse_int*      csr_row_ptr_D,
                                   const rocsparse_int*      csr_col_ind_D,
                                   const rocsparse_mat_descr descr_C,
                                   float*                    csr_val_C,
                                   const rocsparse_int*      csr_row_ptr_C,
                                   rocsparse_int*            csr_col_ind_C,
                                   const rocsparse_mat_info  info_C,
                                   void*                     temp_buffer)
{
    return rocsparse_scsrgemm(handle,
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

template <>
rocsparse_status rocsparse_csrgemm(rocsparse_handle          handle,
                                   rocsparse_operation       trans_A,
                                   rocsparse_operation       trans_B,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   rocsparse_int             k,
                                   const double*             alpha,
                                   const rocsparse_mat_descr descr_A,
                                   rocsparse_int             nnz_A,
                                   const double*             csr_val_A,
                                   const rocsparse_int*      csr_row_ptr_A,
                                   const rocsparse_int*      csr_col_ind_A,
                                   const rocsparse_mat_descr descr_B,
                                   rocsparse_int             nnz_B,
                                   const double*             csr_val_B,
                                   const rocsparse_int*      csr_row_ptr_B,
                                   const rocsparse_int*      csr_col_ind_B,
                                   const double*             beta,
                                   const rocsparse_mat_descr descr_D,
                                   rocsparse_int             nnz_D,
                                   const double*             csr_val_D,
                                   const rocsparse_int*      csr_row_ptr_D,
                                   const rocsparse_int*      csr_col_ind_D,
                                   const rocsparse_mat_descr descr_C,
                                   double*                   csr_val_C,
                                   const rocsparse_int*      csr_row_ptr_C,
                                   rocsparse_int*            csr_col_ind_C,
                                   const rocsparse_mat_info  info_C,
                                   void*                     temp_buffer)
{
    return rocsparse_dcsrgemm(handle,
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

template <>
rocsparse_status rocsparse_csrgemm(rocsparse_handle               handle,
                                   rocsparse_operation            trans_A,
                                   rocsparse_operation            trans_B,
                                   rocsparse_int                  m,
                                   rocsparse_int                  n,
                                   rocsparse_int                  k,
                                   const rocsparse_float_complex* alpha,
                                   const rocsparse_mat_descr      descr_A,
                                   rocsparse_int                  nnz_A,
                                   const rocsparse_float_complex* csr_val_A,
                                   const rocsparse_int*           csr_row_ptr_A,
                                   const rocsparse_int*           csr_col_ind_A,
                                   const rocsparse_mat_descr      descr_B,
                                   rocsparse_int                  nnz_B,
                                   const rocsparse_float_complex* csr_val_B,
                                   const rocsparse_int*           csr_row_ptr_B,
                                   const rocsparse_int*           csr_col_ind_B,
                                   const rocsparse_float_complex* beta,
                                   const rocsparse_mat_descr      descr_D,
                                   rocsparse_int                  nnz_D,
                                   const rocsparse_float_complex* csr_val_D,
                                   const rocsparse_int*           csr_row_ptr_D,
                                   const rocsparse_int*           csr_col_ind_D,
                                   const rocsparse_mat_descr      descr_C,
                                   rocsparse_float_complex*       csr_val_C,
                                   const rocsparse_int*           csr_row_ptr_C,
                                   rocsparse_int*                 csr_col_ind_C,
                                   const rocsparse_mat_info       info_C,
                                   void*                          temp_buffer)
{
    return rocsparse_ccsrgemm(handle,
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

template <>
rocsparse_status rocsparse_csrgemm(rocsparse_handle                handle,
                                   rocsparse_operation             trans_A,
                                   rocsparse_operation             trans_B,
                                   rocsparse_int                   m,
                                   rocsparse_int                   n,
                                   rocsparse_int                   k,
                                   const rocsparse_double_complex* alpha,
                                   const rocsparse_mat_descr       descr_A,
                                   rocsparse_int                   nnz_A,
                                   const rocsparse_double_complex* csr_val_A,
                                   const rocsparse_int*            csr_row_ptr_A,
                                   const rocsparse_int*            csr_col_ind_A,
                                   const rocsparse_mat_descr       descr_B,
                                   rocsparse_int                   nnz_B,
                                   const rocsparse_double_complex* csr_val_B,
                                   const rocsparse_int*            csr_row_ptr_B,
                                   const rocsparse_int*            csr_col_ind_B,
                                   const rocsparse_double_complex* beta,
                                   const rocsparse_mat_descr       descr_D,
                                   rocsparse_int                   nnz_D,
                                   const rocsparse_double_complex* csr_val_D,
                                   const rocsparse_int*            csr_row_ptr_D,
                                   const rocsparse_int*            csr_col_ind_D,
                                   const rocsparse_mat_descr       descr_C,
                                   rocsparse_double_complex*       csr_val_C,
                                   const rocsparse_int*            csr_row_ptr_C,
                                   rocsparse_int*                  csr_col_ind_C,
                                   const rocsparse_mat_info        info_C,
                                   void*                           temp_buffer)
{
    return rocsparse_zcsrgemm(handle,
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

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
// bsric0
template <>
rocsparse_status rocsparse_bsric0_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const float*              bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size)
{
    return rocsparse_sbsric0_buffer_size(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         block_dim,
                                         info,
                                         buffer_size);
}

template <>
rocsparse_status rocsparse_bsric0_buffer_size(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_int             mb,
                                              rocsparse_int             nnzb,
                                              const rocsparse_mat_descr descr,
                                              const double*             bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             block_dim,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size)
{
    return rocsparse_dbsric0_buffer_size(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         block_dim,
                                         info,
                                         buffer_size);
}

template <>
rocsparse_status rocsparse_bsric0_buffer_size(rocsparse_handle               handle,
                                              rocsparse_direction            dir,
                                              rocsparse_int                  mb,
                                              rocsparse_int                  nnzb,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* bsr_val,
                                              const rocsparse_int*           bsr_row_ptr,
                                              const rocsparse_int*           bsr_col_ind,
                                              rocsparse_int                  block_dim,
                                              rocsparse_mat_info             info,
                                              size_t*                        buffer_size)
{
    return rocsparse_cbsric0_buffer_size(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         (const rocsparse_float_complex*)bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         block_dim,
                                         info,
                                         buffer_size);
}

template <>
rocsparse_status rocsparse_bsric0_buffer_size(rocsparse_handle                handle,
                                              rocsparse_direction             dir,
                                              rocsparse_int                   mb,
                                              rocsparse_int                   nnzb,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* bsr_val,
                                              const rocsparse_int*            bsr_row_ptr,
                                              const rocsparse_int*            bsr_col_ind,
                                              rocsparse_int                   block_dim,
                                              rocsparse_mat_info              info,
                                              size_t*                         buffer_size)
{
    return rocsparse_zbsric0_buffer_size(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         (const rocsparse_double_complex*)bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         block_dim,
                                         info,
                                         buffer_size);
}

template <>
rocsparse_status rocsparse_bsric0_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_int             mb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const float*              bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer)
{
    return rocsparse_sbsric0_analysis(handle,
                                      dir,
                                      mb,
                                      nnzb,
                                      descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_bsric0_analysis(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_int             mb,
                                           rocsparse_int             nnzb,
                                           const rocsparse_mat_descr descr,
                                           const double*             bsr_val,
                                           const rocsparse_int*      bsr_row_ptr,
                                           const rocsparse_int*      bsr_col_ind,
                                           rocsparse_int             block_dim,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer)
{
    return rocsparse_dbsric0_analysis(handle,
                                      dir,
                                      mb,
                                      nnzb,
                                      descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_bsric0_analysis(rocsparse_handle               handle,
                                           rocsparse_direction            dir,
                                           rocsparse_int                  mb,
                                           rocsparse_int                  nnzb,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* bsr_val,
                                           const rocsparse_int*           bsr_row_ptr,
                                           const rocsparse_int*           bsr_col_ind,
                                           rocsparse_int                  block_dim,
                                           rocsparse_mat_info             info,
                                           rocsparse_analysis_policy      analysis,
                                           rocsparse_solve_policy         solve,
                                           void*                          temp_buffer)
{
    return rocsparse_cbsric0_analysis(handle,
                                      dir,
                                      mb,
                                      nnzb,
                                      descr,
                                      (const rocsparse_float_complex*)bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_bsric0_analysis(rocsparse_handle                handle,
                                           rocsparse_direction             dir,
                                           rocsparse_int                   mb,
                                           rocsparse_int                   nnzb,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* bsr_val,
                                           const rocsparse_int*            bsr_row_ptr,
                                           const rocsparse_int*            bsr_col_ind,
                                           rocsparse_int                   block_dim,
                                           rocsparse_mat_info              info,
                                           rocsparse_analysis_policy       analysis,
                                           rocsparse_solve_policy          solve,
                                           void*                           temp_buffer)
{
    return rocsparse_zbsric0_analysis(handle,
                                      dir,
                                      mb,
                                      nnzb,
                                      descr,
                                      (const rocsparse_double_complex*)bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      block_dim,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_bsric0(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_int             mb,
                                  rocsparse_int             nnzb,
                                  const rocsparse_mat_descr descr,
                                  float*                    bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_sbsric0(handle,
                             dir,
                             mb,
                             nnzb,
                             descr,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             block_dim,
                             info,
                             policy,
                             temp_buffer);
}

template <>
rocsparse_status rocsparse_bsric0(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_int             mb,
                                  rocsparse_int             nnzb,
                                  const rocsparse_mat_descr descr,
                                  double*                   bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_dbsric0(handle,
                             dir,
                             mb,
                             nnzb,
                             descr,
                             bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             block_dim,
                             info,
                             policy,
                             temp_buffer);
}

template <>
rocsparse_status rocsparse_bsric0(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_int             mb,
                                  rocsparse_int             nnzb,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_float_complex*  bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_cbsric0(handle,
                             dir,
                             mb,
                             nnzb,
                             descr,
                             (rocsparse_float_complex*)bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             block_dim,
                             info,
                             policy,
                             temp_buffer);
}

template <>
rocsparse_status rocsparse_bsric0(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_int             mb,
                                  rocsparse_int             nnzb,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_double_complex* bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             block_dim,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_zbsric0(handle,
                             dir,
                             mb,
                             nnzb,
                             descr,
                             (rocsparse_double_complex*)bsr_val,
                             bsr_row_ptr,
                             bsr_col_ind,
                             block_dim,
                             info,
                             policy,
                             temp_buffer);
}

// bsrilu0
template <>
rocsparse_status rocsparse_bsrilu0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               const float*              bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size)
{
    return rocsparse_sbsrilu0_buffer_size(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          block_dim,
                                          info,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_bsrilu0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               rocsparse_int             mb,
                                               rocsparse_int             nnzb,
                                               const rocsparse_mat_descr descr,
                                               const double*             bsr_val,
                                               const rocsparse_int*      bsr_row_ptr,
                                               const rocsparse_int*      bsr_col_ind,
                                               rocsparse_int             block_dim,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size)
{
    return rocsparse_dbsrilu0_buffer_size(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          block_dim,
                                          info,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_bsrilu0_buffer_size(rocsparse_handle               handle,
                                               rocsparse_direction            dir,
                                               rocsparse_int                  mb,
                                               rocsparse_int                  nnzb,
                                               const rocsparse_mat_descr      descr,
                                               const rocsparse_float_complex* bsr_val,
                                               const rocsparse_int*           bsr_row_ptr,
                                               const rocsparse_int*           bsr_col_ind,
                                               rocsparse_int                  block_dim,
                                               rocsparse_mat_info             info,
                                               size_t*                        buffer_size)
{
    return rocsparse_cbsrilu0_buffer_size(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          (const rocsparse_float_complex*)bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          block_dim,
                                          info,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_bsrilu0_buffer_size(rocsparse_handle                handle,
                                               rocsparse_direction             dir,
                                               rocsparse_int                   mb,
                                               rocsparse_int                   nnzb,
                                               const rocsparse_mat_descr       descr,
                                               const rocsparse_double_complex* bsr_val,
                                               const rocsparse_int*            bsr_row_ptr,
                                               const rocsparse_int*            bsr_col_ind,
                                               rocsparse_int                   block_dim,
                                               rocsparse_mat_info              info,
                                               size_t*                         buffer_size)
{
    return rocsparse_zbsrilu0_buffer_size(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          (const rocsparse_double_complex*)bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          block_dim,
                                          info,
                                          buffer_size);
}

template <>
rocsparse_status rocsparse_bsrilu0_numeric_boost(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 int                enable_boost,
                                                 const float*       boost_tol,
                                                 const float*       boost_val)
{
    return rocsparse_sbsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_bsrilu0_numeric_boost(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 int                enable_boost,
                                                 const double*      boost_tol,
                                                 const double*      boost_val)
{
    return rocsparse_dbsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_bsrilu0_numeric_boost(rocsparse_handle               handle,
                                                 rocsparse_mat_info             info,
                                                 int                            enable_boost,
                                                 const float*                   boost_tol,
                                                 const rocsparse_float_complex* boost_val)
{
    return rocsparse_cbsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_bsrilu0_numeric_boost(rocsparse_handle                handle,
                                                 rocsparse_mat_info              info,
                                                 int                             enable_boost,
                                                 const double*                   boost_tol,
                                                 const rocsparse_double_complex* boost_val)
{
    return rocsparse_zbsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_bsrilu0_analysis(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nnzb,
                                            const rocsparse_mat_descr descr,
                                            const float*              bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer)
{
    return rocsparse_sbsrilu0_analysis(handle,
                                       dir,
                                       mb,
                                       nnzb,
                                       descr,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind,
                                       block_dim,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrilu0_analysis(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_int             mb,
                                            rocsparse_int             nnzb,
                                            const rocsparse_mat_descr descr,
                                            const double*             bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             block_dim,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer)
{
    return rocsparse_dbsrilu0_analysis(handle,
                                       dir,
                                       mb,
                                       nnzb,
                                       descr,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind,
                                       block_dim,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrilu0_analysis(rocsparse_handle               handle,
                                            rocsparse_direction            dir,
                                            rocsparse_int                  mb,
                                            rocsparse_int                  nnzb,
                                            const rocsparse_mat_descr      descr,
                                            const rocsparse_float_complex* bsr_val,
                                            const rocsparse_int*           bsr_row_ptr,
                                            const rocsparse_int*           bsr_col_ind,
                                            rocsparse_int                  block_dim,
                                            rocsparse_mat_info             info,
                                            rocsparse_analysis_policy      analysis,
                                            rocsparse_solve_policy         solve,
                                            void*                          temp_buffer)
{
    return rocsparse_cbsrilu0_analysis(handle,
                                       dir,
                                       mb,
                                       nnzb,
                                       descr,
                                       (const rocsparse_float_complex*)bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind,
                                       block_dim,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrilu0_analysis(rocsparse_handle                handle,
                                            rocsparse_direction             dir,
                                            rocsparse_int                   mb,
                                            rocsparse_int                   nnzb,
                                            const rocsparse_mat_descr       descr,
                                            const rocsparse_double_complex* bsr_val,
                                            const rocsparse_int*            bsr_row_ptr,
                                            const rocsparse_int*            bsr_col_ind,
                                            rocsparse_int                   block_dim,
                                            rocsparse_mat_info              info,
                                            rocsparse_analysis_policy       analysis,
                                            rocsparse_solve_policy          solve,
                                            void*                           temp_buffer)
{
    return rocsparse_zbsrilu0_analysis(handle,
                                       dir,
                                       mb,
                                       nnzb,
                                       descr,
                                       (const rocsparse_double_complex*)bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind,
                                       block_dim,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrilu0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   float*                    bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_sbsrilu0(handle,
                              dir,
                              mb,
                              nnzb,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              info,
                              policy,
                              temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrilu0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   double*                   bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_dbsrilu0(handle,
                              dir,
                              mb,
                              nnzb,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              info,
                              policy,
                              temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrilu0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_float_complex*  bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_cbsrilu0(handle,
                              dir,
                              mb,
                              nnzb,
                              descr,
                              (rocsparse_float_complex*)bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              info,
                              policy,
                              temp_buffer);
}

template <>
rocsparse_status rocsparse_bsrilu0(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nnzb,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_double_complex* bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_zbsrilu0(handle,
                              dir,
                              mb,
                              nnzb,
                              descr,
                              (rocsparse_double_complex*)bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              info,
                              policy,
                              temp_buffer);
}

// csric0
template <>
rocsparse_status rocsparse_csric0_buffer_size(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size)
{
    return rocsparse_scsric0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csric0_buffer_size(rocsparse_handle          handle,
                                              rocsparse_int             m,
                                              rocsparse_int             nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const rocsparse_int*      csr_row_ptr,
                                              const rocsparse_int*      csr_col_ind,
                                              rocsparse_mat_info        info,
                                              size_t*                   buffer_size)
{
    return rocsparse_dcsric0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csric0_buffer_size(rocsparse_handle               handle,
                                              rocsparse_int                  m,
                                              rocsparse_int                  nnz,
                                              const rocsparse_mat_descr      descr,
                                              const rocsparse_float_complex* csr_val,
                                              const rocsparse_int*           csr_row_ptr,
                                              const rocsparse_int*           csr_col_ind,
                                              rocsparse_mat_info             info,
                                              size_t*                        buffer_size)
{
    return rocsparse_ccsric0_buffer_size(handle,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_float_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         buffer_size);
}

template <>
rocsparse_status rocsparse_csric0_buffer_size(rocsparse_handle                handle,
                                              rocsparse_int                   m,
                                              rocsparse_int                   nnz,
                                              const rocsparse_mat_descr       descr,
                                              const rocsparse_double_complex* csr_val,
                                              const rocsparse_int*            csr_row_ptr,
                                              const rocsparse_int*            csr_col_ind,
                                              rocsparse_mat_info              info,
                                              size_t*                         buffer_size)
{
    return rocsparse_zcsric0_buffer_size(handle,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_double_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         buffer_size);
}

template <>
rocsparse_status rocsparse_csric0_analysis(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const float*              csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer)
{
    return rocsparse_scsric0_analysis(handle,
                                      m,
                                      nnz,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_csric0_analysis(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             nnz,
                                           const rocsparse_mat_descr descr,
                                           const double*             csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           const rocsparse_int*      csr_col_ind,
                                           rocsparse_mat_info        info,
                                           rocsparse_analysis_policy analysis,
                                           rocsparse_solve_policy    solve,
                                           void*                     temp_buffer)
{
    return rocsparse_dcsric0_analysis(handle,
                                      m,
                                      nnz,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_csric0_analysis(rocsparse_handle               handle,
                                           rocsparse_int                  m,
                                           rocsparse_int                  nnz,
                                           const rocsparse_mat_descr      descr,
                                           const rocsparse_float_complex* csr_val,
                                           const rocsparse_int*           csr_row_ptr,
                                           const rocsparse_int*           csr_col_ind,
                                           rocsparse_mat_info             info,
                                           rocsparse_analysis_policy      analysis,
                                           rocsparse_solve_policy         solve,
                                           void*                          temp_buffer)
{
    return rocsparse_ccsric0_analysis(handle,
                                      m,
                                      nnz,
                                      descr,
                                      (const rocsparse_float_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_csric0_analysis(rocsparse_handle                handle,
                                           rocsparse_int                   m,
                                           rocsparse_int                   nnz,
                                           const rocsparse_mat_descr       descr,
                                           const rocsparse_double_complex* csr_val,
                                           const rocsparse_int*            csr_row_ptr,
                                           const rocsparse_int*            csr_col_ind,
                                           rocsparse_mat_info              info,
                                           rocsparse_analysis_policy       analysis,
                                           rocsparse_solve_policy          solve,
                                           void*                           temp_buffer)
{
    return rocsparse_zcsric0_analysis(handle,
                                      m,
                                      nnz,
                                      descr,
                                      (const rocsparse_double_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      analysis,
                                      solve,
                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_csric0(rocsparse_handle          handle,
                                  rocsparse_int             m,
                                  rocsparse_int             nnz,
                                  const rocsparse_mat_descr descr,
                                  float*                    csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_scsric0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

template <>
rocsparse_status rocsparse_csric0(rocsparse_handle          handle,
                                  rocsparse_int             m,
                                  rocsparse_int             nnz,
                                  const rocsparse_mat_descr descr,
                                  double*                   csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_dcsric0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

template <>
rocsparse_status rocsparse_csric0(rocsparse_handle          handle,
                                  rocsparse_int             m,
                                  rocsparse_int             nnz,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_float_complex*  csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_ccsric0(handle,
                             m,
                             nnz,
                             descr,
                             (rocsparse_float_complex*)csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             info,
                             policy,
                             temp_buffer);
}

template <>
rocsparse_status rocsparse_csric0(rocsparse_handle          handle,
                                  rocsparse_int             m,
                                  rocsparse_int             nnz,
                                  const rocsparse_mat_descr descr,
                                  rocsparse_double_complex* csr_val,
                                  const rocsparse_int*      csr_row_ptr,
                                  const rocsparse_int*      csr_col_ind,
                                  rocsparse_mat_info        info,
                                  rocsparse_solve_policy    policy,
                                  void*                     temp_buffer)
{
    return rocsparse_zcsric0(handle,
                             m,
                             nnz,
                             descr,
                             (rocsparse_double_complex*)csr_val,
                             csr_row_ptr,
                             csr_col_ind,
                             info,
                             policy,
                             temp_buffer);
}

// csrilu0
template <>
rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const float*              csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size)
{
    return rocsparse_scsrilu0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const double*             csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size)
{
    return rocsparse_dcsrilu0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  nnz,
                                               const rocsparse_mat_descr      descr,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               rocsparse_mat_info             info,
                                               size_t*                        buffer_size)
{
    return rocsparse_ccsrilu0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrilu0_buffer_size(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               rocsparse_int                   nnz,
                                               const rocsparse_mat_descr       descr,
                                               const rocsparse_double_complex* csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               const rocsparse_int*            csr_col_ind,
                                               rocsparse_mat_info              info,
                                               size_t*                         buffer_size)
{
    return rocsparse_zcsrilu0_buffer_size(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
}

template <>
rocsparse_status rocsparse_csrilu0_numeric_boost(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 int                enable_boost,
                                                 const float*       boost_tol,
                                                 const float*       boost_val)
{
    return rocsparse_scsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_csrilu0_numeric_boost(rocsparse_handle   handle,
                                                 rocsparse_mat_info info,
                                                 int                enable_boost,
                                                 const double*      boost_tol,
                                                 const double*      boost_val)
{
    return rocsparse_dcsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_csrilu0_numeric_boost(rocsparse_handle               handle,
                                                 rocsparse_mat_info             info,
                                                 int                            enable_boost,
                                                 const float*                   boost_tol,
                                                 const rocsparse_float_complex* boost_val)
{
    return rocsparse_ccsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_csrilu0_numeric_boost(rocsparse_handle                handle,
                                                 rocsparse_mat_info              info,
                                                 int                             enable_boost,
                                                 const double*                   boost_tol,
                                                 const rocsparse_double_complex* boost_val)
{
    return rocsparse_zcsrilu0_numeric_boost(handle, info, enable_boost, boost_tol, boost_val);
}

template <>
rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             nnz,
                                            const rocsparse_mat_descr descr,
                                            const float*              csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer)
{
    return rocsparse_scsrilu0_analysis(handle,
                                       m,
                                       nnz,
                                       descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             nnz,
                                            const rocsparse_mat_descr descr,
                                            const double*             csr_val,
                                            const rocsparse_int*      csr_row_ptr,
                                            const rocsparse_int*      csr_col_ind,
                                            rocsparse_mat_info        info,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            void*                     temp_buffer)
{
    return rocsparse_dcsrilu0_analysis(handle,
                                       m,
                                       nnz,
                                       descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle               handle,
                                            rocsparse_int                  m,
                                            rocsparse_int                  nnz,
                                            const rocsparse_mat_descr      descr,
                                            const rocsparse_float_complex* csr_val,
                                            const rocsparse_int*           csr_row_ptr,
                                            const rocsparse_int*           csr_col_ind,
                                            rocsparse_mat_info             info,
                                            rocsparse_analysis_policy      analysis,
                                            rocsparse_solve_policy         solve,
                                            void*                          temp_buffer)
{
    return rocsparse_ccsrilu0_analysis(handle,
                                       m,
                                       nnz,
                                       descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_csrilu0_analysis(rocsparse_handle                handle,
                                            rocsparse_int                   m,
                                            rocsparse_int                   nnz,
                                            const rocsparse_mat_descr       descr,
                                            const rocsparse_double_complex* csr_val,
                                            const rocsparse_int*            csr_row_ptr,
                                            const rocsparse_int*            csr_col_ind,
                                            rocsparse_mat_info              info,
                                            rocsparse_analysis_policy       analysis,
                                            rocsparse_solve_policy          solve,
                                            void*                           temp_buffer)
{
    return rocsparse_zcsrilu0_analysis(handle,
                                       m,
                                       nnz,
                                       descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       info,
                                       analysis,
                                       solve,
                                       temp_buffer);
}

template <>
rocsparse_status rocsparse_csrilu0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   float*                    csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_scsrilu0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

template <>
rocsparse_status rocsparse_csrilu0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   double*                   csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_dcsrilu0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

template <>
rocsparse_status rocsparse_csrilu0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_float_complex*  csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_ccsrilu0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

template <>
rocsparse_status rocsparse_csrilu0(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             nnz,
                                   const rocsparse_mat_descr descr,
                                   rocsparse_double_complex* csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_mat_info        info,
                                   rocsparse_solve_policy    policy,
                                   void*                     temp_buffer)
{
    return rocsparse_zcsrilu0(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
}

// gtsv
template <>
rocsparse_status rocsparse_gtsv_buffer_size(rocsparse_handle handle,
                                            rocsparse_int    m,
                                            rocsparse_int    n,
                                            const float*     dl,
                                            const float*     d,
                                            const float*     du,
                                            const float*     B,
                                            rocsparse_int    ldb,
                                            size_t*          buffer_size)
{
    return rocsparse_sgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_buffer_size(rocsparse_handle handle,
                                            rocsparse_int    m,
                                            rocsparse_int    n,
                                            const double*    dl,
                                            const double*    d,
                                            const double*    du,
                                            const double*    B,
                                            rocsparse_int    ldb,
                                            size_t*          buffer_size)
{
    return rocsparse_dgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_buffer_size(rocsparse_handle               handle,
                                            rocsparse_int                  m,
                                            rocsparse_int                  n,
                                            const rocsparse_float_complex* dl,
                                            const rocsparse_float_complex* d,
                                            const rocsparse_float_complex* du,
                                            const rocsparse_float_complex* B,
                                            rocsparse_int                  ldb,
                                            size_t*                        buffer_size)
{
    return rocsparse_cgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_buffer_size(rocsparse_handle                handle,
                                            rocsparse_int                   m,
                                            rocsparse_int                   n,
                                            const rocsparse_double_complex* dl,
                                            const rocsparse_double_complex* d,
                                            const rocsparse_double_complex* du,
                                            const rocsparse_double_complex* B,
                                            rocsparse_int                   ldb,
                                            size_t*                         buffer_size)
{
    return rocsparse_zgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv(rocsparse_handle handle,
                                rocsparse_int    m,
                                rocsparse_int    n,
                                const float*     dl,
                                const float*     d,
                                const float*     du,
                                float*           B,
                                rocsparse_int    ldb,
                                void*            temp_buffer)
{
    return rocsparse_sgtsv(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv(rocsparse_handle handle,
                                rocsparse_int    m,
                                rocsparse_int    n,
                                const double*    dl,
                                const double*    d,
                                const double*    du,
                                double*          B,
                                rocsparse_int    ldb,
                                void*            temp_buffer)
{
    return rocsparse_dgtsv(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv(rocsparse_handle               handle,
                                rocsparse_int                  m,
                                rocsparse_int                  n,
                                const rocsparse_float_complex* dl,
                                const rocsparse_float_complex* d,
                                const rocsparse_float_complex* du,
                                rocsparse_float_complex*       B,
                                rocsparse_int                  ldb,
                                void*                          temp_buffer)
{
    return rocsparse_cgtsv(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv(rocsparse_handle                handle,
                                rocsparse_int                   m,
                                rocsparse_int                   n,
                                const rocsparse_double_complex* dl,
                                const rocsparse_double_complex* d,
                                const rocsparse_double_complex* du,
                                rocsparse_double_complex*       B,
                                rocsparse_int                   ldb,
                                void*                           temp_buffer)
{
    return rocsparse_zgtsv(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

// gtsv_no_pivot
template <>
rocsparse_status rocsparse_gtsv_no_pivot_buffer_size(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     rocsparse_int    n,
                                                     const float*     dl,
                                                     const float*     d,
                                                     const float*     du,
                                                     const float*     B,
                                                     rocsparse_int    ldb,
                                                     size_t*          buffer_size)
{
    return rocsparse_sgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_buffer_size(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     rocsparse_int    n,
                                                     const double*    dl,
                                                     const double*    d,
                                                     const double*    du,
                                                     const double*    B,
                                                     rocsparse_int    ldb,
                                                     size_t*          buffer_size)
{
    return rocsparse_dgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_buffer_size(rocsparse_handle               handle,
                                                     rocsparse_int                  m,
                                                     rocsparse_int                  n,
                                                     const rocsparse_float_complex* dl,
                                                     const rocsparse_float_complex* d,
                                                     const rocsparse_float_complex* du,
                                                     const rocsparse_float_complex* B,
                                                     rocsparse_int                  ldb,
                                                     size_t*                        buffer_size)
{
    return rocsparse_cgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_buffer_size(rocsparse_handle                handle,
                                                     rocsparse_int                   m,
                                                     rocsparse_int                   n,
                                                     const rocsparse_double_complex* dl,
                                                     const rocsparse_double_complex* d,
                                                     const rocsparse_double_complex* du,
                                                     const rocsparse_double_complex* B,
                                                     rocsparse_int                   ldb,
                                                     size_t*                         buffer_size)
{
    return rocsparse_zgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot(rocsparse_handle handle,
                                         rocsparse_int    m,
                                         rocsparse_int    n,
                                         const float*     dl,
                                         const float*     d,
                                         const float*     du,
                                         float*           B,
                                         rocsparse_int    ldb,
                                         void*            temp_buffer)
{
    return rocsparse_sgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot(rocsparse_handle handle,
                                         rocsparse_int    m,
                                         rocsparse_int    n,
                                         const double*    dl,
                                         const double*    d,
                                         const double*    du,
                                         double*          B,
                                         rocsparse_int    ldb,
                                         void*            temp_buffer)
{
    return rocsparse_dgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot(rocsparse_handle               handle,
                                         rocsparse_int                  m,
                                         rocsparse_int                  n,
                                         const rocsparse_float_complex* dl,
                                         const rocsparse_float_complex* d,
                                         const rocsparse_float_complex* du,
                                         rocsparse_float_complex*       B,
                                         rocsparse_int                  ldb,
                                         void*                          temp_buffer)
{
    return rocsparse_cgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot(rocsparse_handle                handle,
                                         rocsparse_int                   m,
                                         rocsparse_int                   n,
                                         const rocsparse_double_complex* dl,
                                         const rocsparse_double_complex* d,
                                         const rocsparse_double_complex* du,
                                         rocsparse_double_complex*       B,
                                         rocsparse_int                   ldb,
                                         void*                           temp_buffer)
{
    return rocsparse_zgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

// gtsv_no_pivot_strided_batch
template <>
rocsparse_status rocsparse_gtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle,
                                                                   rocsparse_int    m,
                                                                   const float*     dl,
                                                                   const float*     d,
                                                                   const float*     du,
                                                                   const float*     x,
                                                                   rocsparse_int    batch_count,
                                                                   rocsparse_int    batch_stride,
                                                                   size_t*          buffer_size)
{
    return rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(
        handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle,
                                                                   rocsparse_int    m,
                                                                   const double*    dl,
                                                                   const double*    d,
                                                                   const double*    du,
                                                                   const double*    x,
                                                                   rocsparse_int    batch_count,
                                                                   rocsparse_int    batch_stride,
                                                                   size_t*          buffer_size)
{
    return rocsparse_dgtsv_no_pivot_strided_batch_buffer_size(
        handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

template <>
rocsparse_status
    rocsparse_gtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle               handle,
                                                      rocsparse_int                  m,
                                                      const rocsparse_float_complex* dl,
                                                      const rocsparse_float_complex* d,
                                                      const rocsparse_float_complex* du,
                                                      const rocsparse_float_complex* x,
                                                      rocsparse_int                  batch_count,
                                                      rocsparse_int                  batch_stride,
                                                      size_t*                        buffer_size)
{
    return rocsparse_cgtsv_no_pivot_strided_batch_buffer_size(
        handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

template <>
rocsparse_status
    rocsparse_gtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle                handle,
                                                      rocsparse_int                   m,
                                                      const rocsparse_double_complex* dl,
                                                      const rocsparse_double_complex* d,
                                                      const rocsparse_double_complex* du,
                                                      const rocsparse_double_complex* x,
                                                      rocsparse_int                   batch_count,
                                                      rocsparse_int                   batch_stride,
                                                      size_t*                         buffer_size)
{
    return rocsparse_zgtsv_no_pivot_strided_batch_buffer_size(
        handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_strided_batch(rocsparse_handle handle,
                                                       rocsparse_int    m,
                                                       const float*     dl,
                                                       const float*     d,
                                                       const float*     du,
                                                       float*           x,
                                                       rocsparse_int    batch_count,
                                                       rocsparse_int    batch_stride,
                                                       void*            temp_buffer)
{
    return rocsparse_sgtsv_no_pivot_strided_batch(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_strided_batch(rocsparse_handle handle,
                                                       rocsparse_int    m,
                                                       const double*    dl,
                                                       const double*    d,
                                                       const double*    du,
                                                       double*          x,
                                                       rocsparse_int    batch_count,
                                                       rocsparse_int    batch_stride,
                                                       void*            temp_buffer)
{
    return rocsparse_dgtsv_no_pivot_strided_batch(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_strided_batch(rocsparse_handle               handle,
                                                       rocsparse_int                  m,
                                                       const rocsparse_float_complex* dl,
                                                       const rocsparse_float_complex* d,
                                                       const rocsparse_float_complex* du,
                                                       rocsparse_float_complex*       x,
                                                       rocsparse_int                  batch_count,
                                                       rocsparse_int                  batch_stride,
                                                       void*                          temp_buffer)
{
    return rocsparse_cgtsv_no_pivot_strided_batch(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

template <>
rocsparse_status rocsparse_gtsv_no_pivot_strided_batch(rocsparse_handle                handle,
                                                       rocsparse_int                   m,
                                                       const rocsparse_double_complex* dl,
                                                       const rocsparse_double_complex* d,
                                                       const rocsparse_double_complex* du,
                                                       rocsparse_double_complex*       x,
                                                       rocsparse_int                   batch_count,
                                                       rocsparse_int                   batch_stride,
                                                       void*                           temp_buffer)
{
    return rocsparse_zgtsv_no_pivot_strided_batch(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */

// nnz
template <>
rocsparse_status rocsparse_nnz(rocsparse_handle          handle,
                               rocsparse_direction       dir,
                               rocsparse_int             m,
                               rocsparse_int             n,
                               const rocsparse_mat_descr descr,
                               const float*              A,
                               rocsparse_int             ld,
                               rocsparse_int*            nnz_per_row_columns,
                               rocsparse_int*            nnz_total_dev_host_ptr)
{
    return rocsparse_snnz(
        handle, dir, m, n, descr, A, ld, nnz_per_row_columns, nnz_total_dev_host_ptr);
}

template <>
rocsparse_status rocsparse_nnz(rocsparse_handle          handle,
                               rocsparse_direction       dir,
                               rocsparse_int             m,
                               rocsparse_int             n,
                               const rocsparse_mat_descr descr,
                               const double*             A,
                               rocsparse_int             ld,
                               rocsparse_int*            nnz_per_row_columns,
                               rocsparse_int*            nnz_total_dev_host_ptr)
{

    return rocsparse_dnnz(
        handle, dir, m, n, descr, A, ld, nnz_per_row_columns, nnz_total_dev_host_ptr);
}

template <>
rocsparse_status rocsparse_nnz(rocsparse_handle               handle,
                               rocsparse_direction            dir,
                               rocsparse_int                  m,
                               rocsparse_int                  n,
                               const rocsparse_mat_descr      descr,
                               const rocsparse_float_complex* A,
                               rocsparse_int                  ld,
                               rocsparse_int*                 nnz_per_row_columns,
                               rocsparse_int*                 nnz_total_dev_host_ptr)
{
    return rocsparse_cnnz(
        handle, dir, m, n, descr, A, ld, nnz_per_row_columns, nnz_total_dev_host_ptr);
}

template <>
rocsparse_status rocsparse_nnz(rocsparse_handle                handle,
                               rocsparse_direction             dir,
                               rocsparse_int                   m,
                               rocsparse_int                   n,
                               const rocsparse_mat_descr       descr,
                               const rocsparse_double_complex* A,
                               rocsparse_int                   ld,
                               rocsparse_int*                  nnz_per_row_columns,
                               rocsparse_int*                  nnz_total_dev_host_ptr)
{

    return rocsparse_znnz(
        handle, dir, m, n, descr, A, ld, nnz_per_row_columns, nnz_total_dev_host_ptr);
}

// dense2csr
template <>
rocsparse_status rocsparse_dense2csr(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const float*              A,
                                     rocsparse_int             ld,
                                     const rocsparse_int*      nnz_per_row,
                                     float*                    csr_val,
                                     rocsparse_int*            csr_row_ptr,
                                     rocsparse_int*            csr_col_ind)
{
    return rocsparse_sdense2csr(
        handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
}

template <>
rocsparse_status rocsparse_dense2csr(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const double*             A,
                                     rocsparse_int             ld,
                                     const rocsparse_int*      nnz_per_row,
                                     double*                   csr_val,
                                     rocsparse_int*            csr_row_ptr,
                                     rocsparse_int*            csr_col_ind)
{
    return rocsparse_ddense2csr(
        handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
}

template <>
rocsparse_status rocsparse_dense2csr(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  n,
                                     const rocsparse_mat_descr      descr,
                                     const rocsparse_float_complex* A,
                                     rocsparse_int                  ld,
                                     const rocsparse_int*           nnz_per_row,
                                     rocsparse_float_complex*       csr_val,
                                     rocsparse_int*                 csr_row_ptr,
                                     rocsparse_int*                 csr_col_ind)
{
    return rocsparse_cdense2csr(
        handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
}

template <>
rocsparse_status rocsparse_dense2csr(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   n,
                                     const rocsparse_mat_descr       descr,
                                     const rocsparse_double_complex* A,
                                     rocsparse_int                   ld,
                                     const rocsparse_int*            nnz_per_row,
                                     rocsparse_double_complex*       csr_val,
                                     rocsparse_int*                  csr_row_ptr,
                                     rocsparse_int*                  csr_col_ind)
{
    return rocsparse_zdense2csr(
        handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
}

// prune_dense2csr_buffer_size
template <>
rocsparse_status rocsparse_prune_dense2csr_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       const float*              A,
                                                       rocsparse_int             lda,
                                                       const float*              threshold,
                                                       const rocsparse_mat_descr descr,
                                                       const float*              csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       const rocsparse_int*      csr_col_ind,
                                                       size_t*                   buffer_size)
{
    return rocsparse_sprune_dense2csr_buffer_size(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size);
}

template <>
rocsparse_status rocsparse_prune_dense2csr_buffer_size(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       const double*             A,
                                                       rocsparse_int             lda,
                                                       const double*             threshold,
                                                       const rocsparse_mat_descr descr,
                                                       const double*             csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       const rocsparse_int*      csr_col_ind,
                                                       size_t*                   buffer_size)
{
    return rocsparse_dprune_dense2csr_buffer_size(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size);
}

// prune_dense2csr_nnz
template <>
rocsparse_status rocsparse_prune_dense2csr_nnz(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const float*              A,
                                               rocsparse_int             lda,
                                               const float*              threshold,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            nnz_total_dev_host_ptr,
                                               void*                     temp_buffer)
{
    return rocsparse_sprune_dense2csr_nnz(
        handle, m, n, A, lda, threshold, descr, csr_row_ptr, nnz_total_dev_host_ptr, temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_dense2csr_nnz(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const double*             A,
                                               rocsparse_int             lda,
                                               const double*             threshold,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            nnz_total_dev_host_ptr,
                                               void*                     temp_buffer)
{
    return rocsparse_dprune_dense2csr_nnz(
        handle, m, n, A, lda, threshold, descr, csr_row_ptr, nnz_total_dev_host_ptr, temp_buffer);
}

// prune_dense2csr
template <>
rocsparse_status rocsparse_prune_dense2csr(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           const float*              A,
                                           rocsparse_int             lda,
                                           const float*              threshold,
                                           const rocsparse_mat_descr descr,
                                           float*                    csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           rocsparse_int*            csr_col_ind,
                                           void*                     temp_buffer)
{
    return rocsparse_sprune_dense2csr(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_dense2csr(rocsparse_handle          handle,
                                           rocsparse_int             m,
                                           rocsparse_int             n,
                                           const double*             A,
                                           rocsparse_int             lda,
                                           const double*             threshold,
                                           const rocsparse_mat_descr descr,
                                           double*                   csr_val,
                                           const rocsparse_int*      csr_row_ptr,
                                           rocsparse_int*            csr_col_ind,
                                           void*                     temp_buffer)
{
    return rocsparse_dprune_dense2csr(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

// prune_dense2csr_by_percentage_buffer_size
template <>
rocsparse_status
    rocsparse_prune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const float*              A,
                                                        rocsparse_int             lda,
                                                        float                     percentage,
                                                        const rocsparse_mat_descr descr,
                                                        const float*              csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        rocsparse_mat_info        info,
                                                        size_t*                   buffer_size)
{
    return rocsparse_sprune_dense2csr_by_percentage_buffer_size(handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                percentage,
                                                                descr,
                                                                csr_val,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                info,
                                                                buffer_size);
}

template <>
rocsparse_status
    rocsparse_prune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             n,
                                                        const double*             A,
                                                        rocsparse_int             lda,
                                                        double                    percentage,
                                                        const rocsparse_mat_descr descr,
                                                        const double*             csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        rocsparse_mat_info        info,
                                                        size_t*                   buffer_size)
{
    return rocsparse_dprune_dense2csr_by_percentage_buffer_size(handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                percentage,
                                                                descr,
                                                                csr_val,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                info,
                                                                buffer_size);
}

// prune_dense2csr_nnz_by_percentage
template <>
rocsparse_status rocsparse_prune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                             rocsparse_int             m,
                                                             rocsparse_int             n,
                                                             const float*              A,
                                                             rocsparse_int             lda,
                                                             float                     percentage,
                                                             const rocsparse_mat_descr descr,
                                                             rocsparse_int*            csr_row_ptr,
                                                             rocsparse_int* nnz_total_dev_host_ptr,
                                                             rocsparse_mat_info info,
                                                             void*              temp_buffer)
{
    return rocsparse_sprune_dense2csr_nnz_by_percentage(handle,
                                                        m,
                                                        n,
                                                        A,
                                                        lda,
                                                        percentage,
                                                        descr,
                                                        csr_row_ptr,
                                                        nnz_total_dev_host_ptr,
                                                        info,
                                                        temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                             rocsparse_int             m,
                                                             rocsparse_int             n,
                                                             const double*             A,
                                                             rocsparse_int             lda,
                                                             double                    percentage,
                                                             const rocsparse_mat_descr descr,
                                                             rocsparse_int*            csr_row_ptr,
                                                             rocsparse_int* nnz_total_dev_host_ptr,
                                                             rocsparse_mat_info info,
                                                             void*              temp_buffer)
{
    return rocsparse_dprune_dense2csr_nnz_by_percentage(handle,
                                                        m,
                                                        n,
                                                        A,
                                                        lda,
                                                        percentage,
                                                        descr,
                                                        csr_row_ptr,
                                                        nnz_total_dev_host_ptr,
                                                        info,
                                                        temp_buffer);
}

// prune_dense2csr_by_percentage
template <>
rocsparse_status rocsparse_prune_dense2csr_by_percentage(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const float*              A,
                                                         rocsparse_int             lda,
                                                         float                     percentage,
                                                         const rocsparse_mat_descr descr,
                                                         float*                    csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         rocsparse_int*            csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         void*                     temp_buffer)
{
    return rocsparse_sprune_dense2csr_by_percentage(handle,
                                                    m,
                                                    n,
                                                    A,
                                                    lda,
                                                    percentage,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    info,
                                                    temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_dense2csr_by_percentage(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const double*             A,
                                                         rocsparse_int             lda,
                                                         double                    percentage,
                                                         const rocsparse_mat_descr descr,
                                                         double*                   csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         rocsparse_int*            csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         void*                     temp_buffer)
{
    return rocsparse_dprune_dense2csr_by_percentage(handle,
                                                    m,
                                                    n,
                                                    A,
                                                    lda,
                                                    percentage,
                                                    descr,
                                                    csr_val,
                                                    csr_row_ptr,
                                                    csr_col_ind,
                                                    info,
                                                    temp_buffer);
}

// dense2coo
template <>
rocsparse_status rocsparse_dense2coo(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const float*              A,
                                     rocsparse_int             ld,
                                     const rocsparse_int*      nnz_per_rows,
                                     float*                    coo_val,
                                     rocsparse_int*            coo_row_ind,
                                     rocsparse_int*            coo_col_ind)
{
    return rocsparse_sdense2coo(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}

template <>
rocsparse_status rocsparse_dense2coo(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const double*             A,
                                     rocsparse_int             ld,
                                     const rocsparse_int*      nnz_per_rows,
                                     double*                   coo_val,
                                     rocsparse_int*            coo_row_ind,
                                     rocsparse_int*            coo_col_ind)
{
    return rocsparse_ddense2coo(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}

template <>
rocsparse_status rocsparse_dense2coo(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  n,
                                     const rocsparse_mat_descr      descr,
                                     const rocsparse_float_complex* A,
                                     rocsparse_int                  ld,
                                     const rocsparse_int*           nnz_per_rows,
                                     rocsparse_float_complex*       coo_val,
                                     rocsparse_int*                 coo_row_ind,
                                     rocsparse_int*                 coo_col_ind)
{
    return rocsparse_cdense2coo(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}

template <>
rocsparse_status rocsparse_dense2coo(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   n,
                                     const rocsparse_mat_descr       descr,
                                     const rocsparse_double_complex* A,
                                     rocsparse_int                   ld,
                                     const rocsparse_int*            nnz_per_rows,
                                     rocsparse_double_complex*       coo_val,
                                     rocsparse_int*                  coo_row_ind,
                                     rocsparse_int*                  coo_col_ind)
{
    return rocsparse_zdense2coo(
        handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind);
}

// coo2dense
template <>
rocsparse_status rocsparse_coo2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     const float*              coo_val,
                                     const rocsparse_int*      coo_row_ind,
                                     const rocsparse_int*      coo_col_ind,
                                     float*                    A,
                                     rocsparse_int             ld)
{
    return rocsparse_scoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}

template <>
rocsparse_status rocsparse_coo2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     const double*             coo_val,
                                     const rocsparse_int*      coo_row_ind,
                                     const rocsparse_int*      coo_col_ind,
                                     double*                   A,
                                     rocsparse_int             ld)
{
    return rocsparse_dcoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}

template <>
rocsparse_status rocsparse_coo2dense(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  n,
                                     rocsparse_int                  nnz,
                                     const rocsparse_mat_descr      descr,
                                     const rocsparse_float_complex* coo_val,
                                     const rocsparse_int*           coo_row_ind,
                                     const rocsparse_int*           coo_col_ind,
                                     rocsparse_float_complex*       A,
                                     rocsparse_int                  ld)
{
    return rocsparse_ccoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}

template <>
rocsparse_status rocsparse_coo2dense(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   n,
                                     rocsparse_int                   nnz,
                                     const rocsparse_mat_descr       descr,
                                     const rocsparse_double_complex* coo_val,
                                     const rocsparse_int*            coo_row_ind,
                                     const rocsparse_int*            coo_col_ind,
                                     rocsparse_double_complex*       A,
                                     rocsparse_int                   ld)
{
    return rocsparse_zcoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, coo_col_ind, A, ld);
}

// dense2csc
template <>
rocsparse_status rocsparse_dense2csc(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const float*              A,
                                     rocsparse_int             ld,
                                     const rocsparse_int*      nnz_per_columns,
                                     float*                    csc_val,
                                     rocsparse_int*            csc_col_ptr,
                                     rocsparse_int*            csc_row_ind)
{
    return rocsparse_sdense2csc(
        handle, m, n, descr, A, ld, nnz_per_columns, csc_val, csc_col_ptr, csc_row_ind);
}

template <>
rocsparse_status rocsparse_dense2csc(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const double*             A,
                                     rocsparse_int             ld,
                                     const rocsparse_int*      nnz_per_columns,
                                     double*                   csc_val,
                                     rocsparse_int*            csc_col_ptr,
                                     rocsparse_int*            csc_row_ind)
{
    return rocsparse_ddense2csc(
        handle, m, n, descr, A, ld, nnz_per_columns, csc_val, csc_col_ptr, csc_row_ind);
}

template <>
rocsparse_status rocsparse_dense2csc(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  n,
                                     const rocsparse_mat_descr      descr,
                                     const rocsparse_float_complex* A,
                                     rocsparse_int                  ld,
                                     const rocsparse_int*           nnz_per_columns,
                                     rocsparse_float_complex*       csc_val,
                                     rocsparse_int*                 csc_col_ptr,
                                     rocsparse_int*                 csc_row_ind)
{
    return rocsparse_cdense2csc(
        handle, m, n, descr, A, ld, nnz_per_columns, csc_val, csc_col_ptr, csc_row_ind);
}

template <>
rocsparse_status rocsparse_dense2csc(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   n,
                                     const rocsparse_mat_descr       descr,
                                     const rocsparse_double_complex* A,
                                     rocsparse_int                   ld,
                                     const rocsparse_int*            nnz_per_columns,
                                     rocsparse_double_complex*       csc_val,
                                     rocsparse_int*                  csc_col_ptr,
                                     rocsparse_int*                  csc_row_ind)
{
    return rocsparse_zdense2csc(
        handle, m, n, descr, A, ld, nnz_per_columns, csc_val, csc_col_ptr, csc_row_ind);
}

// csr2dense
template <>
rocsparse_status rocsparse_csr2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const float*              csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     float*                    A,
                                     rocsparse_int             ld)
{
    return rocsparse_scsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
}

template <>
rocsparse_status rocsparse_csr2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const double*             csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     double*                   A,
                                     rocsparse_int             ld)
{
    return rocsparse_dcsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
}

template <>
rocsparse_status rocsparse_csr2dense(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  n,
                                     const rocsparse_mat_descr      descr,
                                     const rocsparse_float_complex* csr_val,
                                     const rocsparse_int*           csr_row_ptr,
                                     const rocsparse_int*           csr_col_ind,
                                     rocsparse_float_complex*       A,
                                     rocsparse_int                  ld)
{
    return rocsparse_ccsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
}

template <>
rocsparse_status rocsparse_csr2dense(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   n,
                                     const rocsparse_mat_descr       descr,
                                     const rocsparse_double_complex* csr_val,
                                     const rocsparse_int*            csr_row_ptr,
                                     const rocsparse_int*            csr_col_ind,
                                     rocsparse_double_complex*       A,
                                     rocsparse_int                   ld)
{
    return rocsparse_zcsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
}

// csc2dense
template <>
rocsparse_status rocsparse_csc2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const float*              csc_val,
                                     const rocsparse_int*      csc_col_ptr,
                                     const rocsparse_int*      csc_row_ind,
                                     float*                    A,
                                     rocsparse_int             ld)
{
    return rocsparse_scsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, csc_row_ind, A, ld);
}

template <>
rocsparse_status rocsparse_csc2dense(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr descr,
                                     const double*             csc_val,
                                     const rocsparse_int*      csc_col_ptr,
                                     const rocsparse_int*      csc_row_ind,
                                     double*                   A,
                                     rocsparse_int             ld)
{
    return rocsparse_dcsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, csc_row_ind, A, ld);
}

template <>
rocsparse_status rocsparse_csc2dense(rocsparse_handle               handle,
                                     rocsparse_int                  m,
                                     rocsparse_int                  n,
                                     const rocsparse_mat_descr      descr,
                                     const rocsparse_float_complex* csc_val,
                                     const rocsparse_int*           csc_col_ptr,
                                     const rocsparse_int*           csc_row_ind,
                                     rocsparse_float_complex*       A,
                                     rocsparse_int                  ld)
{
    return rocsparse_ccsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, csc_row_ind, A, ld);
}

template <>
rocsparse_status rocsparse_csc2dense(rocsparse_handle                handle,
                                     rocsparse_int                   m,
                                     rocsparse_int                   n,
                                     const rocsparse_mat_descr       descr,
                                     const rocsparse_double_complex* csc_val,
                                     const rocsparse_int*            csc_col_ptr,
                                     const rocsparse_int*            csc_row_ind,
                                     rocsparse_double_complex*       A,
                                     rocsparse_int                   ld)
{
    return rocsparse_zcsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, csc_row_ind, A, ld);
}

// nnz_compress
template <>
rocsparse_status rocsparse_nnz_compress(rocsparse_handle          handle,
                                        rocsparse_int             m,
                                        const rocsparse_mat_descr descr_A,
                                        const float*              csr_val_A,
                                        const rocsparse_int*      csr_row_ptr_A,
                                        rocsparse_int*            nnz_per_row,
                                        rocsparse_int*            nnz_C,
                                        float                     tol)
{
    return rocsparse_snnz_compress(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
}

template <>
rocsparse_status rocsparse_nnz_compress(rocsparse_handle          handle,
                                        rocsparse_int             m,
                                        const rocsparse_mat_descr descr_A,
                                        const double*             csr_val_A,
                                        const rocsparse_int*      csr_row_ptr_A,
                                        rocsparse_int*            nnz_per_row,
                                        rocsparse_int*            nnz_C,
                                        double                    tol)
{

    return rocsparse_dnnz_compress(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
}

template <>
rocsparse_status rocsparse_nnz_compress(rocsparse_handle               handle,
                                        rocsparse_int                  m,
                                        const rocsparse_mat_descr      descr_A,
                                        const rocsparse_float_complex* csr_val_A,
                                        const rocsparse_int*           csr_row_ptr_A,
                                        rocsparse_int*                 nnz_per_row,
                                        rocsparse_int*                 nnz_C,
                                        rocsparse_float_complex        tol)
{
    return rocsparse_cnnz_compress(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
}

template <>
rocsparse_status rocsparse_nnz_compress(rocsparse_handle                handle,
                                        rocsparse_int                   m,
                                        const rocsparse_mat_descr       descr_A,
                                        const rocsparse_double_complex* csr_val_A,
                                        const rocsparse_int*            csr_row_ptr_A,
                                        rocsparse_int*                  nnz_per_row,
                                        rocsparse_int*                  nnz_C,
                                        rocsparse_double_complex        tol)
{

    return rocsparse_znnz_compress(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
}

// csr2csc
template <>
rocsparse_status rocsparse_csr2csc(rocsparse_handle     handle,
                                   rocsparse_int        m,
                                   rocsparse_int        n,
                                   rocsparse_int        nnz,
                                   const float*         csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   float*               csc_val,
                                   rocsparse_int*       csc_row_ind,
                                   rocsparse_int*       csc_col_ptr,
                                   rocsparse_action     copy_values,
                                   rocsparse_index_base idx_base,
                                   void*                temp_buffer)
{
    return rocsparse_scsr2csc(handle,
                              m,
                              n,
                              nnz,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              csc_val,
                              csc_row_ind,
                              csc_col_ptr,
                              copy_values,
                              idx_base,
                              temp_buffer);
}

template <>
rocsparse_status rocsparse_csr2csc(rocsparse_handle     handle,
                                   rocsparse_int        m,
                                   rocsparse_int        n,
                                   rocsparse_int        nnz,
                                   const double*        csr_val,
                                   const rocsparse_int* csr_row_ptr,
                                   const rocsparse_int* csr_col_ind,
                                   double*              csc_val,
                                   rocsparse_int*       csc_row_ind,
                                   rocsparse_int*       csc_col_ptr,
                                   rocsparse_action     copy_values,
                                   rocsparse_index_base idx_base,
                                   void*                temp_buffer)
{
    return rocsparse_dcsr2csc(handle,
                              m,
                              n,
                              nnz,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              csc_val,
                              csc_row_ind,
                              csc_col_ptr,
                              copy_values,
                              idx_base,
                              temp_buffer);
}

template <>
rocsparse_status rocsparse_csr2csc(rocsparse_handle               handle,
                                   rocsparse_int                  m,
                                   rocsparse_int                  n,
                                   rocsparse_int                  nnz,
                                   const rocsparse_float_complex* csr_val,
                                   const rocsparse_int*           csr_row_ptr,
                                   const rocsparse_int*           csr_col_ind,
                                   rocsparse_float_complex*       csc_val,
                                   rocsparse_int*                 csc_row_ind,
                                   rocsparse_int*                 csc_col_ptr,
                                   rocsparse_action               copy_values,
                                   rocsparse_index_base           idx_base,
                                   void*                          temp_buffer)
{
    return rocsparse_ccsr2csc(handle,
                              m,
                              n,
                              nnz,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              csc_val,
                              csc_row_ind,
                              csc_col_ptr,
                              copy_values,
                              idx_base,
                              temp_buffer);
}

template <>
rocsparse_status rocsparse_csr2csc(rocsparse_handle                handle,
                                   rocsparse_int                   m,
                                   rocsparse_int                   n,
                                   rocsparse_int                   nnz,
                                   const rocsparse_double_complex* csr_val,
                                   const rocsparse_int*            csr_row_ptr,
                                   const rocsparse_int*            csr_col_ind,
                                   rocsparse_double_complex*       csc_val,
                                   rocsparse_int*                  csc_row_ind,
                                   rocsparse_int*                  csc_col_ptr,
                                   rocsparse_action                copy_values,
                                   rocsparse_index_base            idx_base,
                                   void*                           temp_buffer)
{
    return rocsparse_zcsr2csc(handle,
                              m,
                              n,
                              nnz,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              csc_val,
                              csc_row_ind,
                              csc_col_ptr,
                              copy_values,
                              idx_base,
                              temp_buffer);
}

// gebsr2gebsc

template <>
rocsparse_status rocsparse_gebsr2gebsc_buffer_size(rocsparse_handle     handle,
                                                   rocsparse_int        mb,
                                                   rocsparse_int        nb,
                                                   rocsparse_int        nnzb,
                                                   const float*         bsr_val,
                                                   const rocsparse_int* bsr_row_ptr,
                                                   const rocsparse_int* bsr_col_ind,
                                                   rocsparse_int        row_block_dim,
                                                   rocsparse_int        col_block_dim,
                                                   size_t*              p_buffer_size)
{
    return rocsparse_sgebsr2gebsc_buffer_size(handle,
                                              mb,
                                              nb,
                                              nnzb,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              row_block_dim,
                                              col_block_dim,
                                              p_buffer_size);
}

template <>
rocsparse_status rocsparse_gebsr2gebsc_buffer_size(rocsparse_handle     handle,
                                                   rocsparse_int        mb,
                                                   rocsparse_int        nb,
                                                   rocsparse_int        nnzb,
                                                   const double*        bsr_val,
                                                   const rocsparse_int* bsr_row_ptr,
                                                   const rocsparse_int* bsr_col_ind,
                                                   rocsparse_int        row_block_dim,
                                                   rocsparse_int        col_block_dim,
                                                   size_t*              p_buffer_size)
{
    return rocsparse_dgebsr2gebsc_buffer_size(handle,
                                              mb,
                                              nb,
                                              nnzb,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              row_block_dim,
                                              col_block_dim,
                                              p_buffer_size);
}

template <>
rocsparse_status rocsparse_gebsr2gebsc_buffer_size(rocsparse_handle               handle,
                                                   rocsparse_int                  mb,
                                                   rocsparse_int                  nb,
                                                   rocsparse_int                  nnzb,
                                                   const rocsparse_float_complex* bsr_val,
                                                   const rocsparse_int*           bsr_row_ptr,
                                                   const rocsparse_int*           bsr_col_ind,
                                                   rocsparse_int                  row_block_dim,
                                                   rocsparse_int                  col_block_dim,
                                                   size_t*                        p_buffer_size)
{
    return rocsparse_cgebsr2gebsc_buffer_size(handle,
                                              mb,
                                              nb,
                                              nnzb,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              row_block_dim,
                                              col_block_dim,
                                              p_buffer_size);
}

template <>
rocsparse_status rocsparse_gebsr2gebsc_buffer_size(rocsparse_handle                handle,
                                                   rocsparse_int                   mb,
                                                   rocsparse_int                   nb,
                                                   rocsparse_int                   nnzb,
                                                   const rocsparse_double_complex* bsr_val,
                                                   const rocsparse_int*            bsr_row_ptr,
                                                   const rocsparse_int*            bsr_col_ind,
                                                   rocsparse_int                   row_block_dim,
                                                   rocsparse_int                   col_block_dim,
                                                   size_t*                         p_buffer_size)
{
    return rocsparse_zgebsr2gebsc_buffer_size(handle,
                                              mb,
                                              nb,
                                              nnzb,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              row_block_dim,
                                              col_block_dim,
                                              p_buffer_size);
}

template <>
rocsparse_status rocsparse_gebsr2gebsc(rocsparse_handle     handle,
                                       rocsparse_int        mb,
                                       rocsparse_int        nb,
                                       rocsparse_int        nnzb,
                                       const float*         bsr_val,
                                       const rocsparse_int* bsr_row_ptr,
                                       const rocsparse_int* bsr_col_ind,
                                       rocsparse_int        row_block_dim,
                                       rocsparse_int        col_block_dim,
                                       float*               bsc_val,
                                       rocsparse_int*       bsc_row_ind,
                                       rocsparse_int*       bsc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
{
    return rocsparse_sgebsr2gebsc(handle,
                                  mb,
                                  nb,
                                  nnzb,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  row_block_dim,
                                  col_block_dim,
                                  bsc_val,
                                  bsc_row_ind,
                                  bsc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_gebsr2gebsc(rocsparse_handle     handle,
                                       rocsparse_int        mb,
                                       rocsparse_int        nb,
                                       rocsparse_int        nnzb,
                                       const double*        bsr_val,
                                       const rocsparse_int* bsr_row_ptr,
                                       const rocsparse_int* bsr_col_ind,
                                       rocsparse_int        row_block_dim,
                                       rocsparse_int        col_block_dim,
                                       double*              bsc_val,
                                       rocsparse_int*       bsc_row_ind,
                                       rocsparse_int*       bsc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
{
    return rocsparse_dgebsr2gebsc(handle,
                                  mb,
                                  nb,
                                  nnzb,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  row_block_dim,
                                  col_block_dim,
                                  bsc_val,
                                  bsc_row_ind,
                                  bsc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_gebsr2gebsc(rocsparse_handle               handle,
                                       rocsparse_int                  mb,
                                       rocsparse_int                  nb,
                                       rocsparse_int                  nnzb,
                                       const rocsparse_float_complex* bsr_val,
                                       const rocsparse_int*           bsr_row_ptr,
                                       const rocsparse_int*           bsr_col_ind,
                                       rocsparse_int                  row_block_dim,
                                       rocsparse_int                  col_block_dim,
                                       rocsparse_float_complex*       bsc_val,
                                       rocsparse_int*                 bsc_row_ind,
                                       rocsparse_int*                 bsc_col_ptr,
                                       rocsparse_action               copy_values,
                                       rocsparse_index_base           idx_base,
                                       void*                          temp_buffer)
{
    return rocsparse_cgebsr2gebsc(handle,
                                  mb,
                                  nb,
                                  nnzb,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  row_block_dim,
                                  col_block_dim,
                                  bsc_val,
                                  bsc_row_ind,
                                  bsc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_gebsr2gebsc(rocsparse_handle                handle,
                                       rocsparse_int                   mb,
                                       rocsparse_int                   nb,
                                       rocsparse_int                   nnzb,
                                       const rocsparse_double_complex* bsr_val,
                                       const rocsparse_int*            bsr_row_ptr,
                                       const rocsparse_int*            bsr_col_ind,
                                       rocsparse_int                   row_block_dim,
                                       rocsparse_int                   col_block_dim,
                                       rocsparse_double_complex*       bsc_val,
                                       rocsparse_int*                  bsc_row_ind,
                                       rocsparse_int*                  bsc_col_ptr,
                                       rocsparse_action                copy_values,
                                       rocsparse_index_base            idx_base,
                                       void*                           temp_buffer)
{
    return rocsparse_zgebsr2gebsc(handle,
                                  mb,
                                  nb,
                                  nnzb,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  row_block_dim,
                                  col_block_dim,
                                  bsc_val,
                                  bsc_row_ind,
                                  bsc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
}

// csr2ell
template <>
rocsparse_status rocsparse_csr2ell(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   const rocsparse_mat_descr csr_descr,
                                   const float*              csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int             ell_width,
                                   float*                    ell_val,
                                   rocsparse_int*            ell_col_ind)
{
    return rocsparse_scsr2ell(handle,
                              m,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind);
}

template <>
rocsparse_status rocsparse_csr2ell(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   const rocsparse_mat_descr csr_descr,
                                   const double*             csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int             ell_width,
                                   double*                   ell_val,
                                   rocsparse_int*            ell_col_ind)
{
    return rocsparse_dcsr2ell(handle,
                              m,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind);
}

template <>
rocsparse_status rocsparse_csr2ell(rocsparse_handle               handle,
                                   rocsparse_int                  m,
                                   const rocsparse_mat_descr      csr_descr,
                                   const rocsparse_float_complex* csr_val,
                                   const rocsparse_int*           csr_row_ptr,
                                   const rocsparse_int*           csr_col_ind,
                                   const rocsparse_mat_descr      ell_descr,
                                   rocsparse_int                  ell_width,
                                   rocsparse_float_complex*       ell_val,
                                   rocsparse_int*                 ell_col_ind)
{
    return rocsparse_ccsr2ell(handle,
                              m,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind);
}

template <>
rocsparse_status rocsparse_csr2ell(rocsparse_handle                handle,
                                   rocsparse_int                   m,
                                   const rocsparse_mat_descr       csr_descr,
                                   const rocsparse_double_complex* csr_val,
                                   const rocsparse_int*            csr_row_ptr,
                                   const rocsparse_int*            csr_col_ind,
                                   const rocsparse_mat_descr       ell_descr,
                                   rocsparse_int                   ell_width,
                                   rocsparse_double_complex*       ell_val,
                                   rocsparse_int*                  ell_col_ind)
{
    return rocsparse_zcsr2ell(handle,
                              m,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind);
}

// csr2hyb
template <>
rocsparse_status rocsparse_csr2hyb(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr descr,
                                   const float*              csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_hyb_mat         hyb,
                                   rocsparse_int             user_ell_width,
                                   rocsparse_hyb_partition   partition_type)
{
    return rocsparse_scsr2hyb(handle,
                              m,
                              n,
                              descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              hyb,
                              user_ell_width,
                              partition_type);
}

template <>
rocsparse_status rocsparse_csr2hyb(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr descr,
                                   const double*             csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_hyb_mat         hyb,
                                   rocsparse_int             user_ell_width,
                                   rocsparse_hyb_partition   partition_type)
{
    return rocsparse_dcsr2hyb(handle,
                              m,
                              n,
                              descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              hyb,
                              user_ell_width,
                              partition_type);
}

template <>
rocsparse_status rocsparse_csr2hyb(rocsparse_handle               handle,
                                   rocsparse_int                  m,
                                   rocsparse_int                  n,
                                   const rocsparse_mat_descr      descr,
                                   const rocsparse_float_complex* csr_val,
                                   const rocsparse_int*           csr_row_ptr,
                                   const rocsparse_int*           csr_col_ind,
                                   rocsparse_hyb_mat              hyb,
                                   rocsparse_int                  user_ell_width,
                                   rocsparse_hyb_partition        partition_type)
{
    return rocsparse_ccsr2hyb(handle,
                              m,
                              n,
                              descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              hyb,
                              user_ell_width,
                              partition_type);
}

template <>
rocsparse_status rocsparse_csr2hyb(rocsparse_handle                handle,
                                   rocsparse_int                   m,
                                   rocsparse_int                   n,
                                   const rocsparse_mat_descr       descr,
                                   const rocsparse_double_complex* csr_val,
                                   const rocsparse_int*            csr_row_ptr,
                                   const rocsparse_int*            csr_col_ind,
                                   rocsparse_hyb_mat               hyb,
                                   rocsparse_int                   user_ell_width,
                                   rocsparse_hyb_partition         partition_type)
{
    return rocsparse_zcsr2hyb(handle,
                              m,
                              n,
                              descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              hyb,
                              user_ell_width,
                              partition_type);
}

// csr2bsr
template <>
rocsparse_status rocsparse_csr2bsr(rocsparse_handle          handle,
                                   rocsparse_direction       direction,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr csr_descr,
                                   const float*              csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_int             block_dim,
                                   const rocsparse_mat_descr bsr_descr,
                                   float*                    bsr_val,
                                   rocsparse_int*            bsr_row_ptr,
                                   rocsparse_int*            bsr_col_ind)
{
    return rocsparse_scsr2bsr(handle,
                              direction,
                              m,
                              n,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              block_dim,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind);
}

template <>
rocsparse_status rocsparse_csr2bsr(rocsparse_handle          handle,
                                   rocsparse_direction       direction,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr csr_descr,
                                   const double*             csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   const rocsparse_int*      csr_col_ind,
                                   rocsparse_int             block_dim,
                                   const rocsparse_mat_descr bsr_descr,
                                   double*                   bsr_val,
                                   rocsparse_int*            bsr_row_ptr,
                                   rocsparse_int*            bsr_col_ind)
{
    return rocsparse_dcsr2bsr(handle,
                              direction,
                              m,
                              n,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              block_dim,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind);
}

template <>
rocsparse_status rocsparse_csr2bsr(rocsparse_handle               handle,
                                   rocsparse_direction            direction,
                                   rocsparse_int                  m,
                                   rocsparse_int                  n,
                                   const rocsparse_mat_descr      csr_descr,
                                   const rocsparse_float_complex* csr_val,
                                   const rocsparse_int*           csr_row_ptr,
                                   const rocsparse_int*           csr_col_ind,
                                   rocsparse_int                  block_dim,
                                   const rocsparse_mat_descr      bsr_descr,
                                   rocsparse_float_complex*       bsr_val,
                                   rocsparse_int*                 bsr_row_ptr,
                                   rocsparse_int*                 bsr_col_ind)
{
    return rocsparse_ccsr2bsr(handle,
                              direction,
                              m,
                              n,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              block_dim,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind);
}

template <>
rocsparse_status rocsparse_csr2bsr(rocsparse_handle                handle,
                                   rocsparse_direction             direction,
                                   rocsparse_int                   m,
                                   rocsparse_int                   n,
                                   const rocsparse_mat_descr       csr_descr,
                                   const rocsparse_double_complex* csr_val,
                                   const rocsparse_int*            csr_row_ptr,
                                   const rocsparse_int*            csr_col_ind,
                                   rocsparse_int                   block_dim,
                                   const rocsparse_mat_descr       bsr_descr,
                                   rocsparse_double_complex*       bsr_val,
                                   rocsparse_int*                  bsr_row_ptr,
                                   rocsparse_int*                  bsr_col_ind)
{
    return rocsparse_zcsr2bsr(handle,
                              direction,
                              m,
                              n,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind,
                              block_dim,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind);
}

// csr2gebsr_buffer_size
template <>
rocsparse_status rocsparse_csr2gebsr_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       direction,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr csr_descr,
                                                 const float*              csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_int             row_block_dim,
                                                 rocsparse_int             col_block_dim,
                                                 size_t*                   p_buffer_size)
{
    return rocsparse_scsr2gebsr_buffer_size(handle,
                                            direction,
                                            m,
                                            n,
                                            csr_descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            row_block_dim,
                                            col_block_dim,
                                            p_buffer_size);
}

template <>
rocsparse_status rocsparse_csr2gebsr_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       direction,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const rocsparse_mat_descr csr_descr,
                                                 const double*             csr_val,
                                                 const rocsparse_int*      csr_row_ptr,
                                                 const rocsparse_int*      csr_col_ind,
                                                 rocsparse_int             row_block_dim,
                                                 rocsparse_int             col_block_dim,
                                                 size_t*                   p_buffer_size)
{
    return rocsparse_dcsr2gebsr_buffer_size(handle,
                                            direction,
                                            m,
                                            n,
                                            csr_descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            row_block_dim,
                                            col_block_dim,
                                            p_buffer_size);
}

template <>
rocsparse_status rocsparse_csr2gebsr_buffer_size(rocsparse_handle               handle,
                                                 rocsparse_direction            direction,
                                                 rocsparse_int                  m,
                                                 rocsparse_int                  n,
                                                 const rocsparse_mat_descr      csr_descr,
                                                 const rocsparse_float_complex* csr_val,
                                                 const rocsparse_int*           csr_row_ptr,
                                                 const rocsparse_int*           csr_col_ind,
                                                 rocsparse_int                  row_block_dim,
                                                 rocsparse_int                  col_block_dim,
                                                 size_t*                        p_buffer_size)
{
    return rocsparse_ccsr2gebsr_buffer_size(handle,
                                            direction,
                                            m,
                                            n,
                                            csr_descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            row_block_dim,
                                            col_block_dim,
                                            p_buffer_size);
}

template <>
rocsparse_status rocsparse_csr2gebsr_buffer_size(rocsparse_handle                handle,
                                                 rocsparse_direction             direction,
                                                 rocsparse_int                   m,
                                                 rocsparse_int                   n,
                                                 const rocsparse_mat_descr       csr_descr,
                                                 const rocsparse_double_complex* csr_val,
                                                 const rocsparse_int*            csr_row_ptr,
                                                 const rocsparse_int*            csr_col_ind,
                                                 rocsparse_int                   row_block_dim,
                                                 rocsparse_int                   col_block_dim,
                                                 size_t*                         p_buffer_size)
{
    return rocsparse_zcsr2gebsr_buffer_size(handle,
                                            direction,
                                            m,
                                            n,
                                            csr_descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            row_block_dim,
                                            col_block_dim,
                                            p_buffer_size);
}

// csr2gebsr
template <>
rocsparse_status rocsparse_csr2gebsr(rocsparse_handle          handle,
                                     rocsparse_direction       direction,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr csr_descr,
                                     const float*              csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     const rocsparse_mat_descr bsr_descr,
                                     float*                    bsr_val,
                                     rocsparse_int*            bsr_row_ptr,
                                     rocsparse_int*            bsr_col_ind,
                                     rocsparse_int             row_block_dim,
                                     rocsparse_int             col_block_dim,
                                     void*                     p_buffer)
{
    return rocsparse_scsr2gebsr(handle,
                                direction,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                p_buffer);
}

template <>
rocsparse_status rocsparse_csr2gebsr(rocsparse_handle          handle,
                                     rocsparse_direction       direction,
                                     rocsparse_int             m,
                                     rocsparse_int             n,
                                     const rocsparse_mat_descr csr_descr,
                                     const double*             csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     const rocsparse_mat_descr bsr_descr,
                                     double*                   bsr_val,
                                     rocsparse_int*            bsr_row_ptr,
                                     rocsparse_int*            bsr_col_ind,
                                     rocsparse_int             row_block_dim,
                                     rocsparse_int             col_block_dim,
                                     void*                     p_buffer)
{
    return rocsparse_dcsr2gebsr(handle,
                                direction,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                p_buffer);
}

template <>
rocsparse_status rocsparse_csr2gebsr(rocsparse_handle               handle,
                                     rocsparse_direction            direction,
                                     rocsparse_int                  m,
                                     rocsparse_int                  n,
                                     const rocsparse_mat_descr      csr_descr,
                                     const rocsparse_float_complex* csr_val,
                                     const rocsparse_int*           csr_row_ptr,
                                     const rocsparse_int*           csr_col_ind,
                                     const rocsparse_mat_descr      bsr_descr,
                                     rocsparse_float_complex*       bsr_val,
                                     rocsparse_int*                 bsr_row_ptr,
                                     rocsparse_int*                 bsr_col_ind,
                                     rocsparse_int                  row_block_dim,
                                     rocsparse_int                  col_block_dim,
                                     void*                          p_buffer)
{
    return rocsparse_ccsr2gebsr(handle,
                                direction,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                p_buffer);
}

template <>
rocsparse_status rocsparse_csr2gebsr(rocsparse_handle                handle,
                                     rocsparse_direction             direction,
                                     rocsparse_int                   m,
                                     rocsparse_int                   n,
                                     const rocsparse_mat_descr       csr_descr,
                                     const rocsparse_double_complex* csr_val,
                                     const rocsparse_int*            csr_row_ptr,
                                     const rocsparse_int*            csr_col_ind,
                                     const rocsparse_mat_descr       bsr_descr,
                                     rocsparse_double_complex*       bsr_val,
                                     rocsparse_int*                  bsr_row_ptr,
                                     rocsparse_int*                  bsr_col_ind,
                                     rocsparse_int                   row_block_dim,
                                     rocsparse_int                   col_block_dim,
                                     void*                           p_buffer)
{
    return rocsparse_zcsr2gebsr(handle,
                                direction,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                p_buffer);
}

// ell2csr
template <>
rocsparse_status rocsparse_ell2csr(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int             ell_width,
                                   const float*              ell_val,
                                   const rocsparse_int*      ell_col_ind,
                                   const rocsparse_mat_descr csr_descr,
                                   float*                    csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   rocsparse_int*            csr_col_ind)
{
    return rocsparse_sell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

template <>
rocsparse_status rocsparse_ell2csr(rocsparse_handle          handle,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr ell_descr,
                                   rocsparse_int             ell_width,
                                   const double*             ell_val,
                                   const rocsparse_int*      ell_col_ind,
                                   const rocsparse_mat_descr csr_descr,
                                   double*                   csr_val,
                                   const rocsparse_int*      csr_row_ptr,
                                   rocsparse_int*            csr_col_ind)
{
    return rocsparse_dell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

template <>
rocsparse_status rocsparse_ell2csr(rocsparse_handle               handle,
                                   rocsparse_int                  m,
                                   rocsparse_int                  n,
                                   const rocsparse_mat_descr      ell_descr,
                                   rocsparse_int                  ell_width,
                                   const rocsparse_float_complex* ell_val,
                                   const rocsparse_int*           ell_col_ind,
                                   const rocsparse_mat_descr      csr_descr,
                                   rocsparse_float_complex*       csr_val,
                                   const rocsparse_int*           csr_row_ptr,
                                   rocsparse_int*                 csr_col_ind)
{
    return rocsparse_cell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

template <>
rocsparse_status rocsparse_ell2csr(rocsparse_handle                handle,
                                   rocsparse_int                   m,
                                   rocsparse_int                   n,
                                   const rocsparse_mat_descr       ell_descr,
                                   rocsparse_int                   ell_width,
                                   const rocsparse_double_complex* ell_val,
                                   const rocsparse_int*            ell_col_ind,
                                   const rocsparse_mat_descr       csr_descr,
                                   rocsparse_double_complex*       csr_val,
                                   const rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*                  csr_col_ind)
{
    return rocsparse_zell2csr(handle,
                              m,
                              n,
                              ell_descr,
                              ell_width,
                              ell_val,
                              ell_col_ind,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

// hyb2csr
template <>
rocsparse_status rocsparse_hyb2csr(rocsparse_handle          handle,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_hyb_mat   hyb,
                                   float*                    csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind,
                                   void*                     temp_buffer)
{
    return rocsparse_shyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

template <>
rocsparse_status rocsparse_hyb2csr(rocsparse_handle          handle,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_hyb_mat   hyb,
                                   double*                   csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind,
                                   void*                     temp_buffer)
{
    return rocsparse_dhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

template <>
rocsparse_status rocsparse_hyb2csr(rocsparse_handle          handle,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_hyb_mat   hyb,
                                   rocsparse_float_complex*  csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind,
                                   void*                     temp_buffer)
{
    return rocsparse_chyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

template <>
rocsparse_status rocsparse_hyb2csr(rocsparse_handle          handle,
                                   const rocsparse_mat_descr descr,
                                   const rocsparse_hyb_mat   hyb,
                                   rocsparse_double_complex* csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind,
                                   void*                     temp_buffer)
{
    return rocsparse_zhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);
}

// bsr2csr
template <>
rocsparse_status rocsparse_bsr2csr(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nb,
                                   const rocsparse_mat_descr bsr_descr,
                                   const float*              bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   const rocsparse_mat_descr csr_descr,
                                   float*                    csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind)
{
    return rocsparse_sbsr2csr(handle,
                              dir,
                              mb,
                              nb,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

template <>
rocsparse_status rocsparse_bsr2csr(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             mb,
                                   rocsparse_int             nb,
                                   const rocsparse_mat_descr bsr_descr,
                                   const double*             bsr_val,
                                   const rocsparse_int*      bsr_row_ptr,
                                   const rocsparse_int*      bsr_col_ind,
                                   rocsparse_int             block_dim,
                                   const rocsparse_mat_descr csr_descr,
                                   double*                   csr_val,
                                   rocsparse_int*            csr_row_ptr,
                                   rocsparse_int*            csr_col_ind)
{
    return rocsparse_dbsr2csr(handle,
                              dir,
                              mb,
                              nb,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

template <>
rocsparse_status rocsparse_bsr2csr(rocsparse_handle               handle,
                                   rocsparse_direction            dir,
                                   rocsparse_int                  mb,
                                   rocsparse_int                  nb,
                                   const rocsparse_mat_descr      bsr_descr,
                                   const rocsparse_float_complex* bsr_val,
                                   const rocsparse_int*           bsr_row_ptr,
                                   const rocsparse_int*           bsr_col_ind,
                                   rocsparse_int                  block_dim,
                                   const rocsparse_mat_descr      csr_descr,
                                   rocsparse_float_complex*       csr_val,
                                   rocsparse_int*                 csr_row_ptr,
                                   rocsparse_int*                 csr_col_ind)
{
    return rocsparse_cbsr2csr(handle,
                              dir,
                              mb,
                              nb,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

template <>
rocsparse_status rocsparse_bsr2csr(rocsparse_handle                handle,
                                   rocsparse_direction             dir,
                                   rocsparse_int                   mb,
                                   rocsparse_int                   nb,
                                   const rocsparse_mat_descr       bsr_descr,
                                   const rocsparse_double_complex* bsr_val,
                                   const rocsparse_int*            bsr_row_ptr,
                                   const rocsparse_int*            bsr_col_ind,
                                   rocsparse_int                   block_dim,
                                   const rocsparse_mat_descr       csr_descr,
                                   rocsparse_double_complex*       csr_val,
                                   rocsparse_int*                  csr_row_ptr,
                                   rocsparse_int*                  csr_col_ind)
{
    return rocsparse_zbsr2csr(handle,
                              dir,
                              mb,
                              nb,
                              bsr_descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              block_dim,
                              csr_descr,
                              csr_val,
                              csr_row_ptr,
                              csr_col_ind);
}

// gebsr2csr
template <>
rocsparse_status rocsparse_gebsr2csr(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_int             mb,
                                     rocsparse_int             nb,
                                     const rocsparse_mat_descr bsr_descr,
                                     const float*              bsr_val,
                                     const rocsparse_int*      bsr_row_ptr,
                                     const rocsparse_int*      bsr_col_ind,
                                     rocsparse_int             row_block_dim,
                                     rocsparse_int             col_block_dim,
                                     const rocsparse_mat_descr csr_descr,
                                     float*                    csr_val,
                                     rocsparse_int*            csr_row_ptr,
                                     rocsparse_int*            csr_col_ind)
{
    return rocsparse_sgebsr2csr(handle,
                                dir,
                                mb,
                                nb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind);
}

template <>
rocsparse_status rocsparse_gebsr2csr(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_int             mb,
                                     rocsparse_int             nb,
                                     const rocsparse_mat_descr bsr_descr,
                                     const double*             bsr_val,
                                     const rocsparse_int*      bsr_row_ptr,
                                     const rocsparse_int*      bsr_col_ind,
                                     rocsparse_int             row_block_dim,
                                     rocsparse_int             col_block_dim,
                                     const rocsparse_mat_descr csr_descr,
                                     double*                   csr_val,
                                     rocsparse_int*            csr_row_ptr,
                                     rocsparse_int*            csr_col_ind)
{
    return rocsparse_dgebsr2csr(handle,
                                dir,
                                mb,
                                nb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind);
}

template <>
rocsparse_status rocsparse_gebsr2csr(rocsparse_handle               handle,
                                     rocsparse_direction            dir,
                                     rocsparse_int                  mb,
                                     rocsparse_int                  nb,
                                     const rocsparse_mat_descr      bsr_descr,
                                     const rocsparse_float_complex* bsr_val,
                                     const rocsparse_int*           bsr_row_ptr,
                                     const rocsparse_int*           bsr_col_ind,
                                     rocsparse_int                  row_block_dim,
                                     rocsparse_int                  col_block_dim,
                                     const rocsparse_mat_descr      csr_descr,
                                     rocsparse_float_complex*       csr_val,
                                     rocsparse_int*                 csr_row_ptr,
                                     rocsparse_int*                 csr_col_ind)
{
    return rocsparse_cgebsr2csr(handle,
                                dir,
                                mb,
                                nb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind);
}

template <>
rocsparse_status rocsparse_gebsr2csr(rocsparse_handle                handle,
                                     rocsparse_direction             dir,
                                     rocsparse_int                   mb,
                                     rocsparse_int                   nb,
                                     const rocsparse_mat_descr       bsr_descr,
                                     const rocsparse_double_complex* bsr_val,
                                     const rocsparse_int*            bsr_row_ptr,
                                     const rocsparse_int*            bsr_col_ind,
                                     rocsparse_int                   row_block_dim,
                                     rocsparse_int                   col_block_dim,
                                     const rocsparse_mat_descr       csr_descr,
                                     rocsparse_double_complex*       csr_val,
                                     rocsparse_int*                  csr_row_ptr,
                                     rocsparse_int*                  csr_col_ind)
{
    return rocsparse_zgebsr2csr(handle,
                                dir,
                                mb,
                                nb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                row_block_dim,
                                col_block_dim,
                                csr_descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind);
}

// gebsr2gebsr_buffer_size
template <>
rocsparse_status rocsparse_gebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr_A,
                                                   const float*              bsr_val_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   rocsparse_int             row_block_dim_A,
                                                   rocsparse_int             col_block_dim_A,
                                                   rocsparse_int             row_block_dim_C,
                                                   rocsparse_int             col_block_dim_C,
                                                   size_t*                   buffer_size)
{
    return rocsparse_sgebsr2gebsr_buffer_size(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              buffer_size);
}

template <>
rocsparse_status rocsparse_gebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr_A,
                                                   const double*             bsr_val_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   rocsparse_int             row_block_dim_A,
                                                   rocsparse_int             col_block_dim_A,
                                                   rocsparse_int             row_block_dim_C,
                                                   rocsparse_int             col_block_dim_C,
                                                   size_t*                   buffer_size)
{
    return rocsparse_dgebsr2gebsr_buffer_size(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              buffer_size);
}

template <>
rocsparse_status rocsparse_gebsr2gebsr_buffer_size(rocsparse_handle               handle,
                                                   rocsparse_direction            dir,
                                                   rocsparse_int                  mb,
                                                   rocsparse_int                  nb,
                                                   rocsparse_int                  nnzb,
                                                   const rocsparse_mat_descr      descr_A,
                                                   const rocsparse_float_complex* bsr_val_A,
                                                   const rocsparse_int*           bsr_row_ptr_A,
                                                   const rocsparse_int*           bsr_col_ind_A,
                                                   rocsparse_int                  row_block_dim_A,
                                                   rocsparse_int                  col_block_dim_A,
                                                   rocsparse_int                  row_block_dim_C,
                                                   rocsparse_int                  col_block_dim_C,
                                                   size_t*                        buffer_size)
{
    return rocsparse_cgebsr2gebsr_buffer_size(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              buffer_size);
}

template <>
rocsparse_status rocsparse_gebsr2gebsr_buffer_size(rocsparse_handle                handle,
                                                   rocsparse_direction             dir,
                                                   rocsparse_int                   mb,
                                                   rocsparse_int                   nb,
                                                   rocsparse_int                   nnzb,
                                                   const rocsparse_mat_descr       descr_A,
                                                   const rocsparse_double_complex* bsr_val_A,
                                                   const rocsparse_int*            bsr_row_ptr_A,
                                                   const rocsparse_int*            bsr_col_ind_A,
                                                   rocsparse_int                   row_block_dim_A,
                                                   rocsparse_int                   col_block_dim_A,
                                                   rocsparse_int                   row_block_dim_C,
                                                   rocsparse_int                   col_block_dim_C,
                                                   size_t*                         buffer_size)
{
    return rocsparse_zgebsr2gebsr_buffer_size(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              buffer_size);
}

// gebsr2gebsr
template <>
rocsparse_status rocsparse_gebsr2gebsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_int             mb,
                                       rocsparse_int             nb,
                                       rocsparse_int             nnzb,
                                       const rocsparse_mat_descr descr_A,
                                       const float*              bsr_val_A,
                                       const rocsparse_int*      bsr_row_ptr_A,
                                       const rocsparse_int*      bsr_col_ind_A,
                                       rocsparse_int             row_block_dim_A,
                                       rocsparse_int             col_block_dim_A,
                                       const rocsparse_mat_descr descr_C,
                                       float*                    bsr_val_C,
                                       rocsparse_int*            bsr_row_ptr_C,
                                       rocsparse_int*            bsr_col_ind_C,
                                       rocsparse_int             row_block_dim_C,
                                       rocsparse_int             col_block_dim_C,
                                       void*                     temp_buffer)
{
    return rocsparse_sgebsr2gebsr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  nnzb,
                                  descr_A,
                                  bsr_val_A,
                                  bsr_row_ptr_A,
                                  bsr_col_ind_A,
                                  row_block_dim_A,
                                  col_block_dim_A,
                                  descr_C,
                                  bsr_val_C,
                                  bsr_row_ptr_C,
                                  bsr_col_ind_C,
                                  row_block_dim_C,
                                  col_block_dim_C,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_gebsr2gebsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       rocsparse_int             mb,
                                       rocsparse_int             nb,
                                       rocsparse_int             nnzb,
                                       const rocsparse_mat_descr descr_A,
                                       const double*             bsr_val_A,
                                       const rocsparse_int*      bsr_row_ptr_A,
                                       const rocsparse_int*      bsr_col_ind_A,
                                       rocsparse_int             row_block_dim_A,
                                       rocsparse_int             col_block_dim_A,
                                       const rocsparse_mat_descr descr_C,
                                       double*                   bsr_val_C,
                                       rocsparse_int*            bsr_row_ptr_C,
                                       rocsparse_int*            bsr_col_ind_C,
                                       rocsparse_int             row_block_dim_C,
                                       rocsparse_int             col_block_dim_C,
                                       void*                     temp_buffer)
{
    return rocsparse_dgebsr2gebsr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  nnzb,
                                  descr_A,
                                  bsr_val_A,
                                  bsr_row_ptr_A,
                                  bsr_col_ind_A,
                                  row_block_dim_A,
                                  col_block_dim_A,
                                  descr_C,
                                  bsr_val_C,
                                  bsr_row_ptr_C,
                                  bsr_col_ind_C,
                                  row_block_dim_C,
                                  col_block_dim_C,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_gebsr2gebsr(rocsparse_handle               handle,
                                       rocsparse_direction            dir,
                                       rocsparse_int                  mb,
                                       rocsparse_int                  nb,
                                       rocsparse_int                  nnzb,
                                       const rocsparse_mat_descr      descr_A,
                                       const rocsparse_float_complex* bsr_val_A,
                                       const rocsparse_int*           bsr_row_ptr_A,
                                       const rocsparse_int*           bsr_col_ind_A,
                                       rocsparse_int                  row_block_dim_A,
                                       rocsparse_int                  col_block_dim_A,
                                       const rocsparse_mat_descr      descr_C,
                                       rocsparse_float_complex*       bsr_val_C,
                                       rocsparse_int*                 bsr_row_ptr_C,
                                       rocsparse_int*                 bsr_col_ind_C,
                                       rocsparse_int                  row_block_dim_C,
                                       rocsparse_int                  col_block_dim_C,
                                       void*                          temp_buffer)
{
    return rocsparse_cgebsr2gebsr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  nnzb,
                                  descr_A,
                                  bsr_val_A,
                                  bsr_row_ptr_A,
                                  bsr_col_ind_A,
                                  row_block_dim_A,
                                  col_block_dim_A,
                                  descr_C,
                                  bsr_val_C,
                                  bsr_row_ptr_C,
                                  bsr_col_ind_C,
                                  row_block_dim_C,
                                  col_block_dim_C,
                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_gebsr2gebsr(rocsparse_handle                handle,
                                       rocsparse_direction             dir,
                                       rocsparse_int                   mb,
                                       rocsparse_int                   nb,
                                       rocsparse_int                   nnzb,
                                       const rocsparse_mat_descr       descr_A,
                                       const rocsparse_double_complex* bsr_val_A,
                                       const rocsparse_int*            bsr_row_ptr_A,
                                       const rocsparse_int*            bsr_col_ind_A,
                                       rocsparse_int                   row_block_dim_A,
                                       rocsparse_int                   col_block_dim_A,
                                       const rocsparse_mat_descr       descr_C,
                                       rocsparse_double_complex*       bsr_val_C,
                                       rocsparse_int*                  bsr_row_ptr_C,
                                       rocsparse_int*                  bsr_col_ind_C,
                                       rocsparse_int                   row_block_dim_C,
                                       rocsparse_int                   col_block_dim_C,
                                       void*                           temp_buffer)
{
    return rocsparse_zgebsr2gebsr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  nnzb,
                                  descr_A,
                                  bsr_val_A,
                                  bsr_row_ptr_A,
                                  bsr_col_ind_A,
                                  row_block_dim_A,
                                  col_block_dim_A,
                                  descr_C,
                                  bsr_val_C,
                                  bsr_row_ptr_C,
                                  bsr_col_ind_C,
                                  row_block_dim_C,
                                  col_block_dim_C,
                                  temp_buffer);
}

// csr2csr_compress
template <>
rocsparse_status rocsparse_csr2csr_compress(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const rocsparse_mat_descr descr_A,
                                            const float*              csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            rocsparse_int             nnz_A,
                                            const rocsparse_int*      nnz_per_row,
                                            float*                    csr_val_C,
                                            rocsparse_int*            csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C,
                                            float                     tol)
{
    return rocsparse_scsr2csr_compress(handle,
                                       m,
                                       n,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       nnz_A,
                                       nnz_per_row,
                                       csr_val_C,
                                       csr_row_ptr_C,
                                       csr_col_ind_C,
                                       tol);
}

template <>
rocsparse_status rocsparse_csr2csr_compress(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            const rocsparse_mat_descr descr_A,
                                            const double*             csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            rocsparse_int             nnz_A,
                                            const rocsparse_int*      nnz_per_row,
                                            double*                   csr_val_C,
                                            rocsparse_int*            csr_row_ptr_C,
                                            rocsparse_int*            csr_col_ind_C,
                                            double                    tol)
{
    return rocsparse_dcsr2csr_compress(handle,
                                       m,
                                       n,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       nnz_A,
                                       nnz_per_row,
                                       csr_val_C,
                                       csr_row_ptr_C,
                                       csr_col_ind_C,
                                       tol);
}

template <>
rocsparse_status rocsparse_csr2csr_compress(rocsparse_handle               handle,
                                            rocsparse_int                  m,
                                            rocsparse_int                  n,
                                            const rocsparse_mat_descr      descr_A,
                                            const rocsparse_float_complex* csr_val_A,
                                            const rocsparse_int*           csr_row_ptr_A,
                                            const rocsparse_int*           csr_col_ind_A,
                                            rocsparse_int                  nnz_A,
                                            const rocsparse_int*           nnz_per_row,
                                            rocsparse_float_complex*       csr_val_C,
                                            rocsparse_int*                 csr_row_ptr_C,
                                            rocsparse_int*                 csr_col_ind_C,
                                            rocsparse_float_complex        tol)
{
    return rocsparse_ccsr2csr_compress(handle,
                                       m,
                                       n,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       nnz_A,
                                       nnz_per_row,
                                       csr_val_C,
                                       csr_row_ptr_C,
                                       csr_col_ind_C,
                                       tol);
}

template <>
rocsparse_status rocsparse_csr2csr_compress(rocsparse_handle                handle,
                                            rocsparse_int                   m,
                                            rocsparse_int                   n,
                                            const rocsparse_mat_descr       descr_A,
                                            const rocsparse_double_complex* csr_val_A,
                                            const rocsparse_int*            csr_row_ptr_A,
                                            const rocsparse_int*            csr_col_ind_A,
                                            rocsparse_int                   nnz_A,
                                            const rocsparse_int*            nnz_per_row,
                                            rocsparse_double_complex*       csr_val_C,
                                            rocsparse_int*                  csr_row_ptr_C,
                                            rocsparse_int*                  csr_col_ind_C,
                                            rocsparse_double_complex        tol)
{
    return rocsparse_zcsr2csr_compress(handle,
                                       m,
                                       n,
                                       descr_A,
                                       csr_val_A,
                                       csr_row_ptr_A,
                                       csr_col_ind_A,
                                       nnz_A,
                                       nnz_per_row,
                                       csr_val_C,
                                       csr_row_ptr_C,
                                       csr_col_ind_C,
                                       tol);
}

// prune_csr2csr_buffer_size
template <>
rocsparse_status rocsparse_prune_csr2csr_buffer_size(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             n,
                                                     rocsparse_int             nnz_A,
                                                     const rocsparse_mat_descr csr_descr_A,
                                                     const float*              csr_val_A,
                                                     const rocsparse_int*      csr_row_ptr_A,
                                                     const rocsparse_int*      csr_col_ind_A,
                                                     const float*              threshold,
                                                     const rocsparse_mat_descr csr_descr_C,
                                                     const float*              csr_val_C,
                                                     const rocsparse_int*      csr_row_ptr_C,
                                                     const rocsparse_int*      csr_col_ind_C,
                                                     size_t*                   buffer_size)
{
    return rocsparse_sprune_csr2csr_buffer_size(handle,
                                                m,
                                                n,
                                                nnz_A,
                                                csr_descr_A,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                threshold,
                                                csr_descr_C,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                buffer_size);
}

template <>
rocsparse_status rocsparse_prune_csr2csr_buffer_size(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             n,
                                                     rocsparse_int             nnz_A,
                                                     const rocsparse_mat_descr csr_descr_A,
                                                     const double*             csr_val_A,
                                                     const rocsparse_int*      csr_row_ptr_A,
                                                     const rocsparse_int*      csr_col_ind_A,
                                                     const double*             threshold,
                                                     const rocsparse_mat_descr csr_descr_C,
                                                     const double*             csr_val_C,
                                                     const rocsparse_int*      csr_row_ptr_C,
                                                     const rocsparse_int*      csr_col_ind_C,
                                                     size_t*                   buffer_size)
{
    return rocsparse_dprune_csr2csr_buffer_size(handle,
                                                m,
                                                n,
                                                nnz_A,
                                                csr_descr_A,
                                                csr_val_A,
                                                csr_row_ptr_A,
                                                csr_col_ind_A,
                                                threshold,
                                                csr_descr_C,
                                                csr_val_C,
                                                csr_row_ptr_C,
                                                csr_col_ind_C,
                                                buffer_size);
}

// prune_csr2csr_nnz
template <>
rocsparse_status rocsparse_prune_csr2csr_nnz(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_mat_descr csr_descr_A,
                                             const float*              csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             const float*              threshold,
                                             const rocsparse_mat_descr csr_descr_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            nnz_total_dev_host_ptr,
                                             void*                     temp_buffer)
{
    return rocsparse_sprune_csr2csr_nnz(handle,
                                        m,
                                        n,
                                        nnz_A,
                                        csr_descr_A,
                                        csr_val_A,
                                        csr_row_ptr_A,
                                        csr_col_ind_A,
                                        threshold,
                                        csr_descr_C,
                                        csr_row_ptr_C,
                                        nnz_total_dev_host_ptr,
                                        temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_csr2csr_nnz(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_mat_descr csr_descr_A,
                                             const double*             csr_val_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             const double*             threshold,
                                             const rocsparse_mat_descr csr_descr_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            nnz_total_dev_host_ptr,
                                             void*                     temp_buffer)
{
    return rocsparse_dprune_csr2csr_nnz(handle,
                                        m,
                                        n,
                                        nnz_A,
                                        csr_descr_A,
                                        csr_val_A,
                                        csr_row_ptr_A,
                                        csr_col_ind_A,
                                        threshold,
                                        csr_descr_C,
                                        csr_row_ptr_C,
                                        nnz_total_dev_host_ptr,
                                        temp_buffer);
}

// prune_csr2csr
template <>
rocsparse_status rocsparse_prune_csr2csr(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         rocsparse_int             nnz_A,
                                         const rocsparse_mat_descr csr_descr_A,
                                         const float*              csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         const rocsparse_int*      csr_col_ind_A,
                                         const float*              threshold,
                                         const rocsparse_mat_descr csr_descr_C,
                                         float*                    csr_val_C,
                                         const rocsparse_int*      csr_row_ptr_C,
                                         rocsparse_int*            csr_col_ind_C,
                                         void*                     temp_buffer)
{
    return rocsparse_sprune_csr2csr(handle,
                                    m,
                                    n,
                                    nnz_A,
                                    csr_descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    threshold,
                                    csr_descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_csr2csr(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         rocsparse_int             nnz_A,
                                         const rocsparse_mat_descr csr_descr_A,
                                         const double*             csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         const rocsparse_int*      csr_col_ind_A,
                                         const double*             threshold,
                                         const rocsparse_mat_descr csr_descr_C,
                                         double*                   csr_val_C,
                                         const rocsparse_int*      csr_row_ptr_C,
                                         rocsparse_int*            csr_col_ind_C,
                                         void*                     temp_buffer)
{
    return rocsparse_dprune_csr2csr(handle,
                                    m,
                                    n,
                                    nnz_A,
                                    csr_descr_A,
                                    csr_val_A,
                                    csr_row_ptr_A,
                                    csr_col_ind_A,
                                    threshold,
                                    csr_descr_C,
                                    csr_val_C,
                                    csr_row_ptr_C,
                                    csr_col_ind_C,
                                    temp_buffer);
}

// prune_csr2csr_by_percentage_buffer_size
template <>
rocsparse_status
    rocsparse_prune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz_A,
                                                      const rocsparse_mat_descr csr_descr_A,
                                                      const float*              csr_val_A,
                                                      const rocsparse_int*      csr_row_ptr_A,
                                                      const rocsparse_int*      csr_col_ind_A,
                                                      float                     percentage,
                                                      const rocsparse_mat_descr csr_descr_C,
                                                      const float*              csr_val_C,
                                                      const rocsparse_int*      csr_row_ptr_C,
                                                      const rocsparse_int*      csr_col_ind_C,
                                                      rocsparse_mat_info        info,
                                                      size_t*                   buffer_size)
{
    return rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle,
                                                              m,
                                                              n,
                                                              nnz_A,
                                                              csr_descr_A,
                                                              csr_val_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              percentage,
                                                              csr_descr_C,
                                                              csr_val_C,
                                                              csr_row_ptr_C,
                                                              csr_col_ind_C,
                                                              info,
                                                              buffer_size);
}

template <>
rocsparse_status
    rocsparse_prune_csr2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                      rocsparse_int             m,
                                                      rocsparse_int             n,
                                                      rocsparse_int             nnz_A,
                                                      const rocsparse_mat_descr csr_descr_A,
                                                      const double*             csr_val_A,
                                                      const rocsparse_int*      csr_row_ptr_A,
                                                      const rocsparse_int*      csr_col_ind_A,
                                                      double                    percentage,
                                                      const rocsparse_mat_descr csr_descr_C,
                                                      const double*             csr_val_C,
                                                      const rocsparse_int*      csr_row_ptr_C,
                                                      const rocsparse_int*      csr_col_ind_C,
                                                      rocsparse_mat_info        info,
                                                      size_t*                   buffer_size)
{
    return rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle,
                                                              m,
                                                              n,
                                                              nnz_A,
                                                              csr_descr_A,
                                                              csr_val_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              percentage,
                                                              csr_descr_C,
                                                              csr_val_C,
                                                              csr_row_ptr_C,
                                                              csr_col_ind_C,
                                                              info,
                                                              buffer_size);
}

// prune_csr2csr_nnz_by_percentage
template <>
rocsparse_status rocsparse_prune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           rocsparse_int             nnz_A,
                                                           const rocsparse_mat_descr csr_descr_A,
                                                           const float*              csr_val_A,
                                                           const rocsparse_int*      csr_row_ptr_A,
                                                           const rocsparse_int*      csr_col_ind_A,
                                                           float                     percentage,
                                                           const rocsparse_mat_descr csr_descr_C,
                                                           rocsparse_int*            csr_row_ptr_C,
                                                           rocsparse_int* nnz_total_dev_host_ptr,
                                                           rocsparse_mat_info info,
                                                           void*              temp_buffer)
{
    return rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                      m,
                                                      n,
                                                      nnz_A,
                                                      csr_descr_A,
                                                      csr_val_A,
                                                      csr_row_ptr_A,
                                                      csr_col_ind_A,
                                                      percentage,
                                                      csr_descr_C,
                                                      csr_row_ptr_C,
                                                      nnz_total_dev_host_ptr,
                                                      info,
                                                      temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_csr2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           rocsparse_int             nnz_A,
                                                           const rocsparse_mat_descr csr_descr_A,
                                                           const double*             csr_val_A,
                                                           const rocsparse_int*      csr_row_ptr_A,
                                                           const rocsparse_int*      csr_col_ind_A,
                                                           double                    percentage,
                                                           const rocsparse_mat_descr csr_descr_C,
                                                           rocsparse_int*            csr_row_ptr_C,
                                                           rocsparse_int* nnz_total_dev_host_ptr,
                                                           rocsparse_mat_info info,
                                                           void*              temp_buffer)
{
    return rocsparse_dprune_csr2csr_nnz_by_percentage(handle,
                                                      m,
                                                      n,
                                                      nnz_A,
                                                      csr_descr_A,
                                                      csr_val_A,
                                                      csr_row_ptr_A,
                                                      csr_col_ind_A,
                                                      percentage,
                                                      csr_descr_C,
                                                      csr_row_ptr_C,
                                                      nnz_total_dev_host_ptr,
                                                      info,
                                                      temp_buffer);
}

// prune_csr2csr
template <>
rocsparse_status rocsparse_prune_csr2csr_by_percentage(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const float*              csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       float                     percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       float*                    csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       rocsparse_int*            csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       void*                     temp_buffer)
{
    return rocsparse_sprune_csr2csr_by_percentage(handle,
                                                  m,
                                                  n,
                                                  nnz_A,
                                                  csr_descr_A,
                                                  csr_val_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  percentage,
                                                  csr_descr_C,
                                                  csr_val_C,
                                                  csr_row_ptr_C,
                                                  csr_col_ind_C,
                                                  info,
                                                  temp_buffer);
}

template <>
rocsparse_status rocsparse_prune_csr2csr_by_percentage(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       rocsparse_int             nnz_A,
                                                       const rocsparse_mat_descr csr_descr_A,
                                                       const double*             csr_val_A,
                                                       const rocsparse_int*      csr_row_ptr_A,
                                                       const rocsparse_int*      csr_col_ind_A,
                                                       double                    percentage,
                                                       const rocsparse_mat_descr csr_descr_C,
                                                       double*                   csr_val_C,
                                                       const rocsparse_int*      csr_row_ptr_C,
                                                       rocsparse_int*            csr_col_ind_C,
                                                       rocsparse_mat_info        info,
                                                       void*                     temp_buffer)
{
    return rocsparse_dprune_csr2csr_by_percentage(handle,
                                                  m,
                                                  n,
                                                  nnz_A,
                                                  csr_descr_A,
                                                  csr_val_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  percentage,
                                                  csr_descr_C,
                                                  csr_val_C,
                                                  csr_row_ptr_C,
                                                  csr_col_ind_C,
                                                  info,
                                                  temp_buffer);
}

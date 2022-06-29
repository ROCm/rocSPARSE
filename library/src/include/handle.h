/*! \file */
/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef HANDLE_H
#define HANDLE_H

#include "rocsparse.h"

#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

/*! \brief typedefs to opaque info structs */
typedef struct _rocsparse_trm_info*     rocsparse_trm_info;
typedef struct _rocsparse_csrmv_info*   rocsparse_csrmv_info;
typedef struct _rocsparse_csrgemm_info* rocsparse_csrgemm_info;

/********************************************************************************
 * \brief rocsparse_handle is a structure holding the rocsparse library context.
 * It must be initialized using rocsparse_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparse_destroy_handle().
 *******************************************************************************/
struct _rocsparse_handle
{
    // constructor
    _rocsparse_handle();
    // destructor
    ~_rocsparse_handle();

    // set stream
    rocsparse_status set_stream(hipStream_t user_stream);
    // get stream
    rocsparse_status get_stream(hipStream_t* user_stream) const;

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    int wavefront_size;
    // asic revision
    int asic_rev;
    // stream ; default stream is system stream NULL
    hipStream_t stream = 0;
    // pointer mode ; default mode is host
    rocsparse_pointer_mode pointer_mode = rocsparse_pointer_mode_host;
    // logging mode
    rocsparse_layer_mode layer_mode;
    // device buffer
    size_t buffer_size;
    void*  buffer;
    // device one
    float*  sone;
    double* done;
    // device complex one
    rocsparse_float_complex*  cone;
    rocsparse_double_complex* zone;

    // logging streams
    std::ofstream log_trace_ofs;
    std::ofstream log_bench_ofs;
    std::ofstream log_debug_ofs;
    std::ostream* log_trace_os{};
    std::ostream* log_bench_os{};
    std::ostream* log_debug_os{};
};

/********************************************************************************
 * \brief rocsparse_mat_descr is a structure holding the rocsparse matrix
 * descriptor. It must be initialized using rocsparse_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocsparse_destroy_mat_descr().
 *******************************************************************************/
struct _rocsparse_mat_descr
{
    // matrix type
    rocsparse_matrix_type type = rocsparse_matrix_type_general;
    // fill mode
    rocsparse_fill_mode fill_mode = rocsparse_fill_mode_lower;
    // diagonal type
    rocsparse_diag_type diag_type = rocsparse_diag_type_non_unit;
    // index base
    rocsparse_index_base base = rocsparse_index_base_zero;
    // storage mode
    rocsparse_storage_mode storage_mode = rocsparse_storage_mode_sorted;
};

/********************************************************************************
 * \brief rocsparse_hyb_mat is a structure holding the rocsparse HYB matrix.
 * It must be initialized using rocsparse_create_hyb_mat() and the returned
 * handle must be passed to all subsequent library function calls that involve
 * the HYB matrix.
 * It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *******************************************************************************/
struct _rocsparse_hyb_mat
{
    // num rows
    rocsparse_int m{};
    // num cols
    rocsparse_int n{};

    // partition type
    rocsparse_hyb_partition partition = rocsparse_hyb_partition_auto;

    // ELL matrix part
    rocsparse_int  ell_nnz{};
    rocsparse_int  ell_width{};
    rocsparse_int* ell_col_ind{};
    void*          ell_val{};

    // COO matrix part
    rocsparse_int  coo_nnz{};
    rocsparse_int* coo_row_ind{};
    rocsparse_int* coo_col_ind{};
    void*          coo_val{};

    rocsparse_datatype data_type_T = rocsparse_datatype_f32_r;
};

/********************************************************************************
 * \brief rocsparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_mat_info().
 *******************************************************************************/
struct _rocsparse_mat_info
{
    // info structs
    rocsparse_trm_info bsrsv_upper_info{};
    rocsparse_trm_info bsrsv_lower_info{};
    rocsparse_trm_info bsrsvt_upper_info{};
    rocsparse_trm_info bsrsvt_lower_info{};
    rocsparse_trm_info bsric0_info{};
    rocsparse_trm_info bsrilu0_info{};
    rocsparse_trm_info bsrsm_upper_info{};
    rocsparse_trm_info bsrsm_lower_info{};
    rocsparse_trm_info bsrsmt_upper_info{};
    rocsparse_trm_info bsrsmt_lower_info{};

    rocsparse_csrmv_info   csrmv_info{};
    rocsparse_trm_info     csric0_info{};
    rocsparse_trm_info     csrilu0_info{};
    rocsparse_trm_info     csrsv_upper_info{};
    rocsparse_trm_info     csrsv_lower_info{};
    rocsparse_trm_info     csrsvt_upper_info{};
    rocsparse_trm_info     csrsvt_lower_info{};
    rocsparse_trm_info     csrsm_upper_info{};
    rocsparse_trm_info     csrsm_lower_info{};
    rocsparse_trm_info     csrsmt_upper_info{};
    rocsparse_trm_info     csrsmt_lower_info{};
    rocsparse_csrgemm_info csrgemm_info{};

    // zero pivot for csrsv, csrsm, csrilu0, csric0
    void* zero_pivot{};

    // numeric boost for ilu0
    int         boost_enable{};
    int         use_double_prec_tol{};
    const void* boost_tol{};
    const void* boost_val{};
};

/********************************************************************************
 * \brief rocsparse_color_info is a structure holding the color info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_color_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_color_info().
 *******************************************************************************/
struct _rocsparse_color_info
{
};

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * rocsparse_create_csrmv_info() routine. It should be destroyed at the end
 * rocsparse_destroy_csrmv_info().
 *******************************************************************************/
struct _rocsparse_csrmv_info
{
    // num row blocks
    size_t size{};
    // row blocks
    void*         row_blocks{};
    unsigned int* wg_flags{};
    void*         wg_ids{};

    // some data to verify correct execution
    rocsparse_operation         trans = rocsparse_operation_none;
    int64_t                     m{};
    int64_t                     n{};
    int64_t                     nnz{};
    int64_t                     max_rows{};
    const _rocsparse_mat_descr* descr{};
    const void*                 csr_row_ptr{};
    const void*                 csr_col_ind{};

    rocsparse_indextype index_type_I = rocsparse_indextype_u16;
    rocsparse_indextype index_type_J = rocsparse_indextype_u16;
};

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * rocsparse_create_csrmv_info() routine. It should be destroyed at the end
 * using rocsparse_destroy_csrmv_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrmv_info(rocsparse_csrmv_info* info);

/********************************************************************************
 * \brief Copy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_csrmv_info(rocsparse_csrmv_info       dest,
                                           const rocsparse_csrmv_info src);

/********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrmv_info(rocsparse_csrmv_info info);

struct _rocsparse_trm_info
{
    // maximum non-zero entries per row
    int64_t max_nnz{};

    // device array to hold row permutation
    void* row_map{};
    // device array to hold pointer to diagonal entry
    void* trm_diag_ind{};
    // device pointers to hold transposed data
    void* trmt_perm{};
    void* trmt_row_ptr{};
    void* trmt_col_ind{};

    // some data to verify correct execution
    int64_t                     m{};
    int64_t                     nnz{};
    const _rocsparse_mat_descr* descr{};
    const void*                 trm_row_ptr{};
    const void*                 trm_col_ind{};

    rocsparse_indextype index_type_I = rocsparse_indextype_u16;
    rocsparse_indextype index_type_J = rocsparse_indextype_u16;
};

/********************************************************************************
 * \brief rocsparse_trm_info is a structure holding the rocsparse bsrsv, csrsv,
 * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
 * csrilu0_analysis and csric0_analysis. It must be initialized using the
 * rocsparse_create_trm_info() routine. It should be destroyed at the end
 * using rocsparse_destroy_trm_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_trm_info(rocsparse_trm_info* info);

/********************************************************************************
 * \brief Copy trm info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_trm_info(rocsparse_trm_info dest, const rocsparse_trm_info src);

/********************************************************************************
 * \brief Destroy trm info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_trm_info(rocsparse_trm_info info);

/********************************************************************************
 * \brief rocsparse_check_trm_shared checks if the given trm info structure
 * shares its meta data with another trm info structure.
 *******************************************************************************/
bool rocsparse_check_trm_shared(const rocsparse_mat_info info, rocsparse_trm_info trm);

/********************************************************************************
 * \brief rocsparse_csrgemm_info is a structure holding the rocsparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the rocsparse_create_csrgemm_info() routine. It should be destroyed at the
 * end using rocsparse_destroy_csrgemm_info().
 *******************************************************************************/
struct _rocsparse_csrgemm_info
{
    // Perform alpha * A * B
    bool mul = true;
    // Perform beta * D
    bool add = true;
};

/********************************************************************************
 * \brief rocsparse_csrgemm_info is a structure holding the rocsparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the rocsparse_create_csrgemm_info() routine. It should be destroyed at the
 * end using rocsparse_destroy_csrgemm_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrgemm_info(rocsparse_csrgemm_info* info);

/********************************************************************************
 * \brief Copy csrgemm info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_csrgemm_info(rocsparse_csrgemm_info       dest,
                                             const rocsparse_csrgemm_info src);

/********************************************************************************
 * \brief Destroy csrgemm info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrgemm_info(rocsparse_csrgemm_info info);

/********************************************************************************
 * \brief ELL format indexing
 *******************************************************************************/
#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

struct _rocsparse_spvec_descr
{
    bool init{};

    int64_t size{};
    int64_t nnz{};

    void* idx_data{};
    void* val_data{};

    rocsparse_indextype idx_type{};
    rocsparse_datatype  data_type{};

    rocsparse_index_base idx_base{};
};

struct _rocsparse_spmat_descr
{
    bool init{};
    bool analysed{};

    int64_t rows{};
    int64_t cols{};
    int64_t nnz{};

    void* row_data{};
    void* col_data{};
    void* ind_data{};
    void* val_data{};

    rocsparse_indextype row_type{};
    rocsparse_indextype col_type{};
    rocsparse_datatype  data_type{};

    rocsparse_index_base idx_base{};
    rocsparse_format     format{};

    rocsparse_mat_descr descr{};
    rocsparse_mat_info  info{};

    rocsparse_direction block_dir{};
    int64_t             block_dim{};
    int64_t             ell_cols{};
    int64_t             ell_width{};

    int64_t batch_count{};
    int64_t batch_stride{};
    int64_t offsets_batch_stride{};
    int64_t columns_values_batch_stride{};
};

struct _rocsparse_dnvec_descr
{
    bool init{};

    int64_t            size{};
    void*              values{};
    rocsparse_datatype data_type{};
};

struct _rocsparse_dnmat_descr
{
    bool init{};

    int64_t rows{};
    int64_t cols{};
    int64_t ld{};

    void* values{};

    rocsparse_datatype data_type{};
    rocsparse_order    order{};

    int64_t batch_count{};
    int64_t batch_stride{};
};

#endif // HANDLE_H

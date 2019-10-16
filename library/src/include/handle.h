/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
typedef struct _rocsparse_csrmv_info*   rocsparse_csrmv_info;
typedef struct _rocsparse_csrtr_info*   rocsparse_csrtr_info;
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
    std::ostream* log_trace_os = nullptr;
    std::ostream* log_bench_os = nullptr;
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
    rocsparse_int m = 0;
    // num cols
    rocsparse_int n = 0;

    // partition type
    rocsparse_hyb_partition partition = rocsparse_hyb_partition_auto;

    // ELL matrix part
    rocsparse_int  ell_nnz     = 0;
    rocsparse_int  ell_width   = 0;
    rocsparse_int* ell_col_ind = nullptr;
    void*          ell_val     = nullptr;

    // COO matrix part
    rocsparse_int  coo_nnz     = 0;
    rocsparse_int* coo_row_ind = nullptr;
    rocsparse_int* coo_col_ind = nullptr;
    void*          coo_val     = nullptr;
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
    rocsparse_csrmv_info   csrmv_info       = nullptr;
    rocsparse_csrtr_info   csrilu0_info     = nullptr;
    rocsparse_csrtr_info   csrsv_upper_info = nullptr;
    rocsparse_csrtr_info   csrsv_lower_info = nullptr;
    rocsparse_csrgemm_info csrgemm_info     = nullptr;
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
    size_t size = 0;
    // row blocks
    unsigned long long* row_blocks = nullptr;

    // some data to verify correct execution
    rocsparse_operation         trans;
    rocsparse_int               m;
    rocsparse_int               n;
    rocsparse_int               nnz;
    const _rocsparse_mat_descr* descr;
    const rocsparse_int*        csr_row_ptr;
    const rocsparse_int*        csr_col_ind;
};

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * rocsparse_create_csrmv_info() routine. It should be destroyed at the end
 * using rocsparse_destroy_csrmv_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrmv_info(rocsparse_csrmv_info* info);

/********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrmv_info(rocsparse_csrmv_info info);

struct _rocsparse_csrtr_info
{
    // maximum non-zero entries per row
    rocsparse_int max_nnz = 0;

    // device array to hold row permutation
    rocsparse_int* row_map = nullptr;
    // device array to hold pointer to diagonal entry
    rocsparse_int* csr_diag_ind = nullptr;
    // device pointer to hold zero pivot
    rocsparse_int* zero_pivot = nullptr;

    // some data to verify correct execution
    rocsparse_int               m;
    rocsparse_int               nnz;
    const _rocsparse_mat_descr* descr;
    const rocsparse_int*        csr_row_ptr;
    const rocsparse_int*        csr_col_ind;
};

/********************************************************************************
 * \brief rocsparse_csrtr_info is a structure holding the rocsparse csrsv and
 * csrilu0 data gathered during csrsv_analysis and csrilu0_analysis. It must be
 * initialized using the rocsparse_create_csrtr_info() routine. It should be
 * destroyed at the end using rocsparse_destroy_csrtr_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_csrtr_info(rocsparse_csrtr_info* info);

/********************************************************************************
 * \brief Destroy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrtr_info(rocsparse_csrtr_info info);

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
 * \brief Destroy csrgemm info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_csrgemm_info(rocsparse_csrgemm_info info);

/********************************************************************************
 * \brief ELL format indexing
 *******************************************************************************/
#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

#endif // HANDLE_H

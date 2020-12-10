/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "testing.hpp"

void testing_spmat_descr_bad_arg(const Arguments& arg)
{
    int64_t rows      = 100;
    int64_t cols      = 100;
    int64_t nnz       = 100;
    int64_t ell_width = 100;

    rocsparse_indextype  itype  = rocsparse_indextype_i32;
    rocsparse_indextype  jtype  = rocsparse_indextype_i32;
    rocsparse_datatype   ttype  = rocsparse_datatype_f32_r;
    rocsparse_index_base base   = rocsparse_index_base_zero;
    rocsparse_format     format = rocsparse_format_csr;

    // Allocate memory on device
    device_vector<int>   row_data(100);
    device_vector<int>   col_data(100);
    device_vector<float> val_data(100);

    if(!row_data || !col_data || !val_data)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    rocsparse_spmat_descr A;

    // rocsparse_create_coo_descr
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            nullptr, rows, cols, nnz, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &A, -1, cols, nnz, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &A, rows, -1, nnz, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &A, rows, cols, -1, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &A, rows, cols, nnz, nullptr, col_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &A, rows, cols, nnz, row_data, nullptr, val_data, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &A, rows, cols, nnz, row_data, col_data, nullptr, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &A, 100, 100, 20000, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_invalid_size);

    // rocsparse_create_csr_descr
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            nullptr, rows, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, -1, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, -1, nnz, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, cols, -1, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, cols, nnz, nullptr, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, cols, nnz, row_data, nullptr, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, cols, nnz, row_data, col_data, nullptr, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, 100, 100, 20000, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, cols, 0, nullptr, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);

    // rocsparse_create_csc_descr
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            nullptr, rows, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, -1, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, -1, nnz, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, cols, -1, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, cols, nnz, nullptr, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, cols, nnz, col_data, nullptr, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, cols, nnz, col_data, row_data, nullptr, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, 100, 100, 20000, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, cols, 0, nullptr, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_invalid_pointer);

    // rocsparse_create_ell_descr
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(
            nullptr, rows, cols, col_data, val_data, ell_width, itype, base, ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(&A, -1, cols, col_data, val_data, ell_width, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(&A, rows, -1, col_data, val_data, ell_width, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_ell_descr(
                                &A, rows, cols, nullptr, val_data, ell_width, itype, base, ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_ell_descr(
                                &A, rows, cols, col_data, nullptr, ell_width, itype, base, ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(&A, rows, cols, col_data, val_data, -1, itype, base, ttype),
        rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(&A, 100, 100, col_data, val_data, 200, itype, base, ttype),
        rocsparse_status_invalid_size);

    // rocsparse_destroy_spmat_descr
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(&A, 0, cols, 0, nullptr, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(&A, rows, 0, 0, nullptr, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_coo_descr(
                                &A, rows, cols, 0, nullptr, nullptr, nullptr, itype, base, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, 0, cols, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, 0, 0, row_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &A, rows, cols, 0, row_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, 0, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, 0, cols, 0, col_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &A, rows, cols, 0, col_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(&A, 0, cols, nullptr, nullptr, ell_width, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(&A, rows, 0, nullptr, nullptr, 0, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(&A, rows, cols, nullptr, nullptr, 0, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    // Create valid descriptor
    rocsparse_spmat_descr coo;
    rocsparse_spmat_descr csr;
    rocsparse_spmat_descr csc;
    rocsparse_spmat_descr ell;

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_coo_descr(
            &coo, rows, cols, nnz, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csr_descr(
            &csr, rows, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_csc_descr(
            &csc, rows, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_ell_descr(
            &ell, rows, cols, col_data, val_data, ell_width, itype, base, ttype),
        rocsparse_status_success);

    // rocsparse_coo_get
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(nullptr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              nullptr,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              nullptr,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              &cols,
                                              nullptr,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              nullptr,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              nullptr,
                                              (void**)&val_data,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              nullptr,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              nullptr,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              nullptr,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_get(coo,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &base,
                                              nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_csr_get
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(nullptr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              nullptr,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              nullptr,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              nullptr,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              nullptr,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              nullptr,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              nullptr,
                                              &itype,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              nullptr,
                                              &jtype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              nullptr,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              nullptr,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_get(csr,
                                              &rows,
                                              &cols,
                                              &nnz,
                                              (void**)&row_data,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &itype,
                                              &jtype,
                                              &base,
                                              nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_ell_get
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_get(nullptr,
                                              &rows,
                                              &cols,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &ell_width,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_get(ell,
                                              nullptr,
                                              &cols,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &ell_width,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_get(ell,
                                              &rows,
                                              nullptr,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &ell_width,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_ell_get(
            ell, &rows, &cols, nullptr, (void**)&val_data, &ell_width, &itype, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_ell_get(
            ell, &rows, &cols, (void**)&col_data, nullptr, &ell_width, &itype, &base, &ttype),
        rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_get(ell,
                                              &rows,
                                              &cols,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              nullptr,
                                              &itype,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_get(ell,
                                              &rows,
                                              &cols,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &ell_width,
                                              nullptr,
                                              &base,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_get(ell,
                                              &rows,
                                              &cols,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &ell_width,
                                              &itype,
                                              nullptr,
                                              &ttype),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_get(ell,
                                              &rows,
                                              &cols,
                                              (void**)&col_data,
                                              (void**)&val_data,
                                              &ell_width,
                                              &itype,
                                              &base,
                                              nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_coo_set_pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_pointers(nullptr, row_data, col_data, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_pointers(coo, nullptr, col_data, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_pointers(coo, row_data, nullptr, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_coo_set_pointers(coo, row_data, col_data, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_csr_set_pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_set_pointers(nullptr, row_data, col_data, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_set_pointers(csr, nullptr, col_data, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_set_pointers(csr, row_data, nullptr, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr_set_pointers(csr, row_data, col_data, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_csc_set_pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_csc_set_pointers(nullptr, col_data, row_data, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csc_set_pointers(csc, nullptr, row_data, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csc_set_pointers(csc, col_data, nullptr, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csc_set_pointers(csc, col_data, row_data, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_ell_set_pointers
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_set_pointers(nullptr, col_data, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_set_pointers(ell, nullptr, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell_set_pointers(ell, col_data, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_size
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(nullptr, &rows, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_format
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(nullptr, &format),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(coo, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_index_base
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(nullptr, &base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(coo, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_values(nullptr, (void**)&val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_values(coo, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_set_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_values(nullptr, val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_set_values(coo, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(coo), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(csc), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(ell), rocsparse_status_success);
}

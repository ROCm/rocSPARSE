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

#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename I, typename J, typename T>
void testing_spmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_spmat_descr A{};
    int64_t               rows      = safe_size;
    int64_t               cols      = safe_size;
    int64_t               nnz       = safe_size;
    int64_t               ell_width = safe_size;
    void*                 row_data  = (void*)0x4;
    void*                 col_data  = (void*)0x4;
    void*                 val_data  = (void*)0x4;
    rocsparse_index_base  base      = rocsparse_index_base_zero;
    rocsparse_format      format    = rocsparse_format_csr;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

#define PARAMS_CREATE_COO &A, rows, cols, nnz, row_data, col_data, val_data, itype, base, ttype
    auto_testing_bad_arg(rocsparse_create_coo_descr, PARAMS_CREATE_COO);
#undef PARAMS_CREATE_COO

#define PARAMS_CREATE_CSR \
    &A, rows, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype
    auto_testing_bad_arg(rocsparse_create_csr_descr, PARAMS_CREATE_CSR);
#undef PARAMS_CREATE_CSR

#define PARAMS_CREATE_CSC \
    &A, rows, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype
    auto_testing_bad_arg(rocsparse_create_csc_descr, PARAMS_CREATE_CSC);
#undef PARAMS_CREATE_CSC

#define PARAMS_CREATE_ELL &A, rows, cols, col_data, val_data, ell_width, itype, base, ttype
    auto_testing_bad_arg(rocsparse_create_ell_descr, PARAMS_CREATE_ELL);
#undef PARAMS_CREATE_ELL

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

    void** row_data_ptr = (void**)0x4;
    void** col_data_ptr = (void**)0x4;
    void** val_data_ptr = (void**)0x4;

#define PARAMS_GET_COO \
    A, &rows, &cols, &nnz, row_data_ptr, col_data_ptr, val_data_ptr, &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_coo_get, PARAMS_GET_COO);
#undef PARAMS_GET_COO

#define PARAMS_GET_CSR \
    A, &rows, &cols, &nnz, row_data_ptr, col_data_ptr, val_data_ptr, &itype, &jtype, &base, &ttype
    auto_testing_bad_arg(rocsparse_csr_get, PARAMS_GET_CSR);
#undef PARAMS_GET_CSR

#define PARAMS_GET_ELL \
    A, &rows, &cols, col_data_ptr, val_data_ptr, &ell_width, &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_ell_get, PARAMS_GET_ELL);
#undef PARAMS_GET_ELL

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

#define INSTANTIATE(ITYPE, JTYPE, TTYPE) \
    template void testing_spmat_descr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

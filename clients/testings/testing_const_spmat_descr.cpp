/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename J, typename T>
void testing_const_spmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_const_spmat_descr A{};
    int64_t                     rows           = safe_size;
    int64_t                     cols           = safe_size;
    int64_t                     nnz            = safe_size;
    rocsparse_direction         bell_block_dir = rocsparse_direction_row;
    int64_t                     bell_block_dim = safe_size;
    int64_t                     bell_cols      = safe_size;
    const void*                 row_data       = (const void*)0x4;
    const void*                 col_data       = (const void*)0x4;
    const void*                 val_data       = (const void*)0x4;
    rocsparse_index_base        base           = rocsparse_index_base_zero;
    rocsparse_format            format         = rocsparse_format_csr;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

#define PARAMS_CREATE_COO &A, rows, cols, nnz, row_data, col_data, val_data, itype, base, ttype
    auto_testing_bad_arg(rocsparse_create_const_coo_descr, PARAMS_CREATE_COO);
#undef PARAMS_CREATE_COO

#define PARAMS_CREATE_CSR \
    &A, rows, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype
    auto_testing_bad_arg(rocsparse_create_const_csr_descr, PARAMS_CREATE_CSR);
#undef PARAMS_CREATE_CSR

#define PARAMS_CREATE_CSC \
    &A, rows, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype
    auto_testing_bad_arg(rocsparse_create_const_csc_descr, PARAMS_CREATE_CSC);
#undef PARAMS_CREATE_CSC

#define PARAMS_CREATE_ELL                                                                       \
    &A, rows, cols, bell_block_dir, bell_block_dim, bell_cols, col_data, val_data, itype, base, \
        ttype
    auto_testing_bad_arg(rocsparse_create_const_bell_descr, PARAMS_CREATE_ELL);
#undef PARAMS_CREATE_ELL

    // rocsparse_destroy_spmat_descr_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(nullptr),
                            rocsparse_status_invalid_pointer);

    // Check valid descriptor creations
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(
                                &A, 0, cols, 0, nullptr, nullptr, nullptr, itype, base, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(
                                &A, rows, 0, 0, nullptr, nullptr, nullptr, itype, base, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(
                                &A, rows, cols, 0, nullptr, nullptr, nullptr, itype, base, ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, 0, cols, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, rows, 0, 0, row_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, rows, cols, 0, row_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &A, rows, cols, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &A, rows, 0, 0, nullptr, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &A, 0, cols, 0, col_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &A, rows, cols, 0, col_data, nullptr, nullptr, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(&A,
                                                              0,
                                                              cols,
                                                              bell_block_dir,
                                                              bell_block_dim,
                                                              bell_cols,
                                                              nullptr,
                                                              nullptr,
                                                              itype,
                                                              base,
                                                              ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_bell_descr(
            &A, rows, 0, bell_block_dir, bell_block_dim, 0, nullptr, nullptr, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(&A,
                                                              rows,
                                                              cols,
                                                              bell_block_dir,
                                                              bell_block_dim,
                                                              0,
                                                              nullptr,
                                                              nullptr,
                                                              itype,
                                                              base,
                                                              ttype),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);

    // Create valid descriptor
    rocsparse_const_spmat_descr coo;
    rocsparse_const_spmat_descr csr;
    rocsparse_const_spmat_descr csc;
    rocsparse_const_spmat_descr ell;

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_coo_descr(
            &coo, rows, cols, nnz, row_data, col_data, val_data, itype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csr_descr(
            &csr, rows, cols, nnz, row_data, col_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(
        rocsparse_create_const_csc_descr(
            &csc, rows, cols, nnz, col_data, row_data, val_data, itype, jtype, base, ttype),
        rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(&ell,
                                                              rows,
                                                              cols,
                                                              bell_block_dir,
                                                              bell_block_dim,
                                                              bell_cols,
                                                              col_data,
                                                              val_data,
                                                              itype,
                                                              base,
                                                              ttype),
                            rocsparse_status_success);

    const void** row_data_ptr = (const void**)0x4;
    const void** col_data_ptr = (const void**)0x4;
    const void** val_data_ptr = (const void**)0x4;

#define PARAMS_GET_COO \
    A, &rows, &cols, &nnz, row_data_ptr, col_data_ptr, val_data_ptr, &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_const_coo_get, PARAMS_GET_COO);
#undef PARAMS_GET_COO

#define PARAMS_GET_CSR \
    A, &rows, &cols, &nnz, row_data_ptr, col_data_ptr, val_data_ptr, &itype, &jtype, &base, &ttype
    auto_testing_bad_arg(rocsparse_const_csr_get, PARAMS_GET_CSR);
#undef PARAMS_GET_CSR

#define PARAMS_GET_ELL                                                                         \
    A, &rows, &cols, &bell_block_dir, &bell_block_dim, &bell_cols, col_data_ptr, val_data_ptr, \
        &itype, &base, &ttype
    auto_testing_bad_arg(rocsparse_const_bell_get, PARAMS_GET_ELL);
#undef PARAMS_GET_ELL

    // rocsparse_spmat_get_size_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(nullptr, &rows, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(coo, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csr, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csr, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csr, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csc, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csc, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(csc, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(ell, nullptr, &cols, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(ell, &rows, nullptr, &nnz),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_size(ell, &rows, &cols, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_format_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(nullptr, &format),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_format(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_spmat_get_index_base_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(nullptr, &base),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_index_base(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    // rocsparse_const_spmat_get_values
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(nullptr, (const void**)&val_data),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_const_spmat_get_values(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    int batch_count = safe_size;

    // rocsparse_spmat_get_strided_batch_ex
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(nullptr, &batch_count),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(coo, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(csr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(csc, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spmat_get_strided_batch(ell, nullptr),
                            rocsparse_status_invalid_pointer);

    // Destroy valid descriptors
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(coo), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(csc), rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(ell), rocsparse_status_success);
}
template <typename I, typename J, typename T>
void testing_const_spmat_descr(const Arguments& arg)
{
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                        \
    template void testing_const_spmat_descr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_const_spmat_descr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
void testing_const_spmat_descr_extra(const Arguments& arg) {}
